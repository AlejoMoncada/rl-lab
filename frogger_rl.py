"""Small testable helper for the Frogger PPO project.

Independent of Jupyter: importing this module never creates directories and
never instantiates an emulator. The notebook (F2) builds on top of it; the
focused unit tests exercise the wrapper configuration and the evaluation
contract without running ALE.

Environment contract (F1)
-------------------------
`make_env()` builds `ALE/Frogger-v5` with the canonical Gymnasium Atari
recipe:

* base env: frameskip=1, repeat_action_probability=0.25  (sticky actions,
  configured in exactly this one layer),
* one `AtariPreprocessing(frame_skip=4, screen_size=84, grayscale_obs=True,
  terminal_on_life_loss=False, scale_obs=False)`  -> uint8 (84, 84),
* `FrameStackObservation(stack_size=4)`                    -> uint8 (4, 84, 84),
  CNN channel-first,
* explicit `TimeLimit(max_episode_steps=...)` measured in *agent steps*
  (the outermost wrapper, so every env.step() the agent issues counts once).

Rewards are the raw ALE rewards: never clipped, never shaped.

Evaluation contract (F1)
------------------------
`evaluate_policy()` runs held-out episodes and returns plain JSON-serializable
metrics. `policy=None` samples random actions from the *actual* action-space
size with a local numpy RNG (never the global RNG); a PPO-like policy is
called as `policy.predict(obs, deterministic=True)`. The environment is
always closed, on success and on exceptions.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import error as gym_errors
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ENV_ID = "ALE/Frogger-v5"

# ALE minimal action set for Frogger (verified against
# env.unwrapped.get_action_meanings()): five indices, 0..4.
ACTION_NAMES = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"]

# Base configuration: ALE created with frameskip=1; the four-frame repeat is
# applied exactly once, by AtariPreprocessing below.
BASE_FRAMESKIP = 1
STICKY_ACTION_PROBABILITY = 0.25

PREPROCESS_FRAME_SKIP = 4
SCREEN_SIZE = 84
STACK_SIZE = 4
OBS_SHAPE = (STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)

# Default evaluation horizon, measured in agent steps.
ENV_MAX_AGENT_STEPS = 2700


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def _make_base_env(render_mode: str | None):
    """Create the raw ALE env; register ale-py lazily when the first try fails."""
    # ale-py 0.12.x ships no gymnasium entry-point plugin, so the explicit
    # import feeds `gym.register_envs(ale_py)` in the retry below. It resolves
    # at runtime from this project's declared ale-py dependency, but pyright
    # (bound to the workspace-root venv, which lacks ale-py) cannot see it.
    import ale_py  # noqa: F401  # type: ignore[reportMissingImports]

    try:
        return gym.make(
            ENV_ID,
            frameskip=BASE_FRAMESKIP,
            repeat_action_probability=STICKY_ACTION_PROBABILITY,
            render_mode=render_mode,
        )
    except gym_errors.Error as first_error:
        # Not registered yet (e.g. older ale-py without plugin entry points).
        # If registration itself conflicts, report the ORIGINAL error.
        try:
            gym.register_envs(ale_py)
        except gym_errors.Error as registration_error:
            raise first_error from registration_error
        return gym.make(
            ENV_ID,
            frameskip=BASE_FRAMESKIP,
            repeat_action_probability=STICKY_ACTION_PROBABILITY,
            render_mode=render_mode,
        )


def build_env(base_env, *, max_episode_steps: int = ENV_MAX_AGENT_STEPS):
    """Apply the Atari wrappers over an already-created base env.

    The base env is closed if any wrapping step fails, so tests that inject a
    fake base can assert resource cleanup on failure.
    """
    if max_episode_steps <= 0:
        raise ValueError(
            f"max_episode_steps must be positive, got {max_episode_steps}"
        )
    try:
        env = AtariPreprocessing(
            base_env,
            frame_skip=PREPROCESS_FRAME_SKIP,
            screen_size=SCREEN_SIZE,
            grayscale_obs=True,
            terminal_on_life_loss=False,
            scale_obs=False,
        )
        env = FrameStackObservation(env, stack_size=STACK_SIZE)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return env
    except BaseException:
        base_env.close()
        raise


def make_env(*, max_episode_steps: int = ENV_MAX_AGENT_STEPS,
             render_mode: str | None = None):
    """Build the canonical Frogger environment.

    Sticky actions are configured in exactly one layer (the base ALE env);
    the frame repeat happens exactly once (AtariPreprocessing); the TimeLimit
    is measured in agent steps because it wraps the outermost env.
    """
    if max_episode_steps <= 0:
        raise ValueError(
            f"max_episode_steps must be positive, got {max_episode_steps}"
        )
    base = _make_base_env(render_mode=render_mode)
    try:
        return build_env(base, max_episode_steps=max_episode_steps)
    except BaseException:
        base.close()
        raise


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _validate_eval_args(episode_seeds, max_episode_steps, random_seed, policy):
    """Validate evaluation inputs *before* any environment is created."""
    if max_episode_steps <= 0:
        raise ValueError(
            f"max_episode_steps must be positive, got {max_episode_steps}"
        )
    if random_seed is not None:
        if not isinstance(random_seed, (int, np.integer)) or isinstance(
            random_seed, bool
        ):
            raise ValueError(
                f"random_seed must be an int or None, got {random_seed!r}"
            )
        if random_seed < 0:
            raise ValueError(
                f"random_seed must be non-negative, got {random_seed}"
            )
    if isinstance(episode_seeds, (str, bytes)):
        raise ValueError("episode_seeds must be an iterable of ints")
    try:
        seeds = list(episode_seeds)
    except TypeError as error:
        raise ValueError(
            f"episode_seeds must be an iterable of ints, got {episode_seeds!r}"
        ) from error
    if not seeds:
        raise ValueError("episode_seeds must contain at least one episode seed")
    for seed in seeds:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise ValueError(
                f"episode seed must be an int, got {seed!r}"
            )
        if seed < 0:
            raise ValueError(f"episode seed must be non-negative, got {seed}")
    if policy is not None and not callable(getattr(policy, "predict", None)):
        raise TypeError(
            "policy must be None (random) or expose predict(obs, deterministic=True)"
        )
    return [int(seed) for seed in seeds]


def _select_action(policy, obs, action_space, rng):
    """Pick the next action: random from the ACTUAL action-space size, or
    PPO-like `predict(obs, deterministic=True)`."""
    if policy is None:
        return int(rng.integers(action_space.n))
    actions, _ = policy.predict(obs, deterministic=True)
    return _as_int(actions)


def _as_int(value):
    """Normalize a single action returned by a policy to a plain int."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if np.size(value) != 1:
            raise ValueError(
                f"predict() must return a single action, got shape {np.shape(value)}"
            )
        value = value.reshape(-1)[0] if isinstance(value, np.ndarray) else value[0]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"cannot interpret action {value!r} as an int")
    return int(value)


def _summarize(episodes):
    """Aggregate per-episode records into plain-JSON statistics."""
    returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.float64)

    def _stats(values):
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "n": len(episodes),
        "return": _stats(returns),
        "length": _stats(lengths),
        "terminated_count": int(
            sum(bool(episode["terminated"]) for episode in episodes)
        ),
        "truncated_count": int(
            sum(bool(episode["truncated"]) for episode in episodes)
        ),
    }


def evaluate_policy(
    policy=None,
    *,
    episode_seeds,
    max_episode_steps: int = ENV_MAX_AGENT_STEPS,
    random_seed: int | None = None,
):
    """Evaluate a policy over held-out episodes and return JSON metrics.

    Parameters
    ----------
    policy : None -> random actions; otherwise an object exposing
        `predict(obs, deterministic=True)` (e.g. a Stable-Baselines3 PPO).
    episode_seeds : iterable of ints, one environment seed per episode.
        Each episode restarts the (single, reused) environment with that seed.
    max_episode_steps : TimeLimit cap in agent steps (default 2700).
    random_seed : seed for the local RNG that drives the random policy and
        per-episode action streams. Callers set training and evaluation seeds.

    Returns a dict of plain ints/floats/bools suitable for json.dumps:
    env_id, policy type, per-episode records (seed, raw return, agent-step
    length, terminated, truncated) and aggregated metrics with mean/std/
    min/max, counts and truncation count. The environment is always closed,
    also on exceptions; dependency/emulation errors are never swallowed.
    """
    seeds = _validate_eval_args(episode_seeds, max_episode_steps, random_seed, policy)
    env = make_env(max_episode_steps=max_episode_steps)
    episodes = []
    try:
        rng_source = np.random.SeedSequence(random_seed)
        for index, episode_seed in enumerate(seeds):
            obs, _ = env.reset(seed=episode_seed)
            # Local RNG per episode: reproducible, never the global RNG.
            policy_rng = np.random.default_rng(rng_source.spawn(1)[0])
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = _select_action(policy, obs, env.action_space, policy_rng)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if episode_length >= max_episode_steps and not (
                    terminated or truncated
                ):
                    # Defensive cap: the TimeLimit already handles this for
                    # make_env() environments; keep it for safety.
                    truncated = True
            episodes.append(
                {
                    "episode": index + 1,
                    "seed": int(episode_seed),
                    "return": float(episode_return),
                    "length": int(episode_length),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
        return {
            "env_id": ENV_ID,
            "policy": "random" if policy is None else "predict",
            "max_episode_steps": int(max_episode_steps),
            "random_seed": int(random_seed) if random_seed is not None else None,
            "num_episodes": len(episodes),
            "metrics": _summarize(episodes),
            "episodes": episodes,
        }
    finally:
        env.close()
