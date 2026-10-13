"""Small testable helper for the Walker2d PPO experiment.

Independent of Jupyter: importing this module never creates an environment
and never creates directories. The notebook builds on top of it.

Environment contract
--------------------
`make_env()` builds `Walker2d-v5` (Gymnasium MuJoCo) with the default Gymnasium
configuration. Observations are float64 vectors of shape (17,) and actions
are float32 vectors of shape (6,) in [-1, 1]. Rewards are raw, never
normalized or clipped here.

Evaluation contract
-------------------
`evaluate_policy()` runs held-out episodes and returns plain
JSON-serializable per-episode records and summary statistics. `policy=None`
samples uniform random actions from the real action-space bounds with a local
NumPy generator (never the global RNG). Otherwise the policy is called as
`policy.predict(obs, deterministic=True)`, optionally on observations
normalized by a frozen `VecNormalize`. The environment is always closed.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ENV_ID = "Walker2d-v5"


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def make_env(seed: int | None = None, render_mode: str | None = None):
    """Build a Walker2d-v5 environment.

    When `seed` is given, only the action space RNG is seeded here; the
    environment itself is seeded by `env.reset(seed=...)` by the caller.
    """
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None:
        try:
            env.action_space.seed(seed)
        except BaseException:
            env.close()
            raise
    return env


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _is_plain_int(value) -> bool:
    """True for Python or NumPy integers, excluding booleans."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_eval_args(policy, episode_seeds, random_seed, vecnormalize):
    """Validate evaluation inputs *before* any environment is created.

    Returns the episode seeds as a list of plain ints.
    """
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
        if not _is_plain_int(seed):
            raise ValueError(f"episode seed must be an int, got {seed!r}")
        if seed < 0:
            raise ValueError(f"episode seed must be non-negative, got {seed}")

    if random_seed is not None:
        if not _is_plain_int(random_seed):
            raise ValueError(f"random_seed must be an int or None, got {random_seed!r}")
        if random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {random_seed}")

    if policy is None:
        if random_seed is None:
            raise ValueError("random_seed is required when policy is None")
        if vecnormalize is not None:
            raise ValueError("vecnormalize requires a policy; random actions ignore it")
    elif not callable(getattr(policy, "predict", None)):
        raise TypeError(
            "policy must be None (random) or expose predict(obs, deterministic=True)"
        )

    if vecnormalize is not None and not callable(
        getattr(vecnormalize, "normalize_obs", None)
    ):
        raise TypeError("vecnormalize must expose normalize_obs(obs)")

    return [int(seed) for seed in seeds]


def _summarize(values) -> dict[str, float]:
    """Population mean and standard deviation as plain floats."""
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def evaluate_policy(
    policy,
    episode_seeds,
    random_seed: int | None = None,
    vecnormalize=None,
) -> dict:
    """Evaluate a policy over held-out episodes and return JSON-ready metrics.

    Parameters
    ----------
    policy : None for uniform random actions, otherwise an object exposing
        `predict(obs, deterministic=True)` (e.g. a Stable-Baselines3 PPO).
    episode_seeds : iterable of non-negative ints, one reset seed per episode.
    random_seed : seed for the local NumPy RNG used by the random policy.
        Required when `policy` is None.
    vecnormalize : optional frozen `VecNormalize`; its `normalize_obs` is
        applied to observations before `predict`. Rewards stay raw.

    Returns a dict of plain ints, floats and bools suitable for `json.dumps`.
    The environment is always closed, including when the policy raises.
    """
    seeds = _validate_eval_args(policy, episode_seeds, random_seed, vecnormalize)

    rng = np.random.default_rng(random_seed) if policy is None else None
    env = make_env()
    episodes = []
    try:
        low = env.action_space.low
        high = env.action_space.high
        for episode_seed in seeds:
            obs, _ = env.reset(seed=episode_seed)
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                if policy is None:
                    action = rng.uniform(low, high).astype(env.action_space.dtype)
                else:
                    policy_obs = (
                        obs if vecnormalize is None else vecnormalize.normalize_obs(obs)
                    )
                    action = policy.predict(policy_obs, deterministic=True)[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
            episodes.append(
                {
                    "seed": int(episode_seed),
                    "return": float(episode_return),
                    "length": int(episode_length),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    finally:
        env.close()

    return {
        "num_episodes": len(episodes),
        "metrics": {
            "return": _summarize([episode["return"] for episode in episodes]),
            "length": _summarize([episode["length"] for episode in episodes]),
        },
        "episodes": episodes,
    }
