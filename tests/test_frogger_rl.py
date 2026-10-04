"""Focused tests for the Frogger PPO helper (stdlib unittest only).

Three groups:

1. Helper/platform facts (no emulator): ENV_ID, the five ALE action names.
2. `make_env` wrapper configuration with an injected fake base env:
   exact wrapper chain, AtariPreprocessing params, FrameStackObservation,
   agent-step TimeLimit, stickiness in exactly one layer, rejection of
   nonpositive caps, and base-env closure when wrapping fails.
3. Deterministic fake-env `evaluate_policy` contract: raw reward summation
   (never clipped), both end flags, bounded stop length, seeded local RNG,
   closure on predict/step failure, and validation of invalid input.
4. Real ALE/Frogger-v5 contract. This test must NOT silently skip: on
   missing dependencies it reports the environmental failure as an error.
"""

import json
import unittest
from typing import cast
from unittest import mock

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit

import frogger_rl


def _wrapper_chain(env: gym.Env) -> list[gym.Env]:
    """Outermost-first list of every layer, base env last (heterogeneous)."""
    layers: list[gym.Env] = [env]
    layer = env
    while isinstance(layer, gym.Wrapper):
        layer = layer.env
        layers.append(layer)
    return layers


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeBase(gym.Env):
    """Minimal duck-typed base env that can host the real wrappers."""

    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
        )
        self.metadata = {"render_modes": []}
        self.render_mode = None
        self.reward_range = (-float("inf"), float("inf"))
        # Emulates an ALE env created with frameskip=1 (AtariPreprocessing
        # rejects wrapping a base env whose _frameskip != 1: no double skip).
        self._frameskip = 1
        self.repeat_action_probability = 0.25
        self.called_close = False

    def get_action_meanings(self):
        # AtariPreprocessing(noop_max>0) asserts the first action is NOOP.
        return list(frogger_rl.ACTION_NAMES)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((210, 160, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((210, 160, 3), dtype=np.uint8), 0.0, False, False, {}

    def close(self):
        self.called_close = True


class _ScriptedFakeEnv:
    """Deterministic evaluation target driven only by the action sequence.

    Rewards are raw floats outside the usual [-1, 1] clip range so the tests
    can prove nothing is clipped or reshaped. Termination/truncation are
    scripted per step index (1-based); `fail_on_step_at` raises on purpose.
    """

    def __init__(
        self,
        rewards=(),
        term_after=None,
        trunc_after=None,
        fail_on_step_at=None,
    ):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.rewards = list(rewards)
        self.term_after = term_after
        self.trunc_after = trunc_after
        self.fail_on_step_at = fail_on_step_at
        self.obs = np.zeros((4, 84, 84), dtype=np.uint8)
        self.steps = 0
        self.actions = []
        self.seeds = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        self.steps = 0
        return self.obs, {}

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        if self.fail_on_step_at is not None and self.steps == self.fail_on_step_at:
            raise RuntimeError("fake env step failure")
        index = self.steps - 1
        reward = self.rewards[index] if index < len(self.rewards) else 0.0
        terminated = self.term_after is not None and self.steps >= self.term_after
        truncated = self.trunc_after is not None and self.steps >= self.trunc_after
        return self.obs, reward, terminated, truncated, {}

    def close(self):
        self.closed = True


class _StubPolicy:
    """PPO-like policy: returns `action` for every observation."""

    def __init__(self, action):
        self.action = action
        self.calls = []

    def predict(self, obs, deterministic=True):
        self.calls.append((obs, deterministic))
        return self.action, None


class _BrokenPolicy:
    def predict(self, obs, deterministic=True):
        raise RuntimeError("predict exploded")


def _patch_eval_env(fake_env):
    """Make evaluate_policy use `fake_env` instead of building a real ALE env."""
    return mock.patch.object(frogger_rl, "make_env", return_value=fake_env)


def _ale_action_meanings(env):
    """Real ALE action meanings for the minimal action set, when available."""
    unwrapped = env.unwrapped
    getter = getattr(unwrapped, "get_action_meanings", None)
    if not callable(getter):
        raise AssertionError(
            "real ALE env has no get_action_meanings(); cannot verify action mapping"
        )
    return list(cast(list, getter()))


# ---------------------------------------------------------------------------
# 1. Helper/platform facts
# ---------------------------------------------------------------------------

class TestPlatformFacts(unittest.TestCase):
    def test_env_id(self):
        self.assertEqual(frogger_rl.ENV_ID, "ALE/Frogger-v5")

    def test_action_names_are_five_ale_indices(self):
        self.assertEqual(len(frogger_rl.ACTION_NAMES), 5)
        self.assertEqual(
            frogger_rl.ACTION_NAMES, ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"]
        )

    def test_default_horizon_is_positive(self):
        self.assertGreater(frogger_rl.ENV_MAX_AGENT_STEPS, 0)


# ---------------------------------------------------------------------------
# 2. make_env wrapper configuration (fake base env)
# ---------------------------------------------------------------------------

class TestMakeEnvConfiguration(unittest.TestCase):
    def _build(self, cap=2700):
        fake = _FakeBase()
        with mock.patch.object(frogger_rl.gym, "make", return_value=fake) as gym_make:
            env = frogger_rl.make_env(max_episode_steps=cap)
        return env, fake, gym_make

    def test_wrapper_chain_and_parameters(self):
        env, fake, gym_make = self._build(cap=2700)
        try:
            layers = _wrapper_chain(env)
            time_limit = layers[0]
            frame_stack = layers[1]
            atari = layers[2]
            # Builtin isinstance() both keeps the runtime assertion and
            # narrows each layer to its concrete wrapper type statically.
            assert isinstance(
                time_limit, TimeLimit
            ), "outermost layer must be the TimeLimit"
            assert isinstance(
                frame_stack, FrameStackObservation
            ), "frame stack must sit over the AtariPreprocessing layer"
            assert isinstance(
                atari, AtariPreprocessing
            ), "AtariPreprocessing must sit over the fake base"
            self.assertIs(layers[3], fake)

            self.assertEqual(atari.frame_skip, 4)
            self.assertEqual(atari.screen_size, (84, 84))  # normalized to a tuple
            self.assertTrue(atari.grayscale_obs)
            self.assertFalse(atari.terminal_on_life_loss)
            self.assertFalse(atari.scale_obs)
            self.assertEqual(frame_stack.stack_size, 4)
            self.assertEqual(time_limit._max_episode_steps, 2700)

            # Base env created with frameskip=1 + sticky actions, render off.
            gym_make.assert_called_once_with(
                frogger_rl.ENV_ID,
                frameskip=1,
                repeat_action_probability=0.25,
                render_mode=None,
            )
        finally:
            env.close()

    def test_stickiness_configured_in_exactly_one_layer(self):
        env, fake, _ = self._build(cap=2700)
        try:
            sticky = [
                layer
                for layer in _wrapper_chain(env)
                if getattr(layer, "repeat_action_probability", None) is not None
            ]
            self.assertEqual(len(sticky), 1)
            sticky_layer = sticky[0]
            # The only configured sticky layer is the fake ALE base, which
            # declares repeat_action_probability on its class: the isinstance
            # is the static proof (no suppression needed).
            assert isinstance(
                sticky_layer, _FakeBase
            ), "the single sticky layer must be the fake base env"
            self.assertIs(sticky_layer, fake)
            self.assertEqual(sticky_layer.repeat_action_probability, 0.25)
        finally:
            env.close()

    def test_observation_space_is_channel_first_uint8(self):
        env, fake, _ = self._build(cap=2700)
        try:
            self.assertEqual(env.observation_space.shape, (4, 84, 84))
            self.assertEqual(env.observation_space.dtype, np.uint8)
        finally:
            env.close()

    def test_rejects_nonpositive_step_caps(self):
        for bad_cap in (0, -1, -2700):
            with self.subTest(cap=bad_cap):
                fake = _FakeBase()
                with mock.patch.object(
                    frogger_rl.gym, "make", return_value=fake
                ) as gym_make, self.assertRaises(ValueError):
                    frogger_rl.make_env(max_episode_steps=bad_cap)
                gym_make.assert_not_called()

    def test_base_env_closed_when_wrapping_fails(self):
        fake = _FakeBase()
        with (
            mock.patch.object(frogger_rl.gym, "make", return_value=fake),
            mock.patch.object(
                frogger_rl, "FrameStackObservation", side_effect=RuntimeError("boom")
            ),
            self.assertRaises(RuntimeError),
        ):
            frogger_rl.make_env(max_episode_steps=2700)
        self.assertTrue(fake.called_close)


# ---------------------------------------------------------------------------
# 3. evaluate_policy contract (fake env, deterministic)
# ---------------------------------------------------------------------------

class TestEvaluatePolicyFakeEnv(unittest.TestCase):
    def test_random_policy_is_deterministic_and_uses_action_space_n(self):
        rewards = [1.0] * 30
        actions_seen = []

        def run():
            fake = _ScriptedFakeEnv(rewards=rewards)
            with _patch_eval_env(fake):
                result = frogger_rl.evaluate_policy(
                    None,
                    episode_seeds=[1, 2],
                    max_episode_steps=30,
                    random_seed=42,
                )
            actions_seen.append(list(fake.actions))
            self.assertEqual(fake.seeds, [1, 2])
            return result

        first = run()
        second = run()
        self.assertEqual(actions_seen[0], actions_seen[1])
        self.assertTrue(actions_seen[0])
        self.assertLessEqual(max(actions_seen[0]), 3)  # from Discrete(4)
        self.assertGreaterEqual(min(actions_seen[0]), 0)
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_random_policy_uses_local_rng_not_global(self):
        def run():
            fake = _ScriptedFakeEnv(rewards=[float(a) for a in range(40)])
            with _patch_eval_env(fake):
                result = frogger_rl.evaluate_policy(
                    None,
                    episode_seeds=[7],
                    max_episode_steps=40,
                    random_seed=11,
                )
            return list(fake.actions), result

        np.random.seed(0)
        actions_a, result_a = run()
        np.random.seed(999)
        actions_b, result_b = run()
        self.assertEqual(actions_a, actions_b)
        self.assertEqual(result_a["metrics"]["return"]["mean"],
                         result_b["metrics"]["return"]["mean"])

    def test_raw_reward_summation_without_clipping(self):
        # Raw floats far outside [-1, 1]; clipping would destroy the total.
        fake = _ScriptedFakeEnv(
            rewards=[100.0, -3.5, 0.5, 2.0], term_after=4
        )
        with _patch_eval_env(fake):
            result = frogger_rl.evaluate_policy(
                None, episode_seeds=[5], random_seed=1
            )
        episode = result["episodes"][0]
        self.assertEqual(episode["return"], 99.0)
        self.assertEqual(episode["length"], 4)
        self.assertTrue(episode["terminated"])
        self.assertFalse(episode["truncated"])
        self.assertEqual(result["metrics"]["terminated_count"], 1)
        self.assertEqual(result["metrics"]["truncated_count"], 0)

    def test_truncated_end_flag_and_count(self):
        fake = _ScriptedFakeEnv(rewards=[1.0] * 10, trunc_after=7)
        with _patch_eval_env(fake):
            result = frogger_rl.evaluate_policy(
                None, episode_seeds=[11], random_seed=3
            )
        episode = result["episodes"][0]
        self.assertTrue(episode["truncated"])
        self.assertFalse(episode["terminated"])
        self.assertEqual(episode["length"], 7)
        self.assertEqual(result["metrics"]["truncated_count"], 1)

    def test_stop_length_when_never_done_hits_cap(self):
        fake = _ScriptedFakeEnv(rewards=[1.5] * 100)
        with _patch_eval_env(fake):
            result = frogger_rl.evaluate_policy(
                None, episode_seeds=[1], max_episode_steps=5, random_seed=3
            )
        episode = result["episodes"][0]
        self.assertEqual(episode["length"], 5)
        self.assertTrue(episode["truncated"])
        self.assertFalse(episode["terminated"])
        self.assertEqual(episode["return"], 7.5)

    def test_predict_policy_single_and_array_actions(self):
        for action in (2, np.array([3]), np.int64(1)):
            with self.subTest(action=action):
                fake = _ScriptedFakeEnv(
                    rewards=[1.0] * 5, term_after=5
                )
                policy = _StubPolicy(action)
                with _patch_eval_env(fake):
                    result = frogger_rl.evaluate_policy(
                        policy,
                        episode_seeds=[21],
                        max_episode_steps=5,
                    )
                self.assertEqual(result["policy"], "predict")
                self.assertEqual(fake.actions, [int(np.asarray(action).reshape(-1)[0] if np.size(action) else action)] * 5)
                self.assertEqual(result["episodes"][0]["return"], 5.0)
                self.assertTrue(all(call[1] for call in policy.calls))  # deterministic=True

    def test_env_closed_when_predict_fails(self):
        fake = _ScriptedFakeEnv(rewards=[1.0] * 10)
        with self.assertRaises(RuntimeError), _patch_eval_env(fake):
            frogger_rl.evaluate_policy(
                _BrokenPolicy(), episode_seeds=[1], max_episode_steps=5
            )
        self.assertTrue(fake.closed)

    def test_env_closed_when_step_fails(self):
        fake = _ScriptedFakeEnv(rewards=[1.0] * 10, fail_on_step_at=3)
        with self.assertRaises(RuntimeError), _patch_eval_env(fake):
            frogger_rl.evaluate_policy(
                None, episode_seeds=[1], max_episode_steps=5, random_seed=0
            )
        self.assertTrue(fake.closed)

    def test_invalid_inputs_rejected_before_env_creation(self):
        cases = [
            ({"episode_seeds": []}, ValueError),
            ({"episode_seeds": 0}, ValueError),
            ({"episode_seeds": ["a"]}, ValueError),
            ({"episode_seeds": [-1]}, ValueError),
            ({"episode_seeds": [1], "max_episode_steps": 0}, ValueError),
            ({"episode_seeds": [1], "max_episode_steps": -5}, ValueError),
            ({"episode_seeds": [1], "random_seed": "x"}, ValueError),
            ({"episode_seeds": [1], "policy": object()}, TypeError),
        ]
        for kwargs, expected in cases:
            with (
                self.subTest(kwargs=kwargs, expected=expected),
                mock.patch.object(frogger_rl, "make_env") as make_env_mock,
                self.assertRaises(expected),
            ):
                frogger_rl.evaluate_policy(**kwargs)
            make_env_mock.assert_not_called()

    def test_result_is_json_serializable(self):
        fake = _ScriptedFakeEnv(rewards=[2.0] * 8, trunc_after=8)
        with _patch_eval_env(fake):
            result = frogger_rl.evaluate_policy(
                None,
                episode_seeds=[1, 2, 3],
                max_episode_steps=8,
                random_seed=4,
            )
        payload = json.dumps(result)
        self.assertIn('"env_id": "ALE/Frogger-v5"', payload)
        self.assertEqual(len(result["episodes"]), 3)
        self.assertEqual(result["num_episodes"], 3)
        self.assertEqual(result["metrics"]["n"], 3)
        self.assertEqual(result["metrics"]["truncated_count"], 3)


# ---------------------------------------------------------------------------
# 4. Real ALE contract (must not silently skip)
# ---------------------------------------------------------------------------

class TestRealAleEnvironment(unittest.TestCase):
    def test_real_ale_reset_step_shape_and_action_mapping(self):
        env = frogger_rl.make_env(max_episode_steps=20)
        try:
            obs, info = env.reset(seed=1234)
            self.assertEqual(obs.shape, (4, 84, 84))
            self.assertEqual(obs.dtype, np.uint8)
            self.assertGreaterEqual(int(obs.min()), 0)
            self.assertLessEqual(int(obs.max()), 255)
            action_space = env.action_space
            assert isinstance(
                action_space, spaces.Discrete
            ), "the ALE action space must be Discrete"
            self.assertEqual(action_space.n, 5)

            # ACTION_NAMES must match the real ALE minimal action set.
            self.assertEqual(frogger_rl.ACTION_NAMES, _ale_action_meanings(env))

            # Every one of the five ALE action indices is accepted.
            for action in range(5):
                with self.subTest(action=action):
                    env.reset(seed=9000 + action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    self.assertEqual(obs.shape, (4, 84, 84))
                    self.assertEqual(obs.dtype, np.uint8)
                    self.assertIsInstance(float(reward), float)
                    self.assertIsInstance(bool(terminated), bool)
                    self.assertIsInstance(bool(truncated), bool)
        finally:
            env.close()

    def test_time_limit_bounds_episode_in_agent_steps(self):
        cap = 20
        env = frogger_rl.make_env(max_episode_steps=cap)
        try:
            env.reset(seed=4321)
            steps = 0
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(0)  # NOOP
                total += float(reward)
                steps += 1
                self.assertEqual(obs.shape, (4, 84, 84))
                self.assertEqual(obs.dtype, np.uint8)
                self.assertLessEqual(steps, cap + 1)  # TimeLimit must bound us
            self.assertEqual(steps, cap)
            self.assertTrue(truncated)
            self.assertFalse(terminated)
            # Raw rewards pass through untouched (NOOP -> 0): nothing clipped.
            self.assertEqual(total, 0.0)
        finally:
            env.close()

    def test_seeded_resets_are_reproducible(self):
        env_a = frogger_rl.make_env(max_episode_steps=50)
        env_b = frogger_rl.make_env(max_episode_steps=50)
        try:
            obs_a, _ = env_a.reset(seed=99)
            obs_b, _ = env_b.reset(seed=99)
            np.testing.assert_array_equal(obs_a, obs_b)
        finally:
            env_a.close()
            env_b.close()


if __name__ == "__main__":
    unittest.main()
