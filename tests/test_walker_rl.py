"""Focused tests for the Walker2d PPO helper (stdlib unittest only).

1. Import has no side effects (no env, no directories).
2. `make_env` returns the Walker2d-v5 observation/action shapes.
3. `evaluate_policy` contract: deterministic random baseline, JSON output,
   population mean/std, input validation before env creation, deterministic
   `predict` calls, and env closure when the policy raises.
"""

import importlib
import json
import sys
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

import walker_rl

_REAL_MAKE_ENV = walker_rl.make_env


class _TrackedEnv:
    """Wraps a real env and records whether close() was called."""

    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.closed = False

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self.closed = True
        self._env.close()


class _RaisingPolicy:
    def predict(self, obs, deterministic=True):
        raise RuntimeError("predict exploded")


class _RecordingPolicy:
    """Returns a zero action for every observation and records the calls."""

    def __init__(self):
        self.calls = []

    def predict(self, obs, deterministic=True):
        self.calls.append(deterministic)
        return np.zeros(6, dtype=np.float32), None


def _tracked_make_env(holder):
    """Return a make_env stand-in that records every env it hands out."""

    def factory(seed=None, render_mode=None):
        env = _TrackedEnv(_REAL_MAKE_ENV(seed=seed, render_mode=render_mode))
        holder.append(env)
        return env

    return factory


class TestImportSideEffects(unittest.TestCase):
    def test_import_creates_no_env_and_no_directories(self):
        sys.modules.pop("walker_rl", None)
        with mock.patch.object(gym, "make") as gym_make, mock.patch(
            "os.makedirs"
        ) as makedirs, mock.patch("pathlib.Path.mkdir") as mkdir:
            importlib.import_module("walker_rl")
        gym_make.assert_not_called()
        makedirs.assert_not_called()
        mkdir.assert_not_called()


class TestMakeEnv(unittest.TestCase):
    def test_env_id_and_shapes(self):
        from walker_rl import ENV_ID, make_env

        self.assertEqual(ENV_ID, "Walker2d-v5")
        env = make_env()
        try:
            self.assertEqual(env.observation_space.shape, (17,))
            self.assertEqual(env.action_space.shape, (6,))
        finally:
            env.close()

    def test_seed_seeds_action_space_only(self):
        from walker_rl import make_env

        env = make_env(seed=123)
        try:
            first = env.action_space.sample()
        finally:
            env.close()
        env_again = make_env(seed=123)
        try:
            second = env_again.action_space.sample()
        finally:
            env_again.close()
        np.testing.assert_array_equal(first, second)


class TestEvaluatePolicy(unittest.TestCase):
    def test_random_evaluation_is_deterministic_for_same_seeds(self):
        from walker_rl import evaluate_policy

        first = evaluate_policy(None, episode_seeds=[0, 1], random_seed=7)
        second = evaluate_policy(None, episode_seeds=[0, 1], random_seed=7)
        self.assertEqual(first, second)

    def test_result_is_json_serializable_and_stats_match_numpy(self):
        from walker_rl import evaluate_policy

        result = evaluate_policy(None, episode_seeds=[0, 1, 2], random_seed=3)
        payload = json.loads(json.dumps(result))
        self.assertEqual(payload, result)

        returns = np.asarray([ep["return"] for ep in result["episodes"]])
        lengths = np.asarray([ep["length"] for ep in result["episodes"]])
        self.assertEqual(result["num_episodes"], 3)
        self.assertEqual(len(result["episodes"]), 3)
        self.assertAlmostEqual(result["metrics"]["return"]["mean"], float(np.mean(returns)))
        self.assertAlmostEqual(result["metrics"]["return"]["std"], float(np.std(returns)))
        self.assertAlmostEqual(result["metrics"]["length"]["mean"], float(np.mean(lengths)))
        self.assertAlmostEqual(result["metrics"]["length"]["std"], float(np.std(lengths)))

        for episode in result["episodes"]:
            self.assertIs(type(episode["seed"]), int)
            self.assertIs(type(episode["return"]), float)
            self.assertIs(type(episode["length"]), int)
            self.assertIs(type(episode["terminated"]), bool)
            self.assertIs(type(episode["truncated"]), bool)

    def test_invalid_inputs_raise_before_env_creation(self):
        from walker_rl import evaluate_policy

        cases = [
            ({"episode_seeds": [], "random_seed": 0}, ValueError),
            ({"episode_seeds": [-1], "random_seed": 0}, ValueError),
            ({"episode_seeds": [0], "random_seed": None}, ValueError),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                with mock.patch("walker_rl.make_env") as make_env_mock:
                    with self.assertRaises(expected):
                        evaluate_policy(None, **kwargs)
                make_env_mock.assert_not_called()

    def test_predict_called_with_deterministic_true(self):
        from walker_rl import evaluate_policy

        policy = _RecordingPolicy()
        result = evaluate_policy(policy, episode_seeds=[0], random_seed=None)
        self.assertGreater(len(policy.calls), 0)
        self.assertTrue(all(policy.calls))
        self.assertEqual(result["num_episodes"], 1)

    def test_env_closed_when_policy_raises(self):
        from walker_rl import evaluate_policy

        holder = []
        with mock.patch("walker_rl.make_env", side_effect=_tracked_make_env(holder)):
            with self.assertRaises(RuntimeError):
                evaluate_policy(_RaisingPolicy(), episode_seeds=[0])
        self.assertEqual(len(holder), 1)
        self.assertTrue(holder[0].closed)


if __name__ == "__main__":
    unittest.main()
