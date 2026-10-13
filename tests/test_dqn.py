"""Pruebas del DQN escrito a mano: buffer, formas de la CNN y el paso TD.

No arrancan el emulador. El contrato de ALE sigue en test_frogger_rl.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import dqn_frogger
import frogger_rl


class TestReplayBuffer(unittest.TestCase):
    def test_sample_shapes_and_raw_reward(self):
        buffer = dqn_frogger.ReplayBuffer(capacity=8, obs_shape=(4, 84, 84))
        obs = np.zeros((4, 84, 84), dtype=np.uint8)
        nxt = np.full((4, 84, 84), 7, dtype=np.uint8)
        buffer.add(obs, 3, 100.0, nxt, False, False)
        buffer.add(obs, 1, -3.5, nxt, True, False)
        batch = buffer.sample(2, np.random.default_rng(0))
        self.assertEqual(batch["obs"].shape, (2, 4, 84, 84))
        self.assertEqual(batch["next_obs"].shape, (2, 4, 84, 84))
        self.assertEqual(batch["obs"].dtype, np.uint8)
        self.assertEqual(batch["actions"].shape, (2,))
        self.assertEqual(batch["rewards"].shape, (2,))
        self.assertEqual(batch["terminated"].shape, (2,))
        self.assertEqual(batch["truncated"].shape, (2,))
        self.assertCountEqual(batch["rewards"].tolist(), [np.float32(100.0), np.float32(-3.5)])
        self.assertEqual(float(buffer.rewards[0]), 100.0)
        self.assertEqual(float(buffer.rewards[1]), -3.5)

    def test_capacity_drops_oldest(self):
        buffer = dqn_frogger.ReplayBuffer(capacity=2, obs_shape=(1, 2, 2))
        blank = np.zeros((1, 2, 2), dtype=np.uint8)
        for reward in (1.0, 2.0, 3.0):
            buffer.add(blank, 0, reward, blank, False, reward == 3.0)
        self.assertEqual(buffer.size, 2)
        self.assertCountEqual(buffer.rewards[: buffer.size].tolist(), [np.float32(2.0), np.float32(3.0)])
        # La transición más nueva conserva truncated=True.
        self.assertTrue(bool(buffer.truncated[buffer.pos - 1]))
        self.assertFalse(bool(buffer.terminated[buffer.pos - 1]))

    def test_terminated_and_truncated_roundtrip(self):
        buffer = dqn_frogger.ReplayBuffer(capacity=4, obs_shape=(1, 1, 1))
        blank = np.zeros((1, 1, 1), dtype=np.uint8)
        buffer.add(blank, 0, 1.0, blank, True, False)
        buffer.add(blank, 1, 2.0, blank, False, True)
        batch = buffer.sample(2, np.random.default_rng(1))
        paired = {
            (float(reward), bool(term), bool(trunc))
            for reward, term, trunc in zip(
                batch["rewards"], batch["terminated"], batch["truncated"]
            )
        }
        self.assertEqual(paired, {(1.0, True, False), (2.0, False, True)})

    def test_sample_is_reproducible_and_rejects_oversized_batches(self):
        buffer = dqn_frogger.ReplayBuffer(capacity=6, obs_shape=(1, 1, 1))
        blank = np.zeros((1, 1, 1), dtype=np.uint8)
        for action in range(5):
            buffer.add(blank, action, float(action), blank, False, False)

        def once(seed):
            return buffer.sample(3, np.random.default_rng(seed))["actions"].tolist()

        self.assertEqual(once(4), once(4))
        with self.assertRaises(ValueError):
            buffer.sample(6, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            buffer.sample(0, np.random.default_rng(0))

    def test_rejects_bad_shapes_and_capacity(self):
        with self.assertRaises(ValueError):
            dqn_frogger.ReplayBuffer(capacity=0)
        buffer = dqn_frogger.ReplayBuffer(capacity=2, obs_shape=(4, 84, 84))
        bad = np.zeros((84, 84, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            buffer.add(bad, 0, 0.0, bad, False, False)
        float_obs = np.zeros((4, 84, 84), dtype=np.float32)
        with self.assertRaises(TypeError):
            buffer.add(float_obs, 0, 0.0, float_obs, False, False)


class TestNatureCNN(unittest.TestCase):
    def test_output_and_feature_shapes(self):
        net = dqn_frogger.NatureCNN(n_actions=5)
        obs = torch.zeros(8, 4, 84, 84, dtype=torch.uint8)
        with torch.no_grad():
            features = net.features(obs.float() / 255.0)
            q_values = net(obs)
        self.assertEqual(tuple(features.shape), (8, 64, 7, 7))
        self.assertEqual(features.numel() // 8, dqn_frogger.NATURE_FEATURE_DIM)
        self.assertEqual(tuple(q_values.shape), (8, 5))

    def test_single_observation_and_uint8_scaling(self):
        torch.manual_seed(0)
        net = dqn_frogger.NatureCNN(n_actions=len(frogger_rl.ACTION_NAMES))
        net.eval()
        uint8_obs = torch.full((2, 4, 84, 84), 255, dtype=torch.uint8)
        float_obs = torch.ones(2, 4, 84, 84)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(uint8_obs), net(float_obs), atol=1e-5))
        policy = dqn_frogger.DQNPolicy(net, torch.device("cpu"))
        action, extra = policy.predict(np.zeros((4, 84, 84), dtype=np.uint8), deterministic=True)
        self.assertIsNone(extra)
        self.assertIn(action, range(5))
        again, _ = policy.predict(np.zeros((4, 84, 84), dtype=np.uint8), deterministic=True)
        self.assertEqual(action, again)

    def test_module_source_does_not_use_stable_baselines3(self):
        source = Path(dqn_frogger.__file__).read_text(encoding="utf-8")
        self.assertNotIn("stable_baselines3", source)
        self.assertNotIn("LunarLander", source)


class _FixedQ(nn.Module):
    """Q constante por acción: la acción 0 vale 10 y el resto 0."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        q_values = torch.zeros(obs.shape[0], 5, device=obs.device)
        q_values[:, 0] = 10.0
        return q_values


class TestBellmanUpdate(unittest.TestCase):
    def test_epsilon_linear_decay(self):
        self.assertEqual(dqn_frogger.epsilon_by_step(0, 1.0, 0.1, 100), 1.0)
        self.assertAlmostEqual(dqn_frogger.epsilon_by_step(50, 1.0, 0.1, 100), 0.55)
        self.assertAlmostEqual(dqn_frogger.epsilon_by_step(100, 1.0, 0.1, 100), 0.1)
        self.assertAlmostEqual(dqn_frogger.epsilon_by_step(250, 1.0, 0.1, 100), 0.1)

    def test_td_targets_mask_termination_but_not_the_formula_inputs(self):
        rewards = torch.tensor([1.0, 2.0])
        next_q = torch.tensor([10.0, 10.0])
        terminated = torch.tensor([1.0, 0.0])
        targets = dqn_frogger.compute_td_targets(rewards, next_q, terminated, gamma=0.5)
        self.assertTrue(torch.equal(targets, torch.tensor([1.0, 7.0])))

    def test_q_learning_targets_use_target_network_max(self):
        target = _FixedQ()
        next_obs = torch.zeros(2, 4, 84, 84)
        rewards = torch.tensor([1.0, 2.0])
        terminated = torch.tensor([1.0, 0.0])
        targets = dqn_frogger.q_learning_targets(
            target, next_obs, rewards, terminated, gamma=0.5
        )
        self.assertTrue(torch.allclose(targets, torch.tensor([1.0, 7.0])))

    def test_optimize_step_changes_online_net_not_target(self):
        torch.manual_seed(1)
        online = dqn_frogger.NatureCNN(5)
        target = dqn_frogger.NatureCNN(5)
        target.load_state_dict(online.state_dict())
        target_before = {name: value.detach().clone() for name, value in target.state_dict().items()}
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        rng = np.random.default_rng(0)
        batch = {
            "obs": rng.integers(0, 255, size=(16, 4, 84, 84), dtype=np.uint8),
            "next_obs": rng.integers(0, 255, size=(16, 4, 84, 84), dtype=np.uint8),
            "actions": rng.integers(0, 5, size=(16,)),
            "rewards": rng.normal(size=(16,)).astype(np.float32),
            "terminated": np.array([1, 0, 0, 1] * 4, dtype=np.float32),
        }
        before = {name: value.detach().clone() for name, value in online.state_dict().items()}
        stats = dqn_frogger.optimize_step(
            online,
            target,
            optimizer,
            batch,
            gamma=0.99,
            grad_clip=10.0,
            device=torch.device("cpu"),
        )
        self.assertTrue(np.isfinite(stats["loss"]))
        changed = any(
            not torch.equal(before[name], value) for name, value in online.state_dict().items()
        )
        self.assertTrue(changed)
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(target_before[name], value))

    def test_checkpoint_roundtrip(self):
        torch.manual_seed(2)
        net = dqn_frogger.NatureCNN(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q_network.pt"
            dqn_frogger.save_checkpoint(
                path,
                q_net=net,
                target_net=None,
                optimizer=None,
                step=12,
                hyperparameters={"lr": 0.0001},
                n_actions=5,
            )
            loaded, payload = dqn_frogger.load_q_network(path, torch.device("cpu"))
        self.assertEqual(payload["step"], 12)
        self.assertEqual(payload["env_id"], "ALE/Frogger-v5")
        obs = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(obs), loaded(obs)))


class TestCli(unittest.TestCase):
    def test_train_overrides(self):
        args = dqn_frogger.parse_args(
            ["train", "--total-steps", "10", "--lr", "0.1", "--device", "cpu", "--gamma", "0.95"]
        )
        self.assertEqual(args.command, "train")
        self.assertEqual(args.total_steps, 10)
        self.assertAlmostEqual(args.lr, 0.1)
        self.assertAlmostEqual(args.gamma, 0.95)
        self.assertEqual(args.device, "cpu")

    def test_eval_requires_checkpoint(self):
        with self.assertRaises(SystemExit):
            dqn_frogger.parse_args(["eval"])

    def test_auto_device_on_this_machine_is_cpu(self):
        # Esta VM no tiene CUDA ni MPS. En Apple Silicon auto debe resolver mps.
        self.assertFalse(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            self.assertEqual(dqn_frogger.resolve_device("auto").type, "mps")
        else:
            self.assertEqual(dqn_frogger.resolve_device("auto").type, "cpu")

    def test_seed_collision_rejected_before_env(self):
        args = dqn_frogger.parse_args(
            [
                "train",
                "--total-steps", "10",
                "--seed", "777001",
                "--eval-episodes", "1",
                "--eval-seed-base", "777001",
                "--buffer-size", "32",
                "--learning-starts", "32",
            ]
        )
        with self.assertRaises(ValueError):
            dqn_frogger._validate_train_args(args)


if __name__ == "__main__":
    unittest.main()
