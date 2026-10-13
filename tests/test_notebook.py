"""F2 tests: notebook schema/compile checks + fresh-kernel bounded smoke run.

Two groups:

1. Static validation without an emulator: the notebook is a valid nbformat 4
   document with python3 kernel metadata, stable unique cell ids, code cells
   that compile, helper usage without duplicated algorithm/framework, the
   required educational topics, and no executed outputs tracked in the source
   (the clean notebook stays unexecuted).
2. A real fresh-kernel smoke: nbclient executes ALL notebook cells with
   FROGGER_SMOKE=1 and a temporary FROGGER_OUTPUT_DIR, on a kernel spawned
   from this project's .venv (temp kernelspec under a temp JUPYTER_PATH, no
   user-installed kernels touched). Every generated artifact is then asserted:
   nonempty curve/comparison CSV/JSON, 3 policies x 2 episodes with raw
   values, stop flags and seeds, the expected rollout-rounded actual step
   count, nonempty model zip with load validation, decodable RGB GIF and
   curve PNG, and all output paths under the temporary output root.
3. Behavioral regression without ALE: the notebook's own config/setup/baseline
   cells are executed with stubbed emulators to prove the seed guards (clear
   collision/negative errors before any emulator) and the training-env close
   guards on exception paths. The smoke additionally requires observed positive
   optimizer updates and evaluates the same held-out seeds for all three
   policies, disjoint from the training seed (behavioral data, not keywords).

These tests never skip on environmental failure: if the kernel cannot resolve
ale_py/SB3 from this project's .venv or ALE cannot create the environment, the
test errors instead of passing hollow.
"""

import ast
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import nbformat
import numpy as np
from gymnasium import spaces
from nbclient import NotebookClient
from PIL import Image

# ---------------------------------------------------------------------------
# Project locations (this file lives in rl-lab/tests/)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_DIR / "ppo_frogger.ipynb"

sys.path.insert(0, str(PROJECT_DIR))
import frogger_rl  # noqa: E402  (needs PROJECT_DIR on sys.path; importing creates no emulator)

KERNEL_NAME = "ppo-frogger-smoke"
CELL_TIMEOUT = 180          # seconds per cell
SMOKE_TIME_BOUND = 600      # seconds for the whole smoke execution


def _load_notebook():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)
    return nb


def _code_sources(nb):
    return [cell.source for cell in nb.cells if cell.cell_type == "code"]


def _cell_source(nb, marker, cell_type="code"):
    for cell in nb.cells:
        if cell.cell_type == cell_type and marker in cell.source:
            return cell.source
    raise AssertionError(f"no {cell_type} cell contains marker {marker!r}")


def _exec_source(source, namespace=None):
    """Execute one notebook cell's source in a fresh namespace (behavioral)."""
    ns = {"__name__": "__test_cell__"}
    if namespace:
        ns.update(namespace)
    exec(compile(source, "<notebook cell>", "exec"), ns)  # noqa: S102 (behavioral notebook-cell test)
    return ns


# ---------------------------------------------------------------------------
# 1b. Behavioral regression: seed guards + training-env close guards
# ---------------------------------------------------------------------------

class TestNotebookSeedValidation(unittest.TestCase):
    """The config cell must validate seeds BEFORE any emulator is created:
    nonnegative TRAIN_SEED, evaluation seeds distinct and disjoint from the
    training seed; the default 2026 stays disjoint from EVAL_SEEDS."""

    def _exec_config(self, train_seed):
        nb = _load_notebook()
        source = _cell_source(nb, "REQUESTED_STEPS = 100_000")
        return _exec_source(
            source,
            {"SMOKE": False, "TRAIN_SEED": train_seed, "frogger_rl": frogger_rl},
        )

    def test_default_train_seed_is_disjoint_from_eval_seeds(self):
        ns = self._exec_config(2026)
        eval_seeds = [int(seed) for seed in ns["EVAL_SEEDS"]]
        self.assertEqual(len(eval_seeds), 20)
        self.assertEqual(len(set(eval_seeds)), len(eval_seeds))
        self.assertGreaterEqual(min(eval_seeds), 0)
        self.assertNotIn(2026, eval_seeds)

    def test_train_seed_collision_with_eval_seeds_raises_clearly(self):
        with self.assertRaisesRegex(ValueError, "held-out"):
            self._exec_config(777001)

    def test_negative_train_seed_raises_clearly(self):
        with self.assertRaisesRegex(ValueError, "no negativo"):
            self._exec_config(-5)


class _FakeTrainEnv(gym.Env):
    """Duck-typed gymnasium env for training-setup regression (no ALE)."""

    def __init__(self, fail_reset_on=None):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.metadata = {"render_modes": []}
        self.render_mode = None
        self.reward_range = (-float("inf"), float("inf"))
        self._frameskip = 1
        self.repeat_action_probability = 0.25
        self.fail_reset_on = fail_reset_on
        self.resets = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        if self.fail_reset_on is not None and self.resets >= self.fail_reset_on:
            raise RuntimeError("reset explosion")
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((4, 84, 84), dtype=np.uint8), 0.0, False, False, {}

    def close(self):
        self.closed = True


class TestNotebookTrainingClose(unittest.TestCase):
    """The training VecEnv must be closed on every exception path: the setup
    cell (explicit reset) and the before-learn baseline cell."""

    _PPO_KWARGS = {
        "n_steps": 64,
        "batch_size": 64,
        "n_epochs": 1,
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    }

    def _exec_setup(self, env_factory):
        nb = _load_notebook()
        source = _cell_source(nb, "train_vecenv = make_vec_env")

        class _StubFrogger:
            ENV_MAX_AGENT_STEPS = 2700

            @staticmethod
            def make_env(**kwargs):
                return env_factory()

        return _exec_source(
            source,
            {
                "frogger_rl": _StubFrogger,
                "EVAL_CAP": 64,
                "TRAIN_SEED": 2026,
                "DEVICE": "cpu",
                "PPO_KWARGS": dict(self._PPO_KWARGS),
            },
        )

    def test_setup_closes_training_vecenv_when_reset_fails(self):
        # DummyVecEnv resets lazily: the cell's explicit reset is the 1st one.
        fake = _FakeTrainEnv(fail_reset_on=1)
        with self.assertRaises(RuntimeError):
            self._exec_setup(lambda: fake)
        self.assertTrue(
            fake.closed, "training VecEnv must be closed when the explicit reset fails"
        )

    def test_before_learn_eval_failure_closes_training_vecenv(self):
        nb = _load_notebook()
        source = _cell_source(nb, "Baseline 2: el PPO sin entrenar")

        class _RaisingEval:
            @staticmethod
            def evaluate_policy(*args, **kwargs):
                raise RuntimeError("eval explosion")

        class _FakePolicy:
            def state_dict(self):
                return {}

        class _Vec:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        vec = _Vec()
        with self.assertRaises(RuntimeError):
            _exec_source(
                source,
                {
                    "frogger_rl": _RaisingEval,
                    "model": SimpleNamespace(policy=_FakePolicy()),
                    "train_vecenv": vec,
                    "EVAL_SEEDS": [1, 2],
                    "EVAL_CAP": 64,
                },
            )
        self.assertTrue(
            vec.closed,
            "before-learn eval failure must close the training VecEnv",
        )


# ---------------------------------------------------------------------------
# 1. Static notebook checks (no emulator)
# ---------------------------------------------------------------------------

class TestNotebookSchema(unittest.TestCase):
    def test_valid_nbformat_python3_kernel_and_stable_ids(self):
        self.assertTrue(NOTEBOOK_PATH.is_file(), "ppo_frogger.ipynb missing")
        nb = _load_notebook()
        self.assertEqual(nb.nbformat, 4)
        self.assertEqual(nb.metadata["kernelspec"]["name"], "python3")
        self.assertEqual(nb.metadata["kernelspec"]["language"], "python")
        ids = [cell.id for cell in nb.cells]
        self.assertTrue(all(ids), "every cell must have a nonempty id")
        self.assertEqual(len(set(ids)), len(ids), "cell ids must be unique")
        self.assertTrue(
            any(cell.cell_type == "code" for cell in nb.cells)
            and any(cell.cell_type == "markdown" for cell in nb.cells),
            "notebook must interleave code and markdown",
        )

    def test_all_code_cells_compile(self):
        nb = _load_notebook()
        for index, source in enumerate(_code_sources(nb)):
            try:
                ast.parse(source, filename=f"<cell {index}>")
                compile(source, f"<cell {index}>", "exec")
            except SyntaxError as error:  # pragma: no cover - failure branch
                self.fail(f"code cell {index} does not compile: {error}")

    def test_uses_helper_without_duplicating_algorithm(self):
        nb = _load_notebook()
        sources = "\n".join(_code_sources(nb))
        self.assertIn("import frogger_rl", sources, "notebook must use the helper")
        self.assertIn('frogger_rl.make_env', sources)
        self.assertIn("frogger_rl.evaluate_policy", sources)
        # The notebook must not reimplement the environment/evaluation contract,
        # nor pull in a different emulator framework.
        for forbidden in (
            "def make_env(",
            "def evaluate_policy(",
            "import stable_retro",
            "from stable_retro",
        ):
            self.assertNotIn(forbidden, sources, f"helper duplicated: {forbidden}")
        self.assertNotIn("retro.make", sources)

    def test_markdown_covers_required_topics(self):
        nb = _load_notebook()
        markdown = "\n".join(
            cell.source for cell in nb.cells if cell.cell_type == "markdown"
        )
        required = [
            # objective / observability
            "parcialmente observable",
            "Objetivo",
            # actions and reward
            "NOOP", "UP", "RIGHT", "LEFT", "DOWN",
            "Recompensa nativa",
            # temporal context
            "cuatro fotogramas",
            # algorithm explainer
            "PPO", "CNN", "Ventaja",
            # honest evaluation semantics
            "truncated", "terminated",
            "Incertidumbre",
        ]
        for topic in required:
            self.assertIn(topic, markdown, f"markdown is missing topic: {topic!r}")
        # Official citations required by the assignment.
        cited = [
            "https://ale.farama.org/environments/frogger/",
            "https://ale.farama.org/",
            "gymnasium.farama.org",
            "AtariPreprocessing",
            "stable-baselines3.readthedocs.io",
            "PPO",
        ]
        for url in cited:
            self.assertIn(url, markdown, f"missing citation: {url}")
        self.assertIn("Atari", markdown)
        self.assertIn("Sega", markdown)  # flag: Atari is not Sega/Genesis

    def test_source_notebook_has_no_executed_outputs(self):
        nb = _load_notebook()
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                self.assertEqual(
                    list(cell.get("outputs") or []),
                    [],
                    f"code cell {index} must stay unexecuted in the source file",
                )


# ---------------------------------------------------------------------------
# 2. Fresh-kernel smoke execution (ALE + SB3 from this project's .venv)
# ---------------------------------------------------------------------------

class TestNotebookSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_env = dict(os.environ)
        cls._old_cwd = Path.cwd()

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._old_env)
        os.chdir(cls._old_cwd)

    def _restore_env_cwd(self, env_backup, cwd):
        os.environ.clear()
        os.environ.update(env_backup)
        os.chdir(cwd)

    def test_fresh_kernel_smoke_end_to_end(self):
        env_backup = dict(os.environ)
        cwd = Path.cwd()
        started = time.monotonic()
        tmp_root = None
        try:
            nb = _load_notebook()
            source_sha_before = hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()

            tmp_root = Path(tempfile.mkdtemp(prefix="frogger_smoke_")).resolve()
            out_root = tmp_root / "output"
            out_root.mkdir(parents=True)

            # Temp kernelspec that spawns THIS project's venv python (fresh kernel,
            # discovered via JUPYTER_PATH; nothing installed in the user home).
            spec_dir = tmp_root / "jupyter" / "kernels" / KERNEL_NAME
            spec_dir.mkdir(parents=True)
            (spec_dir / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "frogger smoke (fresh kernel)",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )

            os.environ["FROGGER_SMOKE"] = "1"
            os.environ["FROGGER_OUTPUT_DIR"] = str(out_root)
            os.environ["JUPYTER_PATH"] = str(tmp_root / "jupyter")
            os.environ.setdefault("MPLBACKEND", "Agg")
            os.chdir(PROJECT_DIR)

            client = NotebookClient(
                nb,
                kernel_name=KERNEL_NAME,
                timeout=CELL_TIMEOUT,
                allow_errors=False,
                resources={"metadata": {"path": str(PROJECT_DIR)}},
            )
            client.execute(cwd=str(PROJECT_DIR))

            # ---------------------------------------------------------------
            # Manifest: the notebook summarizes its own run.
            # ---------------------------------------------------------------
            manifest_path = out_root / "manifest.json"
            self.assertTrue(manifest_path.is_file(), "manifest.json not generated")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIs(manifest["smoke"], True)
            self.assertEqual(manifest["requested_timesteps"], 256)
            # SB3 collects complete rollouts: 256 % 64 == 0 -> exactly 256.
            self.assertEqual(manifest["actual_timesteps"], 256)
            self.assertGreaterEqual(manifest["completed_episodes_training"], 4)
            self.assertIs(manifest["reload_matches"], True)
            # PPO must actually update weights during learn: positive observed
            # optimizer steps and a nonzero max parameter delta vs the pre-learn
            # snapshot (plumbing evidence; updates != useful learning).
            self.assertGreater(
                manifest["optimizer_updates"], 0,
                "PPO must perform at least one observed optimizer step",
            )
            self.assertGreater(
                manifest["max_parameter_delta"], 0.0,
                "post-learn parameters must differ from the pre-learn snapshot",
            )

            # Kernel really resolved ale_py/SB3 from this project's .venv.
            kernel_python = manifest["runtime"]["python_executable"]
            self.assertTrue(
                os.path.exists(kernel_python) and os.path.samefile(kernel_python, sys.executable),
                f"kernel python {kernel_python!r} != test python {sys.executable!r}",
            )
            for key in ("ale_py", "stable_baselines3", "gymnasium", "torch"):
                self.assertTrue(manifest["runtime"].get(key), f"runtime version {key} missing")

            # All declared artifacts must live under the temporary output root.
            all_artifact_paths = list(manifest["artifacts"].values()) + [
                manifest["manifest_json"]
            ]
            for raw_path in all_artifact_paths:
                path = Path(raw_path)
                self.assertTrue(path.resolve().is_relative_to(out_root),
                                f"artifact outside output root: {path}")
                self.assertTrue(path.is_file(), f"artifact missing: {path}")

            artifacts = {name: Path(raw) for name, raw in manifest["artifacts"].items()}

            # ---------------------------------------------------------------
            # Learning curve CSV/PNG
            # ---------------------------------------------------------------
            with open(artifacts["curve_csv"], encoding="utf-8") as fh:
                curve_rows = list(csv.DictReader(fh))
            self.assertGreaterEqual(len(curve_rows), 4,
                                    "curve.csv must hold the real completed episodes")
            for row in curve_rows:
                self.assertGreaterEqual(int(row["timestep"]), 1)
                self.assertGreaterEqual(int(row["episode_length"]), 1)

            curve_img = Image.open(artifacts["curve_png"])
            self.assertGreater(curve_img.size[0] * curve_img.size[1], 0)
            curve_img.close()

            # ---------------------------------------------------------------
            # Comparison CSV/JSON: 3 policies x 2 episodes, raw data + summaries
            # ---------------------------------------------------------------
            with open(artifacts["comparison_csv"], encoding="utf-8") as fh:
                comparison_rows = list(csv.DictReader(fh))
            episode_rows = [r for r in comparison_rows if r["record_type"] == "episode"]
            summary_rows = [r for r in comparison_rows if r["record_type"] == "summary"]
            self.assertEqual(len(episode_rows), 3 * 2,
                             "expected 3 policies x 2 held-out episodes")
            per_policy = {}
            for row in episode_rows:
                per_policy.setdefault(row["policy"], []).append(row)
            self.assertEqual(list(per_policy.keys()),
                             ["random", "ppo_before_learn", "ppo_after_learn"])
            for policy, rows in per_policy.items():
                self.assertEqual(len(rows), 2, f"{policy} must have 2 episodes")
                for row in rows:
                    float(row["return"])        # raw numeric return
                    self.assertGreaterEqual(int(row["length"]), 1)
                    self.assertGreaterEqual(int(row["seed"]), 0)
                    self.assertIn(row["terminated"], {"0", "1"})
                    self.assertIn(row["truncated"], {"0", "1"})
            self.assertEqual(len(summary_rows), 3)

            comparison = json.loads(
                artifacts["comparison_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(comparison["experiment"]["smoke"], True)
            self.assertEqual(comparison["experiment"]["actual_timesteps"], 256)
            self.assertEqual(comparison["experiment"]["requested_timesteps"], 256)
            hp = comparison["experiment"]["hyperparameters"]
            self.assertEqual(hp["n_steps"], 64)
            self.assertEqual(hp["n_epochs"], 1)
            self.assertEqual(comparison["environment"]["obs_shape"], [4, 84, 84])
            self.assertEqual(len(comparison["environment"]["action_names"]), 5)
            self.assertIs(comparison["environment"]["terminal_on_life_loss"], False)
            self.assertGreater(comparison["experiment"]["optimizer_updates"], 0)
            self.assertGreater(comparison["experiment"]["max_parameter_delta"], 0.0)
            eval_seeds_data = comparison["experiment"]["eval_seeds"]
            train_seed_data = comparison["experiment"]["train_seed"]
            self.assertEqual(train_seed_data, 2026)
            self.assertEqual(sorted(eval_seeds_data), [777001, 777002])
            self.assertNotIn(train_seed_data, eval_seeds_data)
            for key in ("random", "ppo_before_learn", "ppo_after_learn"):
                result = comparison["evaluations"][key]
                self.assertEqual(result["num_episodes"], 2)
                self.assertEqual(result["metrics"]["n"], 2)
                self.assertGreaterEqual(result["metrics"]["truncated_count"], 0)
                self.assertEqual(
                    sum(ep["terminated"] for ep in result["episodes"])
                    + sum(ep["truncated"] for ep in result["episodes"]),
                    2, "every episode must end terminated or truncated",
                )
                # Behavioral data evidence: all three policies share the SAME
                # held-out seeds and none collides with the training seed.
                observed_seeds = sorted(ep["seed"] for ep in result["episodes"])
                self.assertEqual(
                    observed_seeds, sorted(eval_seeds_data),
                    f"{key} must run the same held-out seeds",
                )
                self.assertNotIn(train_seed_data, observed_seeds)
                for ep in result["episodes"]:
                    self.assertIsInstance(ep["return"], float)
                    self.assertIsInstance(ep["seed"], int)

            # CSV and JSON must agree per-episode (raw metrics preserved both ways).
            json_episode = comparison["evaluations"]["ppo_after_learn"]["episodes"][0]
            csv_episode = per_policy["ppo_after_learn"][0]
            self.assertAlmostEqual(float(json_episode["return"]),
                                   float(csv_episode["return"]), places=6)
            self.assertEqual(json_episode["length"], int(csv_episode["length"]))
            self.assertEqual(bool(json_episode["terminated"]),
                             csv_episode["terminated"] == "1")
            self.assertEqual(json_episode["seed"], int(csv_episode["seed"]))

            # ---------------------------------------------------------------
            # Model zip: nonempty + load validation in this test process too.
            # ---------------------------------------------------------------
            model_path = artifacts["model_zip"]
            self.assertGreater(model_path.stat().st_size, 0, "model.zip empty")
            from stable_baselines3 import PPO
            reloaded = PPO.load(str(model_path), device="cpu")
            obs = np.zeros((4, 84, 84), dtype="uint8")
            action, _ = reloaded.predict(obs, deterministic=True)
            flat_action = int(np.asarray(action).reshape(-1)[0])
            self.assertIn(flat_action, range(5))
            del reloaded

            # ---------------------------------------------------------------
            # GIF: short, valid, decodable.
            # ---------------------------------------------------------------
            gif_img = Image.open(artifacts["gif"])
            self.assertEqual(gif_img.format, "GIF")
            self.assertIn(gif_img.mode, ("RGB", "P", "RGBA"))
            frame_count = getattr(gif_img, "n_frames", 1)
            self.assertGreaterEqual(frame_count, 1)
            self.assertLessEqual(frame_count, 20, "smoke GIF must stay short")
            gif_img.seek(0)
            gif_img.load()
            gif_img.close()
            self.assertEqual(manifest["gif_frames"], frame_count)

            # ---------------------------------------------------------------
            # The source notebook must not have been modified by execution.
            # ---------------------------------------------------------------
            source_sha_after = hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()
            self.assertEqual(source_sha_before, source_sha_after,
                             "execution must not edit the source notebook")

            elapsed = time.monotonic() - started
            self.assertLess(elapsed, SMOKE_TIME_BOUND,
                            f"whole smoke exceeded {SMOKE_TIME_BOUND}s ({elapsed:.1f}s)")
        finally:
            self._restore_env_cwd(env_backup, cwd)
            if tmp_root is not None:
                import shutil
                shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
