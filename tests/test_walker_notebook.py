"""T2 tests: ppo_walker.ipynb schema/compile checks + fresh-kernel smoke run.

1. Static validation: the notebook is a valid nbformat document, every code
   cell compiles (IPython magics `%`/`!` lines skipped), and the source
   notebook tracks no outputs.
2. Fresh-kernel smoke: `jupyter nbconvert --execute` runs ALL cells with
   WALKER_SMOKE=1 and a temporary WALKER_OUTPUT_DIR. The source notebook is
   never overwritten. The artifacts must exist, config.json must carry the
   S3 hyperparameters and TRAIN_SEED 2026, and both policies must have been
   evaluated on the same held-out seeds.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_DIR / "ppo_walker.ipynb"

SMOKE_TIME_BOUND = 600  # seconds for the whole smoke execution
SMOKE_EVAL_EPISODES = 2
SMOKE_TIMESTEPS = 4096
TRAIN_SEED = 2026

# S3: Schulman et al. (2017) MuJoCo hyperparameters.
EXPECTED_HYPERPARAMETERS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
}
EXPECTED_NET_ARCH = {"pi": [64, 64], "vf": [64, 64]}

REQUIRED_ARTIFACTS = [
    "config.json",
    "comparison.csv",
    "comparison.json",
    "curve.png",
    "curve.csv",
    "model.zip",
    "vecnormalize.pkl",
]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_cells(nb):
    return [cell for cell in nb.cells if cell.cell_type == "code"]


class TestNotebookStatic(unittest.TestCase):
    def setUp(self):
        self.nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    def test_notebook_validates_with_nbformat(self):
        nbformat.validate(self.nb)

    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(_code_cells(self.nb)):
            lines = [
                line
                for line in cell.source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ]
            try:
                compile("\n".join(lines), f"<code cell {index}>", "exec")
            except SyntaxError as error:
                self.fail(f"code cell {index} does not compile: {error}")

    def test_source_notebook_has_no_outputs(self):
        for index, cell in enumerate(_code_cells(self.nb)):
            self.assertEqual(cell.outputs, [], f"cell {index} has outputs")
            self.assertIsNone(cell.execution_count, f"cell {index} was executed")


class TestNotebookSmokeRun(unittest.TestCase):
    def test_fresh_kernel_smoke_run_writes_artifacts(self):
        source_hash_before = _sha256(NOTEBOOK_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "outputs"
            env = dict(os.environ)
            env.pop("TRAIN_SEED", None)
            env.update(
                WALKER_SMOKE="1",
                WALKER_OUTPUT_DIR=str(output_dir),
                PYTHONUTF8="1",
                PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    str(NOTEBOOK_PATH),
                    "--output",
                    "executed",
                    "--output-dir",
                    str(tmp_path),
                ],
                cwd=str(PROJECT_DIR),
                env=env,
                capture_output=True,
                text=True,
                timeout=SMOKE_TIME_BOUND,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"nbconvert failed:\n{result.stderr[-4000:]}",
            )

            # The source notebook must never be overwritten by the run.
            self.assertEqual(_sha256(NOTEBOOK_PATH), source_hash_before)

            for name in REQUIRED_ARTIFACTS:
                path = output_dir / name
                self.assertTrue(path.is_file(), f"missing artifact {name}")
                self.assertGreater(path.stat().st_size, 0, f"empty artifact {name}")

            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["train_seed"], TRAIN_SEED)
            self.assertTrue(config["smoke"])
            self.assertEqual(config["total_timesteps"], SMOKE_TIMESTEPS)
            for key, value in EXPECTED_HYPERPARAMETERS.items():
                self.assertEqual(config["hyperparameters"][key], value, key)
            self.assertEqual(config["hyperparameters"]["policy_kwargs"]["net_arch"], EXPECTED_NET_ARCH)
            self.assertEqual(config["device"], "cpu")
            for package in ("python", "gymnasium", "mujoco", "stable_baselines3", "torch", "numpy"):
                self.assertIn(package, config["package_versions"])

            eval_seeds = [int(seed) for seed in config["eval_seeds"]]
            self.assertEqual(len(eval_seeds), SMOKE_EVAL_EPISODES)
            self.assertNotIn(TRAIN_SEED, eval_seeds)

            comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            policies = comparison["policies"]
            self.assertEqual(set(policies), {"random", "ppo"})
            for name in ("random", "ppo"):
                self.assertEqual(policies[name]["num_episodes"], SMOKE_EVAL_EPISODES)
                self.assertEqual([int(s) for s in policies[name]["seeds"]], eval_seeds, name)

            with (output_dir / "comparison.csv").open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2 * SMOKE_EVAL_EPISODES)
            self.assertEqual({row["policy"] for row in rows}, {"random", "ppo"})

            with (output_dir / "curve.csv").open(encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh))
            self.assertIn("timestep", header)
            self.assertIn("episode_return", header)


if __name__ == "__main__":
    unittest.main()
