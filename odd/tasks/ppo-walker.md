# PPO on Walker2d-v5 (MuJoCo)

Objective: replace the Frogger PPO deliverable with a reproducible SB3 PPO experiment on `Walker2d-v5`, compared against a random baseline. Branch: `AlejoMoncada/PPO`. Delivery: ask-on-risk. Runner: `uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v`.

## Specs

- S1. Environment `Walker2d-v5` (Gymnasium MuJoCo). Add the MuJoCo dependency (`gymnasium[mujoco]`) to `pyproject.toml` and refresh `uv.lock`. Frogger files (`frogger_rl.py`, `ppo_frogger.ipynb`, their tests) stay untouched.
- S2. Helper module `walker_rl.py` (no Jupyter dependency, importing it never creates an env or directories): `make_env(seed=None, render_mode=None)` and `evaluate_policy(policy, episode_seeds, random_seed=None, vecnormalize=None)` where `policy=None` means uniform random actions sampled from the real action space with a local seeded numpy RNG; otherwise `policy.predict(obs, deterministic=True)`, applying frozen `VecNormalize` observation normalization when given. Returns JSON-serializable per-episode raw returns/lengths plus mean and std. Env always closed.
- S3. Notebook `ppo_walker.ipynb` (Spanish prose, English code/identifiers) using SB3 `PPO("MlpPolicy")` with the Schulman et al. (2017) MuJoCo hyperparameters: `n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5`, `policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh)`, `VecNormalize(norm_obs=True, norm_reward=True)` for training, `TOTAL_TIMESTEPS=1_000_000`, `TRAIN_SEED=2026`, `device="cpu"`. All of these shown in one config cell and exported to `results/walker_ppo/config.json` together with package versions — "hiperparámetros, número de pasos y semilla, para que se pueda reproducir".
- S4. Baseline: "el agente aleatorio, con el mismo número de episodios de evaluación, media y desviación." Random and trained PPO evaluated on the same 20 held-out seeds (disjoint from TRAIN_SEED), raw (unnormalized) rewards, table with mean ± std, exported to `comparison.csv`.
- S5. "La curva de aprendizaje, con la recompensa por episodio o la media móvil a lo largo de los timesteps." Raw episode return vs timestep (from Monitor, before VecNormalize) with moving average and a horizontal line for the random baseline mean; saved as `curve.png` and `curve.csv`.
- S6. Theory markdown: progression REINFORCE → Actor-Critic → PPO; "una o dos líneas sobre el clipping y la ventaja" with the clipped objective formula and GAE advantage, citing Schulman et al. (2017).
- S7. One line justifying the environment: continuous control benchmark from the PPO paper; LunarLander and CartPole already covered in class.
- S8. Closing: "una frase sobre si PPO te serviría para el Agente Inversor" (continuous actions ~ portfolio weights; caveats: non-stationarity, sample efficiency).
- S9. Smoke mode `WALKER_SMOKE=1` (e.g. 4096 steps, 2 eval episodes, outputs under `WALKER_OUTPUT_DIR` if set) so tests can execute the notebook end to end quickly. Save model + VecNormalize stats.
- S10. README section documenting Walker2d usage, modes, and outputs.

## Tasks

| ID | Specs | Route | Status | Commit |
|---|---|---|---|---|
| T1 | S1, S2 | delegated (Haiku) | done (superseded by T1b) | 9cc960c |
| T1b | S1, S2 | delegated (Haiku): walker_rl, shapes (17,)/(6,) | done: 8 walker + 21 frogger tests OK | 9cc960c |
| T2 | S3–S9 | delegated (Haiku) | done: 4 notebook tests OK (smoke ~40 s) | see git log |
| T3 | S10 | delegated (Haiku) | done | see git log |
| T4 | S3–S5 full 1M run | delegated (Haiku), needs user go-ahead (long CPU run) | pending | |

## Log

- L1 (2026-10-13, user, pasted checklist): "Un notebook que muestre el entrenamiento con PPO; Stable-Baselines3 está permitido. El desarrollo y los resultados: hiperparámetros, número de pasos y semilla, para que se pueda reproducir. Una comparación contra un baseline. Lo más claro es el agente aleatorio, con el mismo número de episodios de evaluación, media y desviación. La curva de aprendizaje, con la recompensa por episodio o la media móvil a lo largo de los timesteps. [...] una o dos líneas sobre el clipping y la ventaja [...] cerrar con una frase sobre si PPO te serviría para el Agente Inversor. [...] conviene justificar en una línea por qué lo elegiste."
- L2 (2026-10-13, user): "Lunar lander lo vimos en clase me gustaria algo que quizas sea mas un reto [...] tambein Cartpole lo vimos en otor taller" → Walker2d-v5 chosen.
- L3 (2026-10-13, user): "Receurda solo orquesta Haiku para todo lo demás" → all units delegated to Haiku workers.
- L4 (2026-10-13, user): "Ok entonces Walker, o una de la versinoes de Humanoid" → Walker2d-v5 chosen (Humanoid needs far more than 1M steps with PPO; paper shows it separately in Roboschool). FrozenLake and trading rejected: not control envs; trading belongs to the final project.
