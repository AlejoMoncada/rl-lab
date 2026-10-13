# PPO en Walker2d-v5 (MuJoCo)

Experimento reproducible de **PPO** (Stable-Baselines3) sobre **`Walker2d-v5`**
(Gymnasium MuJoCo). El notebook entrena un agente, lo compara contra una política
**aleatoria** sobre los mismos episodios held-out y exporta hiperparámetros, curva de
aprendizaje, comparación, modelo y un GIF corto.

Walker2d es un benchmark de **control continuo** de la familia MuJoCo usado en el
artículo original de PPO (Schulman et al., 2017). LunarLander y CartPole ya los vimos
en clase; este entorno añade un reto de control continuo más exigente.

## Estructura

```
PPO/
├── ppo_walker.ipynb      # notebook del experimento (español; código en inglés)
├── walker_rl.py          # helper de entorno y evaluación (Walker2d-v5), sin dependencia de Jupyter
├── tests/                # pruebas del helper y del notebook (incluye smoke headless)
├── pyproject.toml        # proyecto uv (gymnasium[mujoco], SB3, torch CPU, jupyter…)
└── results/walker_ppo/   # artefactos generados (por omisión; ver Salidas)
```

## Puesta en marcha

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11–3.13.

```bash
uv sync --frozen
uv run jupyter lab ppo_walker.ipynb
```

En Jupyter Lab: **Run All**, o ejecute las celdas en orden de arriba hacia abajo; las
celdas dependen de las anteriores. Las variables de entorno se fijan antes de arrancar
el kernel.

## Modos y variables de entorno

| Variable | Efecto |
|---|---|
| `WALKER_SMOKE=1` | Modo smoke: PPO real pero corto (4 096 pasos y 2 episodios de evaluación). Comprueba que el notebook corre de punta a punta; **no** es evidencia de aprendizaje. |
| `TRAIN_SEED=<int>` | Semilla de entrenamiento (por omisión `2026`). Debe ser un entero no negativo y no puede coincidir con las semillas held-out de evaluación; el notebook lo valida antes de crear el entorno. |
| `WALKER_OUTPUT_DIR=<dir>` | Directorio de salida. Por omisión: `results/walker_ppo/` dentro del proyecto. |

## Hiperparámetros

Tomados de la celda de configuración del notebook (`PPO_KWARGS` y constantes del
experimento). Se exportan a `config.json` junto con las versiones de los paquetes.

| Parámetro | Valor |
|---|---|
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| Red (`policy_kwargs`) | `net_arch` pi = [64, 64], vf = [64, 64]; activación `Tanh` |
| Normalización de entrenamiento | `VecNormalize(norm_obs=True, norm_reward=True)` |
| `TOTAL_TIMESTEPS` | 1 000 000 (modo normal) |
| `TRAIN_SEED` | 2026 |
| `device` | `cpu` |
| Episodios de evaluación | 20 por política |
| Semillas de evaluación | 10000–10019 (disjuntas de `TRAIN_SEED`) |
| `RANDOM_SEED` (política aleatoria) | 424242 |

El retorno de evaluación es **crudo** (sin normalizar), para que el baseline y PPO se
midan en la misma escala.

## Ejecución headless del smoke (nbconvert)

```bash
WALKER_SMOKE=1 WALKER_OUTPUT_DIR=/tmp/walker-smoke \
  uv run jupyter nbconvert --to notebook --execute \
  --output-dir /tmp/walker-smoke ppo_walker.ipynb
```

- Ejecuta el notebook con un kernel nuevo y escribe la copia ejecutada en
  `--output-dir`; el `ppo_walker.ipynb` fuente no se modifica.
- En Windows, con Git Bash, `/tmp/...` se resuelve dentro de Git Bash; use cualquier
  carpeta temporal propia si lo prefiere.

## Salidas

Todo se escribe bajo `WALKER_OUTPUT_DIR` (o `results/walker_ppo/`):

- `config.json` — hiperparámetros, número de pasos, semillas y versiones de paquetes.
- `comparison.csv` / `comparison.json` — evaluación aleatoria y PPO sobre las mismas
  semillas, con media y desviación.
- `curve.csv` / `curve.png` — retorno crudo por episodio contra timesteps, con media
  móvil y la media del agente aleatorio como línea de referencia.
- `model.zip` — pesos del PPO.
- `vecnormalize.pkl` — estadísticas de normalización necesarias para evaluar el modelo.
- `walker.gif` — episodio corto renderizado con la política entrenada.

## Pruebas

```bash
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
```

`tests/test_walker_notebook.py` valida el esquema del notebook, que cada celda compile,
que el notebook fuente no tenga salidas, y ejecuta el smoke completo con kernel nuevo
comprobando los artefactos, la configuración y las semillas held-out compartidas.

## Estado

- Se ha ejecutado el **smoke** (`WALKER_SMOKE=1`), que verifica el pipeline de punta a
  punta.
- Se ejecutó la corrida completa de **1 000 000 pasos** el 2026-10-13 (~23 min de
  entrenamiento en CPU, semilla 2026, 3163 episodios de entrenamiento).

Resultados (20 episodios de evaluación, semillas 10000–10019, retorno crudo):

| Política  | Retorno medio ± desv. | Largo medio ± desv. |
|-----------|-----------------------|---------------------|
| Aleatoria | −0.57 ± 5.59          | 19.7 ± 5.9          |
| PPO       | 4743.5 ± 76.4         | 997.0 ± 13.1        |

- Los artefactos se generan localmente bajo `results/walker_ppo/` y no están versionados
  en git.

## Referencias

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
- [Walker2d en Gymnasium](https://gymnasium.farama.org/environments/mujoco/walker2d/).
- [PPO en Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

---

## Experimento anterior: Frogger (ALE)

Notebook educativo con PPO sobre `ALE/Frogger-v5` (Atari 2600). Compara una política
aleatoria, PPO sin entrenar y PPO entrenado sobre los mismos episodios held-out.
`ALE/Frogger-v5` es Atari, no Sega/Genesis; si la consigna exige Genesis, este notebook
es un complemento, no un reemplazo.

Archivos: `ppo_frogger.ipynb`, `frogger_rl.py`, sus pruebas y `results/frogger_ppo/`.

```bash
uv sync --frozen
uv run jupyter lab ppo_frogger.ipynb
```

| Variable | Efecto |
|---|---|
| `FROGGER_SMOKE=1` | Modo smoke (256 pasos). Prueba la plomería, no el aprendizaje. |
| `TRAIN_SEED=<int>` | Semilla de entrenamiento (por omisión `2026`); no puede coincidir con las semillas de evaluación. |
| `FROGGER_OUTPUT_DIR=<dir>` | Directorio de salida. Por omisión: `results/frogger_ppo/`. |

Smoke headless:

```bash
FROGGER_SMOKE=1 FROGGER_OUTPUT_DIR=/tmp/frogger-out \
  uv run jupyter nbconvert --to notebook --execute \
  --output-dir /tmp/frogger-out ppo_frogger.ipynb
```

Pruebas:

```bash
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
```

Estado: la ejecución local verificada es únicamente un smoke de 256 pasos. No indica
aprendizaje útil. El entrenamiento normal de 100 000 pasos no se ha ejecutado como parte
de estos resultados.
