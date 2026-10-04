# ppo-frogger

Notebook educativo e independiente de *reinforcement learning* con **PPO** sobre
**`ALE/Frogger-v5`** (Atari 2600, Gymnasium Atari). Compara tres políticas sobre los
mismos episodios held-out: **aleatoria**, **PPO sin entrenar** y **PPO después de
entrenar**, y exporta modelo, curva de aprendizaje, comparación y un GIF corto.

> `ALE/Frogger-v5` es **Atari, no Sega/Genesis**. Si la consigna de la clase exige
> específicamente Genesis (*stable-retro*, como el proyecto Airstriker en `ppo-sega`),
> este notebook es un complemento educativo, no un reemplazo.

Este proyecto es independiente: no modifica `ppo-sega` ni ningún otro trabajo previo.

## Estructura

```
ppo-frogger/
├── ppo_frogger.ipynb   # notebook educativo (español), usable con el helper
├── frogger_rl.py       # helper de entorno/evaluación (ALE/Frogger-v5, tested F1)
├── tests/              # pruebas del helper + validación/ejecución smoke del notebook
├── pyproject.toml      # proyecto uv aislado (ale-py, SB3, torch CPU, jupyter…)
└── results/frogger_ppo/  # artefactos generados (por omisión; ver más abajo)
```

## Puesta en marcha

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11–3.13. Para clonar el
repositorio independiente y abrir el notebook:

```bash
git clone https://github.com/AlejoMoncada/frogger-rl-lab.git ppo-frogger
cd ppo-frogger
uv sync --frozen
uv run jupyter lab ppo_frogger.ipynb
```

En Jupyter Lab: **Run All** (o ejecutar las celdas en orden, de arriba hacia abajo);
las celdas dependen de las anteriores (configuración → sondeo → baselines →
entrenamiento → evaluaciones → artefactos). Se puede ejecutar en modo **smoke** o en
modo **normal** (ver abajo) cambiando variables de entorno antes de arrancar el kernel.

## Modos de ejecución

| Variable | Efecto |
|---|---|
| `FROGGER_SMOKE=1` | Modo smoke: PPO real pero diminuto (256 pasos, `n_steps=64`, `n_epochs=1`, 2 episodios de evaluación por política, GIF corto). Prueba la plomería del pipeline, **no** el aprendizaje. |
| `TRAIN_SEED=<int>` | Semilla del entrenamiento (default `2026`). Debe ser un entero **no negativo** y **no puede coincidir** con las semillas held-out de evaluación: el notebook lo valida antes de crear el primer emulador y falla con un mensaje claro. |
| `FROGGER_OUTPUT_DIR=<dir>` | Directorio raíz de salida. Por omisión: `ppo-frogger/results/frogger_ppo/`. |

**Modo normal (predeterminado):** presupuesto de **100 000 pasos solicitados**,
hiperparámetros del notebook (`n_steps=512`, `batch_size=64`, `n_epochs=4`,
`lr=2.5e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`,
`vf_coef=0.5`, `device="cpu"`), 20 episodios de evaluación por política y tapa de
2700 pasos de agente (default del helper; **no** se hereda el tope 450 de otros
experimentos). Para probar otros presupuestos, cambie `REQUESTED_STEPS` y las
variables de la celda *Configuración del experimento*.

## Ejecución headless corta y reproducible (no toca el notebook fuente)

```bash
cd ppo-frogger
FROGGER_SMOKE=1 FROGGER_OUTPUT_DIR=/tmp/frogger-out \
  uv run jupyter nbconvert --to notebook --execute \
  --output-dir /tmp/frogger-out ppo_frogger.ipynb
```

- Ejecuta el notebook **con kernel fresco** (el `python3` del `.venv`) y escribe la
  copia ejecutada en `--output-dir`; el archivo `ppo_frogger.ipynb` fuente no se
  modifica.
- Todos los artefactos se generan bajo `/tmp/frogger-out` (o el
  `FROGGER_OUTPUT_DIR` elegido) gracias a la detección de raíz de proyecto del
  notebook (exige `frogger_rl.py`).
- El tiempo del smoke depende del hardware y del presupuesto; en las pruebas
  queda acotado (180 s por celda, y toda la ejecución con tope). No hay un
  tiempo típico garantizado.

## Salidas

Todo se escribe bajo el directorio de salida y se resume en `manifest.json`:

- `model.zip` — pesos + configuración del PPO (la recarga se valida en el notebook).
- `curve.csv` / `curve.png` — retorno crudo por episodio completado vs. pasos reales,
  con media móvil y líneas de referencia (aleatorio y PPO sin entrenar).
- `comparison.json` / `comparison.csv` — las tres evaluaciones genuinas, ajustes del
  entorno/wrappers, semillas, conteos de truncación/terminación, bandera de smoke,
  hiperparámetros, métricas de actualización observadas (`optimizer_updates`,
  `max_parameter_delta`) y versiones de paquetes.
- `episodio_smoke.gif` (smoke) / `episodio_ppo.gif` (normal) — GIF RGB corto.

## Pruebas

```bash
cd ppo-frogger
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -v
```

`tests/test_frogger_rl.py` valida el helper (21 casos, contrato real ALE incluido);
`tests/test_notebook.py` valida el esquema del notebook (nbformat, ids estables,
compile de celdas, cobertura de temas), prueba las guardas de semillas y de cierre
(reseteando las celdas de configuración/setup del notebook sin emulador) y ejecuta
un **smoke completo con kernel fresco** del notebook con `FROGGER_SMOKE=1` y un
`FROGGER_OUTPUT_DIR` temporal, comprobando los artefactos, las semillas held-out
compartidas por las tres políticas y las actualizaciones de pesos observadas.

## Estado del experimento

La ejecución local verificada es únicamente un **smoke de 256 pasos** con cuatro
actualizaciones del optimizador. Las tres políticas obtuvieron retornos crudos
`[0, 0]` sobre dos episodios de 64 pasos, ambos truncados. Esto comprueba el
funcionamiento del pipeline, **no aprendizaje útil, niveles completados ni
superioridad sobre jugadores humanos**. El entrenamiento normal de 100 000 pasos
no se ha ejecutado como parte de estos resultados.

Los resultados locales, pesos y copias ejecutadas no se incluyen en Git; cada
usuario puede generarlos con los comandos anteriores. El notebook fuente se
publica limpio, sin salidas de ejecución. No se redistribuyen ROMs en este
repositorio; las bibliotecas conservan sus propias licencias.

## Referencias

- [Frogger en Arcade Learning Environment](https://ale.farama.org/environments/frogger/).
- [AtariPreprocessing de Gymnasium](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing).
- [PPO en Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

## Limitaciones honestas

- **Smoke ≠ aprendizaje:** el modo `FROGGER_SMOKE=1` demuestra plomería (flujo de
  observaciones, actualización de gradientes, exportación), nunca desempeño.
- Un episodio **truncado** por el límite de pasos no es una victoria; solo indica que
  el juego seguía vivo al cortar el tope.
- El retorno es **crudo** (sin clipping ni reshaping): puede ser 0 en presupuestos
  cortos; eso es un dato real.
- Para conclusiones firmes, repita el entrenamiento con **varias semillas**
  (`TRAIN_SEED=...`) y compare distribuciones de retornos, no solo medias.
- El entrenamiento normal (100 000 pasos) requiere tiempo de cómputo en CPU; no se
  ejecuta de forma automática en las pruebas.