"""DQN escrito a mano para ALE/Frogger-v5.

PyTorch puro: no importa ni entrena con Stable-Baselines3. El entorno es el
de ``frogger_rl.make_env`` (AtariPreprocessing 84x84 en gris, pila de 4
fotogramas, recompensa cruda, sticky actions en una sola capa).

El bucle es el DQN clásico de Mnih et al. (2015):

* buffer de repetición uniforme,
* política epsilon-greedy con decaimiento lineal,
* red Q y red objetivo (copia dura cada ``target_update`` pasos),
* objetivo de Bellman / TD: ``y = r + gamma * max_a' Q_target(s', a')``
  si el episodio no terminó de verdad; si ``terminated`` es verdadero no hay
  bootstrap. Un corte por ``TimeLimit`` (``truncated``) sí hace bootstrap.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import frogger_rl

NATURE_FEATURE_DIM = 64 * 7 * 7  # 84 -> 20 -> 9 -> 7 con los kernels de Nature


# ---------------------------------------------------------------------------
# Red Q (Nature DQN)
# ---------------------------------------------------------------------------

class NatureCNN(nn.Module):
    """CNN de Nature DQN: tres convoluciones y dos capas lineales.

    La entrada es un stack ``uint8`` de forma ``(N, 4, 84, 84)`` o un float
    ya escalado a ``[0, 1]``. Los enteros se dividen entre 255 dentro de
    ``forward``. La salida es ``(N, n_actions)`` sin activación (valores Q).
    """

    def __init__(self, n_actions: int, in_channels: int = frogger_rl.STACK_SIZE):
        super().__init__()
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        self.n_actions = int(n_actions)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(NATURE_FEATURE_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Q inicial cerca de cero: el sesgo del último lineal no arranca enorme.
        last = self.head[-1]
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.is_floating_point():
            x = obs
        else:
            x = obs.float() / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class DQNPolicy:
    """Adaptador greedy para ``frogger_rl.evaluate_policy`` (epsilon = 0)."""

    def __init__(self, q_net: NatureCNN, device: torch.device | str):
        self.q_net = q_net
        self.device = torch.device(device)
        self.q_net.eval()

    def predict(self, obs, deterministic: bool = True):
        del deterministic  # la evaluación del taller usa la acción greedy
        array = np.asarray(obs)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError(f"obs must have shape (4, 84, 84) or (N, 4, 84, 84), got {array.shape}")
        tensor = torch.as_tensor(np.ascontiguousarray(array), device=self.device)
        with torch.no_grad():
            q_values = self.q_net(tensor)
            action = int(q_values.argmax(dim=1)[0].item())
        return action, None


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Buffer uniforme de transiciones ``(s, a, r, s', terminated, truncated)``.

    Las observaciones se guardan en ``uint8`` (sin escalar). ``terminated``
    corta el bootstrap; ``truncated`` no. La recompensa se guarda tal cual,
    sin clipping.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...] = frogger_rl.OBS_SHAPE):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        obs_arr = _as_uint8_obs(obs, self.obs_shape)
        next_arr = _as_uint8_obs(next_obs, self.obs_shape)
        slot = self.pos
        self.obs[slot] = obs_arr
        self.next_obs[slot] = next_arr
        self.actions[slot] = int(action)
        self.rewards[slot] = np.float32(reward)
        self.terminated[slot] = bool(terminated)
        self.truncated[slot] = bool(truncated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > self.size:
            raise ValueError(
                f"batch_size {batch_size} exceeds stored transitions {self.size}"
            )
        index = rng.choice(self.size, size=batch_size, replace=False)
        return {
            "obs": self.obs[index],
            "next_obs": self.next_obs[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index],
            "terminated": self.terminated[index],
            "truncated": self.truncated[index],
        }


def _as_uint8_obs(obs, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(obs)
    if array.shape != expected_shape:
        raise ValueError(f"observation shape {array.shape} != {expected_shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(
                f"observation dtype must be uint8, got {array.dtype}"
            )
        array = array.astype(np.uint8, copy=False)
    return array


# ---------------------------------------------------------------------------
# Bellman / TD
# ---------------------------------------------------------------------------

def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    """Decaimiento lineal de epsilon. En ``step >= decay_steps`` vale ``end``."""
    if decay_steps <= 0:
        return float(end)
    if step >= decay_steps:
        return float(end)
    fraction = step / float(decay_steps)
    return float(start + fraction * (end - start))


def compute_td_targets(
    rewards: torch.Tensor,
    next_q_max: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Objetivo de Bellman. ``terminated`` en {0, 1} anula el bootstrap."""
    done = terminated.to(dtype=rewards.dtype)
    return rewards + float(gamma) * next_q_max * (1.0 - done)


def q_learning_targets(
    target_net: nn.Module,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """``y = r + gamma * max_a' Q_target(s', a') * (1 - terminated)``."""
    with torch.no_grad():
        next_q_max = target_net(next_obs).max(dim=1).values
        return compute_td_targets(rewards, next_q_max, terminated, gamma)


def optimize_step(
    q_net: NatureCNN,
    target_net: NatureCNN,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    *,
    gamma: float,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """Un paso de gradiente sobre la pérdida de Huber del error TD."""
    obs = torch.as_tensor(np.ascontiguousarray(batch["obs"]), device=device)
    next_obs = torch.as_tensor(np.ascontiguousarray(batch["next_obs"]), device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=device)

    q_values = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    targets = q_learning_targets(target_net, next_obs, rewards, terminated, gamma)
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "mean_q": float(q_values.detach().mean().item()),
        "mean_target": float(targets.detach().mean().item()),
        "grad_norm": float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
    }


def select_action(
    q_net: NatureCNN,
    obs: np.ndarray,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    """Epsilon-greedy. El azar sale de ``rng`` local, no del RNG global."""
    if float(rng.random()) < epsilon:
        return int(rng.integers(n_actions))
    tensor = torch.as_tensor(np.ascontiguousarray(obs)[None, ...], device=device)
    with torch.no_grad():
        return int(q_net(tensor).argmax(dim=1).item())


def max_parameter_delta(current: nn.Module, initial: dict[str, torch.Tensor]) -> float:
    delta = 0.0
    for name, value in current.state_dict().items():
        base = initial[name]
        delta = max(delta, float((value.detach().cpu() - base).abs().max().item()))
    return delta


# ---------------------------------------------------------------------------
# Dispositivo, checkpoints, figuras
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    """``auto`` elige cuda, luego mps si está disponible, luego cpu."""
    requested = name.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió cuda, pero torch.cuda.is_available() es False")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "Se pidió mps, pero torch.backends.mps.is_available() es False"
            )
    return device


def save_checkpoint(
    path: Path,
    *,
    q_net: NatureCNN,
    target_net: NatureCNN | None,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    hyperparameters: dict,
    n_actions: int,
) -> None:
    payload = {
        "algorithm": "dqn",
        "env_id": frogger_rl.ENV_ID,
        "n_actions": int(n_actions),
        "obs_shape": list(frogger_rl.OBS_SHAPE),
        "architecture": "nature_dqn_cnn",
        "step": int(step),
        "hyperparameters": hyperparameters,
        "q_net": q_net.state_dict(),
    }
    if target_net is not None:
        payload["target_net"] = target_net.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_q_network(path: Path, device: torch.device) -> tuple[NatureCNN, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    n_actions = int(payload["n_actions"])
    net = NatureCNN(n_actions=n_actions).to(device)
    net.load_state_dict(payload["q_net"])
    net.eval()
    return net, payload


def save_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 80) -> int:
    if not frames:
        raise ValueError("no hay cuadros para el GIF")
    from PIL import Image

    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return len(images)


def subsample_frames(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    index = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[int(i)] for i in index]


def save_learning_curve(
    path: Path,
    episodes: list[dict],
    loss_log: list[dict],
    *,
    random_mean: float | None,
    title_suffix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), constrained_layout=True)

    ax = axes[0]
    if episodes:
        steps = [ep["timestep"] for ep in episodes]
        returns = [ep["return"] for ep in episodes]
        ax.plot(steps, returns, color="#9bb7d4", linewidth=0.8, label="retorno del episodio")
        window = 20 if len(returns) >= 20 else max(1, len(returns) // 5)
        if window > 1:
            averaged = _moving_average(returns, window)
            ax.plot(
                steps,
                averaged,
                color="#0b3a66",
                linewidth=2.0,
                label=f"media móvil ({window})",
            )
        if random_mean is not None:
            ax.axhline(
                random_mean,
                color="#b45309",
                linestyle="--",
                linewidth=1.4,
                label=f"baseline aleatorio (media eval) = {random_mean:.2f}",
            )
        ax.set_ylabel("retorno crudo")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "sin episodios completos", ha="center", va="center")
    ax.set_xlabel("pasos de agente")
    ax.set_title(f"DQN en ALE/Frogger-v5 — retorno por episodio{title_suffix}")
    ax.grid(True, alpha=0.3)

    ax_loss = axes[1]
    if loss_log:
        ax_loss.plot(
            [row["step"] for row in loss_log],
            [row["mean_loss"] for row in loss_log],
            color="#7f1d1d",
            linewidth=1.2,
            label="pérdida Huber (media del tramo)",
        )
        ax_eps = ax_loss.twinx()
        ax_eps.plot(
            [row["step"] for row in loss_log],
            [row["epsilon"] for row in loss_log],
            color="#047857",
            linewidth=1.2,
            label="epsilon",
        )
        ax_loss.set_ylabel("pérdida")
        ax_eps.set_ylabel("epsilon")
        lines_a, labels_a = ax_loss.get_legend_handles_labels()
        lines_b, labels_b = ax_eps.get_legend_handles_labels()
        ax_loss.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=8)
    else:
        ax_loss.text(0.5, 0.5, "sin actualizaciones", ha="center", va="center")
    ax_loss.set_xlabel("pasos de agente")
    ax_loss.set_title("Pérdida TD y epsilon")
    ax_loss.grid(True, alpha=0.3)

    fig.savefig(path, dpi=120)
    plt.close(fig)


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    window = min(window, len(values))
    averaged = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        averaged.append(running / min(index + 1, window))
    return averaged


def _mean_slice(values: list[float], count: int, last: bool) -> float | None:
    if not values:
        return None
    count = min(count, len(values))
    chunk = values[-count:] if last else values[:count]
    return float(np.mean(chunk))


def runtime_info(device: torch.device) -> dict:
    import ale_py
    import gymnasium

    mps = getattr(torch.backends, "mps", None)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "ale_py": ale_py.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(mps.is_available()) if mps is not None else False,
    }


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------

def record_gif(
    policy: DQNPolicy,
    path: Path,
    *,
    seed: int,
    max_episode_steps: int,
    max_frames: int,
) -> dict:
    env = frogger_rl.make_env(
        max_episode_steps=max_episode_steps,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    try:
        obs, _ = env.reset(seed=seed)
        frames.append(np.asarray(env.render(), dtype=np.uint8))
        while not (terminated or truncated) and episode_length < max_episode_steps:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            frames.append(np.asarray(env.render(), dtype=np.uint8))
    finally:
        env.close()
    saved = subsample_frames(frames, max_frames)
    n_saved = save_gif(saved, path)
    return {
        "path": str(path),
        "frames_captured": len(frames),
        "frames_saved": n_saved,
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "seed": int(seed),
        "max_episode_steps": int(max_episode_steps),
    }


def _episode_reset_seed(train_seed: int, episode_index: int, eval_seeds: list[int]) -> int:
    """Semilla de reset distinta de las held-out.

    Los episodios usan ``train_seed + episode_index``. Si esa secuencia
    alcanzara una semilla de evaluación (entrenamientos muy largos), se
    salta a una banda alta.
    """
    candidate = train_seed + episode_index
    if candidate in eval_seeds:
        candidate = 50_000_000 + train_seed + episode_index
    return int(candidate)


def _hyperparameters_from_args(args: argparse.Namespace) -> dict:
    keys = [
        "total_steps",
        "buffer_size",
        "batch_size",
        "gamma",
        "lr",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "learning_starts",
        "train_freq",
        "target_update",
        "max_episode_steps",
        "seed",
        "eval_episodes",
        "eval_seed_base",
        "eval_max_episode_steps",
        "grad_clip",
        "gif_max_steps",
        "gif_max_frames",
        "device",
    ]
    return {key: getattr(args, key) for key in keys}


def _validate_train_args(args: argparse.Namespace) -> list[int]:
    if args.total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {args.total_steps}")
    if args.buffer_size < args.batch_size:
        raise ValueError(
            f"buffer_size ({args.buffer_size}) must be >= batch_size ({args.batch_size})"
        )
    if args.learning_starts < args.batch_size:
        raise ValueError(
            f"learning_starts ({args.learning_starts}) must be >= batch_size ({args.batch_size})"
        )
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {args.gamma}")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}")
    for name in ("epsilon_start", "epsilon_end"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if args.epsilon_start < args.epsilon_end:
        raise ValueError("epsilon_start must be >= epsilon_end")
    if args.train_freq <= 0 or args.target_update <= 0:
        raise ValueError("train_freq and target_update must be positive")
    if args.max_episode_steps <= 0 or args.eval_max_episode_steps <= 0:
        raise ValueError("episode step caps must be positive")
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}")
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    eval_seeds = [args.eval_seed_base + index for index in range(args.eval_episodes)]
    if min(eval_seeds) < 0:
        raise ValueError("eval seeds must be non-negative")
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("eval seeds must be unique")
    if args.seed in eval_seeds:
        raise ValueError(
            "la semilla de entrenamiento coincide con una semilla de evaluación: "
            f"{args.seed}. Cambie --seed o --eval-seed-base."
        )
    return eval_seeds


def train(args: argparse.Namespace) -> dict:
    eval_seeds = _validate_train_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hyper = _hyperparameters_from_args(args)
    hyper["resolved_device"] = str(device)

    rng = np.random.default_rng(args.seed)
    sample_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
    env = frogger_rl.make_env(max_episode_steps=args.max_episode_steps)
    started = time.perf_counter()
    episodes: list[dict] = []
    loss_log: list[dict] = []
    loss_sum = 0.0
    loss_count = 0
    q_sum = 0.0
    updates = 0
    incomplete = None
    try:
        n_actions = int(env.action_space.n)
        q_net = NatureCNN(n_actions=n_actions).to(device)
        target_net = NatureCNN(n_actions=n_actions).to(device)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()
        q_net.train()
        initial_weights = {
            name: value.detach().cpu().clone() for name, value in q_net.state_dict().items()
        }
        optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
        buffer = ReplayBuffer(args.buffer_size, obs_shape=tuple(frogger_rl.OBS_SHAPE))

        obs, _ = env.reset(seed=args.seed)
        obs = np.array(obs, dtype=np.uint8, copy=True)
        episode_return = 0.0
        episode_length = 0
        episode_index = 0
        print(
            f"DQN {frogger_rl.ENV_ID} device={device} steps={args.total_steps} "
            f"buffer={args.buffer_size} actions={n_actions} obs={obs.shape} {obs.dtype}",
            flush=True,
        )

        for step in range(1, args.total_steps + 1):
            epsilon = epsilon_by_step(
                step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
            )
            action = select_action(q_net, obs, n_actions, epsilon, rng, device)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.array(next_obs, dtype=np.uint8, copy=True)
            buffer.add(obs, action, float(reward), next_obs, terminated, truncated)
            episode_return += float(reward)
            episode_length += 1

            if (
                step > args.learning_starts
                and step % args.train_freq == 0
                and buffer.size >= args.batch_size
            ):
                batch = buffer.sample(args.batch_size, sample_rng)
                stats = optimize_step(
                    q_net,
                    target_net,
                    optimizer,
                    batch,
                    gamma=args.gamma,
                    grad_clip=args.grad_clip,
                    device=device,
                )
                updates += 1
                loss_sum += stats["loss"]
                q_sum += stats["mean_q"]
                loss_count += 1

            if step % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())
                target_net.eval()

            if terminated or truncated:
                episode_index += 1
                episodes.append(
                    {
                        "episode": episode_index,
                        "return": float(episode_return),
                        "length": int(episode_length),
                        "timestep": int(step),
                        "epsilon": float(epsilon),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset(seed=_episode_reset_seed(args.seed, episode_index, eval_seeds))
                obs = np.array(obs, dtype=np.uint8, copy=True)
            else:
                obs = next_obs

            if step % args.log_every == 0 or step == args.total_steps:
                mean_loss = (loss_sum / loss_count) if loss_count else None
                mean_q = (q_sum / loss_count) if loss_count else None
                loss_log.append(
                    {
                        "step": int(step),
                        "mean_loss": None if mean_loss is None else float(mean_loss),
                        "mean_q": None if mean_q is None else float(mean_q),
                        "epsilon": float(epsilon),
                        "episodes": len(episodes),
                        "updates": int(updates),
                        "buffer_size": int(buffer.size),
                    }
                )
                loss_sum = 0.0
                q_sum = 0.0
                loss_count = 0
                elapsed = time.perf_counter() - started
                recent = episodes[-5:]
                recent_mean = (
                    float(np.mean([ep["return"] for ep in recent])) if recent else float("nan")
                )
                print(
                    f"step {step}/{args.total_steps} eps={epsilon:.3f} "
                    f"updates={updates} episodes={len(episodes)} "
                    f"recent_return={recent_mean:.2f} "
                    f"loss={mean_loss if mean_loss is not None else float('nan'):.4f} "
                    f"{step / elapsed:.1f} steps/s",
                    flush=True,
                )

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_checkpoint(
                    output_dir / "checkpoint.pt",
                    q_net=q_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    step=step,
                    hyperparameters=hyper,
                    n_actions=n_actions,
                )

        if episode_length > 0:
            incomplete = {
                "return": float(episode_return),
                "length": int(episode_length),
                "timestep": int(args.total_steps),
            }

        weight_delta = max_parameter_delta(q_net, initial_weights)
        save_checkpoint(
            output_dir / "checkpoint.pt",
            q_net=q_net,
            target_net=target_net,
            optimizer=optimizer,
            step=args.total_steps,
            hyperparameters=hyper,
            n_actions=n_actions,
        )
        save_checkpoint(
            output_dir / "q_network.pt",
            q_net=q_net,
            target_net=None,
            optimizer=None,
            step=args.total_steps,
            hyperparameters=hyper,
            n_actions=n_actions,
        )

        returns = [ep["return"] for ep in episodes]
        metrics: dict = {
            "env_id": frogger_rl.ENV_ID,
            "algorithm": "DQN",
            "implementation": "pytorch-puro, sin stable-baselines3",
            "seed": int(args.seed),
            "hyperparameters": hyper,
            "optimizer_updates": int(updates),
            "max_parameter_delta": float(weight_delta),
            "training": {
                "episodes_completed": len(episodes),
                "timesteps": int(args.total_steps),
                "mean_return_all": None if not returns else float(np.mean(returns)),
                "mean_return_first_20": _mean_slice(returns, 20, last=False),
                "mean_return_last_20": _mean_slice(returns, 20, last=True),
                "mean_length_all": None
                if not episodes
                else float(np.mean([ep["length"] for ep in episodes])),
                "terminated_count": int(sum(ep["terminated"] for ep in episodes)),
                "truncated_count": int(sum(ep["truncated"] for ep in episodes)),
                "incomplete_final_episode": incomplete,
                "episodes": episodes,
                "loss_log": loss_log,
            },
            "runtime": runtime_info(device),
            "artifacts": {},
        }

        print("evaluando política greedy y baseline aleatorio...", flush=True)
        policy = DQNPolicy(q_net, device)
        dqn_eval = frogger_rl.evaluate_policy(
            policy,
            episode_seeds=eval_seeds,
            max_episode_steps=args.eval_max_episode_steps,
            random_seed=args.seed,
        )
        random_eval = frogger_rl.evaluate_policy(
            None,
            episode_seeds=eval_seeds,
            max_episode_steps=args.eval_max_episode_steps,
            random_seed=args.seed,
        )
        dqn_mean = float(dqn_eval["metrics"]["return"]["mean"])
        random_mean = float(random_eval["metrics"]["return"]["mean"])
        metrics["evaluation"] = {
            "eval_seeds": eval_seeds,
            "max_episode_steps": int(args.eval_max_episode_steps),
            "dqn_greedy": dqn_eval,
            "random": random_eval,
            "return_mean_delta_vs_random": dqn_mean - random_mean,
            "beats_random_mean": bool(dqn_mean > random_mean),
        }
        print(
            f"eval greedy mean={dqn_mean:.3f} random mean={random_mean:.3f} "
            f"delta={dqn_mean - random_mean:.3f}",
            flush=True,
        )

        gif_info = record_gif(
            policy,
            output_dir / "eval.gif",
            seed=eval_seeds[0],
            max_episode_steps=args.gif_max_steps,
            max_frames=args.gif_max_frames,
        )
        metrics["gif"] = gif_info

        curve_path = output_dir / "learning_curve.png"
        save_learning_curve(
            curve_path,
            episodes,
            [row for row in loss_log if row["mean_loss"] is not None],
            random_mean=random_mean,
            title_suffix=f" ({args.total_steps} pasos, {device})",
        )
        metrics["wall_time_seconds"] = float(time.perf_counter() - started)
        metrics["artifacts"] = {
            "learning_curve_png": str(curve_path),
            "metrics_json": str(output_dir / "metrics.json"),
            "gif": str(output_dir / "eval.gif"),
            "q_network": str(output_dir / "q_network.pt"),
            "checkpoint": str(output_dir / "checkpoint.pt"),
        }
        metrics["notes"] = (
            "Recompensas crudas de ALE, sin clipping ni reshaping. "
            "La comparación con la política aleatoria usa las mismas semillas "
            "y el mismo TimeLimit. Un delta pequeño o negativo significa que "
            "este presupuesto de pasos no produjo una política mejor que el azar."
        )
        _write_metrics(output_dir / "metrics.json", metrics)
        print(
            f"listo en {metrics['wall_time_seconds']:.1f}s "
            f"updates={updates} max_delta={weight_delta:.6f} -> {output_dir}",
            flush=True,
        )
        return metrics
    finally:
        env.close()


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    net, payload = load_q_network(Path(args.checkpoint), device)
    policy = DQNPolicy(net, device)
    seeds = [args.eval_seed_base + index for index in range(args.episodes)]
    started = time.perf_counter()
    dqn_eval = frogger_rl.evaluate_policy(
        policy,
        episode_seeds=seeds,
        max_episode_steps=args.max_episode_steps,
        random_seed=args.seed,
    )
    result: dict = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload.get("step"),
        "device": str(device),
        "dqn_greedy": dqn_eval,
    }
    if not args.skip_random:
        random_eval = frogger_rl.evaluate_policy(
            None,
            episode_seeds=seeds,
            max_episode_steps=args.max_episode_steps,
            random_seed=args.seed,
        )
        dqn_mean = float(dqn_eval["metrics"]["return"]["mean"])
        random_mean = float(random_eval["metrics"]["return"]["mean"])
        result["random"] = random_eval
        result["return_mean_delta_vs_random"] = dqn_mean - random_mean
        result["beats_random_mean"] = bool(dqn_mean > random_mean)
    if args.gif:
        output_dir = Path(args.output_dir)
        result["gif"] = record_gif(
            policy,
            output_dir / "eval.gif",
            seed=seeds[0],
            max_episode_steps=args.gif_max_steps,
            max_frames=args.gif_max_frames,
        )
    result["wall_time_seconds"] = float(time.perf_counter() - started)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir / "eval_metrics.json", result)
    dqn_mean = result["dqn_greedy"]["metrics"]["return"]["mean"]
    print(f"eval greedy mean return = {dqn_mean:.4f}", flush=True)
    if "random" in result:
        print(
            f"eval random mean return = {result['random']['metrics']['return']['mean']:.4f} "
            f"delta = {result['return_mean_delta_vs_random']:.4f}",
            flush=True,
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dqn_frogger",
        description=(
            "DQN en PyTorch puro para ALE/Frogger-v5 "
            "(replay, epsilon-greedy, red objetivo, Bellman). Sin Stable-Baselines3."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Entrena y guarda curva, métricas, GIF y checkpoint")
    train_parser.add_argument("--total-steps", type=int, default=100_000)
    train_parser.add_argument("--buffer-size", type=int, default=20_000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--epsilon-start", type=float, default=1.0)
    train_parser.add_argument("--epsilon-end", type=float, default=0.1)
    train_parser.add_argument("--epsilon-decay-steps", type=int, default=80_000)
    train_parser.add_argument("--learning-starts", type=int, default=1_000)
    train_parser.add_argument("--train-freq", type=int, default=4)
    train_parser.add_argument("--target-update", type=int, default=2_000)
    train_parser.add_argument("--max-episode-steps", type=int, default=frogger_rl.ENV_MAX_AGENT_STEPS)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--eval-episodes", type=int, default=10)
    train_parser.add_argument("--eval-seed-base", type=int, default=777_001)
    train_parser.add_argument(
        "--eval-max-episode-steps", type=int, default=frogger_rl.ENV_MAX_AGENT_STEPS
    )
    train_parser.add_argument("--grad-clip", type=float, default=10.0)
    train_parser.add_argument("--gif-max-steps", type=int, default=600)
    train_parser.add_argument("--gif-max-frames", type=int, default=180)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--output-dir", type=str, default="entregables/u3_dqn")
    train_parser.add_argument("--log-every", type=int, default=2_000)
    train_parser.add_argument("--checkpoint-every", type=int, default=50_000)
    train_parser.set_defaults(func=train)

    eval_parser = sub.add_parser("eval", help="Carga un checkpoint y evalúa contra el azar")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--eval-seed-base", type=int, default=777_001)
    eval_parser.add_argument("--max-episode-steps", type=int, default=frogger_rl.ENV_MAX_AGENT_STEPS)
    eval_parser.add_argument("--seed", type=int, default=2026)
    eval_parser.add_argument("--device", type=str, default="auto")
    eval_parser.add_argument("--output-dir", type=str, default="entregables/u3_dqn_eval")
    eval_parser.add_argument("--gif", action="store_true")
    eval_parser.add_argument("--gif-max-steps", type=int, default=600)
    eval_parser.add_argument("--gif-max-frames", type=int, default=180)
    eval_parser.add_argument("--skip-random", action="store_true")
    eval_parser.set_defaults(func=evaluate)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
