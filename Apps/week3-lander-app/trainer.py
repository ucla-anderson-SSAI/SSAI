"""
Background PPO trainer for LunarLander-v3 using Stable-Baselines3.

The TrainingManager owns a single live model + a worker thread. The
FastAPI app calls start()/pause()/reset() and polls state().

start() either creates a fresh model (first run, or after reset) or
resumes the existing model with reset_num_timesteps=False so PPO picks
up where it left off. pause() asks the worker to stop on its next
callback tick but leaves the model and state intact, so the user can
roll out the partially-trained policy and then hit Resume to keep going.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

ENV_ID = "LunarLander-v3"
ACTION_NAMES = ["noop", "left engine", "main engine", "right engine"]
SOLVED_THRESHOLD = 200.0  # avg over last 100 episodes


# 8-dim observation space of LunarLander-v3.
# Source: gymnasium.envs.box2d.lunar_lander
STATE_DIMS = [
    {"index": 0, "name": "x position",        "range": "[-1.5, 1.5]",  "desc": "Horizontal offset from landing pad center"},
    {"index": 1, "name": "y position",        "range": "[-1.5, 1.5]",  "desc": "Height above ground (0 = pad level)"},
    {"index": 2, "name": "x velocity",        "range": "[-5, 5]",      "desc": "Horizontal speed"},
    {"index": 3, "name": "y velocity",        "range": "[-5, 5]",      "desc": "Vertical speed (negative = falling)"},
    {"index": 4, "name": "angle",             "range": "[-π, π]",      "desc": "Lander tilt in radians (0 = upright)"},
    {"index": 5, "name": "angular velocity",  "range": "[-5, 5]",      "desc": "How fast the lander is rotating"},
    {"index": 6, "name": "left leg contact",  "range": "{0, 1}",       "desc": "1 if left leg is touching ground"},
    {"index": 7, "name": "right leg contact", "range": "{0, 1}",       "desc": "1 if right leg is touching ground"},
]


@dataclass
class TrainConfig:
    target_episodes: int = 2000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    ent_coef: float = 0.01
    seed: int = 42

    @property
    def total_timesteps(self) -> int:
        # Generous safety ceiling: ~300 steps/episode average. The
        # callback stops training as soon as target_episodes is hit.
        return self.target_episodes * 300


@dataclass
class TrainState:
    is_running: bool = False
    is_done: bool = False        # finished entire run (not just paused)
    is_paused: bool = False      # paused with model still alive
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    timesteps: int = 0
    target_episodes: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    rolling_100: List[float] = field(default_factory=list)
    last_mean_reward: float = 0.0
    solved: bool = False
    solved_at_episode: Optional[int] = None
    config: Optional[TrainConfig] = None


class TrainingManager:
    """Singleton-style manager owning the live model and worker thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = TrainState()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.model = None     # SB3 PPO model
        self.vec_env = None   # The training VecEnv (kept around so resume reuses it)

    # ---------- public API ----------
    def state_dict(self) -> dict:
        with self._lock:
            s = self._state
            elapsed = None
            if s.started_at is not None:
                end = s.finished_at if s.finished_at else time.time()
                elapsed = end - s.started_at
            return {
                "is_running": s.is_running,
                "is_done": s.is_done,
                "is_paused": s.is_paused,
                "error": s.error,
                "timesteps": s.timesteps,
                "target_episodes": s.target_episodes,
                "episode_rewards": list(s.episode_rewards),
                "rolling_100": list(s.rolling_100),
                "last_mean_reward": s.last_mean_reward,
                "solved": s.solved,
                "solved_at_episode": s.solved_at_episode,
                "elapsed_seconds": elapsed,
                "model_ready": self.model is not None,
            }

    def start(self, cfg: TrainConfig) -> bool:
        """Start fresh or resume an existing model."""
        with self._lock:
            if self._state.is_running:
                return False
            resume = self.model is not None
            if resume:
                # Keep accumulated episode_rewards / rolling_100 / timesteps,
                # just flip the flags and update target if needed.
                self._state.is_running = True
                self._state.is_done = False
                self._state.is_paused = False
                self._state.error = None
                if cfg.target_episodes > self._state.target_episodes:
                    self._state.target_episodes = cfg.target_episodes
                if self._state.started_at is None:
                    self._state.started_at = time.time()
                self._state.finished_at = None
                self._state.config = cfg
            else:
                self._state = TrainState(
                    is_running=True,
                    started_at=time.time(),
                    target_episodes=cfg.target_episodes,
                    config=cfg,
                )
            self._stop_flag.clear()

        self._thread = threading.Thread(
            target=self._train_worker, args=(cfg, resume), daemon=True
        )
        self._thread.start()
        return True

    def pause(self) -> bool:
        """Pause: stop the worker on next tick but keep model + state."""
        with self._lock:
            if not self._state.is_running:
                return False
        self._stop_flag.set()
        return True

    def reset(self):
        """Wipe state + model so the user can start a fresh run."""
        with self._lock:
            if self._state.is_running:
                return False
            self._state = TrainState()
            self.model = None
            self.vec_env = None
            return True

    def rollout(self, max_steps: int = 1000, seed: Optional[int] = None) -> dict:
        """Run a single greedy episode against the current model."""
        if self.model is None:
            raise RuntimeError("No model yet. Start training first.")
        import gymnasium as gym
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=seed)
        xs, ys, angles, actions, rewards = [], [], [], [], []
        total = 0.0
        landed = False
        outcome = "timeout"
        for _ in range(max_steps):
            xs.append(float(obs[0]))
            ys.append(float(obs[1]))
            angles.append(float(obs[4]))
            action, _ = self.model.predict(obs, deterministic=True)
            a = int(action)
            actions.append(a)
            obs, r, term, trunc, _ = env.step(a)
            rewards.append(float(r))
            total += float(r)
            if term or trunc:
                landed = bool(r >= 100)
                if landed:
                    outcome = "landed"
                elif term:
                    outcome = "crashed"
                else:
                    outcome = "timeout"
                break
        env.close()
        return {
            "xs": xs, "ys": ys, "angles": angles,
            "actions": actions, "rewards": rewards,
            "total_reward": total, "landed": landed, "outcome": outcome,
        }

    # ---------- worker ----------
    def _train_worker(self, cfg: TrainConfig, resume: bool):
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor

            if not resume:
                def make_env(rank: int, seed: int):
                    def _init():
                        env = gym.make(ENV_ID)
                        env = Monitor(env)
                        return env
                    return _init

                self.vec_env = DummyVecEnv(
                    [make_env(i, cfg.seed) for i in range(cfg.n_envs)]
                )
                self.model = PPO(
                    "MlpPolicy",
                    self.vec_env,
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size,
                    n_epochs=cfg.n_epochs,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                    ent_coef=cfg.ent_coef,
                    seed=cfg.seed,
                    verbose=0,
                    device="cpu",
                )

            manager = self

            class LiveCallback(BaseCallback):
                """Streams progress + episode rewards into manager state."""

                def __init__(self, start_episode: int):
                    super().__init__()
                    self._episode_count = start_episode

                def _on_step(self) -> bool:
                    if manager._stop_flag.is_set():
                        return False

                    infos = self.locals.get("infos", [])
                    new_eps = []
                    for info in infos:
                        if info and "episode" in info:
                            new_eps.append(float(info["episode"]["r"]))

                    if new_eps or self.num_timesteps % 1000 == 0:
                        with manager._lock:
                            s = manager._state
                            s.timesteps = int(self.num_timesteps)
                            for r in new_eps:
                                s.episode_rewards.append(r)
                                self._episode_count += 1
                                window = s.episode_rewards[-100:]
                                avg = float(np.mean(window))
                                s.rolling_100.append(avg)
                                s.last_mean_reward = avg
                                if (not s.solved) and len(window) == 100 and avg >= SOLVED_THRESHOLD:
                                    s.solved = True
                                    s.solved_at_episode = self._episode_count
                            # Stop once we've hit the target episode count
                            if len(s.episode_rewards) >= s.target_episodes:
                                return False
                    return True

            with self._lock:
                start_episode = len(self._state.episode_rewards)
                # How many more timesteps to run this round
                remaining = max(1, cfg.total_timesteps - self._state.timesteps)

            self.model.learn(
                total_timesteps=remaining,
                callback=LiveCallback(start_episode),
                progress_bar=False,
                reset_num_timesteps=not resume,
            )

            # Did training end naturally or via pause?
            paused = self._stop_flag.is_set()
            with self._lock:
                self._state.is_running = False
                if paused:
                    self._state.is_paused = True
                    self._state.is_done = False
                else:
                    self._state.is_paused = False
                    self._state.is_done = True
                    self._state.finished_at = time.time()

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("=" * 60, flush=True)
            print("TRAINING WORKER CRASHED:", flush=True)
            print(tb, flush=True)
            print("=" * 60, flush=True)
            with self._lock:
                self._state.is_running = False
                self._state.is_done = True
                self._state.error = f"{type(e).__name__}: {e}"
                self._state.finished_at = time.time()


# Module-level singleton (lazy)
_manager: Optional[TrainingManager] = None


def get_manager() -> TrainingManager:
    global _manager
    if _manager is None:
        _manager = TrainingManager()
    return _manager
