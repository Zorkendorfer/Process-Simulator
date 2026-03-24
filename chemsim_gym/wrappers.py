"""Observation normalization and episode logging wrappers."""

from __future__ import annotations
import csv
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper


class NormalizeObservation(Wrapper):
    """
    Running mean/variance normalization using Welford's algorithm.

    Normalizes observations to approximately zero mean, unit variance.
    Updates statistics only during training (not during eval if ``training=False``).
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon  = epsilon
        obs_shape = env.observation_space.shape
        self._mean  = np.zeros(obs_shape, dtype=np.float64)
        self._var   = np.ones(obs_shape,  dtype=np.float64)
        self._count = 0
        self.training = True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        norm_obs = self._normalize(obs)
        info["obs_raw"] = obs
        return norm_obs, reward, terminated, truncated, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.training:
            self._count += 1
            delta  = obs - self._mean
            self._mean += delta / self._count
            delta2 = obs - self._mean
            self._var  = (self._var * (self._count - 1) + delta * delta2) / max(self._count, 1)
        return ((obs - self._mean) / np.sqrt(self._var + self.epsilon)).astype(np.float32)


class LoggingWrapper(Wrapper):
    """
    Writes one CSV row per episode for offline analysis.

    Columns: timestamp, episode, step, reward, purity, reboiler_duty_kW,
             distillate_flow, converged, reflux_ratio, distillate_frac,
             T_feed, P_feed
    """

    def __init__(self, env: gym.Env, log_dir: str):
        super().__init__(env)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "episodes.csv")
        self._file   = open(log_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=[
            "timestamp", "episode", "total_steps",
            "reward", "purity", "reboiler_duty_kW",
            "distillate_flow", "converged",
            "reflux_ratio", "distillate_frac", "T_feed", "P_feed",
        ])
        self._writer.writeheader()
        self._episode     = 0
        self._total_steps = 0
        self._ep_reward   = 0.0
        self._last_info: dict = {}

    def reset(self, **kwargs):
        self._episode    += 1
        self._ep_reward   = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward   += reward
        self._total_steps += 1
        self._last_info    = info

        if terminated or truncated:
            phys = info.get("action_physical", [np.nan] * 4)
            self._writer.writerow({
                "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
                "episode":          self._episode,
                "total_steps":      self._total_steps,
                "reward":           round(self._ep_reward, 4),
                "purity":           round(info.get("distillate_purity", np.nan), 4),
                "reboiler_duty_kW": round(info.get("reboiler_duty_kW",  np.nan), 2),
                "distillate_flow":  round(info.get("distillate_flow",   np.nan), 3),
                "converged":        int(info.get("converged", False)),
                "reflux_ratio":     round(phys[0], 3) if len(phys) > 0 else np.nan,
                "distillate_frac":  round(phys[1], 3) if len(phys) > 1 else np.nan,
                "T_feed":           round(phys[2], 1) if len(phys) > 2 else np.nan,
                "P_feed":           round(phys[3], 0) if len(phys) > 3 else np.nan,
            })

        return obs, reward, terminated, truncated, info

    def close(self):
        self._file.close()
        super().close()
