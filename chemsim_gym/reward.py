"""Reward function for the distillation RL environment."""

from dataclasses import dataclass, field
from typing import Tuple, Dict
import numpy as np


@dataclass
class RewardConfig:
    # Target
    target_purity: float = 0.95
    min_distillate_flow: float = 10.0   # mol/s

    # Weights
    w_purity: float = 10.0
    w_energy: float = 1.0    # penalty per MW reboiler duty
    w_flow:   float = 2.0
    w_crash:  float = -20.0  # non-convergence penalty

    # Shape: "step" | "smooth" | "exponential"
    purity_reward_type: str = "smooth"


class RewardFunction:
    def __init__(self, **kwargs):
        self.cfg = RewardConfig(**kwargs)

    def compute(self, result, action_phys: np.ndarray) -> Tuple[float, Dict]:
        """Return (total_reward, breakdown_dict)."""
        cfg = self.cfg

        if not result.converged:
            return cfg.w_crash, {"crash": cfg.w_crash, "total": cfg.w_crash}

        # 1. Purity reward
        purity   = result.distillate_purity
        r_purity = cfg.w_purity * self._purity_reward(purity, cfg.target_purity)

        # 2. Energy penalty (reboiler duty in MW)
        r_energy = -cfg.w_energy * (result.reboiler_duty / 1e6)

        # 3. Flow reward — log-scale above minimum, linear penalty below
        flow = result.distillate_flow
        if flow < cfg.min_distillate_flow:
            r_flow = -cfg.w_flow * (cfg.min_distillate_flow - flow) / cfg.min_distillate_flow
        else:
            r_flow = cfg.w_flow * np.log(flow / cfg.min_distillate_flow + 1.0)

        total = r_purity + r_energy + r_flow
        return total, {
            "purity":  round(float(r_purity), 4),
            "energy":  round(float(r_energy), 4),
            "flow":    round(float(r_flow),   4),
            "total":   round(float(total),    4),
        }

    def _purity_reward(self, purity: float, target: float) -> float:
        """Smooth reward in [0, 1] centred on target purity."""
        t = self.cfg.purity_reward_type
        if t == "step":
            return 1.0 if purity >= target else 0.0
        elif t == "smooth":
            k = 30.0  # steepness — tune in notebook 02
            return float(1.0 / (1.0 + np.exp(-k * (purity - target))))
        elif t == "exponential":
            return float(np.exp(-10.0 * max(0.0, target - purity)))
        return 0.0
