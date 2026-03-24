"""Action and observation space definitions for the process simulator environment."""

import numpy as np
import gymnasium as gym


# Physical bounds for the 4 action variables
ACTION_LOW  = np.array([0.5,  0.1, 280.0, 5e5],  dtype=np.float32)
ACTION_HIGH = np.array([10.0, 0.9, 400.0, 3e6],  dtype=np.float32)


class ProcessSpaces:
    """
    Defines Gym spaces for the distillation environment.

    Action (4 continuous, normalised to [-1, 1]):
        [reflux_ratio, distillate_frac, T_feed, P_feed]

    Observation (2*NC + 6):
        distillate_z (NC), bottoms_z (NC),
        T_top, T_mid, T_bottom, reboiler_duty_MW, condenser_duty_MW, converged
    """

    def __init__(self, n_components: int):
        self.nc = n_components
        self.obs_dim = 2 * n_components + 6

    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32,
        )

    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

    @staticmethod
    def denormalize_action(action: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] to physical units."""
        return ACTION_LOW + (action + 1.0) / 2.0 * (ACTION_HIGH - ACTION_LOW)

    @staticmethod
    def normalize_action(action_phys: np.ndarray) -> np.ndarray:
        """Map physical action to [-1, 1]."""
        return 2.0 * (action_phys - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0
