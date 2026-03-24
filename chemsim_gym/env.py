"""Gymnasium environment wrapping the ChemSim distillation simulator."""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any

from .simulator_bridge import SimulatorBridge
from .reward import RewardFunction, RewardConfig
from .spaces import ProcessSpaces


class ProcessSimEnv(gym.Env):
    """
    OpenAI Gymnasium environment wrapping the ChemSim C++ process simulator.

    Each episode the agent proposes 4 operating conditions (reflux ratio,
    distillate fraction, feed T, feed P).  The simulator evaluates one
    steady-state solve and returns a reward based on purity and energy.

    In bandit mode (max_steps=1) each episode is a single query;
    increase max_steps for sequential refinement.

    Parameters
    ----------
    flowsheet_json : str
        Path to the JSON flowsheet definition.
    component_db : str
        Path to the component database JSON.
    col_unit : str
        Name of the DistillationColumn unit in the flowsheet.
    feed_stream : str
        Name of the feed stream in the flowsheet.
    reward_config : dict
        Keyword arguments forwarded to RewardConfig.
    max_steps : int
        Steps per episode (1 = bandit mode).
    render_mode : str | None
        "ansi" for text summary, None for no rendering.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        flowsheet_json: str,
        component_db: str,
        col_unit: str = "COL1",
        feed_stream: str = "FEED",
        distillate_stream: str = "DISTILLATE",
        bottoms_stream: str = "BOTTOMS",
        reward_config: Optional[Dict] = None,
        max_steps: int = 1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.sim = SimulatorBridge(
            flowsheet_json, component_db,
            col_unit=col_unit,
            feed_stream=feed_stream,
            distillate_stream=distillate_stream,
            bottoms_stream=bottoms_stream,
        )
        self.reward_fn  = RewardFunction(**(reward_config or {}))
        self.max_steps  = max_steps
        self.render_mode = render_mode
        self._step_count = 0

        spaces = ProcessSpaces(self.sim.n_components())
        self.action_space      = spaces.action_space()
        self.observation_space = spaces.observation_space()
        self._spaces = spaces

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        result = self.sim.reset()
        obs  = self._build_obs(result)
        info = {"converged": result.converged, "step": 0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        action_phys = self._spaces.denormalize_action(np.asarray(action, dtype=np.float32))

        result = self.sim.run(
            reflux_ratio    = float(action_phys[0]),
            distillate_frac = float(action_phys[1]),
            T_feed          = float(action_phys[2]),
            P_feed          = float(action_phys[3]),
        )

        obs    = self._build_obs(result)
        reward, reward_info = self.reward_fn.compute(result, action_phys)

        terminated = self._step_count >= self.max_steps
        truncated  = not result.converged   # end early and penalise

        info = {
            "converged":         result.converged,
            "distillate_purity": result.distillate_purity,
            "reboiler_duty_kW":  result.reboiler_duty / 1e3,
            "distillate_flow":   result.distillate_flow,
            "reward_breakdown":  reward_info,
            "action_physical":   action_phys.tolist(),
            "step":              self._step_count,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            return self.sim.summary()
        return None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_obs(self, result) -> np.ndarray:
        obs = np.concatenate([
            result.distillate_z,
            result.bottoms_z,
            [
                result.T_top,
                result.T_mid,
                result.T_bottom,
                result.reboiler_duty  / 1e6,   # → MW
                result.condenser_duty / 1e6,   # → MW (negative)
                float(result.converged),
            ],
        ]).astype(np.float32)
        return obs
