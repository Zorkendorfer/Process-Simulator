"""Thin wrapper around the pybind11 chemsim module for the RL environment."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os

# Allow running from repo root without installing
_repo_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_repo_root, "build"))

# On Windows, the MinGW runtime DLLs must be findable.
# Add the MSYS2 MinGW64 bin directory if it exists.
_mingw_bin = r"C:/msys64/mingw64/bin"
if os.name == "nt" and os.path.isdir(_mingw_bin):
    os.add_dll_directory(_mingw_bin)

import chemsim  # pybind11 module


@dataclass
class SimResult:
    converged: bool
    distillate_z: np.ndarray    # mole fractions
    bottoms_z: np.ndarray
    T_top: float                # K
    T_mid: float                # K
    T_bottom: float             # K
    reboiler_duty: float        # W
    condenser_duty: float       # W  (≤ 0)
    distillate_flow: float      # mol/s
    bottoms_flow: float         # mol/s

    @property
    def distillate_purity(self) -> float:
        """Mole fraction of the lightest component in distillate."""
        return float(self.distillate_z[0]) if len(self.distillate_z) else 0.0

    @property
    def specific_energy(self) -> float:
        """Reboiler duty per mol of distillate [J/mol]."""
        if self.distillate_flow < 1e-10:
            return 1e10
        return self.reboiler_duty / self.distillate_flow


class SimulatorBridge:
    """
    Manages one ChemSim Flowsheet instance.

    The flowsheet JSON must contain a DistillationColumn unit named ``col_unit``
    and a feed stream named ``feed_stream``.
    """

    def __init__(
        self,
        flowsheet_json: str,
        component_db: str,
        col_unit: str = "COL1",
        feed_stream: str = "FEED",
        distillate_stream: str = "DISTILLATE",
        bottoms_stream: str = "BOTTOMS",
    ):
        self._json        = flowsheet_json
        self._db          = component_db
        self._col         = col_unit
        self._feed        = feed_stream
        self._dist_name   = distillate_stream
        self._btms_name   = bottoms_stream

        self.flowsheet    = chemsim.Flowsheet.from_json(flowsheet_json, component_db)
        self._nc          = len(self.flowsheet.get_stream(feed_stream).z)
        self._current: Optional[SimResult] = None

        # Solve once at nominal conditions to populate stream data
        self.flowsheet.solve()
        self._current = self._collect_result(converged=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> SimResult:
        """Reload flowsheet from JSON and solve at nominal conditions.

        Full reload (not just stream reset) guarantees determinism because
        set_param changes to unit ops (reflux ratio, distillate frac) are
        also reset — they are not touched by reset_to_base().
        """
        self.flowsheet = chemsim.Flowsheet.from_json(self._json, self._db)
        try:
            ok = self.flowsheet.solve()
        except Exception:
            ok = False
        self._current = self._collect_result(ok)
        return self._current

    def run(
        self,
        reflux_ratio: float,
        distillate_frac: float,
        T_feed: float,
        P_feed: float,
    ) -> SimResult:
        """Apply operating conditions and run a steady-state solve."""
        try:
            self.flowsheet.set_param(self._col, "refluxRatio",    reflux_ratio)
            self.flowsheet.set_param(self._col, "distillateFrac", distillate_frac)
            self.flowsheet.set_stream_conditions(self._feed, T_feed, P_feed)
            ok = self.flowsheet.solve()
            self._current = self._collect_result(ok)
        except Exception:
            self._current = self._failed_result()
        return self._current

    def current_state(self) -> SimResult:
        return self._current if self._current is not None else self._failed_result()

    def n_components(self) -> int:
        return self._nc

    def summary(self) -> str:
        return self.flowsheet.summary()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _collect_result(self, converged: bool) -> SimResult:
        if not converged:
            return self._failed_result()
        try:
            dist = self.flowsheet.get_stream(self._dist_name)
            btms = self.flowsheet.get_stream(self._btms_name)
            return SimResult(
                converged       = True,
                distillate_z    = np.array(dist.z,         dtype=np.float32),
                bottoms_z       = np.array(btms.z,         dtype=np.float32),
                T_top           = self.flowsheet.get_unit_scalar(self._col, "T_top"),
                T_mid           = self.flowsheet.get_unit_scalar(self._col, "T_mid"),
                T_bottom        = self.flowsheet.get_unit_scalar(self._col, "T_bottom"),
                reboiler_duty   = self.flowsheet.get_unit_scalar(self._col, "reboilerDuty"),
                condenser_duty  = self.flowsheet.get_unit_scalar(self._col, "condenserDuty"),
                distillate_flow = dist.total_flow,
                bottoms_flow    = btms.total_flow,
            )
        except Exception:
            return self._failed_result()

    def _failed_result(self) -> SimResult:
        nc = self._nc
        return SimResult(
            converged       = False,
            distillate_z    = np.ones(nc, dtype=np.float32) / nc,
            bottoms_z       = np.ones(nc, dtype=np.float32) / nc,
            T_top    = 300.0, T_mid = 300.0, T_bottom = 300.0,
            reboiler_duty   = 1e8,
            condenser_duty  = -1e8,
            distillate_flow = 0.0,
            bottoms_flow    = 0.0,
        )
