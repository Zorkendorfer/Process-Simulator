"""Isentropic pump with efficiency."""
from __future__ import annotations

from chemsim.core import Phase, Stream
from chemsim.ops.base import IUnitOp
from chemsim.thermo.flash import FlashCalculator


class PumpOp(IUnitOp):
    def __init__(self, fc: FlashCalculator, P_out: float, eta: float = 0.75):
        if not (0.0 < eta <= 1.0):
            raise ValueError("PumpOp: eta must be in (0, 1]")
        self._fc = fc
        self._P_out = P_out
        self._eta = eta
        self.inlet = Stream()
        self.outlet = Stream()
        self.shaft_work_mol: float = 0.0
        self.shaft_power_W: float = 0.0

    def inlet_ports(self):
        return ["in"]

    def outlet_ports(self):
        return ["out"]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port != "in":
            raise ValueError(f"PumpOp: unknown inlet port '{port}'")
        self.inlet = stream

    def get_outlet(self, port: str) -> Stream:
        if port != "out":
            raise ValueError(f"PumpOp: unknown outlet port '{port}'")
        return self.outlet

    def solve(self) -> None:
        if self.inlet.total_flow <= 0.0:
            raise ValueError("PumpOp: inlet.total_flow must be > 0")
        if self._P_out <= self.inlet.P:
            raise ValueError("PumpOp: P_out must be greater than inlet pressure")

        T_in = self.inlet.T; P_in = self.inlet.P; z = self.inlet.z
        H_in = self._fc.phase_enthalpy(T_in, P_in, z, True)
        S_in = self._fc.phase_entropy(T_in, P_in, z, True)

        r_s = self._fc.flash_PS(self._P_out, S_in, z, T_in)
        H_out_s = self._fc.total_enthalpy(r_s)

        W_s = H_out_s - H_in
        W_act = W_s / self._eta
        H_out = H_in + W_act

        r_out = self._fc.flash_PH(self._P_out, H_out, z, r_s.T)

        import copy
        self.outlet = copy.copy(self.inlet)
        self.outlet.P = r_out.P; self.outlet.T = r_out.T
        self.outlet.vapor_fraction = r_out.beta
        self.outlet.x = r_out.x; self.outlet.y = r_out.y
        self.outlet.phase = (Phase.LIQUID if r_out.beta < 1e-10
                              else Phase.VAPOR if r_out.beta > 1.0 - 1e-10
                              else Phase.MIXED)
        self.outlet.H = self._fc.total_enthalpy(r_out)
        self.outlet.S = self._fc.total_entropy(r_out)
        self.shaft_work_mol = W_act
        self.shaft_power_W  = self.inlet.total_flow * W_act
