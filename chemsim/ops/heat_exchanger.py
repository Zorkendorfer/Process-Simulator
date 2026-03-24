"""Two-stream heat exchanger."""
from __future__ import annotations

import copy
from enum import Enum

from chemsim.core import Phase, Stream
from chemsim.ops.base import IUnitOp
from chemsim.thermo.flash import FlashCalculator


class HXSpec(Enum):
    DUTY = "DUTY"
    HOT_OUTLET_T = "HOT_OUTLET_T"


class HeatExchangerOp(IUnitOp):
    def __init__(self, spec: HXSpec,
                 fc_hot: FlashCalculator,
                 fc_cold: FlashCalculator):
        self._spec = spec
        self._fc_hot = fc_hot
        self._fc_cold = fc_cold
        self.hot_in = Stream()
        self.cold_in = Stream()
        self.hot_out = Stream()
        self.cold_out = Stream()
        self.Q_spec: float = 0.0      # W, used when spec=DUTY
        self.T_hot_out: float = 300.0  # K, used when spec=HOT_OUTLET_T
        self.duty: float = 0.0

    def inlet_ports(self):
        return ["hot_in", "cold_in"]

    def outlet_ports(self):
        return ["hot_out", "cold_out"]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port == "hot_in":
            self.hot_in = stream
        elif port == "cold_in":
            self.cold_in = stream
        else:
            raise ValueError(f"HeatExchangerOp: unknown inlet port '{port}'")

    def get_outlet(self, port: str) -> Stream:
        if port == "hot_out":
            return self.hot_out
        if port == "cold_out":
            return self.cold_out
        raise ValueError(f"HeatExchangerOp: unknown outlet port '{port}'")

    @staticmethod
    def _stream_H(fc: FlashCalculator, s: Stream) -> float:
        r = fc.flash_TP(s.T, s.P, s.z)
        return fc.total_enthalpy(r)

    def solve(self) -> None:
        if self.hot_in.total_flow <= 0.0:
            raise ValueError("HeatExchangerOp: hot_in.total_flow <= 0")
        if self.cold_in.total_flow <= 0.0:
            raise ValueError("HeatExchangerOp: cold_in.total_flow <= 0")

        H_hot_in  = self._stream_H(self._fc_hot,  self.hot_in)
        H_cold_in = self._stream_H(self._fc_cold, self.cold_in)

        if self._spec == HXSpec.DUTY:
            Q = self.Q_spec
        else:
            r_h = self._fc_hot.flash_TP(self.T_hot_out, self.hot_in.P, self.hot_in.z)
            H_hot_out_spec = self._fc_hot.total_enthalpy(r_h)
            Q = self.hot_in.total_flow * (H_hot_in - H_hot_out_spec)

        self.duty = Q

        H_hot_out  = H_hot_in  - Q / self.hot_in.total_flow
        H_cold_out = H_cold_in + Q / self.cold_in.total_flow

        self.hot_out  = self._flashPH_stream(self._fc_hot,  self.hot_in,  H_hot_out)
        self.cold_out = self._flashPH_stream(self._fc_cold, self.cold_in, H_cold_out)

    def _flashPH_stream(self, fc: FlashCalculator,
                        inlet: Stream, H_out: float) -> Stream:
        r = fc.flash_PH(inlet.P, H_out, inlet.z, inlet.T)
        out = copy.copy(inlet)
        out.T = r.T; out.vapor_fraction = r.beta
        out.x = r.x; out.y = r.y
        out.phase = (Phase.LIQUID if r.beta < 1e-10
                     else Phase.VAPOR if r.beta > 1.0 - 1e-10
                     else Phase.MIXED)
        out.H = fc.total_enthalpy(r)
        out.S = fc.total_entropy(r)
        return out
