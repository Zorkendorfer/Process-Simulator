"""Two-outlet flash separator (TP, PH, or PS specification)."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from chemsim.core import Phase, Stream
from chemsim.ops.base import IUnitOp
from chemsim.thermo.flash import FlashCalculator, FlashResult


class FlashSpec(Enum):
    TP = "TP"
    PH = "PH"
    PS = "PS"


class FlashDrumOp(IUnitOp):
    def __init__(self, spec: FlashSpec, fc: FlashCalculator,
                 spec_value: float, P: float):
        """
        spec_value is T for TP, H for PH, S for PS.
        P is always specified.
        """
        self._spec = spec
        self._fc = fc
        self._spec_value = spec_value
        self._P = P
        self.feed = Stream()
        self.vapor = Stream()
        self.liquid = Stream()
        self.duty: float = 0.0

    def inlet_ports(self):
        return ["feed"]

    def outlet_ports(self):
        return ["vapor", "liquid"]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port != "feed":
            raise ValueError(f"FlashDrumOp: unknown inlet port '{port}'")
        self.feed = stream

    def get_outlet(self, port: str) -> Stream:
        if port == "vapor":
            return self.vapor
        if port == "liquid":
            return self.liquid
        raise ValueError(f"FlashDrumOp: unknown outlet port '{port}'")

    def solve(self) -> None:
        if self.feed.total_flow <= 0.0:
            raise ValueError("FlashDrumOp: feed.total_flow must be > 0")

        r_in = self._fc.flash_TP(self.feed.T, self.feed.P, self.feed.z)
        H_in = self._fc.total_enthalpy(r_in)

        if self._spec == FlashSpec.TP:
            r = self._fc.flash_TP(self._spec_value, self._P, self.feed.z)
        elif self._spec == FlashSpec.PH:
            r = self._fc.flash_PH(self._P, self._spec_value, self.feed.z, self.feed.T)
        else:
            r = self._fc.flash_PS(self._P, self._spec_value, self.feed.z, self.feed.T)

        self._populate_outlets(r)
        H_out = self._fc.total_enthalpy(r)
        self.duty = self.feed.total_flow * (H_out - H_in)

    def _populate_outlets(self, r: FlashResult) -> None:
        F = self.feed.total_flow

        self.vapor = Stream()
        self.vapor.T = r.T; self.vapor.P = r.P
        self.vapor.vapor_fraction = 1.0; self.vapor.phase = Phase.VAPOR
        self.vapor.z = self.vapor.y = self.vapor.x = list(r.y)

        self.liquid = Stream()
        self.liquid.T = r.T; self.liquid.P = r.P
        self.liquid.vapor_fraction = 0.0; self.liquid.phase = Phase.LIQUID
        self.liquid.z = self.liquid.x = self.liquid.y = list(r.x)

        self.vapor.total_flow  = F * r.beta
        self.liquid.total_flow = F * (1.0 - r.beta)

        self.vapor.H  = self._fc.phase_enthalpy(r.T, r.P, r.y, False)
        self.liquid.H = self._fc.phase_enthalpy(r.T, r.P, r.x, True)
        self.vapor.S  = self._fc.phase_entropy(r.T, r.P, r.y, False)
        self.liquid.S = self._fc.phase_entropy(r.T, r.P, r.x, True)

        if r.beta >= 1.0 - 1e-10:
            self.vapor.total_flow  = F
            self.liquid.total_flow = 0.0
            self.liquid.z = self.liquid.x = self.liquid.y = list(r.y)
        elif r.beta <= 1e-10:
            self.liquid.total_flow = F
            self.vapor.total_flow  = 0.0
            self.vapor.z = self.vapor.y = self.vapor.x = list(r.x)
