"""Isothermal reactor with specified stoichiometry and extents."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List

from chemsim.core import Phase, Stream
from chemsim.ops.base import IUnitOp
from chemsim.thermo.flash import FlashCalculator


@dataclass
class Reaction:
    nu: List[float]   # stoichiometric coefficients (length = nComp)


class ReactorOp(IUnitOp):
    def __init__(self, fc: FlashCalculator,
                 T_spec: float, P_spec: float):
        self._fc = fc
        self._T_spec = T_spec
        self._P_spec = P_spec
        self._reactions: List[Reaction] = []
        self._extents: List[float] = []
        self.inlet = Stream()
        self.outlet = Stream()
        self.duty: float = 0.0

    def add_reaction(self, rxn: Reaction, extent_mol_s: float) -> None:
        self._reactions.append(rxn)
        self._extents.append(extent_mol_s)

    def inlet_ports(self):
        return ["in"]

    def outlet_ports(self):
        return ["out"]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port != "in":
            raise ValueError(f"ReactorOp: unknown inlet port '{port}'")
        self.inlet = stream

    def get_outlet(self, port: str) -> Stream:
        if port != "out":
            raise ValueError(f"ReactorOp: unknown outlet port '{port}'")
        return self.outlet

    def solve(self) -> None:
        if self.inlet.total_flow <= 0.0:
            raise ValueError("ReactorOp: inlet.total_flow must be > 0")
        n = len(self.inlet.z)
        for rxn in self._reactions:
            if len(rxn.nu) != n:
                raise ValueError("ReactorOp: reaction nu vector length != nComp")

        n_out = [self.inlet.total_flow * self.inlet.z[i] for i in range(n)]
        for k, rxn in enumerate(self._reactions):
            for i in range(n):
                n_out[i] += rxn.nu[i] * self._extents[k]
        for i, val in enumerate(n_out):
            if val < 0.0:
                raise RuntimeError(f"ReactorOp: negative outlet molar flow for component {i}")

        F_out = sum(n_out)
        if F_out <= 0.0:
            raise RuntimeError("ReactorOp: total outlet flow is non-positive")
        z_out = [ni / F_out for ni in n_out]

        r = self._fc.flash_TP(self._T_spec, self._P_spec, z_out)

        self.outlet = copy.copy(self.inlet)
        self.outlet.total_flow = F_out
        self.outlet.z = z_out
        self.outlet.T = r.T; self.outlet.P = self._P_spec
        self.outlet.vapor_fraction = r.beta
        self.outlet.x = r.x; self.outlet.y = r.y
        self.outlet.phase = (Phase.LIQUID if r.beta < 1e-10
                              else Phase.VAPOR if r.beta > 1.0 - 1e-10
                              else Phase.MIXED)
        self.outlet.H = self._fc.total_enthalpy(r)
        self.outlet.S = self._fc.total_entropy(r)

        r_in = self._fc.flash_TP(self.inlet.T, self.inlet.P, self.inlet.z)
        H_in = self._fc.total_enthalpy(r_in)
        self.duty = F_out * self.outlet.H - self.inlet.total_flow * H_in
