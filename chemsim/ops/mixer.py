"""Adiabatic mixer: N inlets → 1 outlet."""
from __future__ import annotations

from typing import Dict, List

from chemsim.core import Phase, Stream
from chemsim.ops.base import IUnitOp
from chemsim.thermo.flash import FlashCalculator


class MixerOp(IUnitOp):
    def __init__(self, fc: FlashCalculator, inlet_ports: List[str]):
        self._fc = fc
        self._inlets: List[str] = list(inlet_ports)
        self._inlet_streams: Dict[str, Stream] = {}
        self._outlet = Stream()

    def inlet_ports(self) -> List[str]:
        return list(self._inlets)

    def outlet_ports(self) -> List[str]:
        return ["out"]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port not in self._inlets:
            raise ValueError(f"MixerOp: unknown inlet port '{port}'")
        self._inlet_streams[port] = stream

    def get_outlet(self, port: str) -> Stream:
        if port != "out":
            raise ValueError(f"MixerOp: unknown outlet port '{port}'")
        return self._outlet

    def solve(self) -> None:
        streams = [self._inlet_streams[p]
                   for p in self._inlets
                   if p in self._inlet_streams
                   and self._inlet_streams[p].total_flow > 0.0]
        if not streams:
            raise ValueError("MixerOp: no active inlet streams")

        nc = streams[0].n_comp()
        F_out = sum(s.total_flow for s in streams)
        n_out = [sum(s.total_flow * s.z[i] for s in streams) for i in range(nc)]
        z_out = [ni / F_out for ni in n_out]
        H_mix = sum(s.total_flow * s.H for s in streams) / F_out

        # flashPH to find outlet T and phase split
        T_guess = sum(s.total_flow * s.T for s in streams) / F_out
        P_out = min(s.P for s in streams)
        r = self._fc.flash_PH(P_out, H_mix, z_out, T_guess)

        out = Stream()
        out.total_flow = F_out
        out.z = z_out
        out.P = P_out
        out.T = r.T
        out.vapor_fraction = r.beta
        out.x = r.x; out.y = r.y
        out.phase = (Phase.LIQUID if r.beta < 1e-10
                     else Phase.VAPOR if r.beta > 1.0 - 1e-10
                     else Phase.MIXED)
        out.H = self._fc.total_enthalpy(r)
        out.S = self._fc.total_entropy(r)
        self._outlet = out
