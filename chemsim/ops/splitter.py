"""Stream splitter: 1 inlet → N outlets with fixed flow fractions."""
from __future__ import annotations

import copy
from typing import List

from chemsim.core import Stream
from chemsim.ops.base import IUnitOp


class SplitterOp(IUnitOp):
    def __init__(self, fractions: List[float]):
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("SplitterOp: fractions must sum to 1")
        self._fractions = list(fractions)
        self._inlet = Stream()
        self._outlets: List[Stream] = [Stream() for _ in fractions]

    def inlet_ports(self) -> List[str]:
        return ["in"]

    def outlet_ports(self) -> List[str]:
        return [f"out{i}" for i in range(len(self._fractions))]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port != "in":
            raise ValueError(f"SplitterOp: unknown inlet port '{port}'")
        self._inlet = stream

    def get_outlet(self, port: str) -> Stream:
        for i, name in enumerate(self.outlet_ports()):
            if port == name:
                return self._outlets[i]
        raise ValueError(f"SplitterOp: unknown outlet port '{port}'")

    def solve(self) -> None:
        for i, frac in enumerate(self._fractions):
            out = copy.copy(self._inlet)
            out.total_flow = self._inlet.total_flow * frac
            self._outlets[i] = out
