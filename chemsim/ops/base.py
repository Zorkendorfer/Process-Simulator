"""Abstract base class for unit operations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from chemsim.core import Stream


class IUnitOp(ABC):
    @abstractmethod
    def inlet_ports(self) -> List[str]: ...

    @abstractmethod
    def outlet_ports(self) -> List[str]: ...

    @abstractmethod
    def set_inlet(self, port: str, stream: Stream) -> None: ...

    @abstractmethod
    def get_outlet(self, port: str) -> Stream: ...

    @abstractmethod
    def solve(self) -> None: ...
