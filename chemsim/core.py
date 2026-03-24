"""Core data types: Component, Stream, Phase, and ideal-gas mixture functions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

R = 8.314462618  # J/mol/K
P_REF = 101325.0  # Pa
T_REF = 298.15   # K


class Phase(Enum):
    VAPOR = "VAPOR"
    LIQUID = "LIQUID"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Component:
    id: str
    name: str
    MW: float       # g/mol
    Tc: float       # K
    Pc: float       # Pa
    omega: float    # acentric factor
    # DIPPR Aly-Lee Cp coefficients [J/kmol/K]
    Cp1: float = 0.0
    Cp2: float = 0.0
    Cp3: float = 0.0
    Cp4: float = 0.0
    Cp5: float = 0.0
    # Antoine vapour pressure (log10 P_vap = A - B/(T+C), P mmHg, T °C)
    Antoine_A: float = 0.0
    Antoine_B: float = 0.0
    Antoine_C: float = 0.0

    @classmethod
    def from_dict(cls, comp_id: str, d: dict) -> Component:
        return cls(
            id=comp_id,
            name=d.get("name", comp_id),
            MW=d["MW"], Tc=d["Tc"], Pc=d["Pc"], omega=d["omega"],
            Cp1=d.get("Cp1", 0.0), Cp2=d.get("Cp2", 0.0),
            Cp3=d.get("Cp3", 0.0), Cp4=d.get("Cp4", 0.0),
            Cp5=d.get("Cp5", 0.0),
            Antoine_A=d.get("Antoine_A", 0.0),
            Antoine_B=d.get("Antoine_B", 0.0),
            Antoine_C=d.get("Antoine_C", 0.0),
        )


@dataclass
class Stream:
    name: str = ""
    T: float = 300.0
    P: float = 101325.0
    total_flow: float = 0.0   # mol/s  (Python API uses snake_case)
    z: List[float] = field(default_factory=list)  # overall mole fractions
    vapor_fraction: float = 0.0  # β
    x: List[float] = field(default_factory=list)  # liquid mole fractions
    y: List[float] = field(default_factory=list)  # vapor mole fractions
    phase: Phase = Phase.UNKNOWN
    H: float = 0.0   # J/mol
    S: float = 0.0   # J/mol/K

    def n_comp(self) -> int:
        return len(self.z)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "T": self.T,
            "P": self.P,
            "total_flow": self.total_flow,
            "z": list(self.z),
            "phase": self.phase.value,
            "vapor_fraction": self.vapor_fraction,
            "x": list(self.x),
            "y": list(self.y),
            "H": self.H,
            "S": self.S,
        }

    def __repr__(self) -> str:
        return (f"<Stream '{self.name}'"
                f" F={self.total_flow:.2f} mol/s"
                f" T={self.T:.2f} K"
                f" P={self.P/1e5:.2f} bar"
                f" phase={self.phase.value}>")

    @staticmethod
    def converged(a: Stream, b: Stream,
                  tol_T: float = 0.01, tol_P: float = 1.0,
                  tol_z: float = 1e-6) -> bool:
        if abs(a.T - b.T) > tol_T:
            return False
        if abs(a.P - b.P) > tol_P:
            return False
        if len(a.z) != len(b.z):
            return False
        return all(abs(a.z[i] - b.z[i]) <= tol_z for i in range(len(a.z)))


# ── Mixture ideal-gas thermodynamics ─────────────────────────────────────────

def ideal_gas_cp(c: Component, T: float) -> float:
    """DIPPR Aly-Lee Cp for one component [J/mol/K]."""
    u = c.Cp3 / T
    v = c.Cp5 / T
    cp_kmol = (c.Cp1
               + c.Cp2 * (u / math.sinh(u)) ** 2
               + c.Cp4 * (v / math.cosh(v)) ** 2)
    return cp_kmol / 1000.0


def ideal_gas_cp_mix(comps: List[Component], z: List[float], T: float) -> float:
    return sum(z[i] * ideal_gas_cp(comps[i], T) for i in range(len(comps)))


def ideal_gas_H_pure(c: Component, T: float, T_ref: float = T_REF) -> float:
    """Analytical integral of Aly-Lee Cp from T_ref to T [J/mol]."""
    def coth(x: float) -> float:
        return math.cosh(x) / math.sinh(x)

    dH = (c.Cp1 * (T - T_ref)
          + c.Cp2 * c.Cp3 * (coth(c.Cp3 / T) - coth(c.Cp3 / T_ref))
          - c.Cp4 * c.Cp5 * (math.tanh(c.Cp5 / T) - math.tanh(c.Cp5 / T_ref)))
    return dH / 1000.0


def ideal_gas_H(comps: List[Component], z: List[float],
                T: float, T_ref: float = T_REF) -> float:
    """Mixture ideal-gas enthalpy relative to T_ref [J/mol]."""
    return sum(z[i] * ideal_gas_H_pure(comps[i], T, T_ref) for i in range(len(comps)))


def ideal_gas_S_pure(c: Component, T: float, T_ref: float = T_REF) -> float:
    """Integral of Cp/T using 20-interval Simpson's rule [J/mol/K]."""
    if abs(T - T_ref) < 1e-10:
        return 0.0
    N = 20
    h = (T - T_ref) / N

    def f(t: float) -> float:
        return ideal_gas_cp(c, t) / t

    s = f(T_ref) + f(T)
    for i in range(1, N):
        s += (2.0 if i % 2 == 0 else 4.0) * f(T_ref + i * h)
    return s * h / 3.0


def ideal_gas_S(comps: List[Component], z: List[float],
                T: float, T_ref: float = T_REF) -> float:
    """Mixture ideal-gas entropy with mixing term (no P term) [J/mol/K]."""
    S = 0.0
    for i in range(len(comps)):
        S += z[i] * ideal_gas_S_pure(comps[i], T, T_ref)
        if z[i] > 1e-15:
            S -= R * z[i] * math.log(z[i])
    return S
