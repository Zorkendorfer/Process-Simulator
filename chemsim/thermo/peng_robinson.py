"""Peng-Robinson equation of state."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from chemsim.core import Component, R


def _cbrt(x: float) -> float:
    """Real cube root (handles negative values)."""
    if x >= 0.0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


class PengRobinson:
    """Peng-Robinson EOS with van der Waals mixing rules."""

    def __init__(self, components: List[Component],
                 kij: Optional[np.ndarray] = None):
        self.comps = components
        n = len(components)
        self.kij: np.ndarray = kij if kij is not None else np.zeros((n, n))

        self._a_c: List[float] = []
        self._b_c: List[float] = []
        self._kappa: List[float] = []
        for c in components:
            self._a_c.append(0.45724 * R ** 2 * c.Tc ** 2 / c.Pc)
            self._b_c.append(0.07780 * R * c.Tc / c.Pc)
            self._kappa.append(0.37464 + 1.54226 * c.omega - 0.26992 * c.omega ** 2)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _alpha(self, i: int, T: float) -> float:
        x = 1.0 + self._kappa[i] * (1.0 - math.sqrt(T / self.comps[i].Tc))
        return x * x

    def _dalpha_dT(self, i: int, T: float) -> float:
        sqrt_TTc = math.sqrt(T / self.comps[i].Tc)
        x = 1.0 + self._kappa[i] * (1.0 - sqrt_TTc)
        return -self._kappa[i] * x / math.sqrt(T * self.comps[i].Tc)

    def _a_i(self, i: int, T: float) -> float:
        return self._a_c[i] * self._alpha(i, T)

    def _mix_a(self, z: List[float], T: float) -> float:
        n = len(self.comps)
        am = 0.0
        for i in range(n):
            for j in range(n):
                am += (z[i] * z[j]
                       * math.sqrt(self._a_i(i, T) * self._a_i(j, T))
                       * (1.0 - self.kij[i, j]))
        return am

    def _dmix_a_dT(self, z: List[float], T: float) -> float:
        n = len(self.comps)
        dam = 0.0
        for i in range(n):
            for j in range(n):
                ai = self._a_i(i, T); aj = self._a_i(j, T)
                dai = self._a_c[i] * self._dalpha_dT(i, T)
                daj = self._a_c[j] * self._dalpha_dT(j, T)
                d_sqrt = (dai * aj + ai * daj) / (2.0 * math.sqrt(ai * aj))
                dam += z[i] * z[j] * d_sqrt * (1.0 - self.kij[i, j])
        return dam

    def _mix_b(self, z: List[float]) -> float:
        return sum(z[i] * self._b_c[i] for i in range(len(self.comps)))

    def _dAdn_i(self, i: int, z: List[float], T: float) -> float:
        """∂(n·a_mix)/∂n_i = 2 Σ_j z_j √(a_i·a_j)(1-kij)"""
        n = len(self.comps)
        val = sum(z[j] * math.sqrt(self._a_i(i, T) * self._a_i(j, T))
                  * (1.0 - self.kij[i, j])
                  for j in range(n))
        return 2.0 * val

    # ── Cubic solver ─────────────────────────────────────────────────────────

    @staticmethod
    def _solve_cubic(p: float, q: float, r: float) -> List[float]:
        """Solve Z³ + p Z² + q Z + r = 0 (Cardano / trig method)."""
        a = q - p * p / 3.0
        b = r + 2.0 * p ** 3 / 27.0 - p * q / 3.0
        disc = b ** 2 / 4.0 + a ** 3 / 27.0

        if disc > 1e-14:
            sq = math.sqrt(disc)
            roots = [_cbrt(-b / 2.0 + sq) + _cbrt(-b / 2.0 - sq) - p / 3.0]
        else:
            r_val = math.sqrt(max(0.0, -a ** 3 / 27.0))
            arg = max(-1.0, min(1.0, -b / (2.0 * r_val))) if r_val > 0 else 0.0
            theta = math.acos(arg)
            m = 2.0 * _cbrt(r_val)
            PI = math.pi
            roots = [
                m * math.cos(theta / 3.0) - p / 3.0,
                m * math.cos((theta + 2.0 * PI) / 3.0) - p / 3.0,
                m * math.cos((theta + 4.0 * PI) / 3.0) - p / 3.0,
            ]

        roots.sort()
        return [z for z in roots if z > 0.0]

    @staticmethod
    def _pick_Z(roots: List[float], liquid: bool) -> float:
        if not roots:
            raise RuntimeError("PengRobinson: no positive Z roots found")
        if len(roots) == 1:
            return roots[0]
        return roots[0] if liquid else roots[-1]

    # ── Public interface ──────────────────────────────────────────────────────

    def compressibility_factors(self, T: float, P: float,
                                z: List[float]) -> Tuple[float, float]:
        am = self._mix_a(z, T)
        bm = self._mix_b(z)
        A = am * P / (R * T) ** 2
        B = bm * P / (R * T)
        p = -(1.0 - B)
        q = A - 3.0 * B ** 2 - 2.0 * B
        r = -(A * B - B ** 2 - B ** 3)
        roots = self._solve_cubic(p, q, r)
        return self._pick_Z(roots, True), self._pick_Z(roots, False)

    def ln_fugacity_coefficients(self, T: float, P: float,
                                 z: List[float], liquid: bool) -> List[float]:
        n = len(self.comps)
        am = self._mix_a(z, T)
        bm = self._mix_b(z)
        A = am * P / (R * T) ** 2
        B = bm * P / (R * T)
        Z_L, Z_V = self.compressibility_factors(T, P, z)
        Z = Z_L if liquid else Z_V
        # Guard against unphysical Z (e.g. liquid root below co-volume at high T)
        Z = max(Z, B + 1e-10)
        sqrt2 = math.sqrt(2.0)

        lnphi = []
        for i in range(n):
            bi_b = self._b_c[i] / bm
            dAdn = self._dAdn_i(i, z, T) * P / (R * T) ** 2
            lnphi_i = (
                bi_b * (Z - 1.0)
                - math.log(Z - B)
                - A / (2.0 * sqrt2 * B)
                  * (dAdn / A - bi_b)
                  * math.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
            )
            lnphi.append(lnphi_i)
        return lnphi

    def enthalpy_departure(self, T: float, P: float,
                           z: List[float], liquid: bool) -> float:
        am = self._mix_a(z, T)
        bm = self._mix_b(z)
        dam = self._dmix_a_dT(z, T)
        B = bm * P / (R * T)
        Z_L, Z_V = self.compressibility_factors(T, P, z)
        Z = Z_L if liquid else Z_V
        Z = max(Z, B + 1e-10)
        sqrt2 = math.sqrt(2.0)
        return (
            R * T * (Z - 1.0)
            - (am - T * dam) / (2.0 * sqrt2 * bm)
              * math.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
        )

    def entropy_departure(self, T: float, P: float,
                          z: List[float], liquid: bool) -> float:
        bm = self._mix_b(z)
        dam = self._dmix_a_dT(z, T)
        B = bm * P / (R * T)
        Z_L, Z_V = self.compressibility_factors(T, P, z)
        Z = Z_L if liquid else Z_V
        Z = max(Z, B + 1e-10)
        sqrt2 = math.sqrt(2.0)
        return (
            R * math.log(Z - B)
            - dam / (2.0 * sqrt2 * bm)
              * math.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
        )
