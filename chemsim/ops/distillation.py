"""Simplified distillation column (Constant Molar Overflow, Wang-Henke)."""
from __future__ import annotations

import math
from typing import List

from scipy.optimize import brentq

from chemsim.core import Component, Phase, Stream
from chemsim.ops.base import IUnitOp
from chemsim.thermo.flash import FlashCalculator


def _wilson_K_i(c: Component, T: float, P: float) -> float:
    return (c.Pc / P) * math.exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T))


def _thomas(lower: List[float], diag: List[float],
            upper: List[float], rhs: List[float]) -> List[float]:
    """Thomas algorithm for tridiagonal system."""
    n = len(diag)
    w = [0.0] * n
    g = [0.0] * n
    x = [0.0] * n

    d0 = diag[0] if abs(diag[0]) > 1e-30 else 1e-30
    w[0] = upper[0] / d0
    g[0] = rhs[0]   / d0

    for j in range(1, n):
        dj = diag[j] - lower[j] * w[j - 1]
        if abs(dj) < 1e-30:
            dj = math.copysign(1e-30, dj) if dj != 0.0 else 1e-30
        w[j] = upper[j] / dj
        g[j] = (rhs[j] - lower[j] * g[j - 1]) / dj

    x[n - 1] = g[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = g[j] - w[j] * x[j + 1]
    return x


class DistillationColumnOp(IUnitOp):
    """
    Inlet:   "feed"
    Outlets: "distillate", "bottoms"

    CMO model with Wilson K-values, Wang-Henke outer iterations.
    """

    def __init__(self, fc: FlashCalculator, comps: List[Component],
                 N_stages: int, feed_stage: int,
                 reflux_ratio: float, distillate_frac: float,
                 P_top: float = 101325.0,
                 feed_quality: float = 1.0,
                 max_iter: int = 50):
        if N_stages < 2:
            raise ValueError("DistillationColumnOp: need at least 2 stages")
        if not (1 <= feed_stage <= N_stages):
            raise ValueError("DistillationColumnOp: feed_stage out of range [1,N]")
        if reflux_ratio <= 0.0:
            raise ValueError("DistillationColumnOp: reflux_ratio must be > 0")
        if not (0.0 < distillate_frac < 1.0):
            raise ValueError("DistillationColumnOp: distillate_frac must be in (0,1)")

        self._fc = fc
        self._comps = comps
        self._N = N_stages
        self._f = feed_stage
        self._R = reflux_ratio
        self._phi = distillate_frac
        self._P = P_top
        self._q = feed_quality
        self._max_iter = max_iter

        self.feed = Stream()
        self.distillate = Stream()
        self.bottoms = Stream()
        self._T_stages: List[float] = []
        self._Q_reboiler: float = 0.0
        self._Q_condenser: float = 0.0

    # ── RL interface ──────────────────────────────────────────────────────────

    def set_reflux_ratio(self, R: float) -> None:
        self._R = R

    def set_distillate_frac(self, phi: float) -> None:
        self._phi = phi

    def T_top(self) -> float:
        return self._T_stages[1] if len(self._T_stages) > 1 else 0.0

    def T_mid(self) -> float:
        mid = self._N // 2
        return self._T_stages[mid] if len(self._T_stages) > mid else 0.0

    def T_bottom(self) -> float:
        return self._T_stages[self._N] if len(self._T_stages) > self._N else 0.0

    def reboiler_duty(self) -> float:
        return self._Q_reboiler

    def condenser_duty(self) -> float:
        return self._Q_condenser

    # ── IUnitOp interface ─────────────────────────────────────────────────────

    def inlet_ports(self):
        return ["feed"]

    def outlet_ports(self):
        return ["distillate", "bottoms"]

    def set_inlet(self, port: str, stream: Stream) -> None:
        if port != "feed":
            raise ValueError(f"DistillationColumnOp: unknown inlet port '{port}'")
        self.feed = stream

    def get_outlet(self, port: str) -> Stream:
        if port == "distillate":
            return self.distillate
        if port == "bottoms":
            return self.bottoms
        raise ValueError(f"DistillationColumnOp: unknown outlet port '{port}'")

    # ── Bubble-T helper ───────────────────────────────────────────────────────

    def _bubble_T(self, x: List[float], P: float) -> float:
        nc = len(self._comps)
        def f(T):
            return sum(_wilson_K_i(self._comps[i], T, P) * x[i] for i in range(nc)) - 1.0
        return brentq(f, 100.0, 800.0, xtol=1e-8)

    # ── Main solve ────────────────────────────────────────────────────────────

    def solve(self) -> None:
        if self.feed.total_flow <= 0.0:
            raise ValueError("DistillationColumnOp: feed.total_flow must be > 0")

        nc = len(self._comps)
        N = self._N
        F = self.feed.total_flow
        D = self._phi * F
        B = (1.0 - self._phi) * F

        # CMO flow rates (1-indexed)
        L  = self._R * D
        V  = L + D
        Lp = L + self._q * F
        Vp = V - (1.0 - self._q) * F
        Bflow = Lp - Vp

        if Vp <= 0.0:
            raise RuntimeError(
                "DistillationColumnOp: stripping-section vapor flow <= 0 "
                "(check R, phi, q)")

        Lflow = [0.0] * (N + 1)
        Vflow = [0.0] * (N + 1)
        for j in range(1, N + 1):
            if j < self._f:
                Lflow[j] = L; Vflow[j] = V
            else:
                Lflow[j] = Lp; Vflow[j] = Vp

        # Initial temperature profile: linear from distillate bubble point to bottoms
        T_dist_approx = self._bubble_T(self.feed.z, self._P)
        xB_approx = self.feed.z
        T_bot_approx = self._bubble_T(xB_approx, self._P)
        T_stages_init = [T_dist_approx + (T_bot_approx - T_dist_approx) * (j - 1) / (N - 1)
                         if N > 1 else T_dist_approx
                         for j in range(1, N + 1)]

        # Initial K from Wilson at stage temperatures (1-indexed: Kst[j][i])
        Kst = [[_wilson_K_i(self._comps[i], T_stages_init[j - 1], self._P)
                for i in range(nc)]
               for j in range(N + 1)]

        # Stage liquid compositions from equilibrium with vapor
        x_stage = [[1.0 / nc] * nc for _ in range(N + 1)]
        for j in range(1, N + 1):
            for i in range(nc):
                x_stage[j][i] = self.feed.z[i] / Kst[j][i]
            s = sum(x_stage[j])
            x_stage[j] = [xi / s for xi in x_stage[j]]

        # Wang-Henke outer loop
        for it in range(self._max_iter):
            for i in range(nc):
                lo = [0.0] * N
                di = [0.0] * N
                up = [0.0] * N
                rhs = [0.0] * N

                for j1 in range(1, N + 1):
                    jj = j1 - 1
                    Ki_j  = Kst[j1][i]
                    Ki_jp1 = Kst[j1 + 1][i] if j1 < N else 0.0
                    L_jm1 = Lflow[j1 - 1] if j1 > 1 else L
                    V_jp1 = Vflow[j1 + 1] if j1 < N else 0.0
                    L_j   = Bflow if j1 == N else Lflow[j1]

                    lo[jj]  = L_jm1
                    di[jj]  = -(Vflow[j1] * Ki_j + L_j)
                    up[jj]  = V_jp1 * Ki_jp1

                    if j1 == self._f:
                        rhs[jj] = -F * self.feed.z[i]

                # Total condenser boundary: x_0 = K_1*x_1
                # lo[0]*x_0 term → lo[0]*K_1*x_1 absorbed into diagonal
                di[0] += lo[0] * Kst[1][i]
                lo[0] = 0.0

                x_i = _thomas(lo, di, up, rhs)
                for j in range(N):
                    x_stage[j + 1][i] = x_i[j]

            # Normalise and compute component flows for mass balance check
            for j in range(1, N + 1):
                s = sum(max(x_stage[j][i], 1e-12) for i in range(nc))
                x_stage[j] = [max(x_stage[j][i], 1e-12) / s for i in range(nc)]

            # Update K (skip on last iteration)
            if it < self._max_iter - 1:
                err = 0.0
                for j in range(1, N + 1):
                    Tj = self._bubble_T(x_stage[j], self._P)
                    for i in range(nc):
                        K_new = _wilson_K_i(self._comps[i], Tj, self._P)
                        err += abs(K_new - Kst[j][i]) / (abs(Kst[j][i]) + 1e-10)
                        Kst[j][i] = K_new
                if err / (N * nc) < 1e-8:
                    break

        # Stage temperatures
        self._T_stages = [0.0] + [self._bubble_T(x_stage[j], self._P)
                                   for j in range(1, N + 1)]

        # Distillate (total condenser) and bottoms compositions
        xD_raw = [Kst[1][i] * x_stage[1][i] for i in range(nc)]
        sD = sum(xD_raw); xD = [v / sD for v in xD_raw]
        xB = list(x_stage[N])

        T_dist    = self._bubble_T(xD, self._P)
        T_bottoms = self._bubble_T(xB, self._P)

        self.distillate = Stream()
        self.distillate.z = self.distillate.x = self.distillate.y = xD
        self.distillate.T = T_dist; self.distillate.P = self._P
        self.distillate.total_flow = D
        self.distillate.vapor_fraction = 0.0
        self.distillate.phase = Phase.LIQUID
        self.distillate.H = self._fc.phase_enthalpy(T_dist, self._P, xD, True)
        self.distillate.S = self._fc.phase_entropy(T_dist, self._P, xD, True)

        self.bottoms = Stream()
        self.bottoms.z = self.bottoms.x = self.bottoms.y = xB
        self.bottoms.T = T_bottoms; self.bottoms.P = self._P
        self.bottoms.total_flow = B
        self.bottoms.vapor_fraction = 0.0
        self.bottoms.phase = Phase.LIQUID
        self.bottoms.H = self._fc.phase_enthalpy(T_bottoms, self._P, xB, True)
        self.bottoms.S = self._fc.phase_entropy(T_bottoms, self._P, xB, True)

        # Heat duties
        r_feed = self._fc.flash_TP(self.feed.T, self.feed.P, self.feed.z)
        H_feed = self._fc.total_enthalpy(r_feed)

        T_top = self._bubble_T(x_stage[1], self._P)
        H_Vtop = self._fc.phase_enthalpy(T_top, self._P, xD, False)
        self._Q_condenser = V * (self.distillate.H - H_Vtop)
        self._Q_reboiler  = (D * self.distillate.H + B * self.bottoms.H
                              - F * H_feed - self._Q_condenser)

    # ── IUnitOp helpers for flowsheet.get_unit_scalar ─────────────────────────

    def get_scalar(self, key: str) -> float:
        scalars = {
            "T_top":         self.T_top,
            "T_mid":         self.T_mid,
            "T_bottom":      self.T_bottom,
            "reboilerDuty":  self.reboiler_duty,
            "condenserDuty": self.condenser_duty,
        }
        if key not in scalars:
            raise KeyError(f"DistillationColumnOp: unknown scalar '{key}'")
        return scalars[key]()
