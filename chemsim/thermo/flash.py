"""Flash calculations: TP, PH, PS, bubble/dew point, stability."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from scipy.optimize import brentq

from chemsim.core import Component, R, P_REF, ideal_gas_H, ideal_gas_S
from chemsim.thermo.peng_robinson import PengRobinson


@dataclass
class FlashResult:
    T: float = 0.0
    P: float = 0.0
    beta: float = 0.0        # vapour fraction
    x: List[float] = field(default_factory=list)  # liquid mole fractions
    y: List[float] = field(default_factory=list)  # vapour mole fractions
    K: List[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    H_total: float = 0.0
    S_total: float = 0.0


class FlashCalculator:
    """Vapour-liquid equilibrium calculations using a PR EOS."""

    def __init__(self, eos: PengRobinson, components: List[Component]):
        self.eos = eos
        self.comps = components

    # ── Wilson K-value estimate ───────────────────────────────────────────────

    def wilson_K(self, T: float, P: float) -> List[float]:
        return [
            (c.Pc / P) * math.exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T))
            for c in self.comps
        ]

    # ── Rachford-Rice ─────────────────────────────────────────────────────────

    def rachford_rice(self, z: List[float], K: List[float]) -> float:
        Kmax = max(K)
        Kmin = min(K)
        if Kmax <= 1.0:
            return 0.0
        if Kmin >= 1.0:
            return 1.0
        # Safe bounds: avoid singularities at beta = 1/(1-K[i])
        lo = 1.0 / (1.0 - Kmax) + 1e-6
        hi = 1.0 / (1.0 - Kmin) - 1e-6

        def rr(beta: float) -> float:
            total = 0.0
            for i in range(len(z)):
                denom = 1.0 + beta * (K[i] - 1.0)
                if abs(denom) < 1e-12:
                    denom = math.copysign(1e-12, denom) if denom != 0.0 else 1e-12
                total += z[i] * (K[i] - 1.0) / denom
            return total

        root = brentq(rr, lo, hi, xtol=1e-10, maxiter=200)
        return max(0.0, min(1.0, root))

    # ── Successive substitution ───────────────────────────────────────────────

    def _successive_substitution(self, T: float, P: float, z: List[float],
                                 K: List[float],
                                 max_iter: int = 200,
                                 tol: float = 1e-10) -> FlashResult:
        n = len(z)
        res = FlashResult(T=T, P=P, x=[0.0] * n, y=[0.0] * n, K=list(K))

        for it in range(1, max_iter + 1):
            beta = self.rachford_rice(z, K)
            x = [z[i] / (1.0 + beta * (K[i] - 1.0)) for i in range(n)]
            y = [K[i] * x[i] for i in range(n)]
            sx = sum(x); sy = sum(y)
            x = [xi / sx for xi in x]
            y = [yi / sy for yi in y]
            res.beta = beta; res.x = x; res.y = y

            lnPhiL = self.eos.ln_fugacity_coefficients(T, P, x, True)
            lnPhiV = self.eos.ln_fugacity_coefficients(T, P, y, False)
            K_new = [math.exp(lnPhiL[i] - lnPhiV[i]) for i in range(n)]

            err = sum(abs(math.log(K_new[i]) - math.log(K[i])) for i in range(n))
            K = K_new
            res.K = K
            res.iterations = it

            if err < tol:
                res.converged = True
                break

        if not res.converged:
            raise RuntimeError("FlashCalculator.flash_TP: did not converge")
        return res

    # ── Public: flash_TP ──────────────────────────────────────────────────────

    def flash_TP(self, T: float, P: float, z: List[float]) -> FlashResult:
        K0 = self.wilson_K(T, P)
        if min(K0) >= 1.0:
            return FlashResult(T=T, P=P, beta=1.0, x=list(z), y=list(z),
                               K=K0, converged=True)
        if max(K0) <= 1.0:
            return FlashResult(T=T, P=P, beta=0.0, x=list(z), y=list(z),
                               K=K0, converged=True)
        return self._successive_substitution(T, P, z, K0)

    # ── Bubble / dew point ────────────────────────────────────────────────────

    def bubble_P(self, T: float, x: List[float]) -> float:
        n = len(self.comps)
        P = sum(x[i] * self.comps[i].Pc
                * math.exp(5.373 * (1.0 + self.comps[i].omega)
                           * (1.0 - self.comps[i].Tc / T))
                for i in range(n))

        for _ in range(200):
            lnPhiL = self.eos.ln_fugacity_coefficients(T, P, x, True)
            K = [math.exp(lnPhiL[i]) for i in range(n)]
            y = [K[i] * x[i] for i in range(n)]
            sumY = sum(y); y = [yi / sumY for yi in y]
            lnPhiV = self.eos.ln_fugacity_coefficients(T, P, y, False)
            K = [math.exp(lnPhiL[i] - lnPhiV[i]) for i in range(n)]
            P_new = P * sum(K[i] * x[i] for i in range(n))
            if abs(P_new - P) / P < 1e-8:
                return P_new
            P = P_new
        raise RuntimeError("FlashCalculator.bubble_P: did not converge")

    def dew_P(self, T: float, y: List[float]) -> float:
        n = len(self.comps)
        P = 1.0 / sum(y[i] / (self.comps[i].Pc
                               * math.exp(5.373 * (1.0 + self.comps[i].omega)
                                          * (1.0 - self.comps[i].Tc / T)))
                      for i in range(n))

        for _ in range(200):
            lnPhiV = self.eos.ln_fugacity_coefficients(T, P, y, False)
            x = []
            for i in range(n):
                lnPhiL = self.eos.ln_fugacity_coefficients(T, P, [1.0 / n] * n, True)
                Ki = math.exp(lnPhiL[i] - lnPhiV[i])
                x.append(y[i] / Ki)
            sumX = sum(x); x = [xi / sumX for xi in x]
            lnPhiL = self.eos.ln_fugacity_coefficients(T, P, x, True)
            sumYoverK = sum(y[i] / math.exp(lnPhiL[i] - lnPhiV[i]) for i in range(n))
            P_new = P / sumYoverK
            if abs(P_new - P) / P < 1e-8:
                return P_new
            P = P_new
        raise RuntimeError("FlashCalculator.dew_P: did not converge")

    # ── Phase enthalpy / entropy ──────────────────────────────────────────────

    def phase_enthalpy(self, T: float, P: float,
                       z: List[float], liquid: bool) -> float:
        return ideal_gas_H(self.comps, z, T) + self.eos.enthalpy_departure(T, P, z, liquid)

    def phase_entropy(self, T: float, P: float,
                      z: List[float], liquid: bool) -> float:
        return (ideal_gas_S(self.comps, z, T)
                - R * math.log(P / P_REF)
                + self.eos.entropy_departure(T, P, z, liquid))

    # ── Total enthalpy / entropy from FlashResult ─────────────────────────────

    def total_enthalpy(self, r: FlashResult) -> float:
        H_igV = ideal_gas_H(self.comps, r.y, r.T)
        H_igL = ideal_gas_H(self.comps, r.x, r.T)
        H_depV = self.eos.enthalpy_departure(r.T, r.P, r.y, False)
        H_depL = self.eos.enthalpy_departure(r.T, r.P, r.x, True)
        return r.beta * (H_igV + H_depV) + (1.0 - r.beta) * (H_igL + H_depL)

    def total_entropy(self, r: FlashResult) -> float:
        S_igV = ideal_gas_S(self.comps, r.y, r.T) - R * math.log(r.P / P_REF)
        S_igL = ideal_gas_S(self.comps, r.x, r.T) - R * math.log(r.P / P_REF)
        S_depV = self.eos.entropy_departure(r.T, r.P, r.y, False)
        S_depL = self.eos.entropy_departure(r.T, r.P, r.x, True)
        return r.beta * (S_igV + S_depV) + (1.0 - r.beta) * (S_igL + S_depL)

    # ── flash_PH ──────────────────────────────────────────────────────────────

    def flash_PH(self, P: float, H_spec: float,
                 z: List[float], T_guess: float) -> FlashResult:
        def H_of_T(T: float) -> float:
            return self.total_enthalpy(self.flash_TP(T, P, z)) - H_spec

        T_lo = max(T_guess * 0.5, 100.0)
        T_hi = min(T_guess * 2.0, 2000.0)
        f_lo, f_hi = H_of_T(T_lo), H_of_T(T_hi)
        for _ in range(30):
            if f_lo * f_hi <= 0.0:
                break
            if abs(f_lo) < abs(f_hi):
                T_lo = max(T_lo * 0.8, 100.0)
            else:
                T_hi = min(T_hi * 1.2, 2000.0)
            f_lo, f_hi = H_of_T(T_lo), H_of_T(T_hi)
        if f_lo * f_hi > 0.0:
            raise RuntimeError("FlashCalculator.flash_PH: cannot bracket temperature")
        T_sol = brentq(H_of_T, T_lo, T_hi, xtol=1e-6, maxiter=200)
        r = self.flash_TP(T_sol, P, z)
        r.H_total = self.total_enthalpy(r)
        r.S_total = self.total_entropy(r)
        return r

    # ── flash_PS ──────────────────────────────────────────────────────────────

    def flash_PS(self, P: float, S_spec: float,
                 z: List[float], T_guess: float) -> FlashResult:
        def S_of_T(T: float) -> float:
            return self.total_entropy(self.flash_TP(T, P, z)) - S_spec

        T_lo = max(T_guess * 0.5, 100.0)
        T_hi = min(T_guess * 2.0, 2000.0)
        f_lo, f_hi = S_of_T(T_lo), S_of_T(T_hi)
        for _ in range(30):
            if f_lo * f_hi <= 0.0:
                break
            if abs(f_lo) < abs(f_hi):
                T_lo = max(T_lo * 0.8, 100.0)
            else:
                T_hi = min(T_hi * 1.2, 2000.0)
            f_lo, f_hi = S_of_T(T_lo), S_of_T(T_hi)
        if f_lo * f_hi > 0.0:
            raise RuntimeError("FlashCalculator.flash_PS: cannot bracket temperature")
        T_sol = brentq(S_of_T, T_lo, T_hi, xtol=1e-9, maxiter=200)
        r = self.flash_TP(T_sol, P, z)
        r.H_total = self.total_enthalpy(r)
        r.S_total = self.total_entropy(r)
        return r

    # ── Michelsen stability ───────────────────────────────────────────────────

    def stability_trial(self, T: float, P: float, z: List[float],
                        w_init: List[float],
                        max_iter: int = 100) -> Tuple[bool, List[float]]:
        n = len(z)
        K_w = self.wilson_K(T, P)
        kw = sum(K_w[i] * w_init[i] for i in range(n))
        vapor_trial = kw > 1.0
        feed_liq = vapor_trial
        trial_liq = not vapor_trial

        lnPhiFeed = self.eos.ln_fugacity_coefficients(T, P, z, feed_liq)
        d = [math.log(z[i]) + lnPhiFeed[i] for i in range(n)]
        W = list(w_init)

        for _ in range(max_iter):
            sumW = sum(W)
            w_norm = [W[i] / sumW for i in range(n)]
            lnPhiTrial = self.eos.ln_fugacity_coefficients(T, P, w_norm, trial_liq)
            W_new = [math.exp(d[i] - lnPhiTrial[i]) for i in range(n)]
            sumW_new = sum(W_new)
            err = sum(abs(W_new[i] / sumW_new - W[i] / sumW) for i in range(n))
            W = W_new
            if err < 1e-10:
                trivial = all(abs(W[i] / sumW_new - z[i]) <= 1e-4 for i in range(n))
                w_out = [W[i] / sumW_new for i in range(n)]
                if trivial:
                    return True, w_out
                return sumW_new <= 1.0, w_out

        sumW_final = sum(W)
        return True, [W[i] / sumW_final for i in range(n)]

    def is_stable(self, T: float, P: float, z: List[float]) -> bool:
        n = len(z)
        K = self.wilson_K(T, P)

        w = [K[i] * z[i] for i in range(n)]
        s = sum(w); w = [wi / s for wi in w]
        stable, _ = self.stability_trial(T, P, z, w)
        if not stable:
            return False

        w = [z[i] / K[i] for i in range(n)]
        s = sum(w); w = [wi / s for wi in w]
        stable, _ = self.stability_trial(T, P, z, w)
        return stable
