#include "chemsim/thermo/FlashCalculator.hpp"
#include "chemsim/core/Mixture.hpp"
#include "chemsim/numerics/BrentSolver.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace chemsim {

static constexpr double R = 8.314462618;

FlashCalculator::FlashCalculator(const EOS& eos,
                                 const std::vector<Component>& comps)
    : eos_(eos), comps_(comps) {}

// ─── Wilson K-value estimate ──────────────────────────────────────────────────
// K_i = (Pc_i/P) * exp(5.373*(1+ω_i)*(1 - Tc_i/T))
std::vector<double> FlashCalculator::wilsonK(double T, double P) const {
    std::vector<double> K;
    K.reserve(comps_.size());
    for (const auto& c : comps_)
        K.push_back((c.Pc / P) * std::exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T)));
    return K;
}

// ─── Rachford-Rice ────────────────────────────────────────────────────────────
double FlashCalculator::rachfordRice(const std::vector<double>& z,
                                     const std::vector<double>& K) const {
    const int n = static_cast<int>(z.size());

    // Check for single-phase trivial: all K>1 → all vapor, all K<1 → all liquid
    auto rr = [&](double beta) {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += z[i] * (K[i] - 1.0) / (1.0 + beta * (K[i] - 1.0));
        return s;
    };

    double Kmax = *std::max_element(K.begin(), K.end());
    double Kmin = *std::min_element(K.begin(), K.end());

    if (Kmax <= 1.0) return 0.0;   // all liquid
    if (Kmin >= 1.0) return 1.0;   // all vapor

    // Valid bracket: (1/(1-Kmax), 1/(1-Kmin)) — shrink slightly
    double lo = 1.0 / (1.0 - Kmax) + 1e-8;
    double hi = 1.0 / (1.0 - Kmin) - 1e-8;

    auto res = BrentSolver::solve(rr, lo, hi);
    return std::clamp(res.root, 0.0, 1.0);
}

// ─── Successive substitution ──────────────────────────────────────────────────
FlashResult FlashCalculator::successiveSubstitution(double T, double P,
                                                     const std::vector<double>& z,
                                                     std::vector<double> K,
                                                     int maxIter,
                                                     double tol) const {
    const int n = static_cast<int>(z.size());
    FlashResult res;
    res.T = T; res.P = P;
    res.x.resize(n); res.y.resize(n); res.K.resize(n);

    for (int iter = 1; iter <= maxIter; ++iter) {
        double beta = rachfordRice(z, K);

        // Compute x_i, y_i
        for (int i = 0; i < n; ++i) {
            res.x[i] = z[i] / (1.0 + beta * (K[i] - 1.0));
            res.y[i] = K[i] * res.x[i];
        }
        res.beta = beta;

        // Normalise (should be ~1, but defend against drift)
        double sx = std::accumulate(res.x.begin(), res.x.end(), 0.0);
        double sy = std::accumulate(res.y.begin(), res.y.end(), 0.0);
        for (int i = 0; i < n; ++i) { res.x[i] /= sx; res.y[i] /= sy; }

        // Update fugacity coefficients
        auto lnPhiL = eos_.lnFugacityCoefficients(T, P, res.x, true);
        auto lnPhiV = eos_.lnFugacityCoefficients(T, P, res.y, false);

        // New K-values: K_i = φ_L_i / φ_V_i
        std::vector<double> K_new(n);
        for (int i = 0; i < n; ++i)
            K_new[i] = std::exp(lnPhiL[i] - lnPhiV[i]);

        // Convergence: sum |ln K_new - ln K_old|
        double err = 0.0;
        for (int i = 0; i < n; ++i)
            err += std::abs(std::log(K_new[i]) - std::log(K[i]));

        K = K_new;
        res.K = K;
        res.iterations = iter;

        if (err < tol) {
            res.converged = true;
            break;
        }
    }

    if (!res.converged)
        throw std::runtime_error("FlashCalculator::flashTP: did not converge");

    return res;
}

// ─── Public: flashTP ──────────────────────────────────────────────────────────
FlashResult FlashCalculator::flashTP(double T, double P,
                                     const std::vector<double>& z) const {
    auto K0 = wilsonK(T, P);

    // Quick single-phase check from Wilson K before running SS:
    // If all K >= 1 → all vapor; if all K <= 1 → all liquid.
    double Kmax = *std::max_element(K0.begin(), K0.end());
    double Kmin = *std::min_element(K0.begin(), K0.end());

    if (Kmin >= 1.0) {
        // All vapor
        FlashResult r;
        r.T = T; r.P = P; r.beta = 1.0; r.converged = true;
        r.x = r.y = z; r.K = K0;
        return r;
    }
    if (Kmax <= 1.0) {
        // All liquid
        FlashResult r;
        r.T = T; r.P = P; r.beta = 0.0; r.converged = true;
        r.x = r.y = z; r.K = K0;
        return r;
    }

    return successiveSubstitution(T, P, z, K0);
}

// ─── Bubble-point pressure ────────────────────────────────────────────────────
double FlashCalculator::bubbleP(double T, const std::vector<double>& x) const {
    // At bubble point: Σ K_i x_i = 1  (β = 0)
    // Iterate: start with Wilson, then update K from EOS
    const int n = static_cast<int>(comps_.size());

    // Initial guess: Wilson at some P
    double P = 1e5;  // start low, let iterations correct
    // Better: P = Σ x_i * Pc_i * exp(5.373*(1+ω)*(1-Tc/T))
    P = 0.0;
    for (const auto& c : comps_)
        P += x[&c - &comps_[0]] * c.Pc * std::exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T));

    for (int iter = 0; iter < 200; ++iter) {
        auto lnPhiL = eos_.lnFugacityCoefficients(T, P, x, true);

        // y_i = x_i * φ_L_i / φ_V_i; but φ_V depends on y
        // Simplified: use Wilson K to get y, then update P
        auto K = wilsonK(T, P);
        for (int i = 0; i < n; ++i)
            K[i] = std::exp(lnPhiL[i]); // initial: assume φ_V = 1

        // y_i = K_i * x_i (unnormalised)
        std::vector<double> y(n);
        double sumY = 0.0;
        for (int i = 0; i < n; ++i) { y[i] = K[i] * x[i]; sumY += y[i]; }
        for (int i = 0; i < n; ++i) y[i] /= sumY;

        // Update vapor fugacities
        auto lnPhiV = eos_.lnFugacityCoefficients(T, P, y, false);
        for (int i = 0; i < n; ++i)
            K[i] = std::exp(lnPhiL[i] - lnPhiV[i]);

        double newSumKx = 0.0;
        for (int i = 0; i < n; ++i) newSumKx += K[i] * x[i];

        double P_new = P * newSumKx;
        if (std::abs(P_new - P) / P < 1e-8) return P_new;
        P = P_new;
    }
    throw std::runtime_error("FlashCalculator::bubbleP: did not converge");
}

double FlashCalculator::dewP(double T, const std::vector<double>& y) const {
    const int n = static_cast<int>(comps_.size());
    // Initial P from Wilson
    double P = 0.0;
    for (const auto& c : comps_) {
        int i = &c - &comps_[0];
        double Kw = (c.Pc) * std::exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T));
        P += y[i] / Kw;
    }
    P = 1.0 / P;

    for (int iter = 0; iter < 200; ++iter) {
        auto lnPhiV = eos_.lnFugacityCoefficients(T, P, y, false);

        std::vector<double> x(n);
        double sumX = 0.0;
        for (int i = 0; i < n; ++i) {
            auto lnPhiL = eos_.lnFugacityCoefficients(T, P,
                              std::vector<double>(n, 1.0/n), true);
            double K = std::exp(lnPhiL[i] - lnPhiV[i]);
            x[i] = y[i] / K;
            sumX += x[i];
        }
        for (auto& xi : x) xi /= sumX;

        auto lnPhiL = eos_.lnFugacityCoefficients(T, P, x, true);
        double sumYoverK = 0.0;
        for (int i = 0; i < n; ++i)
            sumYoverK += y[i] / std::exp(lnPhiL[i] - lnPhiV[i]);

        double P_new = P / sumYoverK;
        if (std::abs(P_new - P) / P < 1e-8) return P_new;
        P = P_new;
    }
    throw std::runtime_error("FlashCalculator::dewP: did not converge");
}

// ─── phaseEnthalpy / phaseEntropy ────────────────────────────────────────────
double FlashCalculator::phaseEnthalpy(double T, double P,
                                      const std::vector<double>& z,
                                      bool liquid) const {
    return Mixture::idealGasH(comps_, z, T)
         + eos_.enthalpyDeparture(T, P, z, liquid);
}

double FlashCalculator::phaseEntropy(double T, double P,
                                     const std::vector<double>& z,
                                     bool liquid) const {
    constexpr double P_ref = 101325.0;
    return Mixture::idealGasS(comps_, z, T)
         - R * std::log(P / P_ref)
         + eos_.entropyDeparture(T, P, z, liquid);
}

// ─── totalEnthalpy ────────────────────────────────────────────────────────────
// H = β*(H_ig,V + H_dep,V) + (1-β)*(H_ig,L + H_dep,L)  [J/mol]
double FlashCalculator::totalEnthalpy(const FlashResult& r) const {
    double H_igV = Mixture::idealGasH(comps_, r.y, r.T);
    double H_igL = Mixture::idealGasH(comps_, r.x, r.T);
    double H_depV = eos_.enthalpyDeparture(r.T, r.P, r.y, false);
    double H_depL = eos_.enthalpyDeparture(r.T, r.P, r.x, true);
    return r.beta * (H_igV + H_depV) + (1.0 - r.beta) * (H_igL + H_depL);
}

// ─── totalEntropy ─────────────────────────────────────────────────────────────
// S = β*(S_ig,V + S_dep,V) + (1-β)*(S_ig,L + S_dep,L)  [J/mol/K]
// Includes -R*ln(P/P_ref) so cross-pressure isentropic calculations are correct.
double FlashCalculator::totalEntropy(const FlashResult& r) const {
    constexpr double P_ref = 101325.0;
    double S_igV = Mixture::idealGasS(comps_, r.y, r.T) - R * std::log(r.P / P_ref);
    double S_igL = Mixture::idealGasS(comps_, r.x, r.T) - R * std::log(r.P / P_ref);
    double S_depV = eos_.entropyDeparture(r.T, r.P, r.y, false);
    double S_depL = eos_.entropyDeparture(r.T, r.P, r.x, true);
    return r.beta * (S_igV + S_depV) + (1.0 - r.beta) * (S_igL + S_depL);
}

// ─── Michelsen stability trial ────────────────────────────────────────────────
// Performs one successive-substitution stability trial starting from w_init.
// Phase of trial is inferred from w_init: if Σ K_i*w_i > 1, vapor-like (trial against liquid feed).
// Returns {stable, converged_w} where stable = (TPD >= 0).
std::pair<bool, std::vector<double>>
FlashCalculator::stabilityTrial(double T, double P,
                                const std::vector<double>& z,
                                const std::vector<double>& w_init,
                                int maxIter) const {
    const int n = static_cast<int>(z.size());
    auto K_w = wilsonK(T, P);

    // Detect trial type: vapor-like if enriched in light components (Σ K_i*w_i > 1)
    double kw = 0.0;
    for (int i = 0; i < n; ++i) kw += K_w[i] * w_init[i];
    const bool vapor_trial = (kw > 1.0);

    // Feed reference: opposite phase to trial
    // vapor trial → liquid feed reference; liquid trial → vapor feed reference
    const bool feed_liq = vapor_trial;
    const bool trial_liq = !vapor_trial;

    auto lnPhiFeed = eos_.lnFugacityCoefficients(T, P, z, feed_liq);
    // d_i = ln(z_i) + lnPhi_i(z, feed_phase)
    std::vector<double> d(n);
    for (int i = 0; i < n; ++i)
        d[i] = std::log(z[i]) + lnPhiFeed[i];

    // W_i: unnormalized trial mole numbers; initialize from w_init (normalized)
    std::vector<double> W = w_init;

    for (int iter = 0; iter < maxIter; ++iter) {
        double sumW = std::accumulate(W.begin(), W.end(), 0.0);
        std::vector<double> w_norm(n);
        for (int i = 0; i < n; ++i) w_norm[i] = W[i] / sumW;

        auto lnPhiTrial = eos_.lnFugacityCoefficients(T, P, w_norm, trial_liq);

        // W_i_new = exp(d_i - lnPhi_i^trial)
        std::vector<double> W_new(n);
        double sumW_new = 0.0;
        for (int i = 0; i < n; ++i) {
            W_new[i] = std::exp(d[i] - lnPhiTrial[i]);
            sumW_new += W_new[i];
        }

        // Convergence on normalized composition
        double err = 0.0;
        for (int i = 0; i < n; ++i)
            err += std::abs(W_new[i] / sumW_new - W[i] / sumW);

        W = W_new;

        if (err < 1e-10) {
            // Check for trivial solution (W/sumW ≈ z)
            bool trivial = true;
            for (int i = 0; i < n; ++i)
                if (std::abs(W[i] / sumW_new - z[i]) > 1e-4) { trivial = false; break; }
            if (trivial) {
                std::vector<double> w_out(n);
                for (int i = 0; i < n; ++i) w_out[i] = W[i] / sumW_new;
                return {true, w_out};
            }

            // At stationarity: TPD = 1 - sumW  (Michelsen's simplification)
            // If sumW > 1 → TPD < 0 → unstable
            std::vector<double> w_out(n);
            for (int i = 0; i < n; ++i) w_out[i] = W[i] / sumW_new;
            return {sumW_new <= 1.0, w_out};
        }
    }

    // Did not converge: assume stable (conservative)
    std::vector<double> w_out(n);
    double sumW_final = std::accumulate(W.begin(), W.end(), 0.0);
    for (int i = 0; i < n; ++i) w_out[i] = W[i] / sumW_final;
    return {true, w_out};
}

// ─── isStable ─────────────────────────────────────────────────────────────────
// Returns true if feed is STABLE (single phase). Tests vapor-like and liquid-like
// trials from Wilson K initial estimates.
bool FlashCalculator::isStable(double T, double P,
                               const std::vector<double>& z) const {
    const int n = static_cast<int>(z.size());
    auto K = wilsonK(T, P);

    // Vapor-like trial: w_i ~ K_i * z_i
    {
        std::vector<double> w(n);
        double s = 0.0;
        for (int i = 0; i < n; ++i) { w[i] = K[i] * z[i]; s += w[i]; }
        for (double& wi : w) wi /= s;
        auto [stable, w_out] = stabilityTrial(T, P, z, w);
        if (!stable) return false;
    }

    // Liquid-like trial: w_i ~ z_i / K_i
    {
        std::vector<double> w(n);
        double s = 0.0;
        for (int i = 0; i < n; ++i) { w[i] = z[i] / K[i]; s += w[i]; }
        for (double& wi : w) wi /= s;
        auto [stable, w_out] = stabilityTrial(T, P, z, w);
        if (!stable) return false;
    }

    return true;
}

// ─── flashPH ─────────────────────────────────────────────────────────────────
// Finds T (and phase split) that gives total enthalpy = H_spec [J/mol].
FlashResult FlashCalculator::flashPH(double P, double H_spec,
                                     const std::vector<double>& z,
                                     double T_guess) const {
    auto H_of_T = [&](double T) -> double {
        auto r = flashTP(T, P, z);
        return totalEnthalpy(r) - H_spec;
    };

    // Build bracket around T_guess by expanding until sign change
    double T_lo = T_guess * 0.5;
    double T_hi = T_guess * 2.0;
    // Clamp to physically reasonable range
    T_lo = std::max(T_lo, 100.0);
    T_hi = std::min(T_hi, 2000.0);

    double f_lo = H_of_T(T_lo);
    double f_hi = H_of_T(T_hi);

    // Expand if needed (H is monotone in T so one expansion direction suffices)
    for (int k = 0; k < 30 && f_lo * f_hi > 0.0; ++k) {
        if (std::abs(f_lo) < std::abs(f_hi))
            T_lo = std::max(T_lo * 0.8, 100.0);
        else
            T_hi = std::min(T_hi * 1.2, 2000.0);
        f_lo = H_of_T(T_lo);
        f_hi = H_of_T(T_hi);
    }
    if (f_lo * f_hi > 0.0)
        throw std::runtime_error("FlashCalculator::flashPH: cannot bracket temperature");

    auto bres = BrentSolver::solve(H_of_T, T_lo, T_hi,
                                   BrentSolver::Options{1e-6, 200});
    auto r = flashTP(bres.root, P, z);
    r.H_total = totalEnthalpy(r);
    r.S_total = totalEntropy(r);
    return r;
}

// ─── flashPS ─────────────────────────────────────────────────────────────────
// Finds T (and phase split) that gives total entropy = S_spec [J/mol/K].
FlashResult FlashCalculator::flashPS(double P, double S_spec,
                                     const std::vector<double>& z,
                                     double T_guess) const {
    auto S_of_T = [&](double T) -> double {
        auto r = flashTP(T, P, z);
        return totalEntropy(r) - S_spec;
    };

    double T_lo = std::max(T_guess * 0.5, 100.0);
    double T_hi = std::min(T_guess * 2.0, 2000.0);

    double f_lo = S_of_T(T_lo);
    double f_hi = S_of_T(T_hi);

    for (int k = 0; k < 30 && f_lo * f_hi > 0.0; ++k) {
        if (std::abs(f_lo) < std::abs(f_hi))
            T_lo = std::max(T_lo * 0.8, 100.0);
        else
            T_hi = std::min(T_hi * 1.2, 2000.0);
        f_lo = S_of_T(T_lo);
        f_hi = S_of_T(T_hi);
    }
    if (f_lo * f_hi > 0.0)
        throw std::runtime_error("FlashCalculator::flashPS: cannot bracket temperature");

    auto bres = BrentSolver::solve(S_of_T, T_lo, T_hi,
                                   BrentSolver::Options{1e-9, 200});
    auto r = flashTP(bres.root, P, z);
    r.H_total = totalEnthalpy(r);
    r.S_total = totalEntropy(r);
    return r;
}

} // namespace chemsim
