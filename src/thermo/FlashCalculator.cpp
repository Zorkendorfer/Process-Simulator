#include "chemsim/thermo/FlashCalculator.hpp"
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

} // namespace chemsim
