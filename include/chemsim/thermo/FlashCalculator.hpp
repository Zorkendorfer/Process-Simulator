#pragma once
#include <vector>
#include "chemsim/thermo/EOS.hpp"
#include "chemsim/core/Component.hpp"

namespace chemsim {

struct FlashResult {
    double T{}, P{};
    double beta{};                  // vapor fraction V/F  [0..1]
    std::vector<double> x;         // liquid mole fractions
    std::vector<double> y;         // vapor mole fractions
    std::vector<double> K;         // equilibrium ratios y_i/x_i
    double H_total{};              // J/mol  total (H_ig + H_dep)
    double S_total{};              // J/mol/K
    bool   converged{false};
    int    iterations{0};
};

class FlashCalculator {
public:
    explicit FlashCalculator(const EOS& eos,
                             const std::vector<Component>& comps);

    // ── TP flash ─────────────────────────────────────────────────────────────
    FlashResult flashTP(double T, double P,
                        const std::vector<double>& z) const;

    // ── PH flash: find T (and phase split) that gives H = H_spec [J/mol] ───
    // H_spec is total (ideal + departure), relative to T_ref=298.15K
    FlashResult flashPH(double P, double H_spec,
                        const std::vector<double>& z,
                        double T_guess = 300.0) const;

    // ── PS flash ─────────────────────────────────────────────────────────────
    FlashResult flashPS(double P, double S_spec,
                        const std::vector<double>& z,
                        double T_guess = 300.0) const;

    // ── Pure-phase bubble/dew pressure ───────────────────────────────────────
    double bubbleP(double T, const std::vector<double>& x) const;
    double dewP   (double T, const std::vector<double>& y) const;

    // ── Enthalpy and entropy of a converged flash ─────────────────────────
    // H = β·(H_ig,V + H_dep,V) + (1-β)·(H_ig,L + H_dep,L)  [J/mol]
    double totalEnthalpy(const FlashResult& r) const;
    double totalEntropy (const FlashResult& r) const;

    // ── Michelsen tangent-plane stability test ────────────────────────────
    // Returns true if feed is STABLE (no second phase can form).
    bool isStable(double T, double P,
                  const std::vector<double>& z) const;

    // One stability trial starting from initial estimate w_init.
    // Returns (stable, refined_w).
    std::pair<bool, std::vector<double>>
    stabilityTrial(double T, double P,
                   const std::vector<double>& z,
                   const std::vector<double>& w_init,
                   int maxIter = 200) const;

private:
    const EOS&                    eos_;
    const std::vector<Component>& comps_;

    // Wilson K-value estimate
    std::vector<double> wilsonK(double T, double P) const;

    // Rachford-Rice equation; returns beta in [0,1]
    double rachfordRice(const std::vector<double>& z,
                        const std::vector<double>& K) const;

    // Successive substitution inner loop
    FlashResult successiveSubstitution(double T, double P,
                                       const std::vector<double>& z,
                                       std::vector<double> K_init,
                                       int maxIter = 300,
                                       double tol  = 1e-10) const;
};

} // namespace chemsim
