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
    double H_total{};
    double S_total{};
    bool   converged{false};
    int    iterations{0};
};

class FlashCalculator {
public:
    explicit FlashCalculator(const EOS& eos,
                             const std::vector<Component>& comps);

    // TP flash: find phase split at given T, P, overall composition z
    FlashResult flashTP(double T, double P,
                        const std::vector<double>& z) const;

    // Bubble-point pressure at given T and liquid composition x
    double bubbleP(double T, const std::vector<double>& x) const;

    // Dew-point pressure at given T and vapor composition y
    double dewP(double T, const std::vector<double>& y) const;

private:
    const EOS&                    eos_;
    const std::vector<Component>& comps_;

    // Wilson K-value initial estimate
    std::vector<double> wilsonK(double T, double P) const;

    // Rachford-Rice: solve for beta given K and z
    // Throws if no valid bracket (all vapor or all liquid)
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
