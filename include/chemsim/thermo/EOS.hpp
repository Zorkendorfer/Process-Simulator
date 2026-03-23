#pragma once
#include <vector>
#include <utility>

namespace chemsim {

class EOS {
public:
    virtual ~EOS() = default;

    // Returns {Z_liquid, Z_vapor}. If single phase, both may be equal.
    virtual std::pair<double,double>
    compressibilityFactors(double T, double P,
                           const std::vector<double>& z) const = 0;

    // ln(phi_i) for each component in the specified phase
    virtual std::vector<double>
    lnFugacityCoefficients(double T, double P,
                           const std::vector<double>& z,
                           bool liquid) const = 0;

    // H - H_ig  [J/mol]
    virtual double
    enthalpyDeparture(double T, double P,
                      const std::vector<double>& z,
                      bool liquid) const = 0;

    // S - S_ig  [J/mol/K]
    virtual double
    entropyDeparture(double T, double P,
                     const std::vector<double>& z,
                     bool liquid) const = 0;
};

} // namespace chemsim
