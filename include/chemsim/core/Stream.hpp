#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace chemsim {

enum class Phase { VAPOR, LIQUID, MIXED, UNKNOWN };

struct Stream {
    std::string name;

    // State variables
    double T{};           // K
    double P{};           // Pa
    double totalFlow{};   // mol/s
    std::vector<double> z;  // overall mole fractions

    // Phase split (filled by flash)
    Phase  phase{Phase::UNKNOWN};
    double vaporFraction{};     // β ∈ [0,1]
    std::vector<double> x;      // liquid mole fractions
    std::vector<double> y;      // vapor mole fractions

    // Thermodynamic properties (set after flash)
    double H{};   // J/mol  total specific enthalpy
    double S{};   // J/mol/K total specific entropy

    // Convenience
    int    nComp()       const { return static_cast<int>(z.size()); }
    double molarFlow(int i) const { return totalFlow * z[i]; }
    bool   isFullyVapor()  const { return vaporFraction >= 1.0 - 1e-10; }
    bool   isFullyLiquid() const { return vaporFraction <= 1e-10; }

    Stream clone() const { return *this; }

    static bool converged(const Stream& a, const Stream& b,
                          double tol_T = 0.01, double tol_P = 1.0,
                          double tol_z = 1e-6) {
        if (std::abs(a.T - b.T) > tol_T) return false;
        if (std::abs(a.P - b.P) > tol_P) return false;
        if (a.z.size() != b.z.size()) return false;
        for (int i = 0; i < static_cast<int>(a.z.size()); ++i)
            if (std::abs(a.z[i] - b.z[i]) > tol_z) return false;
        return true;
    }
};

} // namespace chemsim
