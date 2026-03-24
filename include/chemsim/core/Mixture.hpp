#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include "chemsim/core/Component.hpp"

namespace chemsim {
namespace Mixture {

// Mole-fraction-averaged molecular weight [g/mol]
inline double meanMW(const std::vector<Component>& comps,
                     const std::vector<double>& z) {
    double mw = 0.0;
    for (int i = 0; i < static_cast<int>(comps.size()); ++i)
        mw += z[i] * comps[i].MW;
    return mw;
}

// DIPPR Aly-Lee Cp at temperature T [J/mol/K]
// Cp = C1 + C2*(C3/T/sinh(C3/T))^2 + C4*(C5/T/cosh(C5/T))^2  [J/kmol/K]
// divide by 1000 for J/mol/K
inline double idealGasCp(const Component& c, double T) {
    double u = c.Cp3 / T;
    double v = c.Cp5 / T;
    double cp_kmol = c.Cp1
                   + c.Cp2 * (u / std::sinh(u)) * (u / std::sinh(u))
                   + c.Cp4 * (v / std::cosh(v)) * (v / std::cosh(v));
    return cp_kmol / 1000.0;  // J/mol/K
}

// Mixture ideal-gas Cp [J/mol/K]
inline double idealGasCp(const std::vector<Component>& comps,
                         const std::vector<double>& z, double T) {
    double cp = 0.0;
    for (int i = 0; i < static_cast<int>(comps.size()); ++i)
        cp += z[i] * idealGasCp(comps[i], T);
    return cp;
}

// Integral of Aly-Lee Cp for one component [J/mol]
// Uses analytical result:
//   integral(Cp, T1, T2) =
//     Cp1*(T2-T1)/1000
//   + Cp2*Cp3*(coth(Cp3/T2) - coth(Cp3/T1))/1000
//   - Cp4*Cp5*(tanh(Cp5/T2) - tanh(Cp5/T1))/1000
inline double idealGasH_pure(const Component& c, double T, double T_ref = 298.15) {
    auto coth = [](double x) { return std::cosh(x) / std::sinh(x); };
    auto tanh_ = [](double x) { return std::tanh(x); };

    double dH = c.Cp1 * (T - T_ref)
              + c.Cp2 * c.Cp3 * (coth(c.Cp3 / T) - coth(c.Cp3 / T_ref))
              - c.Cp4 * c.Cp5 * (tanh_(c.Cp5 / T) - tanh_(c.Cp5 / T_ref));
    return dH / 1000.0;   // J/mol
}

// Mixture ideal-gas enthalpy relative to T_ref [J/mol]
inline double idealGasH(const std::vector<Component>& comps,
                        const std::vector<double>& z,
                        double T, double T_ref = 298.15) {
    double H = 0.0;
    for (int i = 0; i < static_cast<int>(comps.size()); ++i)
        H += z[i] * idealGasH_pure(comps[i], T, T_ref);
    return H;
}

// Integral of Cp/T for one component [J/mol/K], using 20-interval Simpson's rule
// Returns ∫(T_ref to T) Cp(t)/t dt
inline double idealGasS_pure(const Component& c, double T, double T_ref = 298.15) {
    if (std::abs(T - T_ref) < 1e-10) return 0.0;
    const int N = 20;  // must be even
    double h = (T - T_ref) / N;
    auto f = [&](double t) { return idealGasCp(c, t) / t; };
    double sum = f(T_ref) + f(T);
    for (int i = 1; i < N; ++i)
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(T_ref + i * h);
    return sum * h / 3.0;
}

// Mixture ideal-gas entropy change relative to T_ref [J/mol/K]
// Includes ideal mixing: S_ig = Σ z_i*∫Cp_i/T dT - R*Σ z_i*ln(z_i)
// P-dependent term (-R*ln(P/Pref)) is NOT included; users must add -R*ln(P/Pref) for absolute S.
inline double idealGasS(const std::vector<Component>& comps,
                        const std::vector<double>& z,
                        double T, double T_ref = 298.15) {
    constexpr double R = 8.314462618;
    double S = 0.0;
    for (int i = 0; i < static_cast<int>(comps.size()); ++i) {
        S += z[i] * idealGasS_pure(comps[i], T, T_ref);
        if (z[i] > 1e-15)
            S -= R * z[i] * std::log(z[i]);  // ideal mixing
    }
    return S;
}

} // namespace Mixture
} // namespace chemsim
