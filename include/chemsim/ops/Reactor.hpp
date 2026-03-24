#pragma once
#include <vector>
#include "chemsim/core/Stream.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

namespace chemsim {

// Stoichiometry of one reaction across all components.
// nu[i] < 0 → reactant,  nu[i] > 0 → product
struct Reaction {
    std::vector<double> nu;    // stoichiometric coefficients [nComp]
};

// Isothermal reactor at (T_spec, P_spec).
// Algorithm:
//   1. n_out,i = F_in * z_in,i  +  Σ_k  nu_k,i * ξ_k   [mol/s]
//   2. z_out   = n_out / Σ n_out
//   3. Outlet  = flashTP(T_spec, P_spec, z_out)
//   4. duty [W] = F_out*H_out - F_in*H_in  (sensible-heat only; no ΔHf° included)
//
// Note: heat of formation is NOT included in enthalpy reference; duty reflects
// only sensible-heat change, not chemical-reaction enthalpy.
class Reactor {
public:
    Reactor(const FlashCalculator& fc, double T_spec, double P_spec);

    // Add a reaction and its extent of reaction [mol/s].
    // extent > 0 means the reaction proceeds forward.
    void addReaction(const Reaction& rxn, double extent_mol_s);

    // ── Inlet ─────────────────────────────────────────────────────────────
    Stream inlet;

    // ── Outlet (filled by solve) ──────────────────────────────────────────
    Stream outlet;

    // ── Heat duty [W] (positive = heat must be added to maintain T_spec) ──
    double duty{};

    void solve();

private:
    const FlashCalculator& fc_;
    double T_spec_, P_spec_;
    std::vector<Reaction> reactions_;
    std::vector<double>   extents_;
};

} // namespace chemsim
