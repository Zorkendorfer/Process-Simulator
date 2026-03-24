#pragma once
#include "chemsim/core/Stream.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

namespace chemsim {

// Two-phase separator. Spec determines how the equilibrium temperature is set.
//   TP  — flash at (T_spec, P_spec)
//   PH  — find T such that total enthalpy = H_spec  [J/mol]
//   PS  — find T such that total entropy  = S_spec  [J/mol/K]
class FlashDrum {
public:
    enum class Spec { TP, PH, PS };

    FlashDrum(Spec spec, const FlashCalculator& fc);

    // ── Inlet (set before solve) ───────────────────────────────────────────
    Stream feed;

    // ── Specification (set the fields that match your Spec) ───────────────
    double T_spec{300.0};  // K   — used by TP
    double P_spec{1e5};    // Pa  — used by TP, PH, PS
    double H_spec{};       // J/mol — used by PH
    double S_spec{};       // J/mol/K — used by PS

    // ── Outlets (filled by solve) ─────────────────────────────────────────
    Stream vapor;
    Stream liquid;

    // ── Heat duty [W] — positive = heat added to drum ─────────────────────
    // Q = feed.totalFlow * (H_out_per_mol - H_in_per_mol)
    double duty{};

    void solve();

private:
    Spec               spec_;
    const FlashCalculator& fc_;

    // Populate vapor/liquid outlet streams from a converged FlashResult
    void populateOutlets(const FlashResult& r);
};

} // namespace chemsim
