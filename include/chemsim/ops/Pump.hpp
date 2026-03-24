#pragma once
#include "chemsim/core/Stream.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

namespace chemsim {

// Isentropic pump with efficiency correction.
// Algorithm:
//   1. S_in  = phaseEntropy(T_in, P_in, z, liquid=true)
//   2. Isentropic outlet: flashPS(P_out, S_in) → H_out_s
//   3. W_s   = H_out_s - H_in          [J/mol, shaft work at 100% efficiency]
//   4. W_act = W_s / eta               [actual work, eta ∈ (0,1]]
//   5. H_out = H_in + W_act
//   6. Outlet: flashPH(P_out, H_out)
class Pump {
public:
    Pump(const FlashCalculator& fc, double P_out, double eta = 0.75);

    // ── Inlet ─────────────────────────────────────────────────────────────
    Stream inlet;

    // ── Outlet (filled by solve) ──────────────────────────────────────────
    Stream outlet;

    // ── Results ───────────────────────────────────────────────────────────
    double shaft_power_W{};   // total shaft power [W], positive = work into fluid
    double shaft_work_mol{};  // specific work [J/mol]

    void solve();

private:
    const FlashCalculator& fc_;
    double P_out_;
    double eta_;
};

} // namespace chemsim
