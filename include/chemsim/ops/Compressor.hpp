#pragma once
#include "chemsim/core/Stream.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

namespace chemsim {

// Isentropic compressor with efficiency correction.
// Identical thermodynamic algorithm to Pump but typically applied to vapor feed.
// The distinction is only physical (pump ≈ liquid, compressor ≈ vapor);
// the code is the same — feed phase is not enforced.
class Compressor {
public:
    Compressor(const FlashCalculator& fc, double P_out, double eta = 0.72);

    Stream inlet;
    Stream outlet;

    double shaft_power_W{};
    double shaft_work_mol{};

    void solve();

private:
    const FlashCalculator& fc_;
    double P_out_;
    double eta_;
};

} // namespace chemsim
