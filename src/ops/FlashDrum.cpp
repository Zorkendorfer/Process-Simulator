#include "chemsim/ops/FlashDrum.hpp"
#include <stdexcept>

namespace chemsim {

FlashDrum::FlashDrum(Spec spec, const FlashCalculator& fc)
    : spec_(spec), fc_(fc) {}

void FlashDrum::populateOutlets(const FlashResult& r) {
    vapor.T = liquid.T = r.T;
    vapor.P = liquid.P = r.P;
    vapor.vaporFraction  = 1.0;
    liquid.vaporFraction = 0.0;
    vapor.phase  = Phase::VAPOR;
    liquid.phase = Phase::LIQUID;

    vapor.z  = vapor.y  = vapor.x  = r.y;
    liquid.z = liquid.x = liquid.y = r.x;

    vapor.totalFlow  = feed.totalFlow * r.beta;
    liquid.totalFlow = feed.totalFlow * (1.0 - r.beta);

    // Single-phase enthalpies for each outlet stream
    vapor.H  = fc_.phaseEnthalpy(r.T, r.P, r.y, false);
    liquid.H = fc_.phaseEnthalpy(r.T, r.P, r.x, true);
    vapor.S  = fc_.phaseEntropy (r.T, r.P, r.y, false);
    liquid.S = fc_.phaseEntropy (r.T, r.P, r.x, true);

    // If fully single-phase, one outlet gets zero flow
    if (r.beta >= 1.0 - 1e-10) {
        vapor.totalFlow  = feed.totalFlow;
        liquid.totalFlow = 0.0;
        liquid.z = liquid.x = liquid.y = r.y;  // copy composition for zero-flow stream
    } else if (r.beta <= 1e-10) {
        liquid.totalFlow = feed.totalFlow;
        vapor.totalFlow  = 0.0;
        vapor.z = vapor.y = vapor.x = r.x;
    }
}

void FlashDrum::solve() {
    if (feed.totalFlow <= 0.0)
        throw std::invalid_argument("FlashDrum: feed.totalFlow must be > 0");
    if (feed.z.empty())
        throw std::invalid_argument("FlashDrum: feed.z is empty");

    // Compute inlet enthalpy (needed for duty in TP mode)
    // Use flash at feed conditions to get H_in
    auto r_in = fc_.flashTP(feed.T, feed.P, feed.z);
    double H_in = fc_.totalEnthalpy(r_in);

    FlashResult r_out;
    switch (spec_) {
    case Spec::TP:
        r_out = fc_.flashTP(T_spec, P_spec, feed.z);
        break;
    case Spec::PH:
        r_out = fc_.flashPH(P_spec, H_spec, feed.z, feed.T);
        break;
    case Spec::PS:
        r_out = fc_.flashPS(P_spec, S_spec, feed.z, feed.T);
        break;
    }

    populateOutlets(r_out);

    double H_out = fc_.totalEnthalpy(r_out);
    duty = feed.totalFlow * (H_out - H_in);   // [W]
}

} // namespace chemsim
