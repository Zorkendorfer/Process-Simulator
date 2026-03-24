#include "chemsim/ops/Pump.hpp"
#include <stdexcept>
#include <cmath>

namespace chemsim {

Pump::Pump(const FlashCalculator& fc, double P_out, double eta)
    : fc_(fc), P_out_(P_out), eta_(eta) {
    if (eta <= 0.0 || eta > 1.0)
        throw std::invalid_argument("Pump: eta must be in (0, 1]");
}

void Pump::solve() {
    if (inlet.totalFlow <= 0.0)
        throw std::invalid_argument("Pump: inlet.totalFlow must be > 0");
    if (P_out_ <= inlet.P)
        throw std::invalid_argument("Pump: P_out must be greater than inlet pressure");

    const double T_in = inlet.T;
    const double P_in = inlet.P;
    const auto&  z    = inlet.z;

    // Inlet enthalpy (liquid assumed for pump)
    double H_in = fc_.phaseEnthalpy(T_in, P_in, z, true);
    double S_in = fc_.phaseEntropy (T_in, P_in, z, true);

    // Isentropic outlet: flashPS at P_out with S_in
    auto r_s = fc_.flashPS(P_out_, S_in, z, T_in);
    double H_out_s = fc_.totalEnthalpy(r_s);

    // Actual enthalpy: W_actual = W_isentropic / eta
    double W_s   = H_out_s - H_in;          // J/mol, isentropic work
    double W_act = W_s / eta_;               // actual work per mol
    double H_out = H_in + W_act;

    // Actual outlet: flashPH at P_out with H_out
    auto r_out = fc_.flashPH(P_out_, H_out, z, r_s.T);

    outlet           = inlet;                // copy name, z, totalFlow
    outlet.P         = r_out.P;
    outlet.T         = r_out.T;
    outlet.vaporFraction = r_out.beta;
    outlet.x         = r_out.x;
    outlet.y         = r_out.y;
    outlet.phase     = (r_out.beta < 1e-10)  ? Phase::LIQUID
                     : (r_out.beta > 1-1e-10) ? Phase::VAPOR
                                               : Phase::MIXED;
    outlet.H = fc_.totalEnthalpy(r_out);
    outlet.S = fc_.totalEntropy (r_out);

    shaft_work_mol  = W_act;
    shaft_power_W   = inlet.totalFlow * W_act;
}

} // namespace chemsim
