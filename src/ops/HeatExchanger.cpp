#include "chemsim/ops/HeatExchanger.hpp"
#include <stdexcept>

namespace chemsim {

HeatExchanger::HeatExchanger(Spec spec,
                             const FlashCalculator& fc_hot,
                             const FlashCalculator& fc_cold)
    : spec_(spec), fc_hot_(fc_hot), fc_cold_(fc_cold) {}

double HeatExchanger::streamH(const FlashCalculator& fc, const Stream& s) {
    // Compute total specific enthalpy [J/mol] from stream conditions
    auto r = fc.flashTP(s.T, s.P, s.z);
    return fc.totalEnthalpy(r);
}

void HeatExchanger::solve() {
    if (hot_in.totalFlow  <= 0.0) throw std::invalid_argument("HeatExchanger: hot_in.totalFlow <= 0");
    if (cold_in.totalFlow <= 0.0) throw std::invalid_argument("HeatExchanger: cold_in.totalFlow <= 0");

    double H_hot_in  = streamH(fc_hot_,  hot_in);
    double H_cold_in = streamH(fc_cold_, cold_in);

    double Q;  // [W], heat transferred hot→cold
    if (spec_ == Spec::DUTY) {
        Q = Q_spec;
    } else {  // HOT_OUTLET_T
        // Flash hot side at specified outlet T to get H_hot_out
        auto r_h = fc_hot_.flashTP(T_hot_out, hot_in.P, hot_in.z);
        double H_hot_out_spec = fc_hot_.totalEnthalpy(r_h);
        Q = hot_in.totalFlow * (H_hot_in - H_hot_out_spec);
    }

    duty = Q;

    // Hot outlet
    double H_hot_out = H_hot_in - Q / hot_in.totalFlow;
    auto r_hot = fc_hot_.flashPH(hot_in.P, H_hot_out, hot_in.z, hot_in.T);
    hot_out           = hot_in;
    hot_out.T         = r_hot.T;
    hot_out.vaporFraction = r_hot.beta;
    hot_out.x         = r_hot.x;
    hot_out.y         = r_hot.y;
    hot_out.phase     = (r_hot.beta < 1e-10)  ? Phase::LIQUID
                      : (r_hot.beta > 1-1e-10) ? Phase::VAPOR
                                               : Phase::MIXED;
    hot_out.H = fc_hot_.totalEnthalpy(r_hot);
    hot_out.S = fc_hot_.totalEntropy (r_hot);

    // Cold outlet
    double H_cold_out = H_cold_in + Q / cold_in.totalFlow;
    auto r_cold = fc_cold_.flashPH(cold_in.P, H_cold_out, cold_in.z, cold_in.T);
    cold_out           = cold_in;
    cold_out.T         = r_cold.T;
    cold_out.vaporFraction = r_cold.beta;
    cold_out.x         = r_cold.x;
    cold_out.y         = r_cold.y;
    cold_out.phase     = (r_cold.beta < 1e-10)  ? Phase::LIQUID
                       : (r_cold.beta > 1-1e-10) ? Phase::VAPOR
                                                 : Phase::MIXED;
    cold_out.H = fc_cold_.totalEnthalpy(r_cold);
    cold_out.S = fc_cold_.totalEntropy (r_cold);
}

} // namespace chemsim
