#pragma once
#include "chemsim/flowsheet/IUnitOp.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"
#include <map>
#include <numeric>

namespace chemsim {

/// Adiabatic mixer — combines N streams into one via energy + material balance.
/// Uses a PH flash to find the mixed outlet temperature.
/// Inlets: user-specified names   Outlet: "out"
class MixerOp : public IUnitOp {
public:
    MixerOp(const FlashCalculator& fc, std::vector<std::string> inlet_ports)
        : fc_(fc), inlet_ports_(std::move(inlet_ports)) {}

    std::vector<std::string> inletPorts()  const override { return inlet_ports_; }
    std::vector<std::string> outletPorts() const override { return {"out"}; }

    void setInlet(const std::string& port, const Stream& s) override {
        inlets_[port] = s;
    }

    const Stream& getOutlet(const std::string& port) const override {
        if (port != "out")
            throw std::invalid_argument("MixerOp: unknown outlet port '" + port + "'");
        return outlet_;
    }

    void solve() override {
        if (inlets_.empty()) throw std::runtime_error("MixerOp: no inlets set");

        const int nc = static_cast<int>(inlets_.begin()->second.z.size());
        double F_out  = 0.0;
        double P_out  = inlets_.begin()->second.P;
        double H_flow = 0.0;   // sum of F_i * H_i  [J/s]
        std::vector<double> n(nc, 0.0);

        for (const auto& [port, s] : inlets_) {
            if (s.totalFlow <= 0.0) continue;   // skip zero-flow streams
            F_out  += s.totalFlow;
            H_flow += s.totalFlow * s.H;
            for (int i = 0; i < nc; ++i)
                n[i] += s.totalFlow * s.z[i];
        }
        if (F_out <= 0.0) throw std::runtime_error("MixerOp: zero total outlet flow");

        std::vector<double> z_out(nc);
        for (int i = 0; i < nc; ++i) z_out[i] = n[i] / F_out;

        double H_out   = H_flow / F_out;
        double T_guess = inlets_.begin()->second.T;
        auto   r       = fc_.flashPH(P_out, H_out, z_out, T_guess);

        outlet_.totalFlow    = F_out;
        outlet_.z            = z_out;
        outlet_.T            = r.T;
        outlet_.P            = P_out;
        outlet_.vaporFraction = r.beta;
        outlet_.x            = r.x;
        outlet_.y            = r.y;
        outlet_.phase        = (r.beta < 1e-10)   ? Phase::LIQUID
                             : (r.beta > 1-1e-10)  ? Phase::VAPOR : Phase::MIXED;
        outlet_.H            = fc_.totalEnthalpy(r);
        outlet_.S            = fc_.totalEntropy (r);
    }

private:
    const FlashCalculator&          fc_;
    std::vector<std::string>        inlet_ports_;
    std::map<std::string, Stream>   inlets_;
    Stream                          outlet_;
};

} // namespace chemsim
