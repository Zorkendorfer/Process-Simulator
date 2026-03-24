#pragma once
#include "chemsim/flowsheet/IUnitOp.hpp"
#include "chemsim/ops/FlashDrum.hpp"

namespace chemsim {

/// IUnitOp wrapper around FlashDrum.
/// Inlet: "feed"    Outlets: "vapor", "liquid"
class FlashDrumOp : public IUnitOp {
public:
    FlashDrumOp(FlashDrum::Spec spec, const FlashCalculator& fc,
                double T_or_H_or_S = 300.0, double P = 1e5)
        : drum_(spec, fc) {
        switch (spec) {
        case FlashDrum::Spec::TP:
            drum_.T_spec = T_or_H_or_S;
            drum_.P_spec = P;
            break;
        case FlashDrum::Spec::PH:
            drum_.H_spec = T_or_H_or_S;
            drum_.P_spec = P;
            break;
        case FlashDrum::Spec::PS:
            drum_.S_spec = T_or_H_or_S;
            drum_.P_spec = P;
            break;
        }
    }

    std::vector<std::string> inletPorts()  const override { return {"feed"}; }
    std::vector<std::string> outletPorts() const override { return {"vapor", "liquid"}; }

    void setInlet(const std::string& port, const Stream& s) override {
        if (port != "feed")
            throw std::invalid_argument("FlashDrumOp: unknown inlet port '" + port + "'");
        drum_.feed = s;
    }

    const Stream& getOutlet(const std::string& port) const override {
        if (port == "vapor")  return drum_.vapor;
        if (port == "liquid") return drum_.liquid;
        throw std::invalid_argument("FlashDrumOp: unknown outlet port '" + port + "'");
    }

    void solve() override { drum_.solve(); }

    FlashDrum& drum() { return drum_; }

private:
    FlashDrum drum_;
};

} // namespace chemsim
