#pragma once
#include "chemsim/flowsheet/IUnitOp.hpp"
#include "chemsim/ops/Pump.hpp"

namespace chemsim {

/// IUnitOp wrapper around Pump.
/// Inlet: "in"    Outlet: "out"
class PumpOp : public IUnitOp {
public:
    PumpOp(const FlashCalculator& fc, double P_out, double eta = 0.75)
        : pump_(fc, P_out, eta) {}

    std::vector<std::string> inletPorts()  const override { return {"in"}; }
    std::vector<std::string> outletPorts() const override { return {"out"}; }

    void setInlet(const std::string& port, const Stream& s) override {
        if (port != "in")
            throw std::invalid_argument("PumpOp: unknown inlet port '" + port + "'");
        pump_.inlet = s;
    }

    const Stream& getOutlet(const std::string& port) const override {
        if (port != "out")
            throw std::invalid_argument("PumpOp: unknown outlet port '" + port + "'");
        return pump_.outlet;
    }

    void solve() override { pump_.solve(); }
    double shaftPower() const { return pump_.shaft_power_W; }

private:
    Pump pump_;
};

} // namespace chemsim
