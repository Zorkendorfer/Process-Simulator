#pragma once
#include "chemsim/flowsheet/IUnitOp.hpp"
#include <numeric>
#include <cmath>

namespace chemsim {

/// Splits one stream into N streams with identical T, P, z.
/// Inlet: "in"    Outlets: "out0", "out1", ...
class SplitterOp : public IUnitOp {
public:
    explicit SplitterOp(std::vector<double> fractions)
        : fracs_(std::move(fractions)) {
        double s = std::accumulate(fracs_.begin(), fracs_.end(), 0.0);
        if (std::abs(s - 1.0) > 1e-6)
            throw std::invalid_argument("SplitterOp: fractions must sum to 1.0");
        outlets_.resize(fracs_.size());
        for (int k = 0; k < (int)fracs_.size(); ++k)
            outlet_names_.push_back("out" + std::to_string(k));
    }

    std::vector<std::string> inletPorts()  const override { return {"in"}; }
    std::vector<std::string> outletPorts() const override { return outlet_names_; }

    void setInlet(const std::string& port, const Stream& s) override {
        if (port != "in")
            throw std::invalid_argument("SplitterOp: unknown inlet port '" + port + "'");
        inlet_ = s;
    }

    const Stream& getOutlet(const std::string& port) const override {
        for (int k = 0; k < (int)outlet_names_.size(); ++k)
            if (outlet_names_[k] == port) return outlets_[k];
        throw std::invalid_argument("SplitterOp: unknown outlet port '" + port + "'");
    }

    void solve() override {
        for (int k = 0; k < (int)fracs_.size(); ++k) {
            outlets_[k]           = inlet_;
            outlets_[k].totalFlow = inlet_.totalFlow * fracs_[k];
        }
    }

private:
    std::vector<double>      fracs_;
    std::vector<std::string> outlet_names_;
    Stream                   inlet_;
    std::vector<Stream>      outlets_;
};

} // namespace chemsim
