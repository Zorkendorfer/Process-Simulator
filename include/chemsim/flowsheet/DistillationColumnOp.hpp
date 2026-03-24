#pragma once
#include "chemsim/flowsheet/IUnitOp.hpp"
#include "chemsim/ops/DistillationColumn.hpp"

namespace chemsim {

/// IUnitOp wrapper around DistillationColumn.
/// Inlet:  "feed"
/// Outlets: "distillate", "bottoms"
class DistillationColumnOp : public IUnitOp {
public:
    DistillationColumnOp(const FlashCalculator& fc,
                         const std::vector<Component>& comps,
                         int N_stages, int feed_stage,
                         double reflux_ratio, double distillate_frac,
                         double P_top        = 101325.0,
                         double feed_quality = 1.0,
                         int    max_iter     = 15)
        : col_(fc, comps, N_stages, feed_stage,
               reflux_ratio, distillate_frac,
               P_top, feed_quality, max_iter) {}

    std::vector<std::string> inletPorts()  const override { return {"feed"}; }
    std::vector<std::string> outletPorts() const override { return {"distillate", "bottoms"}; }

    void setInlet(const std::string& port, const Stream& s) override {
        if (port != "feed")
            throw std::invalid_argument(
                "DistillationColumnOp: unknown inlet port '" + port + "'");
        col_.feed = s;
    }

    const Stream& getOutlet(const std::string& port) const override {
        if (port == "distillate") return col_.distillate;
        if (port == "bottoms")    return col_.bottoms;
        throw std::invalid_argument(
            "DistillationColumnOp: unknown outlet port '" + port + "'");
    }

    void solve() override { col_.solve(); }

    // ── RL interface ─────────────────────────────────────────────────────────
    void setRefluxRatio  (double R)   { col_.setRefluxRatio(R);    }
    void setDistillateFrac(double phi) { col_.setDistillateFrac(phi); }

    double T_top()         const { return col_.T_top();         }
    double T_mid()         const { return col_.T_mid();         }
    double T_bottom()      const { return col_.T_bottom();      }
    double reboilerDuty()  const { return col_.reboilerDuty();  }
    double condenserDuty() const { return col_.condenserDuty(); }

private:
    DistillationColumn col_;
};

} // namespace chemsim
