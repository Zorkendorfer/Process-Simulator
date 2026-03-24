#pragma once
#include <vector>
#include "chemsim/core/Stream.hpp"
#include "chemsim/core/Component.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

namespace chemsim {

// Simplified multi-stage distillation column.
//
// Model assumptions (Phase 3):
//   • Constant Molar Overflow (CMO): L and V are constant within each column section
//   • Total condenser (not a stage): distillate = liquid at bubble point
//   • Partial reboiler counted as stage N
//   • Single feed at stage f (1-indexed from top)
//   • Feed quality q: q=1 → saturated liquid, q=0 → saturated vapor
//   • K-values from Wilson correlation at T_feed; optionally refined by
//     per-stage bubble-point T updates (Wang-Henke inner loop)
//
// Tridiagonal material balance (one system per component):
//   lower[j]*x[j-1] + diag[j]*x[j] + upper[j]*x[j+1] = rhs[j]
//
// Outputs:
//   distillate: liquid stream at P_top,  composition x_D = y_{top-tray}
//   bottoms:    liquid stream at P_top,  composition x_B = x_{reboiler}
//   Q_condenser < 0 (heat removed),  Q_reboiler > 0 (heat added)
class DistillationColumn {
public:
    // N_stages  — number of theoretical stages including reboiler, excluding condenser
    // feed_stage — 1-indexed from top tray (must be in [1, N_stages])
    // reflux_ratio  — L/D
    // distillate_frac — D/F (overall molar distillate fraction)
    // feed_quality   — q  (1=sat. liquid, 0=sat. vapor)
    // max_iter       — outer Wang-Henke iterations (1 = fixed-K simplified model)
    DistillationColumn(const FlashCalculator& fc,
                       const std::vector<Component>& comps,
                       int N_stages,
                       int feed_stage,
                       double reflux_ratio,
                       double distillate_frac,
                       double P_top          = 101325.0,
                       double feed_quality   = 1.0,
                       int    max_iter       = 15);

    // ── Feed (set before solve) ────────────────────────────────────────────
    Stream feed;

    // ── Outlets (filled by solve) ──────────────────────────────────────────
    Stream distillate;
    Stream bottoms;

    double Q_condenser{};   // [W]  heat removed in condenser (≤ 0)
    double Q_reboiler{};    // [W]  heat added  in reboiler   (≥ 0)

    void solve();

    // ── RL interface ──────────────────────────────────────────────────────────
    void setRefluxRatio  (double R)   { R_   = R;   }
    void setDistillateFrac(double phi) { phi_ = phi; }

    // Per-stage temperatures [K], 1-indexed (filled by solve())
    double T_top()    const { return T_stages_.size() > 1 ? T_stages_[1]       : 0.0; }
    double T_mid()    const { return T_stages_.size() > 1 ? T_stages_[N_/2+1]  : 0.0; }
    double T_bottom() const { return T_stages_.size() > 1 ? T_stages_[N_]      : 0.0; }
    double reboilerDuty()  const { return Q_reboiler;  }
    double condenserDuty() const { return Q_condenser; }

private:
    const FlashCalculator&        fc_;
    const std::vector<Component>& comps_;
    int    N_, f_, maxIter_;
    double R_, phi_, P_, q_;

    std::vector<double> T_stages_;   // 1-indexed, size N+1, filled by solve()

    // Wilson K-value for one component at T, P
    static double wilsonK_i(const Component& c, double T, double P);

    // Thomas algorithm: solves tridiagonal lower/diag/upper * x = rhs
    // lower[0] and upper[N-1] are unused (boundary = 0)
    static std::vector<double> thomas(const std::vector<double>& lower,
                                      const std::vector<double>& diag,
                                      const std::vector<double>& upper,
                                      const std::vector<double>& rhs);

    // Bubble-point temperature at P for composition x, using Wilson K
    // Returns T such that Σ K_i(T)*x_i = 1
    double bubbleT(const std::vector<double>& x, double P) const;
};

} // namespace chemsim
