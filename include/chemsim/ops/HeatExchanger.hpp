#pragma once
#include "chemsim/core/Stream.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

namespace chemsim {

// Shell-and-tube heat exchanger.  No pressure drop assumed.
//
// Two specification modes:
//   DUTY          — specify total heat duty Q [W] transferred hot→cold
//   HOT_OUTLET_T  — specify hot-side outlet temperature; duty is computed
//
// Energy balance (sign convention: Q > 0 = heat flows hot → cold):
//   H_hot_out  = H_hot_in  - Q / F_hot   [J/mol]
//   H_cold_out = H_cold_in + Q / F_cold  [J/mol]
// Hot/cold outlets are then computed by flashPH.
//
// Hot and cold sides may be different component mixtures (separate fc instances).
class HeatExchanger {
public:
    enum class Spec { DUTY, HOT_OUTLET_T };

    HeatExchanger(Spec spec,
                  const FlashCalculator& fc_hot,
                  const FlashCalculator& fc_cold);

    // ── Inlets ────────────────────────────────────────────────────────────
    Stream hot_in;
    Stream cold_in;

    // ── Specification ─────────────────────────────────────────────────────
    double Q_spec{};        // [W]   — used by DUTY
    double T_hot_out{};     // [K]   — used by HOT_OUTLET_T

    // ── Outlets (filled by solve) ─────────────────────────────────────────
    Stream hot_out;
    Stream cold_out;

    // ── Actual duty [W] ───────────────────────────────────────────────────
    double duty{};

    void solve();

private:
    Spec spec_;
    const FlashCalculator& fc_hot_;
    const FlashCalculator& fc_cold_;

    // Compute phase enthalpy of a stream using its own fc
    static double streamH(const FlashCalculator& fc, const Stream& s);
};

} // namespace chemsim
