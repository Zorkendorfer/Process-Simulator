#include "chemsim/ops/Reactor.hpp"
#include <stdexcept>
#include <numeric>

namespace chemsim {

Reactor::Reactor(const FlashCalculator& fc, double T_spec, double P_spec)
    : fc_(fc), T_spec_(T_spec), P_spec_(P_spec) {}

void Reactor::addReaction(const Reaction& rxn, double extent_mol_s) {
    reactions_.push_back(rxn);
    extents_.push_back(extent_mol_s);
}

void Reactor::solve() {
    if (inlet.totalFlow <= 0.0)
        throw std::invalid_argument("Reactor: inlet.totalFlow must be > 0");

    const int n = static_cast<int>(inlet.z.size());

    // Check stoichiometry vector lengths
    for (const auto& rxn : reactions_)
        if (static_cast<int>(rxn.nu.size()) != n)
            throw std::invalid_argument("Reactor: reaction nu vector length != nComp");

    // Compute outlet molar flows: n_out,i = F_in*z_in,i + Σ nu_k,i * ξ_k
    std::vector<double> n_out(n);
    for (int i = 0; i < n; ++i) {
        n_out[i] = inlet.totalFlow * inlet.z[i];
        for (int k = 0; k < static_cast<int>(reactions_.size()); ++k)
            n_out[i] += reactions_[k].nu[i] * extents_[k];
        if (n_out[i] < 0.0)
            throw std::runtime_error("Reactor: negative outlet molar flow for component "
                                     + std::to_string(i));
    }

    double F_out = std::accumulate(n_out.begin(), n_out.end(), 0.0);
    if (F_out <= 0.0)
        throw std::runtime_error("Reactor: total outlet flow is non-positive");

    std::vector<double> z_out(n);
    for (int i = 0; i < n; ++i) z_out[i] = n_out[i] / F_out;

    // Isothermal flash at T_spec, P_spec
    auto r = fc_.flashTP(T_spec_, P_spec_, z_out);

    outlet           = inlet;
    outlet.totalFlow = F_out;
    outlet.z         = z_out;
    outlet.T         = r.T;
    outlet.P         = P_spec_;
    outlet.vaporFraction = r.beta;
    outlet.x         = r.x;
    outlet.y         = r.y;
    outlet.phase     = (r.beta < 1e-10)  ? Phase::LIQUID
                     : (r.beta > 1-1e-10) ? Phase::VAPOR
                                          : Phase::MIXED;
    outlet.H = fc_.totalEnthalpy(r);
    outlet.S = fc_.totalEntropy (r);

    // Duty: energy balance (sensible heat change only)
    auto r_in = fc_.flashTP(inlet.T, inlet.P, inlet.z);
    double H_in  = fc_.totalEnthalpy(r_in);
    double H_out = outlet.H;
    duty = F_out * H_out - inlet.totalFlow * H_in;  // [W]
}

} // namespace chemsim
