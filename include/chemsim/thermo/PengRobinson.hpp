#pragma once
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "chemsim/thermo/EOS.hpp"
#include "chemsim/core/Component.hpp"

namespace chemsim {

class PengRobinson : public EOS {
public:
    // kij defaults to zero matrix (no binary interaction parameters)
    explicit PengRobinson(const std::vector<Component>& components,
                          Eigen::MatrixXd kij = {});

    std::pair<double,double>
    compressibilityFactors(double T, double P,
                           const std::vector<double>& z) const override;

    std::vector<double>
    lnFugacityCoefficients(double T, double P,
                           const std::vector<double>& z,
                           bool liquid) const override;

    double enthalpyDeparture(double T, double P,
                             const std::vector<double>& z,
                             bool liquid) const override;

    double entropyDeparture(double T, double P,
                            const std::vector<double>& z,
                            bool liquid) const override;

private:
    std::vector<Component> comps_;
    Eigen::MatrixXd kij_;

    struct PRParams {
        double a;      // a_c = 0.45724 R²Tc²/Pc
        double b;      // b   = 0.07780 RTc/Pc
        double kappa;  // 0.37464 + 1.54226ω - 0.26992ω²
    };
    std::vector<PRParams> params_;

    // α_i(T) = [1 + κ_i (1 - √(T/Tc_i))]²
    double alpha(int i, double T) const;

    // dα/dT
    double dalphadT(int i, double T) const;

    // a_i(T) = a_c,i · α_i(T)
    double a_i(int i, double T) const;

    // Mixture a via vdW mixing rules (with kij)
    double mixA(const std::vector<double>& z, double T) const;

    // da_mix/dT
    double dmixAdT(const std::vector<double>& z, double T) const;

    // Mixture b (linear)
    double mixB(const std::vector<double>& z) const;

    // ∂(n·a_mix)/∂n_i — used in fugacity
    double dAdn_i(int i, const std::vector<double>& z, double T) const;

    // Solve Z³ + p·Z² + q·Z + r = 0; returns real roots ascending
    static std::vector<double> solveCubic(double p, double q, double r);

    // Pick Z for requested phase from root set; prefers lowest (liq) or highest (vap)
    static double pickZ(const std::vector<double>& roots, bool liquid);

    static constexpr double R = 8.314462618; // J/mol/K
};

} // namespace chemsim
