#include "chemsim/thermo/PengRobinson.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace chemsim {

// ─── Constructor ──────────────────────────────────────────────────────────────

PengRobinson::PengRobinson(const std::vector<Component>& components,
                           Eigen::MatrixXd kij)
    : comps_(components)
{
    const int n = static_cast<int>(comps_.size());
    if (kij.size() == 0) {
        kij_ = Eigen::MatrixXd::Zero(n, n);
    } else {
        if (kij.rows() != n || kij.cols() != n)
            throw std::invalid_argument("PengRobinson: kij must be NC×NC");
        kij_ = std::move(kij);
    }

    params_.reserve(n);
    for (const auto& c : comps_) {
        PRParams p;
        p.a     = 0.45724 * R * R * c.Tc * c.Tc / c.Pc;
        p.b     = 0.07780 * R * c.Tc / c.Pc;
        p.kappa = 0.37464 + 1.54226 * c.omega - 0.26992 * c.omega * c.omega;
        params_.push_back(p);
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

double PengRobinson::alpha(int i, double T) const {
    double x = 1.0 + params_[i].kappa * (1.0 - std::sqrt(T / comps_[i].Tc));
    return x * x;
}

double PengRobinson::dalphadT(int i, double T) const {
    double sqrtTTc = std::sqrt(T / comps_[i].Tc);
    double x       = 1.0 + params_[i].kappa * (1.0 - sqrtTTc);
    // dα/dT = -κ * [1 + κ(1-√(T/Tc))] / √(T·Tc)
    return -params_[i].kappa * x / std::sqrt(T * comps_[i].Tc);
}

double PengRobinson::a_i(int i, double T) const {
    return params_[i].a * alpha(i, T);
}

double PengRobinson::mixA(const std::vector<double>& z, double T) const {
    const int n = static_cast<int>(comps_.size());
    double am = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            am += z[i] * z[j] * std::sqrt(a_i(i,T) * a_i(j,T)) * (1.0 - kij_(i,j));
    return am;
}

double PengRobinson::dmixAdT(const std::vector<double>& z, double T) const {
    const int n = static_cast<int>(comps_.size());
    double dam = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double ai = a_i(i, T), aj = a_i(j, T);
            double dai = params_[i].a * dalphadT(i, T);
            double daj = params_[j].a * dalphadT(j, T);
            // d/dT [√(ai·aj)] = (dai·aj + ai·daj) / (2√(ai·aj))
            double d_sqrt = (dai * aj + ai * daj) / (2.0 * std::sqrt(ai * aj));
            dam += z[i] * z[j] * d_sqrt * (1.0 - kij_(i,j));
        }
    }
    return dam;
}

double PengRobinson::mixB(const std::vector<double>& z) const {
    double bm = 0.0;
    for (int i = 0; i < static_cast<int>(comps_.size()); ++i)
        bm += z[i] * params_[i].b;
    return bm;
}

double PengRobinson::dAdn_i(int i, const std::vector<double>& z, double T) const {
    // ∂(n·a_mix)/∂n_i = 2 Σ_j z_j √(a_i·a_j) (1-kij)
    const int n = static_cast<int>(comps_.size());
    double val = 0.0;
    for (int j = 0; j < n; ++j)
        val += z[j] * std::sqrt(a_i(i,T) * a_i(j,T)) * (1.0 - kij_(i,j));
    return 2.0 * val;
}

// ─── Cubic solver ─────────────────────────────────────────────────────────────
// Solve Z³ + p Z² + q Z + r = 0  (Cardano / trig method)
std::vector<double> PengRobinson::solveCubic(double p, double q, double r) {
    // Depress: Z = t - p/3
    double a = q - p * p / 3.0;
    double b = r + 2.0 * p * p * p / 27.0 - p * q / 3.0;
    double disc = b * b / 4.0 + a * a * a / 27.0;

    std::vector<double> roots;

    if (disc > 1e-14) {
        // One real root
        double sqrtDisc = std::sqrt(disc);
        double u = std::cbrt(-b/2.0 + sqrtDisc);
        double v = std::cbrt(-b/2.0 - sqrtDisc);
        roots.push_back(u + v - p/3.0);
    } else {
        // Three real roots (trig method)
        double r_val = std::sqrt(-a * a * a / 27.0);
        double theta = std::acos(std::clamp(-b / (2.0 * r_val), -1.0, 1.0));
        double m = 2.0 * std::cbrt(r_val);
        roots.push_back(m * std::cos(theta / 3.0)             - p/3.0);
        roots.push_back(m * std::cos((theta + 2.0*M_PI) / 3.0) - p/3.0);
        roots.push_back(m * std::cos((theta + 4.0*M_PI) / 3.0) - p/3.0);
    }

    std::sort(roots.begin(), roots.end());
    // Remove physically meaningless roots (Z must be > 0)
    roots.erase(std::remove_if(roots.begin(), roots.end(),
                               [](double z){ return z <= 0.0; }),
                roots.end());
    return roots;
}

double PengRobinson::pickZ(const std::vector<double>& roots, bool liquid) {
    if (roots.empty())
        throw std::runtime_error("PengRobinson: no positive Z roots found");
    if (roots.size() == 1)
        return roots[0];
    // Multiple roots: liquid = smallest, vapor = largest
    return liquid ? roots.front() : roots.back();
}

// ─── Public interface ─────────────────────────────────────────────────────────

std::pair<double,double>
PengRobinson::compressibilityFactors(double T, double P,
                                     const std::vector<double>& z) const {
    double am = mixA(z, T);
    double bm = mixB(z);

    double A = am * P / (R * R * T * T);
    double B = bm * P / (R * T);

    // Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
    double p = -(1.0 - B);
    double q =  (A - 3.0*B*B - 2.0*B);
    double r = -(A*B - B*B - B*B*B);

    auto roots = solveCubic(p, q, r);

    double Z_L = pickZ(roots, true);
    double Z_V = pickZ(roots, false);
    return {Z_L, Z_V};
}

std::vector<double>
PengRobinson::lnFugacityCoefficients(double T, double P,
                                     const std::vector<double>& z,
                                     bool liquid) const {
    const int n = static_cast<int>(comps_.size());
    double am = mixA(z, T);
    double bm = mixB(z);

    double A = am * P / (R * R * T * T);
    double B = bm * P / (R * T);

    auto [ZL, ZV] = compressibilityFactors(T, P, z);
    double Z = liquid ? ZL : ZV;

    const double sqrt2 = std::sqrt(2.0);

    std::vector<double> lnphi(n);
    for (int i = 0; i < n; ++i) {
        double bi_b = params_[i].b / bm;
        double dAdn_i_val = dAdn_i(i, z, T) * P / (R * R * T * T);  // ∂A/∂n_i

        // ln φ_i = (b_i/b)(Z-1) - ln(Z-B) - A/(2√2·B) * [∂A/∂n_i/A - b_i/b] * ln[(Z+(1+√2)B)/(Z+(1-√2)B)]
        lnphi[i] = bi_b * (Z - 1.0)
                 - std::log(Z - B)
                 - A / (2.0 * sqrt2 * B)
                   * (dAdn_i_val / A - bi_b)
                   * std::log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B));
    }
    return lnphi;
}

double PengRobinson::enthalpyDeparture(double T, double P,
                                       const std::vector<double>& z,
                                       bool liquid) const {
    double am  = mixA(z, T);
    double bm  = mixB(z);
    double dam = dmixAdT(z, T);

    double B = bm * P / (R * T);

    auto [ZL, ZV] = compressibilityFactors(T, P, z);
    double Z = liquid ? ZL : ZV;

    const double sqrt2 = std::sqrt(2.0);

    // H_dep = RT(Z-1) - (a - T·da/dT)/(2√2·b) · ln[(Z+(1+√2)B)/(Z+(1-√2)B)]
    double H_dep = R * T * (Z - 1.0)
                 - (am - T * dam) / (2.0 * sqrt2 * bm)
                   * std::log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B));
    return H_dep;
}

double PengRobinson::entropyDeparture(double T, double P,
                                      const std::vector<double>& z,
                                      bool liquid) const {
    double bm  = mixB(z);
    double dam = dmixAdT(z, T);

    double B = bm * P / (R * T);

    auto [ZL, ZV] = compressibilityFactors(T, P, z);
    double Z = liquid ? ZL : ZV;

    const double sqrt2 = std::sqrt(2.0);

    // S_dep = R·ln(Z-B) - dam/(2√2·b) · ln[(Z+(1+√2)B)/(Z+(1-√2)B)]
    double S_dep = R * std::log(Z - B)
                 - dam / (2.0 * sqrt2 * bm)
                   * std::log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B));
    return S_dep;
}

} // namespace chemsim
