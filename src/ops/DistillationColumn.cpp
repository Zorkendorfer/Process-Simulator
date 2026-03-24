#include "chemsim/ops/DistillationColumn.hpp"
#include "chemsim/numerics/BrentSolver.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace chemsim {

DistillationColumn::DistillationColumn(const FlashCalculator& fc,
                                       const std::vector<Component>& comps,
                                       int N, int f,
                                       double R, double phi,
                                       double P_top, double q, int maxIter)
    : fc_(fc), comps_(comps), N_(N), f_(f), maxIter_(maxIter),
      R_(R), phi_(phi), P_(P_top), q_(q) {
    if (N_ < 2)  throw std::invalid_argument("DistillationColumn: need at least 2 stages");
    if (f_ < 1 || f_ > N_)
        throw std::invalid_argument("DistillationColumn: feed_stage out of range [1,N]");
    if (R_ <= 0.0) throw std::invalid_argument("DistillationColumn: reflux_ratio must be > 0");
    if (phi_ <= 0.0 || phi_ >= 1.0)
        throw std::invalid_argument("DistillationColumn: distillate_frac must be in (0,1)");
}

double DistillationColumn::wilsonK_i(const Component& c, double T, double P) {
    return (c.Pc / P) * std::exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T));
}

// Thomas algorithm (0-indexed, size n)
// lower[0] and upper[n-1] are unused.
std::vector<double> DistillationColumn::thomas(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs) {

    const int n = static_cast<int>(diag.size());
    std::vector<double> w(n), g(n), x(n);

    double d0 = diag[0];
    if (std::abs(d0) < 1e-30) d0 = 1e-30;
    w[0] = upper[0] / d0;
    g[0] = rhs[0]   / d0;

    for (int j = 1; j < n; ++j) {
        double dj = diag[j] - lower[j] * w[j-1];
        if (std::abs(dj) < 1e-30) dj = (dj < 0 ? -1.0 : 1.0) * 1e-30;
        w[j] = upper[j] / dj;
        g[j] = (rhs[j] - lower[j] * g[j-1]) / dj;
    }

    x[n-1] = g[n-1];
    for (int j = n-2; j >= 0; --j)
        x[j] = g[j] - w[j] * x[j+1];
    return x;
}

double DistillationColumn::bubbleT(const std::vector<double>& x, double P) const {
    const int nc = static_cast<int>(comps_.size());
    auto f = [&](double T) {
        double s = 0.0;
        for (int i = 0; i < nc; ++i)
            s += wilsonK_i(comps_[i], T, P) * x[i];
        return s - 1.0;
    };
    return BrentSolver::solve(f, 100.0, 800.0).root;
}

void DistillationColumn::solve() {
    if (feed.totalFlow <= 0.0)
        throw std::invalid_argument("DistillationColumn: feed.totalFlow must be > 0");

    const int  nc = static_cast<int>(comps_.size());
    const int  N  = N_;
    const double F = feed.totalFlow;
    const double D = phi_ * F;
    const double B = (1.0 - phi_) * F;

    // CMO flow rates (1-indexed: L[j], V[j] for stage j = 1..N)
    // Stored 1-indexed in vectors of size N+1 (index 0 unused)
    std::vector<double> Lflow(N+1), Vflow(N+1);
    const double L  = R_ * D;
    const double V  = L  + D;
    const double Lp = L  + q_ * F;
    const double Vp = V  - (1.0 - q_) * F;
    // Bottoms flow = stripping liquid - stripping vapor  (L' - V' = B)
    const double Bflow = Lp - Vp;

    if (Vp <= 0.0)
        throw std::runtime_error(
            "DistillationColumn: stripping-section vapor flow <= 0 "
            "(check R, phi, q)");

    for (int j = 1; j <= N; ++j) {
        if (j < f_) { Lflow[j] = L;  Vflow[j] = V;  }   // rectifying
        else        { Lflow[j] = Lp; Vflow[j] = Vp; }   // stripping (feed stage is stripping)
    }

    // Initial K from Wilson at feed T
    // K_stage[j][i]: 1-indexed j, 0-indexed i
    std::vector<std::vector<double>> Kst(N+1, std::vector<double>(nc));
    double T0 = feed.T;
    for (int j = 1; j <= N; ++j)
        for (int i = 0; i < nc; ++i)
            Kst[j][i] = wilsonK_i(comps_[i], T0, P_);

    // Stage compositions x_stage[j][i], 1-indexed j
    std::vector<std::vector<double>> x_stage(N+1, std::vector<double>(nc, 1.0/nc));

    // Wang-Henke outer loop
    for (int iter = 0; iter < maxIter_; ++iter) {

        // Solve tridiagonal material balance for each component
        for (int i = 0; i < nc; ++i) {
            // 0-indexed arrays for Thomas: index 0..N-1 corresponds to stage 1..N
            std::vector<double> lo(N), di(N), up(N), rhs(N, 0.0);

            for (int j1 = 1; j1 <= N; ++j1) {
                int jj = j1 - 1;
                double Ki_j  = Kst[j1][i];
                double Ki_jp1= (j1 < N) ? Kst[j1+1][i] : 0.0;

                double L_jm1 = (j1 > 1) ? Lflow[j1-1] : L;  // L_0 = R*D
                // V_{j+1}: vapor entering stage j1 from below = vapor leaving stage j1+1
                double V_jp1 = (j1 < N) ? Vflow[j1+1] : 0.0; // V_{N+1} = 0

                // Reboiler (j1=N): liquid leaving downward = bottoms B (not L')
                double L_j = (j1 == N) ? Bflow : Lflow[j1];
                lo[jj]  = L_jm1;
                di[jj]  = -(Vflow[j1] * Ki_j + L_j);
                up[jj]  = V_jp1 * Ki_jp1;

                if (j1 == f_)
                    rhs[jj] = -F * feed.z[i];
            }

            // Total condenser boundary: x_{i,0} = K_{i,1} * x_{i,1}
            // lo[0]*x_{0} term → lo[0]*K_{i,1}*x_{1} absorbed into di[0]
            di[0] += lo[0] * Kst[1][i];
            lo[0]  = 0.0;

            auto x_i = thomas(lo, di, up, rhs);
            for (int j = 0; j < N; ++j)
                x_stage[j+1][i] = x_i[j];
        }

        // Normalize
        for (int j = 1; j <= N; ++j) {
            double s = 0.0;
            for (int i = 0; i < nc; ++i) {
                x_stage[j][i] = std::max(x_stage[j][i], 1e-12);
                s += x_stage[j][i];
            }
            for (int i = 0; i < nc; ++i) x_stage[j][i] /= s;
        }

        // Update K from per-stage bubble-point T (skip on last iteration)
        if (iter < maxIter_ - 1) {
            double err = 0.0;
            for (int j = 1; j <= N; ++j) {
                double Tj = bubbleT(x_stage[j], P_);
                for (int i = 0; i < nc; ++i) {
                    double K_new = wilsonK_i(comps_[i], Tj, P_);
                    err += std::abs(K_new - Kst[j][i]) / (std::abs(Kst[j][i]) + 1e-10);
                    Kst[j][i] = K_new;
                }
            }
            if (err / (N * nc) < 1e-6) break;
        }
    }

    // Store per-stage temperatures (1-indexed, index 0 unused)
    T_stages_.assign(N + 1, 0.0);
    for (int j = 1; j <= N; ++j)
        T_stages_[j] = bubbleT(x_stage[j], P_);

    // ── Build distillate and bottoms ─────────────────────────────────────
    // Total condenser: distillate composition = y_{stage 1} = K_{i,1} * x_{1,i}
    std::vector<double> xD(nc), xB(nc);
    {
        double sD = 0.0;
        for (int i = 0; i < nc; ++i) { xD[i] = Kst[1][i] * x_stage[1][i]; sD += xD[i]; }
        for (int i = 0; i < nc; ++i) xD[i] /= sD;
    }
    xB = x_stage[N];

    double T_dist    = bubbleT(xD, P_);
    double T_bottoms = bubbleT(xB, P_);

    distillate = Stream{};
    distillate.z = distillate.x = distillate.y = xD;
    distillate.T = T_dist; distillate.P = P_;
    distillate.totalFlow = D;
    distillate.vaporFraction = 0.0;
    distillate.phase = Phase::LIQUID;
    distillate.H = fc_.phaseEnthalpy(T_dist,    P_, xD, true);
    distillate.S = fc_.phaseEntropy (T_dist,    P_, xD, true);

    bottoms = Stream{};
    bottoms.z = bottoms.x = bottoms.y = xB;
    bottoms.T = T_bottoms; bottoms.P = P_;
    bottoms.totalFlow = B;
    bottoms.vaporFraction = 0.0;
    bottoms.phase = Phase::LIQUID;
    bottoms.H = fc_.phaseEnthalpy(T_bottoms, P_, xB, true);
    bottoms.S = fc_.phaseEntropy (T_bottoms, P_, xB, true);

    // ── Heat duties via overall energy balance ────────────────────────────
    auto r_feed = fc_.flashTP(feed.T, feed.P, feed.z);
    double H_feed = fc_.totalEnthalpy(r_feed);

    // Q_cond from condensing the top vapor stream:
    //   V = (R+1)*D mol/s of vapor at T_top → cooled to distillate
    double T_top = bubbleT(x_stage[1], P_);
    double H_Vtop = fc_.phaseEnthalpy(T_top, P_, xD, false);  // vapor enthalpy at top
    Q_condenser = V * (distillate.H - H_Vtop);    // ≤ 0 (heat removed)

    // Q_reb from overall energy balance: F*H_F + Q_reb = D*H_D + B*H_B + |Q_cond|
    Q_reboiler = D * distillate.H + B * bottoms.H - F * H_feed - Q_condenser;
}

} // namespace chemsim
