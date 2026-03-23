#include <gtest/gtest.h>
#include <cmath>
#include "chemsim/core/Component.hpp"
#include "chemsim/io/ComponentDB.hpp"
#include "chemsim/numerics/BrentSolver.hpp"
#include "chemsim/thermo/PengRobinson.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

#ifndef CHEMSIM_DATA_DIR
#define CHEMSIM_DATA_DIR "data"
#endif
static const std::string DB_PATH = std::string(CHEMSIM_DATA_DIR) + "/components.json";

using namespace chemsim;

// ═══════════════════════════════════════════════════════════
// ComponentDB
// ═══════════════════════════════════════════════════════════

TEST(ComponentDB, LoadMethane) {
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE"});
    ASSERT_EQ(comps.size(), 1u);
    EXPECT_EQ(comps[0].id, "METHANE");
    EXPECT_NEAR(comps[0].Tc,    190.56,   0.01);
    EXPECT_NEAR(comps[0].Pc,    4599200., 1.0);
    EXPECT_NEAR(comps[0].omega, 0.0115,   1e-4);
}

TEST(ComponentDB, UnknownComponentThrows) {
    ComponentDB db(DB_PATH);
    EXPECT_THROW(db.get({"UNOBTANIUM"}), std::runtime_error);
}

TEST(ComponentDB, MultipleComponents) {
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE", "PROPANE"});
    ASSERT_EQ(comps.size(), 3u);
    EXPECT_EQ(comps[0].id, "METHANE");
    EXPECT_EQ(comps[1].id, "ETHANE");
    EXPECT_EQ(comps[2].id, "PROPANE");
}

// ═══════════════════════════════════════════════════════════
// BrentSolver
// ═══════════════════════════════════════════════════════════

TEST(BrentSolver, QuadraticRoot) {
    auto res = BrentSolver::solve([](double x){ return x*x - 2.0; }, 1.0, 2.0);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.root, std::sqrt(2.0), 1e-9);
}

TEST(BrentSolver, CubicRoot) {
    // x³ + x - 1 = 0 → root ≈ 0.6823278038
    auto res = BrentSolver::solve([](double x){ return x*x*x + x - 1.0; }, 0.0, 1.0);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.root, 0.6823278038, 1e-8);
}

TEST(BrentSolver, NoBracketThrows) {
    EXPECT_THROW(
        BrentSolver::solve([](double x){ return x*x + 1.0; }, -1.0, 1.0),
        std::domain_error);
}

TEST(BrentSolver, ExactRootAtEndpoint) {
    auto res = BrentSolver::solve([](double x){ return x - 3.0; }, 3.0, 5.0);
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.root, 3.0, 1e-14);
}

TEST(BrentSolver, TightTolerance) {
    // sin(x) = 0 near π
    auto res = BrentSolver::solve([](double x){ return std::sin(x); }, 3.0, 4.0,
                                  {1e-14, 300});
    EXPECT_NEAR(res.root, M_PI, 1e-12);
}

// ═══════════════════════════════════════════════════════════
// PengRobinson — compressibilityFactors
// ═══════════════════════════════════════════════════════════

// Helper: evaluate the PR cubic polynomial at Z
static double evalCubic(double Z, double A, double B) {
    return Z*Z*Z - (1-B)*Z*Z + (A - 3*B*B - 2*B)*Z - (A*B - B*B - B*B*B);
}

static constexpr double R_gas = 8.314462618;

// Compute dimensionless A, B from component properties
static std::pair<double,double> AB(const Component& c, double T, double P) {
    double kappa = 0.37464 + 1.54226*c.omega - 0.26992*c.omega*c.omega;
    double alpha  = std::pow(1 + kappa*(1 - std::sqrt(T/c.Tc)), 2);
    double ac     = 0.45724*R_gas*R_gas*c.Tc*c.Tc/c.Pc;
    double a      = ac*alpha;
    double b      = 0.07780*R_gas*c.Tc/c.Pc;
    return { a*P/(R_gas*R_gas*T*T), b*P/(R_gas*T) };
}

TEST(PengRobinson, ZSatisfiesCubicMethane200K5MPa) {
    // Verify the returned Z is actually a root of the PR cubic
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE"});
    PengRobinson pr(comps);

    auto [ZL, ZV] = pr.compressibilityFactors(200.0, 5e6, {1.0});
    auto [A, B] = AB(comps[0], 200.0, 5e6);

    // Methane at 200K, 5MPa is slightly supercritical → one real root
    EXPECT_NEAR(evalCubic(ZV, A, B), 0.0, 1e-10);
    EXPECT_NEAR(ZL, ZV, 1e-8);  // single root → both Z values equal
    // PR EOS result: Z ≈ 0.52 (dense supercritical CH4 near critical point)
    EXPECT_GT(ZV, 0.3);
    EXPECT_LT(ZV, 0.8);
}

TEST(PengRobinson, ZSatisfiesCubicEthane300K4MPa) {
    // Ethane at 300K (below Tc=305.32K), 4MPa — dense phase
    ComponentDB db(DB_PATH);
    auto comps = db.get({"ETHANE"});
    PengRobinson pr(comps);

    auto [ZL, ZV] = pr.compressibilityFactors(300.0, 4e6, {1.0});
    auto [A, B] = AB(comps[0], 300.0, 4e6);

    // Both roots (if two exist) must satisfy the cubic
    EXPECT_NEAR(evalCubic(ZL, A, B), 0.0, 1e-10);
    EXPECT_NEAR(evalCubic(ZV, A, B), 0.0, 1e-10);
    EXPECT_GT(ZL, 0.0);
    EXPECT_GT(ZV, 0.0);
}

TEST(PengRobinson, IdealGasLimitZ) {
    // At very low P, Z → 1
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE"});
    PengRobinson pr(comps);

    auto [ZL, ZV] = pr.compressibilityFactors(300.0, 100.0, {1.0});
    EXPECT_NEAR(ZV, 1.0, 1e-4);
}

TEST(PengRobinson, ZPhysicallyReasonable) {
    // Supercritical methane: Z should be less than 1 (attractive forces)
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE"});
    PengRobinson pr(comps);

    auto [ZL, ZV] = pr.compressibilityFactors(200.0, 5e6, {1.0});
    EXPECT_GT(ZV, 0.0);
    EXPECT_LT(ZV, 1.0);
}

TEST(PengRobinson, TwoRootsLiquidAndVapor) {
    // Propane at 250K (below Tc=369.83K), moderate P → two-phase → two Z roots
    ComponentDB db(DB_PATH);
    auto comps = db.get({"PROPANE"});
    PengRobinson pr(comps);

    auto [ZL, ZV] = pr.compressibilityFactors(250.0, 1e5, {1.0});  // 0.1 MPa
    auto [A, B] = AB(comps[0], 250.0, 1e5);

    EXPECT_NEAR(evalCubic(ZL, A, B), 0.0, 1e-10);
    EXPECT_NEAR(evalCubic(ZV, A, B), 0.0, 1e-10);
    EXPECT_LT(ZL, ZV);
    EXPECT_LT(ZL, 0.2);   // liquid Z small
    EXPECT_GT(ZV, 0.8);   // vapor Z near 1 at low P
}

// ═══════════════════════════════════════════════════════════
// PengRobinson — lnFugacityCoefficients
// ═══════════════════════════════════════════════════════════

TEST(PengRobinson, FugacityIdealGasLimit) {
    // At low P: ln(φ) → 0
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE"});
    PengRobinson pr(comps);

    auto lnphi = pr.lnFugacityCoefficients(300.0, 100.0, {1.0}, false);
    EXPECT_NEAR(lnphi[0], 0.0, 1e-4);
}

TEST(PengRobinson, FugacityLiquidLessThanVapor) {
    // For a saturated system below Tc: ln(phi_L) < ln(phi_V) typically,
    // and fugacity f = x*phi*P should be equal at equilibrium.
    // Here we just check that liquid and vapor fugacity coefficients differ.
    ComponentDB db(DB_PATH);
    auto comps = db.get({"PROPANE"});
    PengRobinson pr(comps);

    auto lnphiL = pr.lnFugacityCoefficients(250.0, 5e5, {1.0}, true);
    auto lnphiV = pr.lnFugacityCoefficients(250.0, 5e5, {1.0}, false);
    // Liquid fugacity coefficient substantially larger than vapor in magnitude
    EXPECT_NE(lnphiL[0], lnphiV[0]);
    EXPECT_LT(lnphiV[0], 0.0);  // vapor phi < 1 at moderate pressure
}

TEST(PengRobinson, FugacityEqualityAfterFlash) {
    // After a converged TP flash: f_i^L = f_i^V
    // Test at 200K, 1.5MPa where CH4/C2H6 50/50 is two-phase (confirmed by Wilson K)
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    auto result = fc.flashTP(200.0, 1.5e6, {0.5, 0.5});
    ASSERT_TRUE(result.converged);
    ASSERT_GT(result.beta, 0.0);
    ASSERT_LT(result.beta, 1.0);

    auto lnPhiL = pr.lnFugacityCoefficients(200.0, 1.5e6, result.x, true);
    auto lnPhiV = pr.lnFugacityCoefficients(200.0, 1.5e6, result.y, false);
    for (int i = 0; i < 2; ++i) {
        double fL = result.x[i] * std::exp(lnPhiL[i]);
        double fV = result.y[i] * std::exp(lnPhiV[i]);
        EXPECT_NEAR(fL, fV, 1e-6) << "Fugacity equality failed for component " << i;
    }
}

// ═══════════════════════════════════════════════════════════
// PengRobinson — enthalpyDeparture
// ═══════════════════════════════════════════════════════════

TEST(PengRobinson, EnthalpyDepartureLowPressureLimit) {
    // H_dep → 0 as P → 0
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE"});
    PengRobinson pr(comps);

    double Hdep = pr.enthalpyDeparture(300.0, 100.0, {1.0}, false);
    EXPECT_NEAR(Hdep, 0.0, 0.5);
}

TEST(PengRobinson, EnthalpyDepartureSign) {
    // Dense vapor: H_dep < 0 (attractive forces reduce enthalpy below ideal)
    ComponentDB db(DB_PATH);
    auto comps = db.get({"PROPANE"});
    PengRobinson pr(comps);

    double Hdep = pr.enthalpyDeparture(300.0, 3e6, {1.0}, false);
    EXPECT_LT(Hdep, 0.0);
}

TEST(PengRobinson, EnthalpyDepartureLiquidMoreNegative) {
    // Liquid departure is always more negative than vapor at same T, P
    ComponentDB db(DB_PATH);
    auto comps = db.get({"PROPANE"});
    PengRobinson pr(comps);

    double HdepL = pr.enthalpyDeparture(300.0, 1e5, {1.0}, true);
    double HdepV = pr.enthalpyDeparture(300.0, 1e5, {1.0}, false);
    EXPECT_LT(HdepL, HdepV);
}

// ═══════════════════════════════════════════════════════════
// FlashCalculator — flashTP
// ═══════════════════════════════════════════════════════════

TEST(FlashCalculator, TwoPhaseMethaneEthane200K) {
    // CH4/C2H6 50/50 at 200K, 1.5 MPa: Wilson K confirms two-phase
    // K_CH4 ≈ 2.97, K_C2H6 ≈ 0.11 → clearly two-phase
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    auto result = fc.flashTP(200.0, 1.5e6, {0.5, 0.5});
    ASSERT_TRUE(result.converged);
    EXPECT_GT(result.beta, 0.0);
    EXPECT_LT(result.beta, 1.0);
}

TEST(FlashCalculator, AllVaporLowPressure) {
    // Far above dew point → β = 1
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    auto result = fc.flashTP(300.0, 1e4, {0.5, 0.5});
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.beta, 1.0, 0.01);
}

TEST(FlashCalculator, MaterialBalance) {
    // z_i = β·y_i + (1-β)·x_i for each component
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    std::vector<double> z = {0.5, 0.5};
    auto result = fc.flashTP(200.0, 1.5e6, z);
    ASSERT_TRUE(result.converged);

    for (int i = 0; i < 2; ++i) {
        double z_check = result.beta * result.y[i]
                       + (1.0 - result.beta) * result.x[i];
        EXPECT_NEAR(z_check, z[i], 1e-8)
            << "Material balance failed for component " << i;
    }
}

TEST(FlashCalculator, MoleFractionsSumToOne) {
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    auto result = fc.flashTP(200.0, 1.5e6, {0.5, 0.5});
    ASSERT_TRUE(result.converged);

    double sumX = result.x[0] + result.x[1];
    double sumY = result.y[0] + result.y[1];
    EXPECT_NEAR(sumX, 1.0, 1e-8);
    EXPECT_NEAR(sumY, 1.0, 1e-8);
}

TEST(FlashCalculator, MethaneEnrichedInVapor) {
    // Methane (lighter, supercritical) concentrates in vapor phase
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    auto result = fc.flashTP(200.0, 1.5e6, {0.5, 0.5});
    ASSERT_TRUE(result.converged);
    ASSERT_GT(result.beta, 0.0);
    ASSERT_LT(result.beta, 1.0);

    EXPECT_GT(result.y[0], result.x[0]);  // methane enriched in vapor
    EXPECT_GT(result.x[1], result.y[1]);  // ethane enriched in liquid
}

TEST(FlashCalculator, HigherPressureLessVapor) {
    // For a two-phase system: increasing P towards bubble point reduces β
    ComponentDB db(DB_PATH);
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator fc(pr, comps);

    auto r1 = fc.flashTP(200.0, 0.8e6, {0.5, 0.5});  // lower P → more vapor
    auto r2 = fc.flashTP(200.0, 2.0e6, {0.5, 0.5});  // higher P → less vapor

    ASSERT_TRUE(r1.converged);
    ASSERT_TRUE(r2.converged);
    EXPECT_GT(r1.beta, r2.beta);
}
