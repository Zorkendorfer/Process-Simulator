#include <gtest/gtest.h>
#include <cmath>
#include "chemsim/core/Component.hpp"
#include "chemsim/core/Mixture.hpp"
#include "chemsim/io/ComponentDB.hpp"
#include "chemsim/thermo/PengRobinson.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"

#ifndef CHEMSIM_DATA_DIR
#define CHEMSIM_DATA_DIR "data"
#endif
static const std::string DB_PATH2 = std::string(CHEMSIM_DATA_DIR) + "/components.json";

using namespace chemsim;

// ─── Helpers ─────────────────────────────────────────────────────────────────
static std::vector<Component> loadComps(std::initializer_list<const char*> ids) {
    ComponentDB db(DB_PATH2);
    std::vector<std::string> sv(ids.begin(), ids.end());
    return db.get(sv);
}

// ═══════════════════════════════════════════════════════════
// Mixture ideal-gas enthalpy / entropy
// ═══════════════════════════════════════════════════════════

TEST(MixtureThermo, EnthalpyZeroAtTref) {
    auto comps = loadComps({"METHANE", "PROPANE"});
    std::vector<double> z = {0.5, 0.5};
    double H = Mixture::idealGasH(comps, z, 298.15);
    EXPECT_NEAR(H, 0.0, 1e-8);
}

TEST(MixtureThermo, EnthalpyMonotonicallyIncreasing) {
    auto comps = loadComps({"METHANE"});
    std::vector<double> z = {1.0};
    double H300 = Mixture::idealGasH(comps, z, 300.0);
    double H400 = Mixture::idealGasH(comps, z, 400.0);
    double H500 = Mixture::idealGasH(comps, z, 500.0);
    EXPECT_GT(H400, H300);
    EXPECT_GT(H500, H400);
}

TEST(MixtureThermo, EntropySelfConsistent) {
    // S(T) > S(T_ref) since Cp > 0 → ∫Cp/T dT > 0 for T > T_ref
    auto comps = loadComps({"ETHANE"});
    std::vector<double> z = {1.0};
    double S300 = Mixture::idealGasS(comps, z, 300.0);
    double S500 = Mixture::idealGasS(comps, z, 500.0);
    // Include ideal mixing: for pure component z={1.0}, -R*1*ln(1) = 0, no contribution
    EXPECT_GT(S500, S300);
}

TEST(MixtureThermo, EnthalpyMethaneAt500K) {
    // Spot check: ∫Cp dT for CH4 from 298.15 to 500K
    // Cp1=19795, Cp2=60052, Cp3=1022.7, Cp4=35653, Cp5=479.83 (all J/kmol/K)
    // At 400K: Cp ≈ (19795 + 60052*(1022.7/400/sinh(1022.7/400))^2
    //               + 35653*(479.83/400/cosh(479.83/400))^2) / 1000 J/mol/K
    // Rough check: Cp_CH4 ≈ 36-40 J/mol/K in this range → ΔH(298→500) ≈ 7000-8000 J/mol
    auto comps = loadComps({"METHANE"});
    std::vector<double> z = {1.0};
    double H = Mixture::idealGasH(comps, z, 500.0);
    EXPECT_GT(H, 6000.0);   // lower bound ~6 kJ/mol
    EXPECT_LT(H, 10000.0);  // upper bound
}

// ═══════════════════════════════════════════════════════════
// 4-component flash: C1/C2/C3/C4
// ═══════════════════════════════════════════════════════════

class FourCompFlash : public ::testing::Test {
protected:
    void SetUp() override {
        comps = loadComps({"METHANE", "ETHANE", "PROPANE", "N-BUTANE"});
        eos   = std::make_unique<PengRobinson>(comps);
        flash = std::make_unique<FlashCalculator>(*eos, comps);
        // Feed: 40% CH4, 30% C2H6, 20% C3H8, 10% nC4H10
        z = {0.40, 0.30, 0.20, 0.10};
    }
    std::vector<Component>           comps;
    std::unique_ptr<PengRobinson>    eos;
    std::unique_ptr<FlashCalculator> flash;
    std::vector<double>              z;
};

TEST_F(FourCompFlash, ConvergesAtTwoPhaseCond) {
    // 260 K, 2 MPa: four-component mixture should split into two phases
    auto r = flash->flashTP(260.0, 2e6, z);
    EXPECT_TRUE(r.converged);
    EXPECT_GT(r.beta, 0.0);
    EXPECT_LT(r.beta, 1.0);
}

TEST_F(FourCompFlash, MaterialBalanceGlobal) {
    // β*y_i + (1-β)*x_i = z_i  for all components
    auto r = flash->flashTP(260.0, 2e6, z);
    for (int i = 0; i < 4; ++i) {
        double z_check = r.beta * r.y[i] + (1.0 - r.beta) * r.x[i];
        EXPECT_NEAR(z_check, z[i], 1e-8) << "component " << i;
    }
}

TEST_F(FourCompFlash, MoleFractionsSumToOne) {
    auto r = flash->flashTP(260.0, 2e6, z);
    double sumX = 0.0, sumY = 0.0;
    for (int i = 0; i < 4; ++i) { sumX += r.x[i]; sumY += r.y[i]; }
    EXPECT_NEAR(sumX, 1.0, 1e-8);
    EXPECT_NEAR(sumY, 1.0, 1e-8);
}

TEST_F(FourCompFlash, LightComponentEnrichedInVapor) {
    // CH4 (K >> 1) should have y[0] > z[0] and x[0] < z[0]
    auto r = flash->flashTP(260.0, 2e6, z);
    EXPECT_GT(r.y[0], z[0]);
    EXPECT_LT(r.x[0], z[0]);
    // nC4 (K << 1) should be enriched in liquid
    EXPECT_GT(r.x[3], z[3]);
    EXPECT_LT(r.y[3], z[3]);
}

TEST_F(FourCompFlash, FugacityEqualityAtEquilibrium) {
    // At flash equilibrium: K_i = φ_L_i / φ_V_i  ≡ 1 when multiplied correctly
    // Check: φ_L_i * x_i * P = φ_V_i * y_i * P  ↔  φ_L_i * x_i = φ_V_i * y_i
    auto r = flash->flashTP(260.0, 2e6, z);
    auto lnPhiL = eos->lnFugacityCoefficients(r.T, r.P, r.x, true);
    auto lnPhiV = eos->lnFugacityCoefficients(r.T, r.P, r.y, false);
    for (int i = 0; i < 4; ++i) {
        double fL = std::exp(lnPhiL[i]) * r.x[i];
        double fV = std::exp(lnPhiV[i]) * r.y[i];
        EXPECT_NEAR(fL, fV, 1e-7) << "fugacity equality for component " << i;
    }
}

// ═══════════════════════════════════════════════════════════
// Stability test
// ═══════════════════════════════════════════════════════════

TEST_F(FourCompFlash, StabilityUnstableAtTwoPhaseCond) {
    // At conditions known to produce two-phase flash, feed should be unstable
    bool stable = flash->isStable(260.0, 2e6, z);
    EXPECT_FALSE(stable);
}

TEST_F(FourCompFlash, StabilityStableInSinglePhaseVapor) {
    // High T, low P: all vapor → stable
    bool stable = flash->isStable(500.0, 0.5e6, z);
    EXPECT_TRUE(stable);
}

TEST_F(FourCompFlash, StabilityStableInSinglePhaseLiquid) {
    // Low T, high P: compressed liquid → stable
    bool stable = flash->isStable(150.0, 10e6, z);
    EXPECT_TRUE(stable);
}

// ═══════════════════════════════════════════════════════════
// totalEnthalpy: energy balance
// ═══════════════════════════════════════════════════════════

TEST_F(FourCompFlash, EnthalpyEnergyBalanceConsistency) {
    // H of feed z at T should equal phase-split average
    // H_feed(T) = β*H_V(T,y) + (1-β)*H_L(T,x)  ← that's the definition, so it's exact
    // What we test: totalEnthalpy returns a finite, physically reasonable value
    auto r = flash->flashTP(260.0, 2e6, z);
    double H = flash->totalEnthalpy(r);
    // H_dep is typically negative; H_ig at 260K is negative relative to 298.15K
    // Just check it's finite and in a reasonable range (−50 kJ/mol to +50 kJ/mol)
    EXPECT_FALSE(std::isnan(H));
    EXPECT_FALSE(std::isinf(H));
    EXPECT_GT(H, -50000.0);
    EXPECT_LT(H,  50000.0);
}

TEST_F(FourCompFlash, EnthalpyIncreasesWithT) {
    // H(T2) > H(T1) for T2 > T1 (both two-phase)
    auto r1 = flash->flashTP(240.0, 2e6, z);
    auto r2 = flash->flashTP(270.0, 2e6, z);
    double H1 = flash->totalEnthalpy(r1);
    double H2 = flash->totalEnthalpy(r2);
    EXPECT_GT(H2, H1);
}

// ═══════════════════════════════════════════════════════════
// flashPH round-trip: flashPH(P, H_spec) → T_out should equal T_in
// ═══════════════════════════════════════════════════════════

TEST_F(FourCompFlash, FlashPHRoundTrip) {
    double T_ref = 260.0, P = 2e6;
    auto r_tp = flash->flashTP(T_ref, P, z);
    double H_spec = flash->totalEnthalpy(r_tp);

    auto r_ph = flash->flashPH(P, H_spec, z, T_ref);
    EXPECT_NEAR(r_ph.T, T_ref, 0.01);
    EXPECT_NEAR(r_ph.beta, r_tp.beta, 1e-4);
}

TEST_F(FourCompFlash, FlashPHStoresEnthalpy) {
    auto r_tp = flash->flashTP(260.0, 2e6, z);
    double H_spec = flash->totalEnthalpy(r_tp);
    auto r_ph = flash->flashPH(2e6, H_spec, z, 260.0);
    EXPECT_NEAR(r_ph.H_total, H_spec, 1e-3);
}

// ═══════════════════════════════════════════════════════════
// flashPS round-trip
// ═══════════════════════════════════════════════════════════

TEST_F(FourCompFlash, FlashPSRoundTrip) {
    double T_ref = 260.0, P = 2e6;
    auto r_tp = flash->flashTP(T_ref, P, z);
    double S_spec = flash->totalEntropy(r_tp);

    auto r_ps = flash->flashPS(P, S_spec, z, T_ref);
    EXPECT_NEAR(r_ps.T, T_ref, 0.01);
    EXPECT_NEAR(r_ps.beta, r_tp.beta, 1e-4);
}
