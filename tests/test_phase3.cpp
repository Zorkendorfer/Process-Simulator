#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include "chemsim/io/ComponentDB.hpp"
#include "chemsim/io/FlowsheetParser.hpp"
#include "chemsim/flowsheet/FlashDrumOp.hpp"
#include "chemsim/flowsheet/Flowsheet.hpp"
#include "chemsim/flowsheet/FlowsheetGraph.hpp"
#include "chemsim/flowsheet/MixerOp.hpp"
#include "chemsim/flowsheet/PumpOp.hpp"
#include "chemsim/flowsheet/RecycleSolver.hpp"
#include "chemsim/flowsheet/SplitterOp.hpp"
#include "chemsim/thermo/PengRobinson.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"
#include "chemsim/ops/FlashDrum.hpp"
#include "chemsim/ops/Pump.hpp"
#include "chemsim/ops/Compressor.hpp"
#include "chemsim/ops/HeatExchanger.hpp"
#include "chemsim/ops/Reactor.hpp"
#include "chemsim/ops/DistillationColumn.hpp"

#ifndef CHEMSIM_DATA_DIR
#define CHEMSIM_DATA_DIR "data"
#endif
static const std::string DB3 = std::string(CHEMSIM_DATA_DIR) + "/components.json";

using namespace chemsim;

// ─── Shared fixture ───────────────────────────────────────────────────────────
struct C4System {
    std::vector<Component>        comps;
    std::unique_ptr<PengRobinson> eos;
    std::unique_ptr<FlashCalculator> fc;
    std::vector<double> z4 = {0.40, 0.30, 0.20, 0.10};  // C1/C2/C3/nC4

    C4System() {
        ComponentDB db(DB3);
        comps = db.get({"METHANE","ETHANE","PROPANE","N-BUTANE"});
        eos   = std::make_unique<PengRobinson>(comps);
        fc    = std::make_unique<FlashCalculator>(*eos, comps);
    }

    Stream makeFeed(double T, double P, double F = 100.0) {
        Stream s;
        s.T = T; s.P = P; s.totalFlow = F; s.z = z4;
        return s;
    }
};

// ═══════════════════════════════════════════════════════════
// FlashDrum
// ═══════════════════════════════════════════════════════════

class FlashDrumTest : public ::testing::Test {
protected:
    C4System sys;
};

TEST_F(FlashDrumTest, TP_MassBalanceClosures) {
    FlashDrum drum(FlashDrum::Spec::TP, *sys.fc);
    drum.feed      = sys.makeFeed(270.0, 2e6);
    drum.T_spec    = 260.0;
    drum.P_spec    = 2e6;
    drum.solve();

    double F_out = drum.vapor.totalFlow + drum.liquid.totalFlow;
    EXPECT_NEAR(F_out, drum.feed.totalFlow, 1e-8);
}

TEST_F(FlashDrumTest, TP_ComponentBalance) {
    FlashDrum drum(FlashDrum::Spec::TP, *sys.fc);
    drum.feed   = sys.makeFeed(270.0, 2e6);
    drum.T_spec = 260.0;  drum.P_spec = 2e6;
    drum.solve();

    double Fv = drum.vapor.totalFlow;
    double Fl = drum.liquid.totalFlow;
    for (int i = 0; i < 4; ++i) {
        double z_check = (Fv * drum.vapor.z[i] + Fl * drum.liquid.z[i])
                       / drum.feed.totalFlow;
        EXPECT_NEAR(z_check, sys.z4[i], 1e-8) << "component " << i;
    }
}

TEST_F(FlashDrumTest, TP_SinglePhaseVapor) {
    // High T / low P → all vapor
    FlashDrum drum(FlashDrum::Spec::TP, *sys.fc);
    drum.feed   = sys.makeFeed(400.0, 0.5e6);
    drum.T_spec = 400.0;  drum.P_spec = 0.5e6;
    drum.solve();

    EXPECT_NEAR(drum.vapor.totalFlow, drum.feed.totalFlow, 1e-6);
    EXPECT_NEAR(drum.liquid.totalFlow, 0.0, 1e-6);
}

TEST_F(FlashDrumTest, PH_RoundTrip) {
    // Compute H_spec from a known TP flash, then PH flash should recover same T
    double T_ref = 260.0, P = 2e6;
    auto r = sys.fc->flashTP(T_ref, P, sys.z4);
    double H_spec = sys.fc->totalEnthalpy(r);

    FlashDrum drum(FlashDrum::Spec::PH, *sys.fc);
    drum.feed   = sys.makeFeed(270.0, P);
    drum.P_spec = P;
    drum.H_spec = H_spec;
    drum.solve();

    // Vapor + liquid T should match T_ref
    EXPECT_NEAR(drum.vapor.T, T_ref, 0.01);
}

TEST_F(FlashDrumTest, TP_DutySignMakeSense) {
    // Cooling from 270K to 260K: duty should be negative (heat removed)
    FlashDrum drum(FlashDrum::Spec::TP, *sys.fc);
    drum.feed   = sys.makeFeed(270.0, 2e6);
    drum.T_spec = 260.0;  drum.P_spec = 2e6;
    drum.solve();
    EXPECT_LT(drum.duty, 0.0);
}

// ═══════════════════════════════════════════════════════════
// Pump  — use C3/nC4 at 250K where the feed is clearly all liquid
// ═══════════════════════════════════════════════════════════

class PumpTest : public ::testing::Test {
protected:
    // C3/nC4 50-50 at 250K is well below bubble point → liquid
    std::vector<Component>           comps;
    std::unique_ptr<PengRobinson>    eos;
    std::unique_ptr<FlashCalculator> fc;
    std::vector<double> z = {0.5, 0.5};

    void SetUp() override {
        ComponentDB db(DB3);
        comps = db.get({"PROPANE","N-BUTANE"});
        eos   = std::make_unique<PengRobinson>(comps);
        fc    = std::make_unique<FlashCalculator>(*eos, comps);
    }

    Stream makeLiquidFeed(double T = 250.0, double P = 0.5e6, double F = 100.0) {
        Stream s; s.T=T; s.P=P; s.totalFlow=F; s.z=z; return s;
    }
};

TEST_F(PumpTest, OutletPressureCorrect) {
    Pump pump(*fc, 2e6, 0.75);
    pump.inlet = makeLiquidFeed();
    pump.solve();
    EXPECT_NEAR(pump.outlet.P, 2e6, 1.0);
}

TEST_F(PumpTest, OutletEnthalpyIncreased) {
    Pump pump(*fc, 2e6, 0.75);
    pump.inlet = makeLiquidFeed();
    pump.solve();
    double H_in = fc->phaseEnthalpy(250.0, 0.5e6, z, true);
    EXPECT_GT(pump.outlet.H, H_in);
}

TEST_F(PumpTest, ShaftPowerPositive) {
    Pump pump(*fc, 2e6, 0.75);
    pump.inlet = makeLiquidFeed();
    pump.solve();
    EXPECT_GT(pump.shaft_power_W, 0.0);
}

TEST_F(PumpTest, EnergyBalance) {
    // shaft_power = F * (H_out - H_in)
    Pump pump(*fc, 2e6, 0.75);
    pump.inlet = makeLiquidFeed();
    pump.solve();
    double H_in = fc->phaseEnthalpy(250.0, 0.5e6, z, true);
    double expected_power = pump.inlet.totalFlow * (pump.outlet.H - H_in);
    EXPECT_NEAR(pump.shaft_power_W, expected_power, 1.0);
}

// ═══════════════════════════════════════════════════════════
// Compressor
// ═══════════════════════════════════════════════════════════

class CompressorTest : public ::testing::Test {
protected:
    C4System sys;
};

// Use P_out=1MPa: at 350K/1MPa all C1-C4 K>1, so outlet stays vapor.
TEST_F(CompressorTest, OutletPressureCorrect) {
    Compressor comp(*sys.fc, 1e6, 0.72);
    comp.inlet = sys.makeFeed(350.0, 0.5e6);
    comp.solve();
    EXPECT_NEAR(comp.outlet.P, 1e6, 1.0);
}

TEST_F(CompressorTest, ShaftPowerPositive) {
    Compressor comp(*sys.fc, 1e6, 0.72);
    comp.inlet = sys.makeFeed(350.0, 0.5e6);
    comp.solve();
    EXPECT_GT(comp.shaft_power_W, 0.0);
}

TEST_F(CompressorTest, LowerEfficiencyMoreWork) {
    Compressor c1(*sys.fc, 1e6, 0.80);
    Compressor c2(*sys.fc, 1e6, 0.60);
    auto feed = sys.makeFeed(350.0, 0.5e6);
    c1.inlet = c2.inlet = feed;
    c1.solve(); c2.solve();
    EXPECT_GT(c2.shaft_power_W, c1.shaft_power_W);
}

// ═══════════════════════════════════════════════════════════
// HeatExchanger
// ═══════════════════════════════════════════════════════════

class HXTest : public ::testing::Test {
protected:
    C4System sys;
};

TEST_F(HXTest, DutySpec_EnergyBalance) {
    // Q transferred hot→cold: hot loses Q, cold gains Q
    double Q = 1e6;  // 1 MW
    HeatExchanger hx(HeatExchanger::Spec::DUTY, *sys.fc, *sys.fc);
    hx.hot_in  = sys.makeFeed(350.0, 2e6);
    hx.cold_in = sys.makeFeed(200.0, 2e6);
    hx.Q_spec  = Q;
    hx.solve();

    double H_hot_in  = sys.fc->totalEnthalpy(sys.fc->flashTP(350.0, 2e6, sys.z4));
    double H_cold_in = sys.fc->totalEnthalpy(sys.fc->flashTP(200.0, 2e6, sys.z4));

    // Energy removed from hot side
    double Q_hot = hx.hot_in.totalFlow  * (H_hot_in  - hx.hot_out.H);
    // Energy gained by cold side
    double Q_cold = hx.cold_in.totalFlow * (hx.cold_out.H - H_cold_in);

    EXPECT_NEAR(Q_hot,  Q, 1.0);
    EXPECT_NEAR(Q_cold, Q, 1.0);
    EXPECT_NEAR(Q_hot, Q_cold, 1.0);
}

TEST_F(HXTest, DutySpec_HotOutletCooled) {
    HeatExchanger hx(HeatExchanger::Spec::DUTY, *sys.fc, *sys.fc);
    hx.hot_in  = sys.makeFeed(350.0, 2e6);
    hx.cold_in = sys.makeFeed(200.0, 2e6);
    hx.Q_spec  = 5e5;
    hx.solve();
    EXPECT_LT(hx.hot_out.T, hx.hot_in.T);
    EXPECT_GT(hx.cold_out.T, hx.cold_in.T);
}

TEST_F(HXTest, HotOutletT_Spec) {
    HeatExchanger hx(HeatExchanger::Spec::HOT_OUTLET_T, *sys.fc, *sys.fc);
    hx.hot_in      = sys.makeFeed(350.0, 2e6);
    hx.cold_in     = sys.makeFeed(200.0, 2e6);
    hx.T_hot_out   = 280.0;
    hx.solve();
    EXPECT_NEAR(hx.hot_out.T, 280.0, 0.5);
}

// ═══════════════════════════════════════════════════════════
// Reactor
// ═══════════════════════════════════════════════════════════

class ReactorTest : public ::testing::Test {
protected:
    // C3H8 → C2H4 + CH4  (hypothetical isothermal cracking, single reaction)
    // Components: METHANE(0), ETHANE(1), PROPANE(2), N-BUTANE(3)
    // Reaction: C3H8 → CH4 + C2H4 — but we don't have C2H4, so use C2H6 as proxy
    // nu: +1 CH4, +1 C2H6, -1 C3H8, 0 nC4
    C4System sys;
};

TEST_F(ReactorTest, MolarFlowBalance) {
    // nu = [+1, +1, -1, 0], extent = 10 mol/s from feed of 100 mol/s with z=[0.4,0.3,0.2,0.1]
    Reactor rxr(*sys.fc, 300.0, 2e6);
    Reaction rxn; rxn.nu = {1.0, 1.0, -1.0, 0.0};
    rxr.addReaction(rxn, 10.0);
    rxr.inlet = sys.makeFeed(300.0, 2e6, 100.0);
    rxr.solve();

    // Expected n_out: [40+10, 30+10, 20-10, 10] = [50, 40, 10, 10]
    // F_out = 110
    EXPECT_NEAR(rxr.outlet.totalFlow, 110.0, 1e-8);
    EXPECT_NEAR(rxr.outlet.z[0] * rxr.outlet.totalFlow, 50.0, 1e-8);  // CH4
    EXPECT_NEAR(rxr.outlet.z[1] * rxr.outlet.totalFlow, 40.0, 1e-8);  // C2H6
    EXPECT_NEAR(rxr.outlet.z[2] * rxr.outlet.totalFlow, 10.0, 1e-8);  // C3H8
}

TEST_F(ReactorTest, CompositionsSumToOne) {
    Reactor rxr(*sys.fc, 300.0, 2e6);
    Reaction rxn; rxn.nu = {1.0, 0.0, -1.0, 0.0};
    rxr.addReaction(rxn, 5.0);
    rxr.inlet = sys.makeFeed(300.0, 2e6);
    rxr.solve();

    double sum = 0.0;
    for (double zi : rxr.outlet.z) sum += zi;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(ReactorTest, NegativeFlowThrows) {
    Reactor rxr(*sys.fc, 300.0, 2e6);
    Reaction rxn; rxn.nu = {0.0, 0.0, -1.0, 0.0};
    rxr.addReaction(rxn, 1000.0);  // consumes far more C3H8 than available
    rxr.inlet = sys.makeFeed(300.0, 2e6, 100.0);
    EXPECT_THROW(rxr.solve(), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════
// DistillationColumn
// ═══════════════════════════════════════════════════════════

class DistColTest : public ::testing::Test {
protected:
    // Binary C1/C2 separation, 50/50 feed
    std::vector<Component> comps2;
    std::unique_ptr<PengRobinson>    eos2;
    std::unique_ptr<FlashCalculator> fc2;
    std::vector<double> z2 = {0.5, 0.5};

    void SetUp() override {
        ComponentDB db(DB3);
        comps2 = db.get({"METHANE","ETHANE"});
        eos2   = std::make_unique<PengRobinson>(comps2);
        fc2    = std::make_unique<FlashCalculator>(*eos2, comps2);
    }
};

TEST_F(DistColTest, TotalMaterialBalance) {
    DistillationColumn col(*fc2, comps2, 10, 5, 2.0, 0.5, 2e6, 1.0, 10);
    col.feed.T = 210.0; col.feed.P = 2e6;
    col.feed.totalFlow = 100.0; col.feed.z = z2;
    col.solve();

    double F_out = col.distillate.totalFlow + col.bottoms.totalFlow;
    EXPECT_NEAR(F_out, col.feed.totalFlow, 1e-6);
}

TEST_F(DistColTest, ComponentMaterialBalance) {
    DistillationColumn col(*fc2, comps2, 10, 5, 2.0, 0.5, 2e6, 1.0, 10);
    col.feed.T = 210.0; col.feed.P = 2e6;
    col.feed.totalFlow = 100.0; col.feed.z = z2;
    col.solve();

    double D = col.distillate.totalFlow;
    double B = col.bottoms.totalFlow;
    for (int i = 0; i < 2; ++i) {
        double z_check = (D * col.distillate.z[i] + B * col.bottoms.z[i])
                       / col.feed.totalFlow;
        // Simplified CMO model: balance closes to ~1e-3 (limited by K-value accuracy)
        EXPECT_NEAR(z_check, z2[i], 2e-3) << "component " << i;
    }
}

TEST_F(DistColTest, LightKeyEnrichedInDistillate) {
    // CH4 (more volatile) should be enriched in distillate
    DistillationColumn col(*fc2, comps2, 10, 5, 2.0, 0.5, 2e6, 1.0, 10);
    col.feed.T = 210.0; col.feed.P = 2e6;
    col.feed.totalFlow = 100.0; col.feed.z = z2;
    col.solve();

    EXPECT_GT(col.distillate.z[0], z2[0]);  // CH4 enriched in distillate
    EXPECT_GT(col.bottoms.z[1],    z2[1]);  // C2H6 enriched in bottoms
}

TEST_F(DistColTest, DistillateCompositionsSumToOne) {
    DistillationColumn col(*fc2, comps2, 10, 5, 2.0, 0.5, 2e6, 1.0, 10);
    col.feed.T = 210.0; col.feed.P = 2e6;
    col.feed.totalFlow = 100.0; col.feed.z = z2;
    col.solve();

    double sumD = 0.0, sumB = 0.0;
    for (int i = 0; i < 2; ++i) { sumD += col.distillate.z[i]; sumB += col.bottoms.z[i]; }
    EXPECT_NEAR(sumD, 1.0, 1e-8);
    EXPECT_NEAR(sumB, 1.0, 1e-8);
}

TEST_F(DistColTest, CondenserDutyNegative) {
    DistillationColumn col(*fc2, comps2, 10, 5, 2.0, 0.5, 2e6, 1.0, 10);
    col.feed.T = 210.0; col.feed.P = 2e6;
    col.feed.totalFlow = 100.0; col.feed.z = z2;
    col.solve();
    EXPECT_LT(col.Q_condenser, 0.0);  // heat removed
    EXPECT_GT(col.Q_reboiler,  0.0);  // heat added
}

// Phase 4: flowsheet layer

static Stream flashedStream(const FlashCalculator& fc,
                            double T, double P, double F,
                            const std::vector<double>& z) {
    Stream s;
    s.T = T;
    s.P = P;
    s.totalFlow = F;
    s.z = z;

    auto r = fc.flashTP(T, P, z);
    s.vaporFraction = r.beta;
    s.x = r.x;
    s.y = r.y;
    s.phase = (r.beta < 1e-10) ? Phase::LIQUID
            : (r.beta > 1.0 - 1e-10) ? Phase::VAPOR
            : Phase::MIXED;
    s.H = fc.totalEnthalpy(r);
    s.S = fc.totalEntropy(r);
    return s;
}

class FlowsheetOpsTest : public ::testing::Test {
protected:
    C4System sys;
};

TEST_F(FlowsheetOpsTest, MixerOpCombinesMaterialAndKeepsSharedState) {
    MixerOp mixer(*sys.fc, {"feed_a", "feed_b"});
    auto a = flashedStream(*sys.fc, 260.0, 2e6, 40.0, sys.z4);
    auto b = flashedStream(*sys.fc, 260.0, 2e6, 60.0, sys.z4);

    mixer.setInlet("feed_a", a);
    mixer.setInlet("feed_b", b);
    mixer.solve();

    const auto& out = mixer.getOutlet("out");
    EXPECT_NEAR(out.totalFlow, 100.0, 1e-8);
    EXPECT_NEAR(out.T, 260.0, 0.05);
    EXPECT_NEAR(out.P, 2e6, 1.0);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(out.z[i], sys.z4[i], 1e-8);
}

TEST_F(FlowsheetOpsTest, SplitterOpPreservesCompositionAndSplitsFlow) {
    SplitterOp splitter({0.25, 0.75});
    auto feed = flashedStream(*sys.fc, 260.0, 2e6, 80.0, sys.z4);

    splitter.setInlet("in", feed);
    splitter.solve();

    const auto& out0 = splitter.getOutlet("out0");
    const auto& out1 = splitter.getOutlet("out1");
    EXPECT_NEAR(out0.totalFlow, 20.0, 1e-8);
    EXPECT_NEAR(out1.totalFlow, 60.0, 1e-8);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out0.z[i], feed.z[i], 1e-12);
        EXPECT_NEAR(out1.z[i], feed.z[i], 1e-12);
    }
}

TEST_F(FlowsheetOpsTest, FlashDrumOpDelegatesToUnderlyingUnit) {
    FlashDrumOp drum(FlashDrum::Spec::TP, *sys.fc, 260.0, 2e6);
    drum.setInlet("feed", sys.makeFeed(270.0, 2e6, 100.0));
    drum.solve();

    double F_out = drum.getOutlet("vapor").totalFlow + drum.getOutlet("liquid").totalFlow;
    EXPECT_NEAR(F_out, 100.0, 1e-8);
}

TEST_F(FlowsheetOpsTest, PumpOpDelegatesToUnderlyingUnit) {
    ComponentDB db(DB3);
    auto comps = db.get({"PROPANE", "N-BUTANE"});
    PengRobinson eos(comps);
    FlashCalculator fc(eos, comps);
    PumpOp pump(fc, 2e6, 0.75);

    pump.setInlet("in", flashedStream(fc, 250.0, 0.5e6, 50.0, {0.5, 0.5}));
    pump.solve();

    EXPECT_NEAR(pump.getOutlet("out").P, 2e6, 1.0);
    EXPECT_GT(pump.shaftPower(), 0.0);
}

TEST(FlowsheetGraphTest, TopologicalSortSupportsFeedsAndProducts) {
    FlowsheetGraph graph;
    graph.addUnit("MIX", std::make_unique<SplitterOp>(std::vector<double>{0.5, 0.5}));
    graph.addUnit("PUMP", std::make_unique<SplitterOp>(std::vector<double>{0.5, 0.5}));

    // Build a simple DAG with externally supplied streams.
    graph.connect("FEED", "", "", "MIX", "in");
    graph.connect("MID", "MIX", "out0", "PUMP", "in");
    graph.connect("PRODUCT", "PUMP", "out0", "", "");

    auto order = graph.topoSort();
    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], "MIX");
    EXPECT_EQ(order[1], "PUMP");
}

TEST(FlowsheetGraphTest, DetectsRecycleAndSelectsTearStream) {
    FlowsheetGraph graph;
    graph.addUnit("MIX", std::make_unique<SplitterOp>(std::vector<double>{0.2, 0.8}));
    graph.addUnit("SPLIT", std::make_unique<SplitterOp>(std::vector<double>{0.5, 0.5}));

    graph.connect("MIXOUT", "MIX", "out0", "SPLIT", "in");
    graph.connect("RECYCLE", "SPLIT", "out1", "MIX", "in");

    auto sccs = graph.findSCCs();
    ASSERT_EQ(sccs.size(), 1u);
    EXPECT_EQ(sccs[0].size(), 2u);

    auto tears = graph.selectTearStreams();
    ASSERT_EQ(tears.size(), 1u);
    EXPECT_TRUE(tears[0] == "MIXOUT" || tears[0] == "RECYCLE");
}

class RecycleFlowsheetTest : public ::testing::Test {
protected:
    void SetUp() override {
        ComponentDB db(DB3);
        comps = db.get({"METHANE", "ETHANE", "PROPANE", "N-BUTANE"});
        eos = std::make_unique<PengRobinson>(comps);
        fc = std::make_unique<FlashCalculator>(*eos, comps);
        z = {0.40, 0.30, 0.20, 0.10};
    }

    std::vector<Component> comps;
    std::unique_ptr<PengRobinson> eos;
    std::unique_ptr<FlashCalculator> fc;
    std::vector<double> z;
};

TEST_F(RecycleFlowsheetTest, RecycleSolverConvergesSimpleLoop) {
    FlowsheetGraph graph;
    graph.addUnit("MIX", std::make_unique<MixerOp>(*fc, std::vector<std::string>{"fresh", "recycle"}));
    graph.addUnit("SPLIT", std::make_unique<SplitterOp>(std::vector<double>{0.2, 0.8}));

    graph.connect("FEED", "", "", "MIX", "fresh");
    graph.connect("RECYCLE", "SPLIT", "out1", "MIX", "recycle");
    graph.connect("MIXOUT", "MIX", "out", "SPLIT", "in");
    graph.connect("PRODUCT", "SPLIT", "out0", "", "");

    auto feed = flashedStream(*fc, 260.0, 2e6, 100.0, z);
    auto guess = flashedStream(*fc, 260.0, 2e6, 100.0, z);

    RecycleSolver solver(graph, RecycleSolver::Options{100, 0.01, 1.0, 1e-6, 1e-6, 1.0});
    auto result = solver.solve({
        {"FEED", feed},
        {"RECYCLE", guess},
        {"MIXOUT", guess}
    });

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.streams.at("PRODUCT").totalFlow, 100.0, 1e-4);
    EXPECT_NEAR(result.streams.at("RECYCLE").totalFlow, 400.0, 1e-3);
    EXPECT_NEAR(result.streams.at("MIXOUT").totalFlow, 500.0, 1e-3);
}

TEST_F(RecycleFlowsheetTest, FlowsheetTopLevelSolvesSimpleRecycle) {
    Flowsheet fs(comps);
    fs.addStream("FEED", 260.0, 2e6, 100.0,
                 {{"METHANE", 0.40}, {"ETHANE", 0.30}, {"PROPANE", 0.20}, {"N-BUTANE", 0.10}});
    fs.addStream("RECYCLE", 260.0, 2e6, 100.0,
                 {{"METHANE", 0.40}, {"ETHANE", 0.30}, {"PROPANE", 0.20}, {"N-BUTANE", 0.10}});
    fs.addStream("MIXOUT", 260.0, 2e6, 100.0,
                 {{"METHANE", 0.40}, {"ETHANE", 0.30}, {"PROPANE", 0.20}, {"N-BUTANE", 0.10}});

    fs.addUnit("MIX", std::make_unique<MixerOp>(*fc, std::vector<std::string>{"fresh", "recycle"}));
    fs.addUnit("SPLIT", std::make_unique<SplitterOp>(std::vector<double>{0.2, 0.8}));

    fs.connect("FEED", "", "", "MIX", "fresh");
    fs.connect("RECYCLE", "SPLIT", "out1", "MIX", "recycle");
    fs.connect("MIXOUT", "MIX", "out", "SPLIT", "in");
    fs.connect("PRODUCT", "SPLIT", "out0", "", "");

    EXPECT_TRUE(fs.solve());
    EXPECT_NEAR(fs.getStream("PRODUCT").totalFlow, 100.0, 1e-4);
    EXPECT_NEAR(fs.getStream("RECYCLE").totalFlow, 400.0, 1e-3);
}

TEST_F(RecycleFlowsheetTest, JsonParserBuildsAndSolvesRecycleFlowsheet) {
    const auto json_path = std::filesystem::temp_directory_path() / "chemsim_phase4_flowsheet.json";
    std::ofstream out(json_path);
    out << R"json(
{
  "components": ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"],
  "streams": {
    "FEED": {
      "T": 260.0,
      "P": 2000000.0,
      "flow": 100.0,
      "composition": {
        "METHANE": 0.40,
        "ETHANE": 0.30,
        "PROPANE": 0.20,
        "N-BUTANE": 0.10
      }
    },
    "RECYCLE": {
      "T": 260.0,
      "P": 2000000.0,
      "flow": 100.0,
      "composition": {
        "METHANE": 0.40,
        "ETHANE": 0.30,
        "PROPANE": 0.20,
        "N-BUTANE": 0.10
      }
    },
    "MIXOUT": {
      "T": 260.0,
      "P": 2000000.0,
      "flow": 100.0,
      "composition": {
        "METHANE": 0.40,
        "ETHANE": 0.30,
        "PROPANE": 0.20,
        "N-BUTANE": 0.10
      }
    }
  },
  "units": {
    "MIX": {
      "type": "Mixer",
      "inlet_ports": ["fresh", "recycle"]
    },
    "SPLIT": {
      "type": "Splitter",
      "fractions": [0.2, 0.8]
    }
  },
  "connections": [
    {"stream": "FEED", "to": "MIX", "to_port": "fresh"},
    {"stream": "RECYCLE", "from": "SPLIT", "from_port": "out1", "to": "MIX", "to_port": "recycle"},
    {"stream": "MIXOUT", "from": "MIX", "from_port": "out", "to": "SPLIT", "to_port": "in"},
    {"stream": "PRODUCT", "from": "SPLIT", "from_port": "out0"}
  ]
}
)json";
    out.close();

    auto fs = Flowsheet::fromJSON(json_path.string(), DB3);
    EXPECT_TRUE(fs.solve());
    EXPECT_NEAR(fs.getStream("PRODUCT").totalFlow, 100.0, 1e-4);
    EXPECT_NEAR(fs.getStream("RECYCLE").totalFlow, 400.0, 1e-3);

    std::filesystem::remove(json_path);
}
