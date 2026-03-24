#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "chemsim/flowsheet/Flowsheet.hpp"
#include "chemsim/flowsheet/MixerOp.hpp"
#include "chemsim/flowsheet/SplitterOp.hpp"
#include "chemsim/io/ComponentDB.hpp"

#ifndef CHEMSIM_DATA_DIR
#define CHEMSIM_DATA_DIR "data"
#endif

using namespace chemsim;

static const std::string DB5 = std::string(CHEMSIM_DATA_DIR) + "/components.json";

namespace {

Flowsheet makeRecycleFlowsheet() {
    ComponentDB db(DB5);
    auto comps = db.get({"METHANE", "ETHANE", "PROPANE", "N-BUTANE"});
    Flowsheet fs(comps);

    const std::map<std::string, double> z = {
        {"METHANE", 0.40},
        {"ETHANE", 0.30},
        {"PROPANE", 0.20},
        {"N-BUTANE", 0.10}
    };

    fs.addStream("FEED", 260.0, 2e6, 100.0, z);
    fs.addStream("RECYCLE", 260.0, 2e6, 100.0, z);
    fs.addStream("MIXOUT", 260.0, 2e6, 100.0, z);

    fs.addUnit("MIX",
               std::make_unique<MixerOp>(fs.flashCalculator(),
                                         std::vector<std::string>{"fresh", "recycle"}));
    fs.addUnit("SPLIT",
               std::make_unique<SplitterOp>(std::vector<double>{0.2, 0.8}));

    fs.connect("FEED", "", "", "MIX", "fresh");
    fs.connect("RECYCLE", "SPLIT", "out1", "MIX", "recycle");
    fs.connect("MIXOUT", "MIX", "out", "SPLIT", "in");
    fs.connect("PRODUCT", "SPLIT", "out0", "", "");

    EXPECT_TRUE(fs.solve());
    return fs;
}

} // namespace

TEST(FlowsheetPolish, SummaryIncludesCoreStreamInfo) {
    auto fs = makeRecycleFlowsheet();
    const auto text = fs.summary();

    EXPECT_NE(text.find("ChemSim Flowsheet Summary"), std::string::npos);
    EXPECT_NE(text.find("PRODUCT"), std::string::npos);
    EXPECT_NE(text.find("RECYCLE"), std::string::npos);
    EXPECT_NE(text.find("phase="), std::string::npos);
}

TEST(FlowsheetPolish, PrintSummaryWritesToStream) {
    auto fs = makeRecycleFlowsheet();
    std::ostringstream out;
    fs.printSummary(out);
    EXPECT_EQ(out.str(), fs.summary());
}

TEST(FlowsheetPolish, ResultsJsonContainsSolvedStreams) {
    auto fs = makeRecycleFlowsheet();
    auto json = fs.resultsAsJson();

    ASSERT_TRUE(json.contains("streams"));
    ASSERT_TRUE(json["streams"].contains("PRODUCT"));
    EXPECT_NEAR(json["streams"]["PRODUCT"]["totalFlow"].get<double>(), 100.0, 1e-4);
    EXPECT_TRUE(json["streams"]["PRODUCT"].contains("phase"));
}

TEST(FlowsheetPolish, ExportResultsWritesJsonFile) {
    auto fs = makeRecycleFlowsheet();
    const auto path = std::filesystem::temp_directory_path() / "chemsim_results_phase5.json";

    fs.exportResults(path.string());

    std::ifstream in(path);
    ASSERT_TRUE(in.is_open());
    nlohmann::json json;
    in >> json;

    EXPECT_TRUE(json.contains("components"));
    EXPECT_TRUE(json.contains("streams"));
    EXPECT_TRUE(json["streams"].contains("FEED"));
    EXPECT_TRUE(json["streams"].contains("PRODUCT"));

    in.close();
    std::filesystem::remove(path);
}

TEST(FlowsheetPolish, UniqueJsonConstructionSolvesRecycleFlowsheet) {
    auto fs = Flowsheet::fromJSONUnique("examples/simple_recycle.json", DB5);

    ASSERT_TRUE(fs);
    EXPECT_TRUE(fs->solve());
    EXPECT_NEAR(fs->getStream("PRODUCT").totalFlow, 100.0, 1e-4);
    EXPECT_NEAR(fs->getStream("RECYCLE").totalFlow, 400.0, 1e-3);
}
