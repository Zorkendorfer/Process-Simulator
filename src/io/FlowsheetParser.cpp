#include "chemsim/io/FlowsheetParser.hpp"
#include "chemsim/flowsheet/DistillationColumnOp.hpp"
#include "chemsim/flowsheet/FlashDrumOp.hpp"
#include "chemsim/flowsheet/MixerOp.hpp"
#include "chemsim/flowsheet/PumpOp.hpp"
#include "chemsim/flowsheet/SplitterOp.hpp"
#include "chemsim/io/ComponentDB.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace chemsim {

namespace {

std::string asStringOrEmpty(const nlohmann::json& j, const char* key) {
    if (!j.contains(key) || j.at(key).is_null()) return {};
    return j.at(key).get<std::string>();
}

} // namespace

std::unique_ptr<Flowsheet> FlowsheetParser::parseFileUnique(
    const std::string& json_path,
    const std::string& component_db_path) {
    std::ifstream file(json_path);
    if (!file.is_open())
        throw std::runtime_error("FlowsheetParser: cannot open " + json_path);

    nlohmann::json j;
    file >> j;

    const auto component_ids = j.at("components").get<std::vector<std::string>>();
    ComponentDB db(component_db_path);
    auto flowsheet = std::make_unique<Flowsheet>(db.get(component_ids));

    if (j.contains("streams")) {
        for (auto& [name, stream_json] : j.at("streams").items()) {
            flowsheet->addStream(
                name,
                stream_json.at("T").get<double>(),
                stream_json.at("P").get<double>(),
                stream_json.at("flow").get<double>(),
                stream_json.at("composition").get<std::map<std::string, double>>());
        }
    }

    for (auto& [name, unit_json] : j.at("units").items()) {
        const auto type = unit_json.at("type").get<std::string>();

        if (type == "Mixer") {
            flowsheet->addUnit(
                name,
                std::make_unique<MixerOp>(
                    flowsheet->flashCalculator(),
                    unit_json.at("inlet_ports").get<std::vector<std::string>>()));
        } else if (type == "Splitter") {
            flowsheet->addUnit(
                name,
                std::make_unique<SplitterOp>(
                    unit_json.at("fractions").get<std::vector<double>>()));
        } else if (type == "FlashDrum") {
            const auto spec_name = unit_json.at("spec").get<std::string>();
            FlashDrum::Spec spec = FlashDrum::Spec::TP;
            double spec_value = 0.0;

            if (spec_name == "TP") {
                spec = FlashDrum::Spec::TP;
                spec_value = unit_json.at("T").get<double>();
            } else if (spec_name == "PH") {
                spec = FlashDrum::Spec::PH;
                spec_value = unit_json.at("H").get<double>();
            } else if (spec_name == "PS") {
                spec = FlashDrum::Spec::PS;
                spec_value = unit_json.at("S").get<double>();
            } else {
                throw std::invalid_argument(
                    "FlowsheetParser: unknown flash drum spec '" + spec_name + "'");
            }

            flowsheet->addUnit(
                name,
                std::make_unique<FlashDrumOp>(
                    spec,
                    flowsheet->flashCalculator(),
                    spec_value,
                    unit_json.at("P").get<double>()));
        } else if (type == "Pump") {
            flowsheet->addUnit(
                name,
                std::make_unique<PumpOp>(
                    flowsheet->flashCalculator(),
                    unit_json.at("P_out").get<double>(),
                    unit_json.value("eta", 0.75)));
        } else if (type == "DistillationColumn") {
            flowsheet->addUnit(
                name,
                std::make_unique<DistillationColumnOp>(
                    flowsheet->flashCalculator(),
                    flowsheet->components(),
                    unit_json.at("N_stages").get<int>(),
                    unit_json.at("feed_stage").get<int>(),
                    unit_json.at("reflux_ratio").get<double>(),
                    unit_json.at("distillate_frac").get<double>(),
                    unit_json.value("P_top", 101325.0),
                    unit_json.value("feed_quality", 1.0),
                    unit_json.value("max_iter", 15)));
        } else {
            throw std::invalid_argument("FlowsheetParser: unknown unit type '" + type + "'");
        }
    }

    for (const auto& connection : j.at("connections")) {
        flowsheet->connect(
            connection.at("stream").get<std::string>(),
            asStringOrEmpty(connection, "from"),
            asStringOrEmpty(connection, "from_port"),
            asStringOrEmpty(connection, "to"),
            asStringOrEmpty(connection, "to_port"));
    }

    return flowsheet;
}

Flowsheet FlowsheetParser::parseFile(const std::string& json_path,
                                     const std::string& component_db_path) {
    std::ifstream file(json_path);
    if (!file.is_open())
        throw std::runtime_error("FlowsheetParser: cannot open " + json_path);

    nlohmann::json j;
    file >> j;

    const auto component_ids = j.at("components").get<std::vector<std::string>>();
    ComponentDB db(component_db_path);
    Flowsheet flowsheet(db.get(component_ids));

    if (j.contains("streams")) {
        for (auto& [name, stream_json] : j.at("streams").items()) {
            flowsheet.addStream(
                name,
                stream_json.at("T").get<double>(),
                stream_json.at("P").get<double>(),
                stream_json.at("flow").get<double>(),
                stream_json.at("composition").get<std::map<std::string, double>>());
        }
    }

    for (auto& [name, unit_json] : j.at("units").items()) {
        const auto type = unit_json.at("type").get<std::string>();

        if (type == "Mixer") {
            flowsheet.addUnit(
                name,
                std::make_unique<MixerOp>(
                    flowsheet.flashCalculator(),
                    unit_json.at("inlet_ports").get<std::vector<std::string>>()));
        } else if (type == "Splitter") {
            flowsheet.addUnit(
                name,
                std::make_unique<SplitterOp>(
                    unit_json.at("fractions").get<std::vector<double>>()));
        } else if (type == "FlashDrum") {
            const auto spec_name = unit_json.at("spec").get<std::string>();
            FlashDrum::Spec spec = FlashDrum::Spec::TP;
            double spec_value = 0.0;

            if (spec_name == "TP") {
                spec = FlashDrum::Spec::TP;
                spec_value = unit_json.at("T").get<double>();
            } else if (spec_name == "PH") {
                spec = FlashDrum::Spec::PH;
                spec_value = unit_json.at("H").get<double>();
            } else if (spec_name == "PS") {
                spec = FlashDrum::Spec::PS;
                spec_value = unit_json.at("S").get<double>();
            } else {
                throw std::invalid_argument(
                    "FlowsheetParser: unknown flash drum spec '" + spec_name + "'");
            }

            flowsheet.addUnit(
                name,
                std::make_unique<FlashDrumOp>(
                    spec,
                    flowsheet.flashCalculator(),
                    spec_value,
                    unit_json.at("P").get<double>()));
        } else if (type == "Pump") {
            flowsheet.addUnit(
                name,
                std::make_unique<PumpOp>(
                    flowsheet.flashCalculator(),
                    unit_json.at("P_out").get<double>(),
                    unit_json.value("eta", 0.75)));
        } else if (type == "DistillationColumn") {
            flowsheet.addUnit(
                name,
                std::make_unique<DistillationColumnOp>(
                    flowsheet.flashCalculator(),
                    flowsheet.components(),
                    unit_json.at("N_stages").get<int>(),
                    unit_json.at("feed_stage").get<int>(),
                    unit_json.at("reflux_ratio").get<double>(),
                    unit_json.at("distillate_frac").get<double>(),
                    unit_json.value("P_top", 101325.0),
                    unit_json.value("feed_quality", 1.0),
                    unit_json.value("max_iter", 15)));
        } else {
            throw std::invalid_argument("FlowsheetParser: unknown unit type '" + type + "'");
        }
    }

    for (const auto& connection : j.at("connections")) {
        flowsheet.connect(
            connection.at("stream").get<std::string>(),
            asStringOrEmpty(connection, "from"),
            asStringOrEmpty(connection, "from_port"),
            asStringOrEmpty(connection, "to"),
            asStringOrEmpty(connection, "to_port"));
    }

    return flowsheet;
}

} // namespace chemsim
