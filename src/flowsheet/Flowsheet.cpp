#include "chemsim/flowsheet/Flowsheet.hpp"
#include "chemsim/io/FlowsheetParser.hpp"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace chemsim {

namespace {

std::string phaseToString(Phase phase) {
    switch (phase) {
    case Phase::VAPOR: return "VAPOR";
    case Phase::LIQUID: return "LIQUID";
    case Phase::MIXED: return "MIXED";
    case Phase::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

nlohmann::json streamToJson(const Stream& stream) {
    return {
        {"name", stream.name},
        {"T", stream.T},
        {"P", stream.P},
        {"totalFlow", stream.totalFlow},
        {"z", stream.z},
        {"phase", phaseToString(stream.phase)},
        {"vaporFraction", stream.vaporFraction},
        {"x", stream.x},
        {"y", stream.y},
        {"H", stream.H},
        {"S", stream.S}
    };
}

} // namespace

Flowsheet::Flowsheet(std::vector<Component> components)
    : components_(std::move(components)) {
    if (components_.empty())
        throw std::invalid_argument("Flowsheet: components must not be empty");
    eos_ = std::make_unique<PengRobinson>(components_);
    flash_ = std::make_unique<FlashCalculator>(*eos_, components_);
}

std::vector<double> Flowsheet::compositionVector(
    const std::map<std::string, double>& composition) const {
    std::vector<double> z(components_.size(), 0.0);
    double sum = 0.0;

    for (std::size_t i = 0; i < components_.size(); ++i) {
        const auto it = composition.find(components_[i].id);
        if (it == composition.end())
            throw std::invalid_argument(
                "Flowsheet: missing composition for component '" + components_[i].id + "'");
        z[i] = it->second;
        sum += z[i];
    }

    if (sum <= 0.0)
        throw std::invalid_argument("Flowsheet: composition sum must be > 0");

    for (double& zi : z) zi /= sum;
    return z;
}

void Flowsheet::initializeThermo(Stream& stream) const {
    if (stream.totalFlow <= 0.0 || stream.z.empty()) return;

    auto flash_result = flash_->flashTP(stream.T, stream.P, stream.z);
    stream.vaporFraction = flash_result.beta;
    stream.x = flash_result.x;
    stream.y = flash_result.y;
    stream.phase = (flash_result.beta < 1e-10) ? Phase::LIQUID
                 : (flash_result.beta > 1.0 - 1e-10) ? Phase::VAPOR
                 : Phase::MIXED;
    stream.H = flash_->totalEnthalpy(flash_result);
    stream.S = flash_->totalEntropy(flash_result);
}

Stream& Flowsheet::addStream(const std::string& name,
                             double T, double P, double flow,
                             const std::map<std::string, double>& composition) {
    Stream stream;
    stream.name = name;
    stream.T = T;
    stream.P = P;
    stream.totalFlow = flow;
    stream.z = compositionVector(composition);
    initializeThermo(stream);

    streams_[name] = stream;
    graph_.setFeed(name, stream);
    return streams_.at(name);
}

void Flowsheet::addUnit(const std::string& name, std::unique_ptr<IUnitOp> op) {
    if (name.empty())
        throw std::invalid_argument("Flowsheet: unit name must not be empty");
    graph_.addUnit(name, std::move(op));
    unit_names_.push_back(name);
}

void Flowsheet::ensureStreamExists(const std::string& name) {
    if (name.empty()) return;
    if (!streams_.count(name)) {
        Stream s;
        s.name = name;
        streams_[name] = s;
    }
}

void Flowsheet::connect(const std::string& stream_name,
                        const std::string& from_unit, const std::string& from_port,
                        const std::string& to_unit,   const std::string& to_port) {
    ensureStreamExists(stream_name);
    graph_.connect(stream_name, from_unit, from_port, to_unit, to_port);
}

bool Flowsheet::solve() {
    return solve(RecycleSolver::Options{});
}

bool Flowsheet::solve(RecycleSolver::Options opts) {
    RecycleSolver solver(graph_, opts);
    auto result = solver.solve(streams_);
    streams_ = std::move(result.streams);
    return result.converged;
}

const Stream& Flowsheet::getStream(const std::string& name) const {
    return streams_.at(name);
}

Stream& Flowsheet::getStream(const std::string& name) {
    return streams_.at(name);
}

IUnitOp& Flowsheet::getUnit(const std::string& name) {
    return graph_.unit(name);
}

const IUnitOp& Flowsheet::getUnit(const std::string& name) const {
    return graph_.unit(name);
}

std::vector<std::string> Flowsheet::streamNames() const {
    std::vector<std::string> names;
    names.reserve(streams_.size());
    for (const auto& [name, _] : streams_)
        names.push_back(name);
    return names;
}

std::string Flowsheet::summary() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "ChemSim Flowsheet Summary\n";
    out << "Components:";
    for (const auto& component : components_)
        out << ' ' << component.id;
    out << "\nStreams (" << streams_.size() << ")\n";
    for (const auto& [name, stream] : streams_) {
        out << "  " << name
            << ": F=" << stream.totalFlow << " mol/s"
            << ", T=" << stream.T << " K"
            << ", P=" << stream.P << " Pa"
            << ", phase=" << phaseToString(stream.phase)
            << ", beta=" << stream.vaporFraction
            << '\n';
    }
    return out.str();
}

void Flowsheet::printSummary(std::ostream& os) const {
    os << summary();
}

nlohmann::json Flowsheet::resultsAsJson() const {
    nlohmann::json result;
    result["components"] = nlohmann::json::array();
    for (const auto& component : components_) {
        result["components"].push_back({
            {"id", component.id},
            {"name", component.name},
            {"MW", component.MW},
            {"Tc", component.Tc},
            {"Pc", component.Pc},
            {"omega", component.omega}
        });
    }

    result["streams"] = nlohmann::json::object();
    for (const auto& [name, stream] : streams_)
        result["streams"][name] = streamToJson(stream);
    return result;
}

void Flowsheet::exportResults(const std::string& json_path) const {
    std::ofstream out(json_path);
    if (!out.is_open())
        throw std::runtime_error("Flowsheet: cannot open results path " + json_path);
    out << resultsAsJson().dump(2);
}

Flowsheet Flowsheet::fromJSON(const std::string& json_path,
                              const std::string& component_db_path) {
    return FlowsheetParser::parseFile(json_path, component_db_path);
}

std::unique_ptr<Flowsheet> Flowsheet::fromJSONUnique(const std::string& json_path,
                                                     const std::string& component_db_path) {
    return FlowsheetParser::parseFileUnique(json_path, component_db_path);
}

} // namespace chemsim
