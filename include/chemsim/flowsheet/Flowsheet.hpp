#pragma once
#include "chemsim/core/Component.hpp"
#include "chemsim/core/Stream.hpp"
#include "chemsim/flowsheet/FlowsheetGraph.hpp"
#include "chemsim/flowsheet/RecycleSolver.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"
#include "chemsim/thermo/PengRobinson.hpp"
#include <iosfwd>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace chemsim {

class Flowsheet {
public:
    Flowsheet(std::vector<Component> components);

    Stream& addStream(const std::string& name,
                      double T, double P, double flow,
                      const std::map<std::string, double>& composition);

    void addUnit(const std::string& name, std::unique_ptr<IUnitOp> op);

    template <typename T, typename... Args>
    T& addUnit(const std::string& name, Args&&... args) {
        auto op = std::make_unique<T>(std::forward<Args>(args)...);
        T& ref = *op;
        addUnit(name, std::move(op));
        return ref;
    }

    void connect(const std::string& stream_name,
                 const std::string& from_unit, const std::string& from_port,
                 const std::string& to_unit,   const std::string& to_port);

    bool solve();
    bool solve(RecycleSolver::Options opts);

    const Stream& getStream(const std::string& name) const;
    Stream& getStream(const std::string& name);

    IUnitOp& getUnit(const std::string& name);
    const IUnitOp& getUnit(const std::string& name) const;

    const std::vector<Component>& components() const { return components_; }
    const FlashCalculator& flashCalculator() const { return *flash_; }
    std::vector<std::string> streamNames() const;
    std::string summary() const;
    void printSummary(std::ostream& os) const;
    nlohmann::json resultsAsJson() const;
    void exportResults(const std::string& json_path) const;

    // ── RL interface ──────────────────────────────────────────────────────────
    // Set a named parameter on a unit op (e.g. "refluxRatio", "distillateFrac")
    void setParam(const std::string& unit_name,
                  const std::string& param_name, double value);

    // Update an existing feed stream's T and P (composition/flow unchanged)
    void setStreamConditions(const std::string& stream_name, double T, double P);

    // Reset all streams to their state at the time they were first added
    void resetToBase();

    static Flowsheet fromJSON(const std::string& json_path,
                              const std::string& component_db_path);
    static std::unique_ptr<Flowsheet> fromJSONUnique(const std::string& json_path,
                                                     const std::string& component_db_path);

private:
    std::vector<Component> components_;
    std::unique_ptr<PengRobinson> eos_;
    std::unique_ptr<FlashCalculator> flash_;
    FlowsheetGraph graph_;
    std::map<std::string, Stream> streams_;
    std::map<std::string, Stream> base_streams_;   // snapshot for resetToBase()
    std::vector<std::string> unit_names_;

    std::vector<double> compositionVector(
        const std::map<std::string, double>& composition) const;
    void initializeThermo(Stream& stream) const;
    void ensureStreamExists(const std::string& name);
};

} // namespace chemsim
