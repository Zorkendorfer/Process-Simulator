#pragma once
#include "chemsim/core/Component.hpp"
#include "chemsim/core/Stream.hpp"
#include "chemsim/flowsheet/FlowsheetGraph.hpp"
#include "chemsim/flowsheet/RecycleSolver.hpp"
#include "chemsim/thermo/FlashCalculator.hpp"
#include "chemsim/thermo/PengRobinson.hpp"
#include <map>
#include <memory>
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

    static Flowsheet fromJSON(const std::string& json_path,
                              const std::string& component_db_path);

private:
    std::vector<Component> components_;
    std::unique_ptr<PengRobinson> eos_;
    std::unique_ptr<FlashCalculator> flash_;
    FlowsheetGraph graph_;
    std::map<std::string, Stream> streams_;
    std::vector<std::string> unit_names_;

    std::vector<double> compositionVector(
        const std::map<std::string, double>& composition) const;
    void initializeThermo(Stream& stream) const;
    void ensureStreamExists(const std::string& name);
};

} // namespace chemsim
