#pragma once
#include "chemsim/flowsheet/Flowsheet.hpp"
#include <memory>
#include <string>

namespace chemsim {

class FlowsheetParser {
public:
    static Flowsheet parseFile(const std::string& json_path,
                               const std::string& component_db_path);
    static std::unique_ptr<Flowsheet> parseFileUnique(const std::string& json_path,
                                                      const std::string& component_db_path);
};

} // namespace chemsim
