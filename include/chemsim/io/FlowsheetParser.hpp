#pragma once
#include "chemsim/flowsheet/Flowsheet.hpp"
#include <string>

namespace chemsim {

class FlowsheetParser {
public:
    static Flowsheet parseFile(const std::string& json_path,
                               const std::string& component_db_path);
};

} // namespace chemsim
