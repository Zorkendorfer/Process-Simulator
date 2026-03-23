#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>
#include "chemsim/core/Component.hpp"

namespace chemsim {

class ComponentDB {
public:
    explicit ComponentDB(const std::string& jsonPath) {
        std::ifstream f(jsonPath);
        if (!f.is_open())
            throw std::runtime_error("ComponentDB: cannot open " + jsonPath);
        nlohmann::json j;
        f >> j;
        for (auto& [id, data] : j.items())
            db_[id] = Component::fromJSON(id, data);
    }

    // Return ordered vector of components for the given IDs
    std::vector<Component> get(const std::vector<std::string>& ids) const {
        std::vector<Component> result;
        result.reserve(ids.size());
        for (const auto& id : ids) {
            auto it = db_.find(id);
            if (it == db_.end())
                throw std::runtime_error("ComponentDB: unknown component '" + id + "'");
            result.push_back(it->second);
        }
        return result;
    }

    bool has(const std::string& id) const { return db_.count(id) > 0; }

private:
    std::unordered_map<std::string, Component> db_;
};

} // namespace chemsim
