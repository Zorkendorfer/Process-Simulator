#pragma once
#include <string>
#include <nlohmann/json.hpp>

namespace chemsim {

struct Component {
    // Identity
    std::string id;     // "METHANE"
    std::string name;   // "Methane"
    double MW{};        // g/mol

    // Critical properties
    double Tc{};        // K
    double Pc{};        // Pa
    double omega{};     // acentric factor

    // Ideal gas Cp: A + B*T + C*T^2 + D*T^3  [J/mol/K]
    double Cp_A{}, Cp_B{}, Cp_C{}, Cp_D{};

    // Antoine: log10(P_sat/mmHg) = A - B/(T/°C + C)
    double Antoine_A{}, Antoine_B{}, Antoine_C{};

    static Component fromJSON(const std::string& id, const nlohmann::json& j) {
        Component c;
        c.id      = id;
        c.name    = j.at("name").get<std::string>();
        c.MW      = j.at("MW").get<double>();
        c.Tc      = j.at("Tc").get<double>();
        c.Pc      = j.at("Pc").get<double>();
        c.omega   = j.at("omega").get<double>();
        c.Cp_A    = j.at("Cp_A").get<double>();
        c.Cp_B    = j.at("Cp_B").get<double>();
        c.Cp_C    = j.at("Cp_C").get<double>();
        c.Cp_D    = j.at("Cp_D").get<double>();
        c.Antoine_A = j.at("Antoine_A").get<double>();
        c.Antoine_B = j.at("Antoine_B").get<double>();
        c.Antoine_C = j.at("Antoine_C").get<double>();
        return c;
    }
};

} // namespace chemsim
