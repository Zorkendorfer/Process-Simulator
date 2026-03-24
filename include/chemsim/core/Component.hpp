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

    // DIPPR ideal-gas Cp (Aly-Lee form, equation 107, J/kmol/K):
    //   Cp = Cp1 + Cp2*(Cp3/T/sinh(Cp3/T))^2 + Cp4*(Cp5/T/cosh(Cp5/T))^2
    // Analytical integral for H:
    //   integral(Cp,T1,T2) = Cp1*(T2-T1)
    //                       + Cp2*Cp3*(coth(Cp3/T2) - coth(Cp3/T1))
    //                       - Cp4*Cp5*(tanh(Cp5/T2) - tanh(Cp5/T1))
    double Cp1{}, Cp2{}, Cp3{}, Cp4{}, Cp5{};

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
        c.Cp1     = j.at("Cp1").get<double>();
        c.Cp2     = j.at("Cp2").get<double>();
        c.Cp3     = j.at("Cp3").get<double>();
        c.Cp4     = j.at("Cp4").get<double>();
        c.Cp5     = j.at("Cp5").get<double>();
        c.Antoine_A = j.at("Antoine_A").get<double>();
        c.Antoine_B = j.at("Antoine_B").get<double>();
        c.Antoine_C = j.at("Antoine_C").get<double>();
        return c;
    }
};

} // namespace chemsim
