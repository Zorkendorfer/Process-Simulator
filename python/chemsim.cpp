#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <iomanip>

#include "chemsim/core/Component.hpp"
#include "chemsim/core/Stream.hpp"
#include "chemsim/flowsheet/Flowsheet.hpp"
#include "chemsim/flowsheet/FlashDrumOp.hpp"
#include "chemsim/flowsheet/MixerOp.hpp"
#include "chemsim/flowsheet/SplitterOp.hpp"
#include "chemsim/flowsheet/PumpOp.hpp"
#include "chemsim/io/ComponentDB.hpp"

namespace py = pybind11;
using namespace chemsim;

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string phaseStr(Phase p) {
    switch (p) {
    case Phase::VAPOR:   return "VAPOR";
    case Phase::LIQUID:  return "LIQUID";
    case Phase::MIXED:   return "MIXED";
    default:             return "UNKNOWN";
    }
}

static py::dict streamToDict(const Stream& s) {
    py::dict d;
    d["name"]           = s.name;
    d["T"]              = s.T;
    d["P"]              = s.P;
    d["total_flow"]     = s.totalFlow;
    d["z"]              = s.z;
    d["x"]              = s.x;
    d["y"]              = s.y;
    d["vapor_fraction"] = s.vaporFraction;
    d["phase"]          = phaseStr(s.phase);
    d["H"]              = s.H;
    d["S"]              = s.S;
    return d;
}

// ── Module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(chemsim, m) {
    m.doc() = R"doc(
ChemSim — C++ process simulator Python bindings.

Quick start::

    import chemsim

    # Programmatic construction
    fs = chemsim.Flowsheet.create(["METHANE", "ETHANE"],
                                   db_path="data/components.json")
    fs.add_stream("FEED", T=250.0, P=2e6, flow=100.0,
                  z={"METHANE": 0.6, "ETHANE": 0.4})
    fs.add_flash_drum("FLASH", T=250.0, P=2e6)
    fs.connect("FEED",  to_unit="FLASH", to_port="feed")
    fs.connect("VAPOR", from_unit="FLASH", from_port="vapor")
    fs.connect("LIQ",   from_unit="FLASH", from_port="liquid")
    fs.solve()

    import pandas as pd
    df = pd.DataFrame(fs.results_table())

    # JSON-driven construction
    fs2 = chemsim.Flowsheet.from_json("examples/simple_recycle.json",
                                       "data/components.json")
    fs2.solve()
    print(fs2.summary())
)doc";

    // ── Phase enum ─────────────────────────────────────────────────────────
    py::enum_<Phase>(m, "Phase")
        .value("VAPOR",   Phase::VAPOR)
        .value("LIQUID",  Phase::LIQUID)
        .value("MIXED",   Phase::MIXED)
        .value("UNKNOWN", Phase::UNKNOWN)
        .export_values();

    // ── Stream ──────────────────────────────────────────────────────────────
    py::class_<Stream>(m, "Stream",
        "Material stream with thermodynamic state.")
        .def_property_readonly("name",
            [](const Stream& s){ return s.name; })
        .def_property_readonly("T",
            [](const Stream& s){ return s.T; },
            "Temperature [K]")
        .def_property_readonly("P",
            [](const Stream& s){ return s.P; },
            "Pressure [Pa]")
        .def_property_readonly("total_flow",
            [](const Stream& s){ return s.totalFlow; },
            "Molar flow rate [mol/s]")
        .def_property_readonly("z",
            [](const Stream& s){ return s.z; },
            "Overall mole fractions")
        .def_property_readonly("x",
            [](const Stream& s){ return s.x; },
            "Liquid mole fractions")
        .def_property_readonly("y",
            [](const Stream& s){ return s.y; },
            "Vapor mole fractions")
        .def_property_readonly("phase",
            [](const Stream& s){ return s.phase; })
        .def_property_readonly("vapor_fraction",
            [](const Stream& s){ return s.vaporFraction; },
            "Vapor fraction β ∈ [0, 1]")
        .def_property_readonly("H",
            [](const Stream& s){ return s.H; },
            "Specific enthalpy [J/mol]")
        .def_property_readonly("S",
            [](const Stream& s){ return s.S; },
            "Specific entropy [J/mol·K]")
        .def("to_dict", &streamToDict,
            "Return all stream properties as a Python dict (pandas-friendly).")
        .def("__repr__", [](const Stream& s) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2);
            ss << "<Stream '" << s.name << "'"
               << " F=" << s.totalFlow << " mol/s"
               << " T=" << s.T << " K"
               << " P=" << s.P/1e5 << " bar"
               << " phase=" << phaseStr(s.phase) << ">";
            return ss.str();
        });

    // ── Flowsheet ───────────────────────────────────────────────────────────
    py::class_<Flowsheet>(m, "Flowsheet",
        "Top-level process flowsheet: add streams, unit ops, connect, solve.")

        // ── Construction ───────────────────────────────────────────────────
        .def_static("create",
            [](const std::vector<std::string>& component_ids,
               const std::string& db_path) {
                ComponentDB db(db_path);
                return std::make_unique<Flowsheet>(db.get(component_ids));
            },
            py::arg("component_ids"),
            py::arg("db_path"),
            R"doc(
Create a Flowsheet from a list of component IDs and a component database.

Parameters
----------
component_ids : list[str]
    Component identifiers (e.g. ``["METHANE", "ETHANE"]``).
db_path : str
    Path to the JSON component database (e.g. ``"data/components.json"``).
)doc")

        .def_static("from_json",
            [](const std::string& json_path,
               const std::string& component_db_path) {
                return Flowsheet::fromJSONUnique(json_path, component_db_path);
            },
            py::arg("json_path"),
            py::arg("component_db_path"),
            "Load and construct a Flowsheet from a JSON specification file.")

        // ── Build ──────────────────────────────────────────────────────────
        .def("add_stream",
            [](Flowsheet& fs,
               const std::string& name,
               double T, double P, double flow,
               const std::map<std::string, double>& z) -> Stream& {
                return fs.addStream(name, T, P, flow, z);
            },
            py::arg("name"),
            py::arg("T"),
            py::arg("P"),
            py::arg("flow"),
            py::arg("z"),
            py::return_value_policy::reference_internal,
            R"doc(
Add a feed stream.

Parameters
----------
name : str   Stream identifier.
T    : float  Temperature [K].
P    : float  Pressure [Pa].
flow : float  Molar flow rate [mol/s].
z    : dict   Composition as ``{component_id: mole_fraction}``.
)doc")

        // Flash drum variants
        .def("add_flash_drum",
            [](Flowsheet& fs, const std::string& name, double T, double P) {
                fs.addUnit(name,
                    std::make_unique<FlashDrumOp>(
                        FlashDrum::Spec::TP, fs.flashCalculator(), T, P));
            },
            py::arg("name"), py::arg("T"), py::arg("P"),
            "Add a TP flash drum (temperature + pressure spec).")

        .def("add_flash_drum_ph",
            [](Flowsheet& fs, const std::string& name, double H, double P) {
                fs.addUnit(name,
                    std::make_unique<FlashDrumOp>(
                        FlashDrum::Spec::PH, fs.flashCalculator(), H, P));
            },
            py::arg("name"), py::arg("H"), py::arg("P"),
            "Add a PH flash drum (enthalpy + pressure spec). H in J/mol.")

        .def("add_flash_drum_ps",
            [](Flowsheet& fs, const std::string& name, double S, double P) {
                fs.addUnit(name,
                    std::make_unique<FlashDrumOp>(
                        FlashDrum::Spec::PS, fs.flashCalculator(), S, P));
            },
            py::arg("name"), py::arg("S"), py::arg("P"),
            "Add a PS flash drum (entropy + pressure spec). S in J/mol·K.")

        .def("add_mixer",
            [](Flowsheet& fs, const std::string& name,
               const std::vector<std::string>& inlet_ports) {
                fs.addUnit(name,
                    std::make_unique<MixerOp>(fs.flashCalculator(), inlet_ports));
            },
            py::arg("name"),
            py::arg("inlet_ports"),
            R"doc(
Add an adiabatic mixer.

Parameters
----------
name         : str        Unit identifier.
inlet_ports  : list[str]  Names of the inlet ports (e.g. ``["feed", "recycle"]``).
Outlet port is always ``"out"``.
)doc")

        .def("add_splitter",
            [](Flowsheet& fs, const std::string& name,
               const std::vector<double>& fractions) {
                fs.addUnit(name, std::make_unique<SplitterOp>(fractions));
            },
            py::arg("name"),
            py::arg("fractions"),
            R"doc(
Add a stream splitter.

Parameters
----------
name      : str         Unit identifier.
fractions : list[float] Split fractions (must sum to 1).
Outlet ports are ``"out0"``, ``"out1"``, etc.
)doc")

        .def("add_pump",
            [](Flowsheet& fs, const std::string& name,
               double P_out, double eta) {
                fs.addUnit(name,
                    std::make_unique<PumpOp>(fs.flashCalculator(), P_out, eta));
            },
            py::arg("name"),
            py::arg("P_out"),
            py::arg("eta") = 0.75,
            "Add an isentropic pump. P_out in Pa, eta ∈ (0, 1].")

        // ── Connect ────────────────────────────────────────────────────────
        .def("connect",
            [](Flowsheet& fs,
               const std::string& stream_name,
               const std::string& from_unit, const std::string& from_port,
               const std::string& to_unit,   const std::string& to_port) {
                fs.connect(stream_name, from_unit, from_port, to_unit, to_port);
            },
            py::arg("stream_name"),
            py::arg("from_unit") = "",
            py::arg("from_port") = "",
            py::arg("to_unit")   = "",
            py::arg("to_port")   = "",
            R"doc(
Connect a stream between two unit ports.

Use empty strings for feed streams (no producing unit) or terminal product streams::

    # Feed → unit
    fs.connect("FEED", to_unit="FLASH1", to_port="feed")

    # Unit → unit
    fs.connect("MID", from_unit="FLASH1", from_port="vapor",
                      to_unit="COND", to_port="feed")

    # Unit → product (just record for bookkeeping)
    fs.connect("VAPOR_PROD", from_unit="FLASH1", from_port="vapor")
)doc")

        // ── Solve ──────────────────────────────────────────────────────────
        .def("solve",
            py::overload_cast<>(&Flowsheet::solve),
            "Solve the flowsheet (with recycle convergence if needed). "
            "Returns True if converged.")

        .def("solve_with",
            [](Flowsheet& fs, int max_iter,
               double tol_T, double tol_P, double tol_z, double relaxation) {
                RecycleSolver::Options opts;
                opts.maxIter    = max_iter;
                opts.tol_T      = tol_T;
                opts.tol_P      = tol_P;
                opts.tol_z      = tol_z;
                opts.relaxation = relaxation;
                return fs.solve(opts);
            },
            py::arg("max_iter")   = 100,
            py::arg("tol_T")      = 0.01,
            py::arg("tol_P")      = 1.0,
            py::arg("tol_z")      = 1e-6,
            py::arg("relaxation") = 1.0,
            "Solve with custom convergence options.")

        // ── Results ────────────────────────────────────────────────────────
        .def("stream_names", &Flowsheet::streamNames,
            "Return list of all stream names.")

        .def("get_stream",
            py::overload_cast<const std::string&>(&Flowsheet::getStream, py::const_),
            py::return_value_policy::reference_internal,
            "Return a Stream object by name.")

        .def("component_ids",
            [](const Flowsheet& fs) {
                std::vector<std::string> ids;
                for (const auto& c : fs.components()) ids.push_back(c.id);
                return ids;
            },
            "Return the ordered list of component IDs in this flowsheet.")

        .def("results_table",
            [](const Flowsheet& fs) {
                py::list rows;
                for (const auto& name : fs.streamNames()) {
                    const Stream& s = fs.getStream(name);
                    py::dict row = streamToDict(s);
                    rows.append(row);
                }
                return rows;
            },
            R"doc(
Return all stream results as a list of dicts — ready for ``pd.DataFrame``::

    import pandas as pd
    df = pd.DataFrame(fs.results_table())
)doc")

        .def("summary", &Flowsheet::summary,
            "Return a human-readable flowsheet summary string.")

        .def("export_results", &Flowsheet::exportResults,
            py::arg("json_path"),
            "Write all stream results to a JSON file.")

        .def("results_as_json",
            [](const Flowsheet& fs) { return fs.resultsAsJson().dump(2); },
            "Return results as a pretty-printed JSON string.")

        .def("__repr__", [](const Flowsheet& fs) {
            auto names = fs.streamNames();
            std::ostringstream ss;
            ss << "<Flowsheet streams=[";
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (i) ss << ", ";
                ss << '\'' << names[i] << '\'';
            }
            ss << "]>";
            return ss.str();
        });
}
