#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "chemsim/core/Stream.hpp"
#include "chemsim/flowsheet/Flowsheet.hpp"

namespace py = pybind11;
using namespace chemsim;

PYBIND11_MODULE(chemsim, m) {
    m.doc() = "ChemSim process simulator bindings";

    py::enum_<Phase>(m, "Phase")
        .value("VAPOR", Phase::VAPOR)
        .value("LIQUID", Phase::LIQUID)
        .value("MIXED", Phase::MIXED)
        .value("UNKNOWN", Phase::UNKNOWN);

    py::class_<Stream>(m, "Stream")
        .def_property_readonly("name", [](const Stream& s) { return s.name; })
        .def_property_readonly("T", [](const Stream& s) { return s.T; })
        .def_property_readonly("P", [](const Stream& s) { return s.P; })
        .def_property_readonly("total_flow", [](const Stream& s) { return s.totalFlow; })
        .def_property_readonly("z", [](const Stream& s) { return s.z; })
        .def_property_readonly("x", [](const Stream& s) { return s.x; })
        .def_property_readonly("y", [](const Stream& s) { return s.y; })
        .def_property_readonly("phase", [](const Stream& s) { return s.phase; })
        .def_property_readonly("vapor_fraction", [](const Stream& s) { return s.vaporFraction; })
        .def_property_readonly("H", [](const Stream& s) { return s.H; })
        .def_property_readonly("S", [](const Stream& s) { return s.S; });

    py::class_<Flowsheet>(m, "Flowsheet")
        .def("solve", py::overload_cast<>(&Flowsheet::solve))
        .def("stream_names", &Flowsheet::streamNames)
        .def("get_stream",
             py::overload_cast<const std::string&>(&Flowsheet::getStream, py::const_),
             py::return_value_policy::reference_internal)
        .def("summary", &Flowsheet::summary)
        .def("export_results", &Flowsheet::exportResults)
        .def("results_as_json",
             [](const Flowsheet& fs) { return fs.resultsAsJson().dump(2); })
        .def_static("from_json",
                    [](const std::string& json_path, const std::string& component_db_path) {
                        return Flowsheet::fromJSONUnique(json_path, component_db_path);
                    },
                    py::arg("json_path"), py::arg("component_db_path"));
}
