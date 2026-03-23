# ChemSim — C++ Process Simulator: Architecture & Implementation Plan

## Project Overview

A steady-state chemical process simulator with:
- Peng-Robinson equation of state
- Flash calculations (TP, PH, PS flash)
- Core unit operations (flash drum, heat exchanger, pump, compressor, reactor, distillation)
- Sequential modular flowsheet solver with recycle convergence
- Python bindings via pybind11
- Clean CLI + JSON input format

**Target**: Solve a flash-distillation-recycle flowsheet with ~20 components.

---

## Technology Stack

| Component | Choice | Reason |
|---|---|---|
| Core language | C++17 | Performance, industry standard for solvers |
| Linear algebra | Eigen 3.4 | Sparse solvers, no manual matrix code |
| Python bindings | pybind11 | Standard in scientific C++ |
| Build system | CMake 3.20+ | Cross-platform, industry standard |
| Testing | GoogleTest | Unit + integration tests |
| JSON I/O | nlohmann/json | Header-only, clean API |
| Logging | spdlog | Fast, header-only |

---

## Repository Structure

```
chemsim/
├── CMakeLists.txt
├── README.md
├── data/
│   └── components.json          ← Pure component database
├── include/
│   └── chemsim/
│       ├── core/
│       │   ├── Component.hpp
│       │   ├── Stream.hpp
│       │   └── Mixture.hpp
│       ├── thermo/
│       │   ├── EOS.hpp
│       │   ├── PengRobinson.hpp
│       │   ├── FlashCalculator.hpp
│       │   └── Enthalpy.hpp
│       ├── unitops/
│       │   ├── UnitOp.hpp
│       │   ├── FlashDrum.hpp
│       │   ├── HeatExchanger.hpp
│       │   ├── Pump.hpp
│       │   ├── Compressor.hpp
│       │   ├── Reactor.hpp
│       │   └── DistillationColumn.hpp
│       ├── flowsheet/
│       │   ├── Flowsheet.hpp
│       │   ├── FlowsheetGraph.hpp
│       │   └── RecycleSolver.hpp
│       ├── numerics/
│       │   ├── NewtonSolver.hpp
│       │   ├── BrentSolver.hpp
│       │   └── BroydenSolver.hpp
│       └── io/
│           ├── ComponentDB.hpp
│           └── FlowsheetParser.hpp
├── src/
│   ├── core/
│   ├── thermo/
│   ├── unitops/
│   ├── flowsheet/
│   ├── numerics/
│   └── io/
├── python/
│   ├── chemsim.cpp              ← pybind11 module definition
│   └── chemsim/
│       └── __init__.py
├── tests/
│   ├── test_thermo.cpp
│   ├── test_flash.cpp
│   ├── test_unitops.cpp
│   └── test_flowsheet.cpp
├── examples/
│   ├── simple_flash.json
│   ├── heat_exchanger_network.json
│   └── distillation_recycle.json
└── third_party/
    ├── eigen/
    ├── pybind11/
    ├── nlohmann/
    └── googletest/
```

---

## Class Design

### 1. Core Layer

```cpp
// include/chemsim/core/Component.hpp
class Component {
public:
    // Identity
    std::string id;          // "METHANE", "ETHANE" etc.
    std::string name;
    double MW;               // g/mol

    // Critical properties (for EOS)
    double Tc;               // K
    double Pc;               // Pa
    double omega;            // acentric factor

    // DIPPR correlations for ideal gas Cp
    // Cp_ig = A + B*T + C*T^2 + D*T^3  [J/mol/K]
    double Cp_A, Cp_B, Cp_C, Cp_D;

    // Antoine coefficients for vapor pressure
    double Antoine_A, Antoine_B, Antoine_C;

    // Constructor from JSON
    static Component fromJSON(const nlohmann::json& j);
};
```

```cpp
// include/chemsim/core/Stream.hpp
enum class Phase { VAPOR, LIQUID, MIXED, UNKNOWN };

class Stream {
public:
    std::string name;
    
    // State variables
    double T;                         // K
    double P;                         // Pa
    double totalFlow;                 // mol/s
    std::vector<double> z;            // overall mole fractions
    
    // Phase split results (set after flash)
    Phase phase;
    double vaporFraction;             // 0..1
    std::vector<double> x;           // liquid mole fractions
    std::vector<double> y;           // vapor mole fractions
    
    // Thermodynamic properties (set after flash)
    double H;                         // J/mol  specific enthalpy
    double S;                         // J/mol/K specific entropy
    
    // Convenience
    int nComp() const { return z.size(); }
    double molarFlow(int i) const { return totalFlow * z[i]; }
    bool isFullyVapor() const { return vaporFraction >= 1.0 - 1e-10; }
    bool isFullyLiquid() const { return vaporFraction <= 1e-10; }
    
    // Deep copy (important for recycle solver)
    Stream clone() const;
    
    // Convergence check between two stream states
    static bool converged(const Stream& a, const Stream& b,
                          double tol_T=0.01, double tol_P=1.0,
                          double tol_z=1e-6);
};
```

```cpp
// include/chemsim/core/Mixture.hpp
// Helper for mixture property calculations
// Mostly free functions operating on vectors of components + compositions

namespace Mixture {
    double meanMW(const std::vector<Component>& comps,
                  const std::vector<double>& z);
    
    double idealGasCp(const std::vector<Component>& comps,
                      const std::vector<double>& z, double T);
    
    // Ideal gas enthalpy relative to 298.15 K reference
    double idealGasH(const std::vector<Component>& comps,
                     const std::vector<double>& z,
                     double T, double T_ref = 298.15);
}
```

---

### 2. Thermodynamics Layer

```cpp
// include/chemsim/thermo/EOS.hpp
// Abstract base — allows swapping PR for SRK later

class EOS {
public:
    virtual ~EOS() = default;
    
    // Compressibility factor: solve cubic Z^3 + bZ^2 + cZ + d = 0
    // Returns {Z_liquid, Z_vapor} — caller picks based on phase
    virtual std::pair<double,double> 
    compressibilityFactors(double T, double P,
                           const std::vector<double>& z) const = 0;
    
    // Fugacity coefficients ln(phi_i) for each component
    virtual std::vector<double>
    lnFugacityCoefficients(double T, double P,
                           const std::vector<double>& z,
                           bool liquid) const = 0;
    
    // Departure enthalpy H - H_ig [J/mol]
    virtual double
    enthalpyDeparture(double T, double P,
                      const std::vector<double>& z,
                      bool liquid) const = 0;
    
    // Departure entropy S - S_ig [J/mol/K]
    virtual double
    entropyDeparture(double T, double P,
                     const std::vector<double>& z,
                     bool liquid) const = 0;
};
```

```cpp
// include/chemsim/thermo/PengRobinson.hpp

class PengRobinson : public EOS {
public:
    explicit PengRobinson(const std::vector<Component>& components,
                          const Eigen::MatrixXd& kij = {});
    // kij: binary interaction parameters (NC x NC), default 0
    
    std::pair<double,double>
    compressibilityFactors(double T, double P,
                           const std::vector<double>& z) const override;
    
    std::vector<double>
    lnFugacityCoefficients(double T, double P,
                           const std::vector<double>& z,
                           bool liquid) const override;
    
    double enthalpyDeparture(double T, double P,
                             const std::vector<double>& z,
                             bool liquid) const override;
    
    double entropyDeparture(double T, double P,
                            const std::vector<double>& z,
                            bool liquid) const override;

private:
    const std::vector<Component>& comps_;
    Eigen::MatrixXd kij_;
    
    // Per-component PR parameters
    struct PRParams {
        double a;       // a(Tc, Pc)
        double b;       // b(Tc, Pc)
        double kappa;   // kappa(omega)
    };
    std::vector<PRParams> params_;
    
    // Alpha function: alpha_i(T) = [1 + kappa_i*(1 - sqrt(T/Tc_i))]^2
    double alpha(int i, double T) const;
    
    // Mixture a and b via van der Waals mixing rules
    double mixA(const std::vector<double>& z, double T) const;
    double mixB(const std::vector<double>& z) const;
    
    // Solve cubic: returns real roots sorted ascending
    std::vector<double> solveCubic(double b, double c, double d) const;
};
```

```cpp
// include/chemsim/thermo/FlashCalculator.hpp

struct FlashResult {
    double T, P;
    double beta;                    // vapor fraction V/F
    std::vector<double> x;         // liquid mole fractions
    std::vector<double> y;         // vapor mole fractions
    std::vector<double> K;         // equilibrium ratios y_i/x_i
    double H_total;                // J/mol
    double S_total;                // J/mol/K
    bool converged;
    int iterations;
};

class FlashCalculator {
public:
    explicit FlashCalculator(const EOS& eos,
                             const std::vector<Component>& comps);
    
    // TP flash: given T, P, z — find phase split
    FlashResult flashTP(double T, double P,
                        const std::vector<double>& z) const;
    
    // PH flash: given P, H_spec, z — find T and phase split
    // Outer loop on T, inner TP flash
    FlashResult flashPH(double P, double H_spec,
                        const std::vector<double>& z,
                        double T_guess = 300.0) const;
    
    // PS flash: given P, S_spec, z
    FlashResult flashPS(double P, double S_spec,
                        const std::vector<double>& z,
                        double T_guess = 300.0) const;
    
    // Bubble point T at given P
    double bubbleT(double P, const std::vector<double>& x) const;
    
    // Dew point T at given P
    double dewT(double P, const std::vector<double>& y) const;

private:
    const EOS& eos_;
    const std::vector<Component>& comps_;
    
    // Wilson K-value initial estimate
    std::vector<double> wilsonK(double T, double P) const;
    
    // Rachford-Rice equation: sum_i[ z_i*(K_i-1)/(1+beta*(K_i-1)) ] = 0
    // Returns beta given K values — Brent's method on bounded interval
    double rachfordRice(const std::vector<double>& z,
                        const std::vector<double>& K) const;
    
    // Inner successive substitution loop
    FlashResult successiveSubstitution(double T, double P,
                                       const std::vector<double>& z,
                                       std::vector<double> K_init,
                                       int maxIter = 200,
                                       double tol = 1e-10) const;
    
    // Stability test (Michelsen): true if single phase is stable
    bool isStable(double T, double P,
                  const std::vector<double>& z) const;
};
```

---

### 3. Numerics Layer

```cpp
// include/chemsim/numerics/BrentSolver.hpp

class BrentSolver {
public:
    struct Options {
        double tol = 1e-10;
        int maxIter = 200;
    };
    
    struct Result {
        double root;
        double residual;
        int iterations;
        bool converged;
    };
    
    // Solve f(x) = 0 on bracketed interval [a, b]
    // Throws if f(a)*f(b) > 0 (no sign change)
    template<typename F>
    static Result solve(F&& f, double a, double b,
                        Options opts = {});
};
```

```cpp
// include/chemsim/numerics/NewtonSolver.hpp

class NewtonSolver {
public:
    struct Options {
        double tol = 1e-8;          // ||F||_inf convergence
        int maxIter = 100;
        double dampingFactor = 1.0; // 1.0 = full Newton step
        bool lineSearch = true;
    };
    
    struct Result {
        Eigen::VectorXd x;
        double residualNorm;
        int iterations;
        bool converged;
    };
    
    // Solve F(x) = 0 where F: R^n -> R^n
    // Jacobian computed by finite differences if not provided
    template<typename F>
    static Result solve(F&& F_func,
                        const Eigen::VectorXd& x0,
                        Options opts = {});
    
    // With analytical Jacobian
    template<typename F, typename J>
    static Result solve(F&& F_func, J&& J_func,
                        const Eigen::VectorXd& x0,
                        Options opts = {});
};
```

```cpp
// include/chemsim/numerics/BroydenSolver.hpp
// Quasi-Newton for flowsheet recycle convergence
// Operates on stream variable vectors, not raw Eigen types

class BroydenSolver {
public:
    struct Options {
        double tol = 1e-5;
        int maxIter = 50;
    };
    
    // x: tear stream variable vector
    // G(x): one pass through the flowsheet returning updated x
    // Returns converged x
    template<typename G>
    static Eigen::VectorXd solve(G&& G_func,
                                 const Eigen::VectorXd& x0,
                                 Options opts = {});
};
```

---

### 4. Unit Operations Layer

```cpp
// include/chemsim/unitops/UnitOp.hpp

class UnitOp {
public:
    std::string name;
    
    virtual ~UnitOp() = default;
    
    // Set inlet streams (pointers — flowsheet owns streams)
    virtual void setInlets(std::vector<Stream*> inlets) = 0;
    
    // Get outlet streams (computed after solve())
    virtual std::vector<Stream*> getOutlets() = 0;
    
    // Run the unit operation calculation
    // Throws UnitOpException on failure
    virtual void solve() = 0;
    
    // Validate configuration before solve
    virtual void validate() const = 0;
    
    // Summary string for logging
    virtual std::string summary() const = 0;
    
protected:
    const FlashCalculator* flash_;   // injected by flowsheet
    
    // Mix multiple inlet streams into one (material + energy balance)
    static Stream mixStreams(const std::vector<Stream*>& inlets);
};
```

```cpp
// include/chemsim/unitops/FlashDrum.hpp

class FlashDrum : public UnitOp {
public:
    // Specification modes
    enum class Spec { TP, PH, PQ };  // Q = vapor fraction
    
    // TP specification
    FlashDrum(std::string name, double T_spec, double P_spec);
    // PH specification (adiabatic flash)
    FlashDrum(std::string name, double P_spec);
    // PQ specification (partial condenser etc.)
    FlashDrum(std::string name, double P_spec, double Q_spec);
    
    void setInlets(std::vector<Stream*>) override;
    std::vector<Stream*> getOutlets() override;
    void solve() override;
    void validate() const override;
    std::string summary() const override;

private:
    Spec spec_;
    double T_spec_, P_spec_, Q_spec_;
    Stream* inlet_ = nullptr;
    Stream vapor_, liquid_;          // outlet streams
    double duty_ = 0.0;             // heat added [W]
};
```

```cpp
// include/chemsim/unitops/HeatExchanger.hpp

class HeatExchanger : public UnitOp {
public:
    enum class Spec { 
        OUTLET_T,        // specify hot or cold outlet T
        DUTY,            // specify heat duty [W]
        APPROACH_T       // specify minimum approach temperature
    };
    
    // Hot side outlet T specification
    HeatExchanger(std::string name, double P_hot, double P_cold,
                  double T_hot_out);
    
    void setInlets(std::vector<Stream*>) override;  // expects 2 inlets
    std::vector<Stream*> getOutlets() override;     // returns 2 outlets
    void solve() override;
    void validate() const override;
    std::string summary() const override;

private:
    double P_hot_, P_cold_;
    double T_hot_out_;
    Stream *hotIn_ = nullptr, *coldIn_ = nullptr;
    Stream hotOut_, coldOut_;
    double duty_ = 0.0;
};
```

```cpp
// include/chemsim/unitops/Pump.hpp

class Pump : public UnitOp {
public:
    Pump(std::string name, double P_out, double efficiency = 0.75);
    
    void setInlets(std::vector<Stream*>) override;
    std::vector<Stream*> getOutlets() override;
    void solve() override;
    void validate() const override;
    std::string summary() const override;

private:
    double P_out_, eta_;
    Stream* inlet_ = nullptr;
    Stream outlet_;
    double work_ = 0.0;  // shaft work [W], positive = into fluid
};
```

```cpp
// include/chemsim/unitops/Compressor.hpp
// Same interface as Pump but for vapor streams
// Uses isentropic work: H2s from PS flash, then correct for eta

class Compressor : public UnitOp {
public:
    Compressor(std::string name, double P_out, double efficiency = 0.75);
    
    void setInlets(std::vector<Stream*>) override;
    std::vector<Stream*> getOutlets() override;
    void solve() override;
    void validate() const override;
    std::string summary() const override;

private:
    double P_out_, eta_;
    Stream* inlet_ = nullptr;
    Stream outlet_;
    double work_ = 0.0;
};
```

```cpp
// include/chemsim/unitops/Reactor.hpp

class Reactor : public UnitOp {
public:
    // Reaction specification
    struct Reaction {
        std::vector<int> compIndices;    // component indices
        std::vector<double> stoich;      // stoichiometry (neg = reactant)
        double conversion;               // fractional conversion of limiting
        int limitingComp;                // index of limiting reactant
    };
    
    enum class Spec { ISOTHERMAL, ADIABATIC };
    
    Reactor(std::string name, double T_spec, double P_spec,
            std::vector<Reaction> reactions, Spec spec = Spec::ISOTHERMAL);
    
    void setInlets(std::vector<Stream*>) override;
    std::vector<Stream*> getOutlets() override;
    void solve() override;
    void validate() const override;
    std::string summary() const override;

private:
    double T_spec_, P_spec_;
    std::vector<Reaction> reactions_;
    Spec spec_;
    Stream* inlet_ = nullptr;
    Stream outlet_;
    double duty_ = 0.0;
};
```

```cpp
// include/chemsim/unitops/DistillationColumn.hpp
// Inside-out algorithm (Boston & Sullivan 1974)
// Solves MESH equations: Material, Equilibrium, Summation, Heat

class DistillationColumn : public UnitOp {
public:
    struct FeedSpec {
        int stage;           // 1-indexed from top (condenser = stage 1)
        Stream* stream;
    };
    
    struct Spec {
        // Two degrees of freedom — choose two:
        double refluxRatio = -1;         // L/D
        double boilupRatio = -1;         // V/B
        double distillateFlow = -1;      // mol/s
        double bottomsFlow = -1;
        double condenser_T = -1;         // K (partial condenser)
        double reboiler_duty = -1;       // W
    };
    
    DistillationColumn(std::string name,
                       int nStages,           // total including condenser + reboiler
                       std::vector<FeedSpec> feeds,
                       Spec spec,
                       double P,              // uniform column pressure [Pa]
                       bool totalCondenser = true);
    
    void setInlets(std::vector<Stream*>) override;
    std::vector<Stream*> getOutlets() override;  // {distillate, bottoms}
    void solve() override;
    void validate() const override;
    std::string summary() const override;

private:
    int nStages_;
    std::vector<FeedSpec> feeds_;
    Spec spec_;
    double P_;
    bool totalCondenser_;
    Stream distillate_, bottoms_;
    double condenser_duty_ = 0.0;
    double reboiler_duty_ = 0.0;
    
    // MESH equation residuals — called by Newton solver
    Eigen::VectorXd meshResiduals(const Eigen::VectorXd& vars) const;
    
    // Initialize with simple shortcut (Fenske-Underwood-Gilliland)
    Eigen::VectorXd initializeVariables() const;
};
```

---

### 5. Flowsheet Layer

```cpp
// include/chemsim/flowsheet/FlowsheetGraph.hpp
// Directed graph: nodes = unit ops, edges = streams

class FlowsheetGraph {
public:
    void addUnit(UnitOp* unit);
    void addStream(Stream* stream, UnitOp* from, UnitOp* to);
    
    // Topological sort (Kahn's algorithm)
    // Returns ordered list of units to solve
    // Throws if graph has no DAG structure after tear stream removal
    std::vector<UnitOp*> topologicalOrder() const;
    
    // Find recycle loops using Tarjan's SCC algorithm
    // Returns groups of units that form SCCs (size > 1 = recycle)
    std::vector<std::vector<UnitOp*>> findRecycles() const;
    
    // Identify tear streams (one stream per recycle loop cut)
    std::vector<Stream*> selectTearStreams() const;

private:
    std::vector<UnitOp*> units_;
    std::vector<Stream*> streams_;
    // adjacency: unit -> list of (stream, downstream unit)
    std::map<UnitOp*, std::vector<std::pair<Stream*, UnitOp*>>> adj_;
};
```

```cpp
// include/chemsim/flowsheet/RecycleSolver.hpp

class RecycleSolver {
public:
    struct Options {
        double tol = 1e-5;        // convergence on tear stream vars
        int maxIter = 100;
        bool useBroyden = true;   // false = direct substitution
    };
    
    RecycleSolver(FlowsheetGraph& graph, Options opts = {});
    
    // Converge all recycle loops
    // Returns true if converged
    bool solve();
    
private:
    FlowsheetGraph& graph_;
    Options opts_;
    
    // Convert tear streams to/from state vector for Broyden
    Eigen::VectorXd streamsToVector(const std::vector<Stream*>& tears) const;
    void vectorToStreams(const Eigen::VectorXd& v,
                        std::vector<Stream*>& tears) const;
    
    // One sequential modular pass through flowsheet
    // Returns updated tear stream states
    Eigen::VectorXd onePass(const Eigen::VectorXd& tearVars,
                            const std::vector<Stream*>& tears);
};
```

```cpp
// include/chemsim/flowsheet/Flowsheet.hpp
// Top-level object — owns everything

class Flowsheet {
public:
    explicit Flowsheet(const std::string& componentDBPath);
    
    // --- Building the flowsheet ---
    Stream& addStream(const std::string& name,
                      double T, double P, double flow,
                      const std::map<std::string,double>& composition);
    
    template<typename T, typename... Args>
    T& addUnit(Args&&... args);  // Creates unit op, adds to graph
    
    void connect(const std::string& streamName,
                 const std::string& fromUnit,
                 const std::string& toUnit);
    
    // --- Solving ---
    bool solve(RecycleSolver::Options opts = {});
    
    // --- Results ---
    const Stream& getStream(const std::string& name) const;
    const UnitOp& getUnit(const std::string& name) const;
    void printSummary() const;
    void exportResults(const std::string& jsonPath) const;
    
    // --- I/O ---
    static Flowsheet fromJSON(const std::string& jsonPath);

private:
    std::vector<Component> comps_;
    std::unique_ptr<PengRobinson> eos_;
    std::unique_ptr<FlashCalculator> flash_;
    std::map<std::string, Stream> streams_;
    std::map<std::string, std::unique_ptr<UnitOp>> units_;
    FlowsheetGraph graph_;
    
    // Inject flash calculator into all unit ops after construction
    void injectCalculators();
};
```

---

### 6. I/O Layer

#### Input JSON Format

```json
{
  "components": ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"],
  
  "streams": {
    "FEED": {
      "T": 350.0,
      "P": 2000000,
      "flow": 100.0,
      "composition": {
        "METHANE": 0.4,
        "ETHANE": 0.3,
        "PROPANE": 0.2,
        "N-BUTANE": 0.1
      }
    }
  },
  
  "units": {
    "FLASH1": {
      "type": "FlashDrum",
      "spec": "TP",
      "T": 280.0,
      "P": 1500000
    },
    "HX1": {
      "type": "HeatExchanger",
      "P_hot": 1500000,
      "P_cold": 1500000,
      "T_hot_out": 320.0
    },
    "COL1": {
      "type": "DistillationColumn",
      "nStages": 20,
      "feedStage": 10,
      "P": 1200000,
      "refluxRatio": 2.5,
      "distillateFlow": 40.0,
      "totalCondenser": true
    }
  },
  
  "connections": [
    {"stream": "FEED",    "from": null,    "to": "FLASH1"},
    {"stream": "VAP1",    "from": "FLASH1","to": "HX1"},
    {"stream": "LIQ1",    "from": "FLASH1","to": "COL1"},
    {"stream": "HOT_OUT", "from": "HX1",  "to": null},
    {"stream": "DIST",    "from": "COL1", "to": null},
    {"stream": "BTMS",    "from": "COL1", "to": null}
  ]
}
```

#### Component Database Format (data/components.json)

```json
{
  "METHANE": {
    "name": "Methane",
    "MW": 16.043,
    "Tc": 190.56,
    "Pc": 4599200,
    "omega": 0.0115,
    "Cp_A": 19.89, "Cp_B": 5.024e-2, "Cp_C": 1.269e-5, "Cp_D": -11.01e-9,
    "Antoine_A": 6.61184, "Antoine_B": 389.93, "Antoine_C": 266.69
  }
}
```

---

### 7. Python Bindings

```cpp
// python/chemsim.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "chemsim/flowsheet/Flowsheet.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chemsim, m) {
    m.doc() = "ChemSim: C++ process simulator with Python bindings";
    
    py::class_<Stream>(m, "Stream")
        .def_readonly("T", &Stream::T)
        .def_readonly("P", &Stream::P)
        .def_readonly("totalFlow", &Stream::totalFlow)
        .def_readonly("vaporFraction", &Stream::vaporFraction)
        .def_readonly("z", &Stream::z)
        .def_readonly("x", &Stream::x)
        .def_readonly("y", &Stream::y)
        .def_readonly("H", &Stream::H);
    
    py::class_<Flowsheet>(m, "Flowsheet")
        .def(py::init<const std::string&>())
        .def("addStream",   &Flowsheet::addStream)
        .def("connect",     &Flowsheet::connect)
        .def("solve",       &Flowsheet::solve)
        .def("getStream",   &Flowsheet::getStream)
        .def("printSummary",&Flowsheet::printSummary)
        .def_static("fromJSON", &Flowsheet::fromJSON);
}
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(ChemSim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Dependencies
find_package(Eigen3 3.4 REQUIRED)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/googletest)

# Core library
file(GLOB_RECURSE CHEMSIM_SOURCES "src/**/*.cpp")
add_library(chemsim_core STATIC ${CHEMSIM_SOURCES})
target_include_directories(chemsim_core PUBLIC include)
target_link_libraries(chemsim_core PUBLIC Eigen3::Eigen)

# Python module
pybind11_add_module(chemsim python/chemsim.cpp)
target_link_libraries(chemsim PRIVATE chemsim_core)

# Tests
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests chemsim_core gtest gtest_main)
add_test(NAME all_tests COMMAND run_tests)

# CLI executable
add_executable(chemsim_cli src/main.cpp)
target_link_libraries(chemsim_cli chemsim_core)
```

---

## Implementation Phases

### Phase 1 — Numerics + Pure Thermo (Weeks 1–3)
Goal: Flash a pure component and a binary mixture correctly.

**Order of implementation:**
1. `Component` class + `ComponentDB` loader (read from JSON)
2. `BrentSolver` — test against known roots
3. `PengRobinson::compressibilityFactors()` — test: Z for pure methane at 200K, 5MPa
4. `PengRobinson::lnFugacityCoefficients()` — test: fugacity of pure component equals Psat from Antoine at bubble point
5. `PengRobinson::enthalpyDeparture()` — test: H_dep → 0 as P → 0
6. `FlashCalculator::flashTP()` for binary — test: methane/ethane at known conditions vs literature

**Validation benchmarks:**
- Methane Z-factor at 200K, 5MPa: compare against NIST WebBook
- Methane/Ethane bubble point at 250K: compare against Perry's or NIST
- Rachford-Rice: analytically verify for trivial cases (pure component)

---

### Phase 2 — Multicomponent Flash + Enthalpy (Weeks 4–5)
Goal: Fully converged TP flash for 4+ component mixture.

1. Extend flash to multicomponent with successive substitution
2. Michelsen stability test (prevent trivial solutions)
3. `flashPH()` — outer T loop using `BrentSolver`, inner `flashTP()`
4. `flashPS()` — same structure
5. `Stream` class with full state tracking
6. `Mixture::idealGasH()` using DIPPR Cp correlations

**Validation:**
- Reproduce flash results from HYSYS or literature for C1-C4 mixture
- Energy balance: H_feed = H_vapor + H_liquid at adiabatic flash

---

### Phase 3 — Unit Operations (Weeks 6–9)
Goal: Each unit op solves correctly in isolation.

Implement in this order (simplest first):
1. `FlashDrum` (TP spec) — wraps `flashTP()`
2. `FlashDrum` (PH spec) — wraps `flashPH()`  
3. `Pump` — isentropic work via PS flash + efficiency correction
4. `Compressor` — same as pump
5. `HeatExchanger` — energy balance, inner PH flash on cold side
6. `Reactor` — stoichiometry update, isothermal flash on product
7. `DistillationColumn` — hardest, implement last

**Distillation strategy:**
- Start with simplified model: fixed K-values (from Wilson), solve material balance only
- Then add full MESH equations with Newton solver
- Variables: {L_j, V_j, T_j, x_ij} for j=1..N stages
- Initialize with Fenske-Underwood shortcut

---

### Phase 4 — Flowsheet Solver (Weeks 10–12)
Goal: Solve a flowsheet with one recycle loop.

1. `FlowsheetGraph` — add units, streams, connect
2. Topological sort (Kahn's algorithm)
3. Tarjan SCC for recycle detection
4. Tear stream selection
5. `RecycleSolver` with direct substitution first
6. Broyden acceleration
7. `Flowsheet` top-level class
8. JSON parser for flowsheet input

**Test flowsheet:**
- Simple flash + recycle (reflux loop without column internals)
- Verify: tear stream converges, material balance closes

---

### Phase 5 — Python Bindings + Polish (Weeks 13–14)
1. pybind11 module
2. Results export to JSON
3. `printSummary()` formatted output
4. Complete test suite
5. README + example notebooks
6. GitHub CI (compile + run tests on push)

---

## Key Algorithms — Implementation Notes

### Rachford-Rice Equation
```
g(beta) = sum_i [ z_i * (K_i - 1) / (1 + beta*(K_i - 1)) ] = 0

Bounds: beta in (beta_min, beta_max) where
  beta_min = 1/(1 - K_max)  + epsilon
  beta_max = 1/(1 - K_min)  - epsilon
```
Use `BrentSolver` on this interval. Guaranteed to converge.

### Successive Substitution (Flash Inner Loop)
```
1. Compute K_i from fugacity coefficients: K_i = phi_L_i / phi_V_i
2. Solve Rachford-Rice for beta
3. Compute x_i = z_i / (1 + beta*(K_i-1))
4. Compute y_i = K_i * x_i
5. Compute new phi_L_i, phi_V_i from EOS
6. Check: sum_i |ln(K_i_new) - ln(K_i_old)| < tol
7. Accelerate with DQFM (dominant eigenvalue) if slow
```

### PR EOS Cubic Solve
```
Z^3 - (1-B)*Z^2 + (A-3B^2-2B)*Z - (AB-B^2-B^3) = 0

where A = aP/(R²T²),  B = bP/(RT)

Three real roots: smallest = liquid Z, largest = vapor Z
If only one real root: single phase
```

### Inside-Out Distillation (Boston-Sullivan)
```
Outer loop variables: {K_ij} — K-value parameters per stage
Inner loop: solve linearized MESH equations given K_ij
  M equations: L_j*x_ij + V_j*y_ij = F_j*z_ij + L_{j+1}*x_{j+1,i} + V_{j-1}*y_{j-1,i}
  E equations: y_ij = K_ij * x_ij
  S equations: sum_i x_ij = 1,  sum_i y_ij = 1
  H equations: L_j*h_j + V_j*H_j = F_j*H_Fj + L_{j+1}*h_{j+1} + V_{j-1}*H_{j-1} + Q_j
Outer: update K_ij from EOS using new T_j, x_ij, y_ij
Converge when K_ij stop changing
```

---

## Testing Strategy

### Unit tests for each class
```cpp
// tests/test_thermo.cpp — example
TEST(PengRobinson, MethaneCompressibility) {
    ComponentDB db("data/components.json");
    auto comps = db.get({"METHANE"});
    PengRobinson pr(comps);
    
    auto [Z_L, Z_V] = pr.compressibilityFactors(200.0, 5e6, {1.0});
    
    // Compare against NIST: Z ~ 0.8672 at 200K, 5MPa for CH4
    EXPECT_NEAR(Z_V, 0.8672, 0.005);
}

TEST(FlashCalculator, BinaryBubblePoint) {
    ComponentDB db("data/components.json");
    auto comps = db.get({"METHANE", "ETHANE"});
    PengRobinson pr(comps);
    FlashCalculator flash(pr, comps);
    
    // Methane/Ethane 50/50 at 250K — bubble P should be ~1.8 MPa
    auto result = flash.flashTP(250.0, 1.8e6, {0.5, 0.5});
    EXPECT_NEAR(result.beta, 0.0, 0.05);  // near bubble point
}
```

### Integration test: full flowsheet
```cpp
TEST(Flowsheet, SimpleFlashDrum) {
    Flowsheet fs("data/components.json");
    fs.addStream("FEED", 350.0, 2e6, 100.0, 
                 {{"METHANE",0.4},{"ETHANE",0.3},
                  {"PROPANE",0.2},{"N-BUTANE",0.1}});
    
    auto& drum = fs.addUnit<FlashDrum>("DRUM1", 280.0, 1.5e6);
    fs.connect("FEED", "", "DRUM1");
    fs.connect("VAP", "DRUM1", "");
    fs.connect("LIQ", "DRUM1", "");
    
    EXPECT_TRUE(fs.solve());
    
    auto& vap = fs.getStream("VAP");
    EXPECT_GT(vap.vaporFraction, 0.99);       // should be all vapor
    EXPECT_GT(vap.z[0], 0.55);               // methane enriched in vapor
}
```

---

## First Commands to Run in Claude Code

When you start a session, paste this as context:

```
Project: ChemSim — C++ process simulator
Goal this session: [e.g. "Implement PengRobinson::compressibilityFactors() and unit test against NIST data"]

Stack: C++17, Eigen 3.4, pybind11, GoogleTest, nlohmann/json, CMake
Architecture doc: [paste relevant section above]

Ground rules:
- No raw pointers where unique_ptr works
- All numerical methods throw on non-convergence (don't return silently wrong answers)
- Every class gets a unit test before moving to next class
- Validate against literature/NIST at each phase before proceeding
```

---

## Reference Data for Validation

### Pure component PR parameters
| Component | Tc (K) | Pc (MPa) | ω |
|---|---|---|---|
| Methane | 190.56 | 4.599 | 0.0115 |
| Ethane | 305.32 | 4.872 | 0.0995 |
| Propane | 369.83 | 4.248 | 0.1523 |
| n-Butane | 425.12 | 3.796 | 0.2002 |
| n-Pentane | 469.70 | 3.370 | 0.2515 |
| Benzene | 562.05 | 4.895 | 0.2103 |
| Toluene | 591.75 | 4.108 | 0.2641 |
| Water | 647.10 | 22.064 | 0.3449 |
| CO2 | 304.13 | 7.375 | 0.2239 |
| H2S | 373.10 | 8.940 | 0.0942 |

### NIST validation points for PR EOS
- Methane at 200K, 5 MPa: Z = 0.8672
- Ethane at 300K, 4 MPa: Z ≈ 0.8128
- Propane bubble point at 300K: P_bub ≈ 0.993 MPa

---

## Estimated Timeline (Part-time, ~10 hrs/week)

| Phase | Content | Duration |
|---|---|---|
| 1 | Numerics + pure PR EOS | 3 weeks |
| 2 | Multicomponent flash + enthalpy | 2 weeks |
| 3 | Unit operations (excl. distillation) | 2 weeks |
| 3b | Distillation column | 2 weeks |
| 4 | Flowsheet solver + recycles | 3 weeks |
| 5 | Python bindings + polish | 2 weeks |
| **Total** | | **~14 weeks** |

After Phase 2 you have something demonstrable. After Phase 4 you have a complete portfolio project.
