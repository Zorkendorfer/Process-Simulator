#pragma once
#include "chemsim/flowsheet/FlowsheetGraph.hpp"
#include <map>
#include <string>
#include <vector>

namespace chemsim {

struct RecycleSolveResult {
    bool converged{false};
    int iterations{0};
    std::vector<std::string> tear_streams;
    std::map<std::string, Stream> streams;
};

class RecycleSolver {
public:
    struct Options {
        int maxIter{100};
        double tol_T{0.01};
        double tol_P{1.0};
        double tol_flow{1e-6};
        double tol_z{1e-6};
        double relaxation{1.0};
    };

    explicit RecycleSolver(const FlowsheetGraph& graph);
    RecycleSolver(const FlowsheetGraph& graph, Options opts);

    RecycleSolveResult solve(const std::map<std::string, Stream>& initial_streams) const;

private:
    const FlowsheetGraph& graph_;
    Options opts_;

    bool streamConverged(const Stream& a, const Stream& b) const;
    Stream blend(const Stream& old_stream, const Stream& new_stream) const;
};

} // namespace chemsim
