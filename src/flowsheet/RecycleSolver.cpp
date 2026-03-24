#include "chemsim/flowsheet/RecycleSolver.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace chemsim {

RecycleSolver::RecycleSolver(const FlowsheetGraph& graph)
    : RecycleSolver(graph, Options{}) {}

RecycleSolver::RecycleSolver(const FlowsheetGraph& graph, Options opts)
    : graph_(graph), opts_(opts) {
    if (opts_.maxIter <= 0)
        throw std::invalid_argument("RecycleSolver: maxIter must be > 0");
    if (opts_.relaxation <= 0.0 || opts_.relaxation > 1.0)
        throw std::invalid_argument("RecycleSolver: relaxation must be in (0, 1]");
}

bool RecycleSolver::streamConverged(const Stream& a, const Stream& b) const {
    if (std::abs(a.totalFlow - b.totalFlow) > opts_.tol_flow) return false;
    return Stream::converged(a, b, opts_.tol_T, opts_.tol_P, opts_.tol_z);
}

Stream RecycleSolver::blend(const Stream& old_stream, const Stream& new_stream) const {
    if (opts_.relaxation >= 1.0) return new_stream;

    auto blend_scalar = [&](double old_value, double new_value) {
        return old_value + opts_.relaxation * (new_value - old_value);
    };

    Stream blended = new_stream;
    blended.T = blend_scalar(old_stream.T, new_stream.T);
    blended.P = blend_scalar(old_stream.P, new_stream.P);
    blended.totalFlow = blend_scalar(old_stream.totalFlow, new_stream.totalFlow);
    blended.vaporFraction = blend_scalar(old_stream.vaporFraction, new_stream.vaporFraction);
    blended.H = blend_scalar(old_stream.H, new_stream.H);
    blended.S = blend_scalar(old_stream.S, new_stream.S);

    auto blend_vector = [&](std::vector<double>& target, const std::vector<double>& from_old,
                            const std::vector<double>& from_new) {
        if (from_old.size() != from_new.size()) return;
        target.resize(from_new.size());
        for (std::size_t i = 0; i < from_new.size(); ++i)
            target[i] = blend_scalar(from_old[i], from_new[i]);
    };

    blend_vector(blended.z, old_stream.z, new_stream.z);
    blend_vector(blended.x, old_stream.x, new_stream.x);
    blend_vector(blended.y, old_stream.y, new_stream.y);
    return blended;
}

RecycleSolveResult RecycleSolver::solve(
    const std::map<std::string, Stream>& initial_streams) const {

    RecycleSolveResult result;
    result.tear_streams = graph_.selectTearStreams();

    std::set<std::string> tear_set(result.tear_streams.begin(), result.tear_streams.end());
    const auto order = graph_.topoSort(tear_set);

    if (result.tear_streams.empty()) {
        result.streams = graph_.evaluate(order, {}, initial_streams);
        result.converged = true;
        result.iterations = 1;
        return result;
    }

    auto tear_values = initial_streams;
    for (const auto& tear : result.tear_streams) {
        if (!tear_values.count(tear))
            throw std::runtime_error(
                "RecycleSolver: missing initial guess for tear stream '" + tear + "'");
    }

    for (int iter = 1; iter <= opts_.maxIter; ++iter) {
        auto pass_streams = graph_.evaluate(order, tear_set, tear_values);

        bool converged = true;
        for (const auto& tear : result.tear_streams) {
            const auto& old_stream = tear_values.at(tear);
            const auto it_new = pass_streams.find(tear);
            if (it_new == pass_streams.end())
                throw std::runtime_error(
                    "RecycleSolver: evaluation did not produce tear stream '" + tear + "'");

            if (!streamConverged(old_stream, it_new->second))
                converged = false;

            tear_values[tear] = blend(old_stream, it_new->second);
            pass_streams[tear] = tear_values[tear];
        }

        result.streams = std::move(pass_streams);
        result.iterations = iter;
        result.converged = converged;
        if (converged) return result;
    }

    return result;
}

} // namespace chemsim
