#include "chemsim/flowsheet/FlowsheetGraph.hpp"
#include <algorithm>
#include <queue>
#include <unordered_map>

namespace chemsim {

// ── Build ────────────────────────────────────────────────────────────────────

void FlowsheetGraph::addUnit(const std::string& unit_id,
                              std::unique_ptr<IUnitOp> op) {
    if (units_.count(unit_id))
        throw std::invalid_argument("FlowsheetGraph: duplicate unit id '" + unit_id + "'");
    units_[unit_id] = std::move(op);
}

void FlowsheetGraph::setFeed(const std::string& stream_id, const Stream& s) {
    feeds_[stream_id] = s;
}

void FlowsheetGraph::connect(const std::string& stream_id,
                              const std::string& from_unit,
                              const std::string& from_port,
                              const std::string& to_unit,
                              const std::string& to_port) {
    if (stream_id.empty())
        throw std::invalid_argument("FlowsheetGraph::connect: stream id must not be empty");
    if (from_unit.empty() && to_unit.empty())
        throw std::invalid_argument("FlowsheetGraph::connect: connection must have a source or destination unit");
    if (!from_unit.empty() && !units_.count(from_unit))
        throw std::invalid_argument("FlowsheetGraph::connect: unknown unit '" + from_unit + "'");
    if (!to_unit.empty() && !units_.count(to_unit))
        throw std::invalid_argument("FlowsheetGraph::connect: unknown unit '" + to_unit + "'");
    for (const auto& c : conns_) {
        if (c.stream_id == stream_id)
            throw std::invalid_argument("FlowsheetGraph::connect: duplicate stream id '" + stream_id + "'");
    }

    if (!from_unit.empty()) {
        const auto ports = units_.at(from_unit)->outletPorts();
        if (std::find(ports.begin(), ports.end(), from_port) == ports.end())
            throw std::invalid_argument("FlowsheetGraph::connect: unknown outlet port '" + from_port +
                                        "' on unit '" + from_unit + "'");
    }

    if (!to_unit.empty()) {
        const auto ports = units_.at(to_unit)->inletPorts();
        if (std::find(ports.begin(), ports.end(), to_port) == ports.end())
            throw std::invalid_argument("FlowsheetGraph::connect: unknown inlet port '" + to_port +
                                        "' on unit '" + to_unit + "'");
    }

    conns_.push_back({stream_id, from_unit, from_port, to_unit, to_port});
}

// ── Adjacency ────────────────────────────────────────────────────────────────

std::map<std::string, std::vector<std::string>>
FlowsheetGraph::adjacency(const std::set<std::string>& removed) const {
    std::map<std::string, std::vector<std::string>> adj;
    for (const auto& [id, _] : units_) adj[id];   // ensure all nodes present

    for (const auto& c : conns_) {
        if (removed.count(c.stream_id)) continue;
        if (c.from_unit.empty() || c.to_unit.empty()) continue;
        adj[c.from_unit].push_back(c.to_unit);
    }
    return adj;
}

// ── Kahn's topological sort ───────────────────────────────────────────────────

std::vector<std::string> FlowsheetGraph::topoSort(
    const std::set<std::string>& removed_streams) const {

    auto adj = adjacency(removed_streams);

    // Compute in-degrees
    std::map<std::string, int> indegree;
    for (const auto& [u, _] : adj) indegree[u] = 0;
    for (const auto& [u, neighbors] : adj)
        for (const auto& v : neighbors)
            ++indegree[v];

    std::queue<std::string> q;
    for (const auto& [u, d] : indegree)
        if (d == 0) q.push(u);

    std::vector<std::string> order;
    while (!q.empty()) {
        auto u = q.front(); q.pop();
        order.push_back(u);
        for (const auto& v : adj.at(u)) {
            if (--indegree[v] == 0) q.push(v);
        }
    }

    if ((int)order.size() != (int)units_.size())
        throw std::runtime_error(
            "FlowsheetGraph::topoSort: cycle detected — add tear streams first");
    return order;
}

// ── Tarjan SCC ───────────────────────────────────────────────────────────────

void FlowsheetGraph::tarjanDFS(
    const std::string& u,
    const std::map<std::string, std::vector<std::string>>& adj,
    std::map<std::string, int>& index_map,
    std::map<std::string, int>& lowlink,
    std::map<std::string, bool>& on_stack,
    std::vector<std::string>& stk,
    int& idx,
    std::vector<std::vector<std::string>>& sccs) const {

    index_map[u] = lowlink[u] = idx++;
    stk.push_back(u);
    on_stack[u] = true;

    for (const auto& v : adj.at(u)) {
        if (!index_map.count(v)) {
            tarjanDFS(v, adj, index_map, lowlink, on_stack, stk, idx, sccs);
            lowlink[u] = std::min(lowlink[u], lowlink[v]);
        } else if (on_stack[v]) {
            lowlink[u] = std::min(lowlink[u], index_map[v]);
        }
    }

    // Root of an SCC
    if (lowlink[u] == index_map[u]) {
        std::vector<std::string> scc;
        while (true) {
            auto w = stk.back(); stk.pop_back();
            on_stack[w] = false;
            scc.push_back(w);
            if (w == u) break;
        }
        // Only keep non-trivial SCCs (size > 1, or self-loop)
        bool self_loop = false;
        for (const auto& v : adj.at(u))
            if (v == u) { self_loop = true; break; }
        if (scc.size() > 1 || self_loop)
            sccs.push_back(std::move(scc));
    }
}

std::vector<std::vector<std::string>> FlowsheetGraph::findSCCs() const {
    auto adj = adjacency({});
    std::map<std::string, int>  index_map, lowlink;
    std::map<std::string, bool> on_stack;
    std::vector<std::string>    stk;
    std::vector<std::vector<std::string>> sccs;
    int idx = 0;

    for (const auto& [u, _] : units_)
        if (!index_map.count(u))
            tarjanDFS(u, adj, index_map, lowlink, on_stack, stk, idx, sccs);

    return sccs;
}

// ── Tear stream selection ─────────────────────────────────────────────────────

std::vector<std::string> FlowsheetGraph::selectTearStreams() const {
    auto sccs = findSCCs();
    std::vector<std::string> tears;

    for (const auto& scc : sccs) {
        std::set<std::string> scc_set(scc.begin(), scc.end());
        // Find the first connection whose both endpoints are in this SCC.
        // That stream becomes the tear stream.
        for (const auto& c : conns_) {
            if (scc_set.count(c.from_unit) && scc_set.count(c.to_unit)) {
                tears.push_back(c.stream_id);
                break;
            }
        }
    }
    return tears;
}

// ── Evaluation ───────────────────────────────────────────────────────────────

std::map<std::string, Stream> FlowsheetGraph::evaluate(
    const std::vector<std::string>& unit_order,
    const std::set<std::string>&    /*tear_streams*/,
    const std::map<std::string, Stream>& known_streams) const {

    auto stream_map = known_streams;  // start with feeds + tear stream values

    for (const auto& uid : unit_order) {
        auto& op = *units_.at(uid);

        // Feed inlets from stream_map
        for (const auto& port : op.inletPorts()) {
            bool found_connection = false;
            for (const auto& c : conns_) {
                if (c.to_unit == uid && c.to_port == port) {
                    found_connection = true;
                    auto it = stream_map.find(c.stream_id);
                    if (it == stream_map.end())
                        throw std::runtime_error(
                            "FlowsheetGraph::evaluate: stream '" + c.stream_id +
                            "' required by unit '" + uid + "' port '" + port +
                            "' has no value yet");
                    op.setInlet(port, it->second);
                    break;
                }
            }
            if (!found_connection) {
                throw std::runtime_error(
                    "FlowsheetGraph::evaluate: unit '" + uid +
                    "' inlet port '" + port + "' is not connected");
            }
        }

        try {
            op.solve();
        } catch (const std::exception& ex) {
            throw std::runtime_error(
                "FlowsheetGraph::evaluate: unit '" + uid + "' failed: " + ex.what());
        }

        // Collect outlets
        for (const auto& port : op.outletPorts()) {
            for (const auto& c : conns_) {
                if (c.from_unit == uid && c.from_port == port) {
                    stream_map[c.stream_id] = op.getOutlet(port);
                    break;
                }
            }
        }
    }

    return stream_map;
}

} // namespace chemsim
