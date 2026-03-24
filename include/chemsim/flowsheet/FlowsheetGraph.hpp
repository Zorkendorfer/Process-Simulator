#pragma once
#include "chemsim/flowsheet/IUnitOp.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace chemsim {

/// Directed graph of unit operations connected by named streams.
///
/// Nodes  = unit operations (IUnitOp)
/// Edges  = streams (each stream has one source port and one sink port)
/// Feeds  = streams with no producing unit (externally set)
///
/// Evaluation order is determined by topological sort after tear-stream removal.
class FlowsheetGraph {
public:
    // ── Build ─────────────────────────────────────────────────────────────────

    /// Register a unit operation.
    void addUnit(const std::string& unit_id, std::unique_ptr<IUnitOp> op);

    /// Declare a feed stream (no producing unit).
    void setFeed(const std::string& stream_id, const Stream& s);

    /// Connect an output port to an input port.
    /// The stream_id is the name given to this connection.
    void connect(const std::string& stream_id,
                 const std::string& from_unit, const std::string& from_port,
                 const std::string& to_unit,   const std::string& to_port);

    // ── Analysis ──────────────────────────────────────────────────────────────

    /// Kahn's topological sort, ignoring edges on removed_streams.
    /// Throws std::runtime_error if cycles remain after removal.
    std::vector<std::string> topoSort(
        const std::set<std::string>& removed_streams = {}) const;

    /// Tarjan SCC on the unit graph.
    /// Returns list of SCCs in reverse topological order;
    /// trivial SCCs (single node, no self-loop) are omitted.
    std::vector<std::vector<std::string>> findSCCs() const;

    /// For each non-trivial SCC, pick one stream (back-edge) as a tear stream.
    /// Returns stream IDs.
    std::vector<std::string> selectTearStreams() const;

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Evaluate units in unit_order.
    /// known_streams must contain values for all feed streams + all tear streams.
    /// Returns the updated stream map (all stream values after evaluation).
    std::map<std::string, Stream> evaluate(
        const std::vector<std::string>& unit_order,
        const std::set<std::string>&    tear_streams,
        const std::map<std::string, Stream>& known_streams) const;

    // ── Accessors ─────────────────────────────────────────────────────────────

    const std::map<std::string, Stream>& feeds() const { return feeds_; }

    /// True if stream_id is a declared feed.
    bool isFeed(const std::string& sid) const {
        return feeds_.count(sid) > 0;
    }

    IUnitOp& unit(const std::string& id) { return *units_.at(id); }
    const IUnitOp& unit(const std::string& id) const { return *units_.at(id); }
    bool hasUnit(const std::string& id) const { return units_.count(id) > 0; }

private:
    struct Connection {
        std::string stream_id;
        std::string from_unit, from_port;
        std::string to_unit,   to_port;
    };

    std::map<std::string, std::unique_ptr<IUnitOp>> units_;
    std::map<std::string, Stream>                   feeds_;
    std::vector<Connection>                         conns_;

    // Helper: unit-to-unit adjacency (omitting removed streams)
    std::map<std::string, std::vector<std::string>>
    adjacency(const std::set<std::string>& removed = {}) const;

    // Tarjan DFS helper
    void tarjanDFS(const std::string& u,
                   const std::map<std::string, std::vector<std::string>>& adj,
                   std::map<std::string, int>& index_map,
                   std::map<std::string, int>& lowlink,
                   std::map<std::string, bool>& on_stack,
                   std::vector<std::string>& stack,
                   int& idx,
                   std::vector<std::vector<std::string>>& sccs) const;
};

} // namespace chemsim
