"""Directed flowsheet graph: topological sort, SCC detection, tear streams."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from chemsim.core import Stream
from chemsim.ops.base import IUnitOp


@dataclass
class Connection:
    stream_id: str
    from_unit: str
    from_port: str
    to_unit: str
    to_port: str


class FlowsheetGraph:
    def __init__(self):
        self._units: Dict[str, IUnitOp] = {}
        self._feeds: Dict[str, Stream] = {}
        self._conns: List[Connection] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def add_unit(self, unit_id: str, op: IUnitOp) -> None:
        if unit_id in self._units:
            raise ValueError(f"FlowsheetGraph: duplicate unit id '{unit_id}'")
        self._units[unit_id] = op

    def set_feed(self, stream_id: str, stream: Stream) -> None:
        self._feeds[stream_id] = stream

    def connect(self, stream_id: str,
                from_unit: str = "", from_port: str = "",
                to_unit: str = "", to_port: str = "") -> None:
        if not stream_id:
            raise ValueError("FlowsheetGraph.connect: stream id must not be empty")
        if not from_unit and not to_unit:
            raise ValueError("FlowsheetGraph.connect: must have source or destination unit")
        if from_unit and from_unit not in self._units:
            raise ValueError(f"FlowsheetGraph.connect: unknown unit '{from_unit}'")
        if to_unit and to_unit not in self._units:
            raise ValueError(f"FlowsheetGraph.connect: unknown unit '{to_unit}'")
        if any(c.stream_id == stream_id for c in self._conns):
            raise ValueError(f"FlowsheetGraph.connect: duplicate stream id '{stream_id}'")
        if from_unit:
            ports = self._units[from_unit].outlet_ports()
            if from_port not in ports:
                raise ValueError(
                    f"FlowsheetGraph.connect: unknown outlet port '{from_port}' "
                    f"on unit '{from_unit}'")
        if to_unit:
            ports = self._units[to_unit].inlet_ports()
            if to_port not in ports:
                raise ValueError(
                    f"FlowsheetGraph.connect: unknown inlet port '{to_port}' "
                    f"on unit '{to_unit}'")
        self._conns.append(Connection(stream_id, from_unit, from_port, to_unit, to_port))

    def feeds(self) -> Dict[str, Stream]:
        return dict(self._feeds)

    def unit(self, name: str) -> IUnitOp:
        return self._units[name]

    # ── Adjacency ─────────────────────────────────────────────────────────────

    def _adjacency(self, removed: Set[str] = frozenset()) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {uid: [] for uid in self._units}
        for c in self._conns:
            if c.stream_id in removed:
                continue
            if c.from_unit and c.to_unit:
                adj[c.from_unit].append(c.to_unit)
        return adj

    # ── Kahn's topological sort ───────────────────────────────────────────────

    def topo_sort(self, removed_streams: Set[str] = frozenset()) -> List[str]:
        adj = self._adjacency(removed_streams)
        indegree: Dict[str, int] = {u: 0 for u in adj}
        for u, neighbors in adj.items():
            for v in neighbors:
                indegree[v] += 1

        q: deque = deque(u for u, d in indegree.items() if d == 0)
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)

        if len(order) != len(self._units):
            raise RuntimeError(
                "FlowsheetGraph.topo_sort: cycle detected — add tear streams first")
        return order

    # ── Tarjan SCC ────────────────────────────────────────────────────────────

    def _tarjan_dfs(self, u: str, adj: Dict[str, List[str]],
                    index_map: Dict[str, int], lowlink: Dict[str, int],
                    on_stack: Dict[str, bool], stk: List[str],
                    idx_ref: List[int],
                    sccs: List[List[str]]) -> None:
        index_map[u] = lowlink[u] = idx_ref[0]
        idx_ref[0] += 1
        stk.append(u); on_stack[u] = True

        for v in adj.get(u, []):
            if v not in index_map:
                self._tarjan_dfs(v, adj, index_map, lowlink,
                                 on_stack, stk, idx_ref, sccs)
                lowlink[u] = min(lowlink[u], lowlink[v])
            elif on_stack.get(v, False):
                lowlink[u] = min(lowlink[u], index_map[v])

        if lowlink[u] == index_map[u]:
            scc: List[str] = []
            while True:
                w = stk.pop()
                on_stack[w] = False
                scc.append(w)
                if w == u:
                    break
            self_loop = u in adj.get(u, [])
            if len(scc) > 1 or self_loop:
                sccs.append(scc)

    def find_sccs(self) -> List[List[str]]:
        adj = self._adjacency()
        index_map: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        on_stack: Dict[str, bool] = {}
        stk: List[str] = []
        sccs: List[List[str]] = []
        idx_ref = [0]
        for u in self._units:
            if u not in index_map:
                self._tarjan_dfs(u, adj, index_map, lowlink,
                                 on_stack, stk, idx_ref, sccs)
        return sccs

    # ── Tear stream selection ─────────────────────────────────────────────────

    def select_tear_streams(self) -> List[str]:
        sccs = self.find_sccs()
        tears: List[str] = []
        for scc in sccs:
            scc_set = set(scc)
            for c in self._conns:
                if c.from_unit in scc_set and c.to_unit in scc_set:
                    tears.append(c.stream_id)
                    break
        return tears

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, unit_order: List[str],
                 tear_streams: Set[str],
                 known_streams: Dict[str, Stream]) -> Dict[str, Stream]:
        stream_map = dict(known_streams)

        for uid in unit_order:
            op = self._units[uid]

            for port in op.inlet_ports():
                found = False
                for c in self._conns:
                    if c.to_unit == uid and c.to_port == port:
                        found = True
                        if c.stream_id not in stream_map:
                            raise RuntimeError(
                                f"FlowsheetGraph.evaluate: stream '{c.stream_id}' "
                                f"required by unit '{uid}' port '{port}' has no value")
                        op.set_inlet(port, stream_map[c.stream_id])
                        break
                if not found:
                    raise RuntimeError(
                        f"FlowsheetGraph.evaluate: unit '{uid}' "
                        f"inlet port '{port}' is not connected")

            try:
                op.solve()
            except Exception as ex:
                raise RuntimeError(
                    f"FlowsheetGraph.evaluate: unit '{uid}' failed: {ex}") from ex

            for port in op.outlet_ports():
                for c in self._conns:
                    if c.from_unit == uid and c.from_port == port:
                        stream_map[c.stream_id] = op.get_outlet(port)
                        break

        return stream_map
