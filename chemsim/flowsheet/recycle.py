"""Successive-substitution recycle solver."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List

from chemsim.core import Stream
from chemsim.flowsheet.graph import FlowsheetGraph


@dataclass
class RecycleOptions:
    max_iter: int = 100
    tol_T: float = 0.01
    tol_P: float = 1.0
    tol_z: float = 1e-6
    tol_flow: float = 1e-4
    relaxation: float = 1.0


@dataclass
class RecycleSolveResult:
    converged: bool = False
    iterations: int = 0
    streams: Dict[str, Stream] = field(default_factory=dict)
    tear_streams: List[str] = field(default_factory=list)


class RecycleSolver:
    def __init__(self, graph: FlowsheetGraph,
                 opts: RecycleOptions = RecycleOptions()):
        if opts.max_iter <= 0:
            raise ValueError("RecycleSolver: max_iter must be > 0")
        if not (0.0 < opts.relaxation <= 1.0):
            raise ValueError("RecycleSolver: relaxation must be in (0, 1]")
        self._graph = graph
        self._opts = opts

    def _stream_converged(self, a: Stream, b: Stream) -> bool:
        if abs(a.total_flow - b.total_flow) > self._opts.tol_flow:
            return False
        return Stream.converged(a, b,
                                self._opts.tol_T,
                                self._opts.tol_P,
                                self._opts.tol_z)

    def _blend(self, old: Stream, new: Stream) -> Stream:
        if self._opts.relaxation >= 1.0:
            return new
        alpha = self._opts.relaxation
        blended = copy.copy(new)
        blended.T           = old.T           + alpha * (new.T - old.T)
        blended.P           = old.P           + alpha * (new.P - old.P)
        blended.total_flow  = old.total_flow  + alpha * (new.total_flow - old.total_flow)
        blended.vapor_fraction = old.vapor_fraction + alpha * (new.vapor_fraction - old.vapor_fraction)
        blended.H           = old.H           + alpha * (new.H - old.H)
        blended.S           = old.S           + alpha * (new.S - old.S)

        def blend_vec(ov, nv):
            if not ov or len(ov) != len(nv):
                return list(nv)
            return [ov[i] + alpha * (nv[i] - ov[i]) for i in range(len(nv))]

        blended.z = blend_vec(old.z, new.z)
        blended.x = blend_vec(old.x, new.x)
        blended.y = blend_vec(old.y, new.y)
        return blended

    def solve(self, initial_streams: Dict[str, Stream]) -> RecycleSolveResult:
        result = RecycleSolveResult()
        result.tear_streams = self._graph.select_tear_streams()
        tear_set = set(result.tear_streams)
        order = self._graph.topo_sort(tear_set)

        if not result.tear_streams:
            result.streams   = self._graph.evaluate(order, set(), initial_streams)
            result.converged = True
            result.iterations = 1
            return result

        tear_values = dict(initial_streams)
        for tear in result.tear_streams:
            if tear not in tear_values:
                feeds = self._graph.feeds()
                if not feeds:
                    raise RuntimeError(
                        f"RecycleSolver: missing initial guess for tear stream '{tear}' "
                        "and no feeds available")
                init = copy.copy(next(iter(feeds.values())))
                init.name = tear
                tear_values[tear] = init

        for it in range(1, self._opts.max_iter + 1):
            pass_streams = self._graph.evaluate(order, tear_set, tear_values)

            converged = True
            for tear in result.tear_streams:
                old = tear_values[tear]
                new = pass_streams.get(tear)
                if new is None:
                    raise RuntimeError(
                        f"RecycleSolver: evaluation did not produce tear stream '{tear}'")
                if not self._stream_converged(old, new):
                    converged = False
                tear_values[tear] = self._blend(old, new)
                pass_streams[tear] = tear_values[tear]

            result.streams    = pass_streams
            result.iterations = it
            result.converged  = converged
            if converged:
                return result

        return result
