"""Top-level Flowsheet orchestrator."""
from __future__ import annotations

import copy
import json
from typing import Dict, List, Optional

from chemsim.core import Component, Phase, Stream
from chemsim.flowsheet.graph import FlowsheetGraph
from chemsim.flowsheet.recycle import RecycleOptions, RecycleSolver
from chemsim.ops.base import IUnitOp
from chemsim.ops.distillation import DistillationColumnOp
from chemsim.thermo.flash import FlashCalculator
from chemsim.thermo.peng_robinson import PengRobinson


class Flowsheet:
    def __init__(self, components: List[Component]):
        if not components:
            raise ValueError("Flowsheet: components must not be empty")
        self._components = list(components)
        self._eos = PengRobinson(self._components)
        self._flash = FlashCalculator(self._eos, self._components)
        self._graph = FlowsheetGraph()
        self._streams: Dict[str, Stream] = {}
        self._base_streams: Dict[str, Stream] = {}

    # ── Accessors ─────────────────────────────────────────────────────────────

    def components(self) -> List[Component]:
        return list(self._components)

    def flash_calculator(self) -> FlashCalculator:
        return self._flash

    # ── Build ─────────────────────────────────────────────────────────────────

    def add_stream(self, name: str, T: float, P: float, flow: float,
                   composition: Dict[str, float]) -> Stream:
        z = self._composition_vector(composition)
        s = Stream(name=name, T=T, P=P, total_flow=flow, z=z)
        self._initialize_thermo(s)
        self._streams[name] = s
        self._base_streams[name] = copy.deepcopy(s)
        self._graph.set_feed(name, s)
        return self._streams[name]

    def add_unit(self, name: str, op: IUnitOp) -> None:
        if not name:
            raise ValueError("Flowsheet: unit name must not be empty")
        self._graph.add_unit(name, op)

    def connect(self, stream_name: str,
                from_unit: str = "", from_port: str = "",
                to_unit: str = "", to_port: str = "") -> None:
        if stream_name and stream_name not in self._streams:
            s = Stream(name=stream_name)
            self._streams[stream_name] = s
        self._graph.connect(stream_name, from_unit, from_port, to_unit, to_port)

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self, opts: Optional[RecycleOptions] = None) -> bool:
        solver = RecycleSolver(self._graph, opts or RecycleOptions())
        result = solver.solve(self._streams)
        self._streams = result.streams
        return result.converged

    # ── Stream / unit access ──────────────────────────────────────────────────

    def get_stream(self, name: str) -> Stream:
        return self._streams[name]

    def stream_names(self) -> List[str]:
        return list(self._streams.keys())

    def get_unit(self, name: str) -> IUnitOp:
        return self._graph.unit(name)

    # ── RL interface ──────────────────────────────────────────────────────────

    def set_param(self, unit_name: str, param_name: str, value: float) -> None:
        op = self._graph.unit(unit_name)
        if isinstance(op, DistillationColumnOp):
            if param_name == "refluxRatio":
                op.set_reflux_ratio(value); return
            if param_name == "distillateFrac":
                op.set_distillate_frac(value); return
        raise ValueError(
            f"Flowsheet.set_param: unknown unit '{unit_name}' "
            f"or param '{param_name}'")

    def set_stream_conditions(self, stream_name: str, T: float, P: float) -> None:
        s = self._streams[stream_name]
        s.T = T; s.P = P
        self._initialize_thermo(s)
        self._graph.set_feed(stream_name, s)

    def reset_to_base(self) -> None:
        self._streams = copy.deepcopy(self._base_streams)
        for name, s in self._base_streams.items():
            self._graph.set_feed(name, copy.deepcopy(s))

    def get_unit_scalar(self, unit_name: str, key: str) -> float:
        op = self._graph.unit(unit_name)
        if isinstance(op, DistillationColumnOp):
            return op.get_scalar(key)
        raise ValueError(
            f"Flowsheet.get_unit_scalar: unknown unit '{unit_name}' or key '{key}'")

    # ── Output ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["ChemSim Flowsheet Summary"]
        lines.append("Components: " + " ".join(c.id for c in self._components))
        lines.append(f"Streams ({len(self._streams)})")
        for name, s in self._streams.items():
            lines.append(
                f"  {name}: F={s.total_flow:.3f} mol/s"
                f", T={s.T:.3f} K"
                f", P={s.P:.3f} Pa"
                f", phase={s.phase.value}"
                f", beta={s.vapor_fraction:.3f}")
        return "\n".join(lines)

    def results_as_json(self) -> str:
        data = {
            "components": [
                {"id": c.id, "name": c.name, "MW": c.MW,
                 "Tc": c.Tc, "Pc": c.Pc, "omega": c.omega}
                for c in self._components
            ],
            "streams": {name: s.to_dict()
                        for name, s in self._streams.items()},
        }
        return json.dumps(data, indent=2)

    def export_results(self, json_path: str) -> None:
        with open(json_path, "w") as f:
            f.write(self.results_as_json())

    def results_table(self) -> List[dict]:
        return [s.to_dict() for s in self._streams.values()]

    def __repr__(self) -> str:
        names = list(self._streams.keys())
        return f"<Flowsheet streams={names}>"

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def create(component_ids: List[str], db_path: str) -> Flowsheet:
        from chemsim.io.component_db import ComponentDB
        db = ComponentDB(db_path)
        return Flowsheet(db.get(component_ids))

    @staticmethod
    def from_json(json_path: str, component_db_path: str) -> Flowsheet:
        from chemsim.io.parser import FlowsheetParser
        return FlowsheetParser.parse_file(json_path, component_db_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _composition_vector(self, composition: Dict[str, float]) -> List[float]:
        z = []
        total = 0.0
        for c in self._components:
            val = composition.get(c.id)
            if val is None:
                raise ValueError(
                    f"Flowsheet: missing composition for component '{c.id}'")
            z.append(val)
            total += val
        if total <= 0.0:
            raise ValueError("Flowsheet: composition sum must be > 0")
        return [zi / total for zi in z]

    def _initialize_thermo(self, s: Stream) -> None:
        if s.total_flow <= 0.0 or not s.z:
            return
        r = self._flash.flash_TP(s.T, s.P, s.z)
        s.vapor_fraction = r.beta
        s.x = r.x; s.y = r.y
        s.phase = (Phase.LIQUID if r.beta < 1e-10
                   else Phase.VAPOR if r.beta > 1.0 - 1e-10
                   else Phase.MIXED)
        s.H = self._flash.total_enthalpy(r)
        s.S = self._flash.total_entropy(r)
