"""JSON flowsheet parser: deserialises a flowsheet spec into a Flowsheet."""
from __future__ import annotations

import json
from pathlib import Path

from chemsim.io.component_db import ComponentDB
from chemsim.ops.distillation import DistillationColumnOp
from chemsim.ops.flash_drum import FlashDrumOp, FlashSpec
from chemsim.ops.mixer import MixerOp
from chemsim.ops.pump import PumpOp
from chemsim.ops.splitter import SplitterOp


class FlowsheetParser:
    @staticmethod
    def parse_file(json_path: str, component_db_path: str):
        """Parse a flowsheet JSON file and return a Flowsheet."""
        from chemsim.flowsheet.flowsheet import Flowsheet

        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"FlowsheetParser: cannot open '{json_path}'")
        with open(path) as f:
            j = json.load(f)

        component_ids = j["components"]
        db = ComponentDB(component_db_path)
        flowsheet = Flowsheet(db.get(component_ids))

        # Streams
        for name, sj in j.get("streams", {}).items():
            flowsheet.add_stream(
                name,
                T=sj["T"], P=sj["P"], flow=sj["flow"],
                composition=sj["composition"])

        # Units
        for name, uj in j["units"].items():
            unit_type = uj["type"]

            if unit_type == "Mixer":
                op = MixerOp(flowsheet.flash_calculator(),
                             uj["inlet_ports"])

            elif unit_type == "Splitter":
                op = SplitterOp(uj["fractions"])

            elif unit_type == "FlashDrum":
                spec_name = uj["spec"]
                P = uj["P"]
                if spec_name == "TP":
                    spec, val = FlashSpec.TP, uj["T"]
                elif spec_name == "PH":
                    spec, val = FlashSpec.PH, uj["H"]
                elif spec_name == "PS":
                    spec, val = FlashSpec.PS, uj["S"]
                else:
                    raise ValueError(
                        f"FlowsheetParser: unknown flash drum spec '{spec_name}'")
                op = FlashDrumOp(spec, flowsheet.flash_calculator(), val, P)

            elif unit_type == "Pump":
                op = PumpOp(flowsheet.flash_calculator(),
                            uj["P_out"],
                            uj.get("eta", 0.75))

            elif unit_type == "DistillationColumn":
                op = DistillationColumnOp(
                    flowsheet.flash_calculator(),
                    flowsheet.components(),
                    N_stages=uj["N_stages"],
                    feed_stage=uj["feed_stage"],
                    reflux_ratio=uj["reflux_ratio"],
                    distillate_frac=uj["distillate_frac"],
                    P_top=uj.get("P_top", 101325.0),
                    feed_quality=uj.get("feed_quality", 1.0),
                    max_iter=uj.get("max_iter", 15))

            else:
                raise ValueError(
                    f"FlowsheetParser: unknown unit type '{unit_type}'")

            flowsheet.add_unit(name, op)

        # Connections
        for conn in j["connections"]:
            flowsheet.connect(
                conn["stream"],
                from_unit=conn.get("from", ""),
                from_port=conn.get("from_port", ""),
                to_unit=conn.get("to", ""),
                to_port=conn.get("to_port", ""))

        return flowsheet
