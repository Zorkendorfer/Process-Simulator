"""
ChemSim — pure-Python steady-state process simulator.

Quick start::

    import chemsim

    # Load from JSON spec
    fs = chemsim.Flowsheet.from_json("examples/simple_recycle.json",
                                      "data/components.json")
    fs.solve()
    print(fs.summary())

    # Programmatic construction
    fs2 = chemsim.Flowsheet.create(["METHANE", "ETHANE"],
                                    db_path="data/components.json")
    fs2.add_stream("FEED", T=250.0, P=2e6, flow=100.0,
                   composition={"METHANE": 0.6, "ETHANE": 0.4})
"""

from chemsim.core import Component, Phase, Stream
from chemsim.flowsheet.flowsheet import Flowsheet
from chemsim.thermo.peng_robinson import PengRobinson
from chemsim.thermo.flash import FlashCalculator, FlashResult
from chemsim.ops import (
    IUnitOp, MixerOp, SplitterOp, FlashDrumOp,
    PumpOp, CompressorOp, HeatExchangerOp,
    ReactorOp, Reaction, DistillationColumnOp,
)
from chemsim.io import ComponentDB, FlowsheetParser

__all__ = [
    # Core
    "Component", "Phase", "Stream",
    # Flowsheet
    "Flowsheet",
    # Thermo
    "PengRobinson", "FlashCalculator", "FlashResult",
    # Unit ops
    "IUnitOp", "MixerOp", "SplitterOp", "FlashDrumOp",
    "PumpOp", "CompressorOp", "HeatExchangerOp",
    "ReactorOp", "Reaction", "DistillationColumnOp",
    # I/O
    "ComponentDB", "FlowsheetParser",
]
