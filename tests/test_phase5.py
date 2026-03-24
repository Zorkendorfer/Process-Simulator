"""Phase 5: Flowsheet assembly, JSON serialisation, parser."""
import json
import math
import tempfile
from pathlib import Path
import pytest

from chemsim.flowsheet.flowsheet import Flowsheet
from chemsim.io.component_db import ComponentDB
from chemsim.thermo.peng_robinson import PengRobinson
from chemsim.thermo.flash import FlashCalculator
from chemsim.ops.mixer import MixerOp
from chemsim.ops.splitter import SplitterOp


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "components.json"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_flowsheet_simple_recycle():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    ok = fs.solve()
    assert ok


def test_flowsheet_summary_contains_streams():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    fs.solve()
    summary = fs.summary()
    assert "FEED" in summary
    assert "ChemSim" in summary


def test_flowsheet_results_as_json():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    fs.solve()
    j = json.loads(fs.results_as_json())
    assert "streams" in j
    assert "components" in j
    assert "FEED" in j["streams"]


def test_flowsheet_export_results():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    fs.solve()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    fs.export_results(path)
    with open(path) as f:
        data = json.load(f)
    assert "streams" in data


def test_flowsheet_distillation_recycle():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "distillation_recycle.json"),
        str(DB_PATH))
    ok = fs.solve()
    assert ok
    dist = fs.get_stream("DISTILLATE")
    btms = fs.get_stream("BOTTOMS")
    assert dist.total_flow > 0.0
    assert btms.total_flow > 0.0


def test_flowsheet_programmatic_mixer_splitter():
    """Build a flowsheet programmatically: Feed → Mixer → Splitter → recycle."""
    db = ComponentDB(str(DB_PATH))
    comps = db.get(["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    fs = Flowsheet(comps)

    fs.add_stream("FEED", T=260.0, P=2e6, flow=100.0,
                  composition={"METHANE": 0.4, "ETHANE": 0.3,
                               "PROPANE": 0.2, "N-BUTANE": 0.1})
    fs.add_stream("RECYCLE", T=260.0, P=2e6, flow=100.0,
                  composition={"METHANE": 0.4, "ETHANE": 0.3,
                               "PROPANE": 0.2, "N-BUTANE": 0.1})

    fc = fs.flash_calculator()
    fs.add_unit("MIX",   MixerOp(fc, ["fresh", "recycle"]))
    fs.add_unit("SPLIT", SplitterOp([0.2, 0.8]))

    fs.connect("FEED",    to_unit="MIX",   to_port="fresh")
    fs.connect("RECYCLE", from_unit="SPLIT", from_port="out1",
                          to_unit="MIX",   to_port="recycle")
    fs.connect("MIXOUT",  from_unit="MIX",   from_port="out",
                          to_unit="SPLIT", to_port="in")
    fs.connect("PRODUCT", from_unit="SPLIT", from_port="out0")

    ok = fs.solve()
    assert ok

    prod = fs.get_stream("PRODUCT")
    assert prod.total_flow > 0.0


def test_flowsheet_stream_names():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    names = fs.stream_names()
    assert "FEED" in names


def test_flowsheet_reset_to_base():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    fs.solve()
    T_before = fs.get_stream("FEED").T
    fs.set_stream_conditions("FEED", 300.0, 3e6)
    fs.reset_to_base()
    T_after = fs.get_stream("FEED").T
    assert abs(T_after - T_before) < 0.1


def test_flowsheet_results_table():
    fs = Flowsheet.from_json(
        str(EXAMPLES / "simple_recycle.json"),
        str(DB_PATH))
    fs.solve()
    table = fs.results_table()
    assert isinstance(table, list)
    assert len(table) > 0
    assert "total_flow" in table[0]
