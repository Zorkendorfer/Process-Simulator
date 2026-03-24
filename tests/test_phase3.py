"""Phase 3: Unit operations (FlashDrum, Pump, HX, Reactor, Distillation, Recycle)."""
import math
import pytest

from chemsim.io.component_db import ComponentDB
from chemsim.thermo.peng_robinson import PengRobinson
from chemsim.thermo.flash import FlashCalculator
from chemsim.core import Stream, Phase
from chemsim.ops.flash_drum import FlashDrumOp, FlashSpec
from chemsim.ops.pump import PumpOp
from chemsim.ops.heat_exchanger import HeatExchangerOp, HXSpec
from chemsim.ops.reactor import ReactorOp, Reaction
from chemsim.ops.distillation import DistillationColumnOp
from chemsim.flowsheet.graph import FlowsheetGraph
from chemsim.flowsheet.recycle import RecycleSolver, RecycleOptions


def make_fc(db_path, ids):
    db = ComponentDB(db_path)
    comps = db.get(ids)
    return FlashCalculator(PengRobinson(comps), comps), comps


def make_stream(fc, comps, T, P, flow, z_dict):
    ids = [c.id for c in comps]
    z = [z_dict[cid] for cid in ids]
    s_total = sum(z); z = [zi / s_total for zi in z]
    s = Stream(T=T, P=P, total_flow=flow, z=z)
    r = fc.flash_TP(T, P, z)
    s.vapor_fraction = r.beta
    s.x = r.x; s.y = r.y
    s.phase = (Phase.LIQUID if r.beta < 1e-10
               else Phase.VAPOR if r.beta > 1 - 1e-10
               else Phase.MIXED)
    s.H = fc.total_enthalpy(r)
    s.S = fc.total_entropy(r)
    return s


# ── FlashDrum ─────────────────────────────────────────────────────────────────

def test_flash_drum_TP(db_path):
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    feed = make_stream(fc, comps, 260.0, 2e6, 100.0,
                       {"METHANE": 0.4, "ETHANE": 0.3, "PROPANE": 0.2, "N-BUTANE": 0.1})
    drum = FlashDrumOp(FlashSpec.TP, fc, 260.0, 2e6)
    drum.feed = feed
    drum.solve()

    # Flows sum to feed
    assert abs(drum.vapor.total_flow + drum.liquid.total_flow - 100.0) < 1e-4
    # Compositions valid
    assert abs(sum(drum.vapor.z) - 1.0) < 1e-6
    assert abs(sum(drum.liquid.z) - 1.0) < 1e-6


def test_flash_drum_PH(db_path):
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    feed = make_stream(fc, comps, 260.0, 2e6, 100.0,
                       {"METHANE": 0.4, "ETHANE": 0.3, "PROPANE": 0.2, "N-BUTANE": 0.1})
    H_target = feed.H
    drum = FlashDrumOp(FlashSpec.PH, fc, H_target, 2e6)
    drum.feed = feed
    drum.solve()
    # Temperature should be close to original (same H, same P)
    assert abs(drum.vapor.T - 260.0) < 1.0


# ── Pump ──────────────────────────────────────────────────────────────────────

def test_pump_pressure_increase(db_path):
    # All-liquid conditions: propane-rich at 200 K, 5 MPa (well below bubble point)
    fc, comps = make_fc(db_path, ["PROPANE", "N-BUTANE"])
    feed = make_stream(fc, comps, 200.0, 5e6, 50.0,
                       {"PROPANE": 0.5, "N-BUTANE": 0.5})
    pump = PumpOp(fc, P_out=10e6)
    pump.inlet = feed
    pump.solve()

    assert abs(pump.outlet.P - 10e6) < 1.0
    assert pump.shaft_work_mol > 0.0   # pump requires work input


def test_pump_shaft_power(db_path):
    fc, comps = make_fc(db_path, ["PROPANE", "N-BUTANE"])
    feed = make_stream(fc, comps, 200.0, 5e6, 50.0,
                       {"PROPANE": 0.5, "N-BUTANE": 0.5})
    pump = PumpOp(fc, P_out=10e6)
    pump.inlet = feed
    pump.solve()
    assert pump.shaft_power_W == pytest.approx(50.0 * pump.shaft_work_mol, rel=1e-6)


# ── Heat Exchanger ────────────────────────────────────────────────────────────

def test_heat_exchanger_energy_balance(db_path):
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE", "PROPANE"])
    hot  = make_stream(fc, comps, 400.0, 2e6, 50.0,
                       {"METHANE": 0.5, "ETHANE": 0.3, "PROPANE": 0.2})
    cold = make_stream(fc, comps, 250.0, 2e6, 50.0,
                       {"METHANE": 0.5, "ETHANE": 0.3, "PROPANE": 0.2})

    hx = HeatExchangerOp(HXSpec.DUTY, fc, fc)
    hx.hot_in = hot; hx.cold_in = cold
    hx.Q_spec = 1e5   # 100 kW
    hx.solve()

    # Energy balance: Q gained by cold = Q_spec; Q lost by hot = Q_spec
    delta_H_cold = cold.total_flow * (hx.cold_out.H - cold.H)
    delta_H_hot  = hot.total_flow  * (hot.H - hx.hot_out.H)
    assert abs(delta_H_cold - 1e5) < 50.0
    assert abs(delta_H_hot  - 1e5) < 50.0


# ── Reactor ───────────────────────────────────────────────────────────────────

def test_reactor_stoichiometry(db_path):
    # A → B: components METHANE, ETHANE; extent 5 mol/s
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE"])
    feed = make_stream(fc, comps, 300.0, 1e5, 100.0,
                       {"METHANE": 1.0, "ETHANE": 0.0})

    # nu = [-1, +1]: 1 mol/s methane → 1 mol/s ethane (conceptually)
    rxn = Reaction(nu=[-1.0, 1.0])
    reactor = ReactorOp(fc, T_spec=300.0, P_spec=1e5)
    reactor.inlet = feed
    reactor.add_reaction(rxn, extent_mol_s=10.0)
    reactor.solve()

    # Outlet: 90 mol/s methane, 10 mol/s ethane
    assert abs(reactor.outlet.total_flow - 100.0) < 1e-6
    assert abs(reactor.outlet.z[0] - 0.90) < 1e-6   # 90/100
    assert abs(reactor.outlet.z[1] - 0.10) < 1e-6   # 10/100


# ── Distillation Column ───────────────────────────────────────────────────────

def test_distillation_basic(db_path):
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE", "PROPANE"])
    feed = make_stream(fc, comps, 250.0, 2e6, 100.0,
                       {"METHANE": 0.5, "ETHANE": 0.3, "PROPANE": 0.2})

    col = DistillationColumnOp(fc, comps,
                               N_stages=10, feed_stage=5,
                               reflux_ratio=2.0, distillate_frac=0.4,
                               P_top=2e6)
    col.feed = feed
    col.solve()

    assert abs(col.distillate.total_flow - 40.0) < 1e-4
    assert abs(col.bottoms.total_flow    - 60.0) < 1e-4
    # Mass balance: z_D*D + z_B*B = z_F*F for each component
    nc = len(comps)
    for i in range(nc):
        lhs = col.distillate.z[i] * 40.0 + col.bottoms.z[i] * 60.0
        assert abs(lhs - feed.z[i] * 100.0) < 0.5


def test_distillation_light_in_distillate(db_path):
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE", "PROPANE"])
    feed = make_stream(fc, comps, 250.0, 2e6, 100.0,
                       {"METHANE": 0.5, "ETHANE": 0.3, "PROPANE": 0.2})
    col = DistillationColumnOp(fc, comps, 10, 5, 2.0, 0.4, 2e6)
    col.feed = feed; col.solve()
    # Methane should be enriched in distillate
    assert col.distillate.z[0] > feed.z[0]
    # Propane should be enriched in bottoms
    assert col.bottoms.z[2] > feed.z[2]


def test_distillation_duties(db_path):
    fc, comps = make_fc(db_path, ["METHANE", "ETHANE", "PROPANE"])
    # Use feed at bubble point (saturated liquid) for realistic column operation
    from scipy.optimize import brentq
    from chemsim.ops.distillation import _wilson_K_i
    z_dict = {"METHANE": 0.5, "ETHANE": 0.3, "PROPANE": 0.2}
    z = [z_dict[c.id] for c in comps]
    def bubble_T(x, P):
        def f(T):
            return sum(_wilson_K_i(comps[i], T, P) * x[i] for i in range(len(comps))) - 1.0
        return brentq(f, 100.0, 500.0)
    T_bubble = bubble_T(z, 2e6)
    feed = make_stream(fc, comps, T_bubble, 2e6, 100.0, z_dict)
    col = DistillationColumnOp(fc, comps, 10, 5, 2.0, 0.4, 2e6)
    col.feed = feed; col.solve()
    assert col.reboiler_duty() > 0.0      # reboiler adds heat
    assert col.condenser_duty() < 0.0    # condenser removes heat


# ── Recycle Solver ────────────────────────────────────────────────────────────

def test_recycle_solver_simple(db_path):
    """Simple mixer-splitter recycle loop converges."""
    from chemsim.flowsheet.flowsheet import Flowsheet
    from pathlib import Path
    json_path = str(Path(db_path).parent.parent / "examples" / "simple_recycle.json")

    fs = Flowsheet.from_json(json_path, db_path)
    ok = fs.solve()
    assert ok, "Simple recycle flowsheet should converge"

    # Feed mass balance: PRODUCT + RECYCLE should account for all FEED
    feed = fs.get_stream("FEED")
    prod = fs.get_stream("PRODUCT")
    # All flow eventually exits as PRODUCT (splitter fractions 0.2 out, 0.8 recycle)
    assert prod.total_flow > 0.0
