"""Phase 2: Ideal-gas thermo, flash calculations, stability."""
import math
import pytest

from chemsim.io.component_db import ComponentDB
from chemsim.thermo.peng_robinson import PengRobinson
from chemsim.thermo.flash import FlashCalculator
from chemsim.core import R, ideal_gas_H, ideal_gas_S


def make_flash(db_path, ids):
    db = ComponentDB(db_path)
    comps = db.get(ids)
    pr = PengRobinson(comps)
    return FlashCalculator(pr, comps), comps


def test_ideal_gas_H_reference(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE"])
    # H at T_ref = 298.15 K should be ~0
    H = ideal_gas_H(comps, [1.0], 298.15)
    assert abs(H) < 1e-6


def test_ideal_gas_H_increases_with_T(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE", "ETHANE"])
    z = [0.5, 0.5]
    H1 = ideal_gas_H(comps, z, 300.0)
    H2 = ideal_gas_H(comps, z, 500.0)
    assert H2 > H1


def test_ideal_gas_S_reference(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE"])
    # S contribution from integral alone (no mixing) at T_ref should be ~0
    from chemsim.core import ideal_gas_S_pure
    S = ideal_gas_S_pure(comps[0], 298.15)
    assert abs(S) < 1e-6


def test_flash_TP_two_phase(db_path):
    fc, comps = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    T = 260.0; P = 2e6
    r = fc.flash_TP(T, P, z)

    assert r.converged
    assert 0.0 < r.beta < 1.0, "Should be two-phase at 260K, 2MPa"

    # Material balance: z_i = beta*y_i + (1-beta)*x_i
    for i in range(len(z)):
        z_calc = r.beta * r.y[i] + (1.0 - r.beta) * r.x[i]
        assert abs(z_calc - z[i]) < 1e-6

    # Mole fraction closure
    assert abs(sum(r.x) - 1.0) < 1e-6
    assert abs(sum(r.y) - 1.0) < 1e-6


def test_flash_TP_light_components_in_vapor(db_path):
    fc, comps = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    r = fc.flash_TP(260.0, 2e6, z)
    # Light components should be enriched in vapour
    assert r.y[0] > r.x[0]   # methane enriched in vapor
    assert r.x[3] > r.y[3]   # n-butane enriched in liquid


def test_fugacity_equality_at_equilibrium(db_path):
    fc, comps = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    r = fc.flash_TP(260.0, 2e6, z)

    pr = fc.eos
    lnPhiL = pr.ln_fugacity_coefficients(r.T, r.P, r.x, True)
    lnPhiV = pr.ln_fugacity_coefficients(r.T, r.P, r.y, False)

    nc = len(z)
    for i in range(nc):
        # f_L_i = x_i * phi_L_i * P  =  f_V_i = y_i * phi_V_i * P
        ln_fL = math.log(r.x[i]) + lnPhiL[i]
        ln_fV = math.log(r.y[i]) + lnPhiV[i]
        assert abs(ln_fL - ln_fV) < 1e-5


def test_flash_all_vapor(db_path):
    fc, _ = make_flash(db_path, ["METHANE"])
    r = fc.flash_TP(600.0, 1e5, [1.0])
    assert r.beta == 1.0 or r.beta > 0.99


def test_flash_all_liquid(db_path):
    fc, _ = make_flash(db_path, ["N-BUTANE"])
    r = fc.flash_TP(250.0, 5e6, [1.0])
    assert r.beta < 0.01


def test_stability_two_phase_feed(db_path):
    fc, _ = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    assert not fc.is_stable(260.0, 2e6, z)   # two-phase → unstable


def test_stability_single_phase_vapor(db_path):
    fc, _ = make_flash(db_path, ["METHANE"])
    assert fc.is_stable(600.0, 1e5, [1.0])   # supercritical gas → stable


def test_total_enthalpy(db_path):
    fc, _ = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    r = fc.flash_TP(260.0, 2e6, z)
    H = fc.total_enthalpy(r)
    assert math.isfinite(H)


def test_flash_PH_round_trip(db_path):
    fc, _ = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    T0 = 260.0; P = 2e6
    r0 = fc.flash_TP(T0, P, z)
    H0 = fc.total_enthalpy(r0)
    r1 = fc.flash_PH(P, H0, z, T0)
    assert abs(r1.T - T0) < 0.5


def test_flash_PS_round_trip(db_path):
    fc, _ = make_flash(db_path, ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    z = [0.4, 0.3, 0.2, 0.1]
    T0 = 260.0; P = 2e6
    r0 = fc.flash_TP(T0, P, z)
    S0 = fc.total_entropy(r0)
    r1 = fc.flash_PS(P, S0, z, T0)
    assert abs(r1.T - T0) < 0.5
