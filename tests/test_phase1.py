"""Phase 1: ComponentDB, Brent solver, Peng-Robinson EOS basics."""
import math
import pytest
from scipy.optimize import brentq

from chemsim.io.component_db import ComponentDB
from chemsim.thermo.peng_robinson import PengRobinson
from chemsim.core import R


def test_component_db_loads(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE", "ETHANE", "PROPANE"])
    assert len(comps) == 3
    assert comps[0].id == "METHANE"
    assert comps[1].id == "ETHANE"
    assert comps[2].id == "PROPANE"


def test_component_properties(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE"])
    c = comps[0]
    assert abs(c.MW    - 16.043) < 1e-3
    assert abs(c.Tc    - 190.564) < 1e-3
    assert abs(c.Pc    - 4599200.0) < 1.0
    assert abs(c.omega - 0.0115) < 1e-4


def test_brent_quadratic():
    # x² - 4 = 0 → x = 2
    root = brentq(lambda x: x**2 - 4.0, 1.0, 3.0)
    assert abs(root - 2.0) < 1e-8


def test_brent_cubic():
    # x³ - 6x² + 11x - 6 = 0 → roots at 1, 2, 3
    root = brentq(lambda x: x**3 - 6*x**2 + 11*x - 6, 0.5, 1.5)
    assert abs(root - 1.0) < 1e-8
    root = brentq(lambda x: x**3 - 6*x**2 + 11*x - 6, 1.5, 2.5)
    assert abs(root - 2.0) < 1e-8


def test_pr_z_factors_methane(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE"])
    pr = PengRobinson(comps)

    T = 300.0    # K, above Tc
    P = 1e6      # 10 bar
    z = [1.0]
    Z_L, Z_V = pr.compressibility_factors(T, P, z)

    # Single-component above Tc → both roots should be the same (one root)
    # At T=300K, P=1MPa methane is a supercritical gas
    assert Z_V > 0.8   # close to 1 for moderate pressure gas
    assert Z_L > 0.0


def test_pr_z_factors_high_pressure(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE", "ETHANE", "PROPANE", "N-BUTANE"])
    pr = PengRobinson(comps)

    T = 260.0; P = 2e6
    z = [0.4, 0.3, 0.2, 0.1]
    Z_L, Z_V = pr.compressibility_factors(T, P, z)
    assert Z_L > 0.0
    assert Z_V >= Z_L


def test_pr_ideal_gas_limit(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE"])
    pr = PengRobinson(comps)

    T = 600.0; P = 100.0   # very low pressure → Z → 1
    z = [1.0]
    Z_L, Z_V = pr.compressibility_factors(T, P, z)
    assert abs(Z_V - 1.0) < 0.01


def test_pr_fugacity_single_phase(db_path):
    db = ComponentDB(db_path)
    comps = db.get(["METHANE"])
    pr = PengRobinson(comps)

    T = 200.0; P = 5e6; z = [1.0]
    lnphi_L = pr.ln_fugacity_coefficients(T, P, z, True)
    lnphi_V = pr.ln_fugacity_coefficients(T, P, z, False)
    assert len(lnphi_L) == 1
    assert len(lnphi_V) == 1
    # Both should be finite
    assert math.isfinite(lnphi_L[0])
    assert math.isfinite(lnphi_V[0])
