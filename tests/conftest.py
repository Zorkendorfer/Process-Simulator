"""Shared fixtures for ChemSim tests."""
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB   = str(REPO / "data" / "components.json")


@pytest.fixture
def db_path():
    return DB


@pytest.fixture
def component_ids_4():
    return ["METHANE", "ETHANE", "PROPANE", "N-BUTANE"]


@pytest.fixture
def component_ids_3():
    return ["METHANE", "ETHANE", "PROPANE"]
