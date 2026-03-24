# ChemSim — Pure Python Process Simulator

## ✅ No C++ Dependencies

ChemSim is a **100% pure Python** steady-state process simulator. No C++ code, no compiled extensions, no pybind11.

## Project Structure

```
Process-Simulator/
├── chemsim/                    # Core simulator (pure Python)
│   ├── __init__.py
│   ├── core.py                 # Stream, Component, Phase classes
│   ├── flowsheet/
│   │   ├── flowsheet.py        # Top-level orchestrator
│   │   ├── graph.py            # Graph-based flowsheet
│   │   └── recycle.py          # Recycle solver
│   ├── ops/                    # Unit operations
│   │   ├── base.py             # IUnitOp interface
│   │   ├── distillation.py     # DistillationColumnOp
│   │   ├── flash_drum.py       # FlashDrumOp
│   │   ├── pump.py             # PumpOp
│   │   ├── heat_exchanger.py   # HeatExchangerOp
│   │   └── reactor.py          # ReactorOp
│   └── thermo/                 # Thermodynamics
│       ├── peng_robinson.py    # Peng-Robinson EOS
│       └── flash.py            # Flash calculator
├── chemsim_gym/                # RL environment (pure Python)
│   ├── env.py                  # Gymnasium environment
│   ├── simulator_bridge.py     # Simulator interface
│   ├── reward.py               # Reward functions
│   ├── spaces.py               # Action/observation spaces
│   └── wrappers.py             # Normalization
├── training/                   # RL training
│   ├── train.py                # PPO training script
│   ├── evaluate.py             # Evaluation script
│   └── config/default.yaml     # Hyperparameters
├── analysis/                   # Analysis tools
│   └── reward_landscape.py     # Grid scan visualization
├── tests/                      # Test suite
│   ├── test_phase1.py          # ComponentDB, PR EOS
│   ├── test_phase2.py          # Flash calculations
│   ├── test_phase3.py          # Unit operations
│   └── test_phase5.py          # Flowsheets
├── examples/                   # Example flowsheets
│   ├── distillation_recycle.json
│   └── simple_recycle.json
└── data/                       # Component databases
    ├── components.json         # Main database (13 components)
    └── components_handbook_candidates.json
```

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn pyyaml
pip install gymnasium stable-baselines3 tensorboard  # For RL

# Install chemsim (optional, as package)
pip install -e .
```

## Quick Start

### Basic Simulation

```python
from chemsim import Flowsheet

# Load from JSON
fs = Flowsheet.from_json(
    "examples/simple_recycle.json",
    "data/components.json"
)
fs.solve()
print(fs.summary())

# Or create programmatically
fs2 = Flowsheet.create(["METHANE", "ETHANE"], db_path="data/components.json")
fs2.add_stream("FEED", T=250.0, P=2e6, flow=100.0,
               composition={"METHANE": 0.6, "ETHANE": 0.4})
```

### RL Environment

```python
from chemsim_gym.env import ProcessSimEnv

env = ProcessSimEnv(
    flowsheet_json="examples/distillation_recycle.json",
    component_db="data/components.json",
)

obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward:.3f}")
print(f"Purity: {info['distillate_purity']:.3f}")
print(f"Energy: {info['reboiler_duty_kW']:.1f} kW")
```

### Train RL Agent

```bash
# Grid scan first (understand reward landscape)
PYTHONPATH=. python analysis/reward_landscape.py --n-reflux 20 --n-dist 20

# Train PPO agent
PYTHONPATH=. python training/train.py --config training/config/default.yaml

# Evaluate trained agent
PYTHONPATH=. python training/evaluate.py --model checkpoints/ppo_baseline/final_model.zip
```

## Available Components

The `data/components.json` database includes 13 components:

| Component | Formula | MW | Tc (K) | Pc (MPa) |
|-----------|---------|-----|--------|----------|
| Methane | CH4 | 16.04 | 190.6 | 4.60 |
| Ethane | C2H6 | 30.07 | 305.3 | 4.87 |
| Propane | C3H8 | 44.10 | 369.8 | 4.25 |
| n-Butane | C4H10 | 58.12 | 425.1 | 3.80 |
| n-Pentane | C5H12 | 72.15 | 469.7 | 3.37 |
| CO2 | CO2 | 44.01 | 304.2 | 7.38 |
| Water | H2O | 18.02 | 647.1 | 22.06 |
| Toluene | C7H8 | 92.14 | 591.8 | 4.11 |
| Benzene | C6H6 | 78.11 | 562.1 | 4.90 |
| **Aniline** | C6H5NH2 | 93.13 | 699.0 | 5.30 |
| **Acetone** | C3H6O | 58.08 | 508.1 | 4.69 |
| **Ethanol** | C2H5OH | 46.07 | 514.0 | 6.14 |
| **n-Hexane** | C6H14 | 86.18 | 507.6 | 3.02 |

**Bold** = Recently added from NIST/Perry's Handbook

## Unit Operations

| Unit | Description |
|------|-------------|
| FlashDrum | TP/PH/PS flash calculations |
| Pump | Liquid pressure increase |
| HeatExchanger | Two-stream heat exchange |
| Reactor | Stoichiometric reactor |
| DistillationColumn | Multi-stage distillation (Wang-Henke) |
| Mixer | Stream mixing |
| Splitter | Stream splitting |

## Thermodynamics

- **Equation of State**: Peng-Robinson
- **Flash Calculations**: TP, PH, PS, bubble/dew point
- **Stability Analysis**: Michelsen stability test
- **Properties**: Enthalpy, entropy, fugacity, K-values

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific phase
python -m pytest tests/test_phase3.py -v  # Unit operations
```

**Test Status**: ✅ 40/40 tests passing

## Performance

- Flash calculation: ~1-5 ms
- Distillation column (10 stages): ~10-50 ms
- Full flowsheet with recycle: ~50-200 ms
- RL training (500k steps, 4 envs): ~2 hours

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
1. All tests pass (`pytest tests/`)
2. New components include critical properties (Tc, Pc, ω)
3. New unit operations have test coverage
