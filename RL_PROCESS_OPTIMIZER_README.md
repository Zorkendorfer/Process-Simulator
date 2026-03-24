# RL Process Optimizer

Reinforcement Learning-based process optimization for the ChemSim process simulator.

## Overview

This package provides a Gymnasium environment wrapping the **pure Python** ChemSim distillation simulator, along with training scripts for optimizing operating conditions using PPO (Proximal Policy Optimization).

**✅ Pure Python**: No C++ dependencies. The entire simulator and RL environment run in Python.

**Goal**: Train an RL agent to find optimal operating conditions for a distillation-recycle system that:
- Maximizes product purity (distillate composition)
- Minimizes energy consumption (reboiler duty)
- Maintains product flow constraints

## Quick Start

### 1. Environment Sanity Check

```bash
cd /Users/jonas/ChemSim/Process-Simulator
PYTHONPATH=. .venv-mac-test/bin/python -c "
from chemsim_gym.env import ProcessSimEnv
env = ProcessSimEnv(
    flowsheet_json='examples/distillation_recycle.json',
    component_db='data/components.json',
)
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, _, _, info = env.step(action)
print(f'Reward: {reward:.3f}, Purity: {info[\"distillate_purity\"]:.3f}')
"
```

### 2. Grid Scan Analysis (Recommended First)

Before training, understand the reward landscape:

```bash
PYTHONPATH=. .venv-mac-test/bin/python analysis/reward_landscape.py \
    --n-reflux 20 --n-dist 20 \
    --output-dir analysis_output
```

This generates:
- `reward_landscape.png` - Contour plots of purity, energy, reward
- `pareto_front.png` - Pareto optimal operating points
- `grid_scan_results.json` - Raw data for analysis

### 3. Training PPO Agent

```bash
PYTHONPATH=. .venv-mac-test/bin/python training/train.py \
    --config training/config/default.yaml
```

Monitor training in TensorBoard:
```bash
tensorboard --logdir runs/ppo_baseline
```

### 4. Evaluate Trained Agent

```bash
PYTHONPATH=. .venv-mac-test/bin/python training/evaluate.py \
    --model checkpoints/ppo_baseline/final_model.zip \
    --n-episodes 100 \
    --output-dir evaluation_output
```

## Environment Details

### Action Space (4 continuous variables, normalized to [-1, 1])

| Action | Physical Range | Description |
|--------|---------------|-------------|
| 0 | 0.5 – 10.0 | Reflux ratio (L/D) |
| 1 | 0.1 – 0.9 | Distillate fraction (D/F) |
| 2 | 280 – 400 K | Feed temperature |
| 3 | 0.5 – 3.0 MPa | Feed pressure |

### Observation Space (12 dimensions for 3-component system)

| Observation | Dimensions | Description |
|-------------|------------|-------------|
| distillate_z | NC | Distillate mole fractions |
| bottoms_z | NC | Bottoms mole fractions |
| T_top, T_mid, T_bottom | 3 | Column temperatures (K) |
| reboiler_duty_MW | 1 | Reboiler duty (MW) |
| condenser_duty_MW | 1 | Condenser duty (MW) |
| converged | 1 | Convergence flag (0/1) |

### Reward Function

```
reward = w_purity * purity_reward 
       - w_energy * (reboiler_duty_MW)
       + w_flow * flow_reward
       + w_crash (if not converged)
```

Default weights:
- `w_purity = 10.0` (reward for meeting 95% purity target)
- `w_energy = 1.0` (penalty per MW of reboiler duty)
- `w_flow = 2.0` (reward for maintaining flow ≥10 mol/s)
- `w_crash = -20.0` (large penalty for convergence failure)

## Configuration

Edit `training/config/default.yaml` to customize:

```yaml
# Reward tuning
reward:
  target_purity: 0.95      # Increase for higher purity requirements
  w_energy: 2.0            # Increase to penalize energy more
  purity_reward_type: "smooth"  # or "step" or "exponential"

# Training hyperparameters
total_timesteps: 500_000   # Increase for better convergence
n_envs: 8                  # More parallel envs = faster training
ent_coef: 0.02             # Increase for more exploration
```

## Project Structure

```
Process-Simulator/
├── chemsim_gym/
│   ├── env.py                  # Gymnasium environment
│   ├── simulator_bridge.py     # Python↔ChemSim bridge
│   ├── reward.py               # Reward function
│   ├── spaces.py               # Action/observation spaces
│   └── wrappers.py             # Normalization wrappers
├── training/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── config/
│       └── default.yaml        # Default hyperparameters
├── analysis/
│   └── reward_landscape.py     # Grid scan visualization
├── examples/
│   └── distillation_recycle.json  # Example flowsheet
└── data/
    └── components.json         # Component database
```

## Grid Scan Results (Sample)

Running a 15×15 grid scan (225 operating points):

```
Convergence rate: 100.0%
Purity range: [0.557, 1.000]
Energy range: [-1319, 9584] kW
Reward range: [-4.92, 12.47]

Best operating point:
  Reflux ratio: 0.50
  Distillate frac: 0.44
  Purity: 1.000
  Energy: -609 kW (net cooling)
  Reward: 12.47
```

**Insight**: Low reflux ratio with moderate distillate fraction achieves perfect separation with minimal energy—this is the optimal region the RL agent should discover.

## Troubleshooting

### Environment crashes on reset
- Check that `examples/distillation_recycle.json` exists
- Verify `data/components.json` has required components (METHANE, ETHANE, PROPANE)

### Agent doesn't converge
- Increase `ent_coef` to 0.02–0.05 for more exploration
- Increase `total_timesteps` to 1M+
- Check reward landscape—optimum may be in narrow region

### Low convergence rate
- Reduce action space bounds (especially reflux ratio)
- Increase `w_crash` penalty magnitude
- Use curriculum learning (start with easy targets)

## Advanced: Multi-Objective Optimization

To explore the Pareto front between purity and energy:

```bash
# Train multiple agents with different weight ratios
for w_energy in 0.5 1.0 2.0 5.0; do
    PYTHONPATH=. .venv-mac-test/bin/python training/train.py \
        --config training/config/default.yaml \
        --total-timesteps 200000
    # (Modify config to change w_energy between runs)
done

# Compare Pareto fronts
PYTHONPATH=. .venv-mac-test/bin/python analysis/pareto_front.py \
    --results-dir checkpoints/
```

## References

1. Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. Brockman et al. "OpenAI Gym" (2016)
3. Raffin et al. "Stable-Baselines3: Reliable Reinforcement Learning Implementations" (2021)
