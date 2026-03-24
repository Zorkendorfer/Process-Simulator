# RL Process Optimizer — Architecture & Implementation Plan

## Concept

Wrap the **pure Python ChemSim simulator** as an OpenAI Gym-compatible RL environment.
Train a PPO agent to find optimal operating conditions for a distillation-recycle system:
- Maximize product purity
- Minimize reboiler duty (energy cost)
- Satisfy product flow constraints

The agent manipulates: reflux ratio, reboiler duty, feed temperature, feed pressure.
It observes: product compositions, temperatures, duties, convergence state.

**Status: ✅ IMPLEMENTED** — Pure Python implementation complete.

---

## Repository Structure (Updated)

```
Process-Simulator/          ← your existing pure-Python simulator
├── chemsim/                    ← Pure Python simulator (no C++)
│   ├── __init__.py
│   ├── core.py                 ← Stream, Component, Phase
│   ├── flowsheet/
│   │   ├── flowsheet.py        ← Top-level orchestrator
│   │   ├── graph.py            ← Graph-based flowsheet
│   │   └── recycle.py          ← Recycle solver
│   ├── ops/
│   │   ├── base.py             ← IUnitOp interface
│   │   ├── distillation.py     ← DistillationColumnOp
│   │   ├── flash_drum.py
│   │   ├── pump.py
│   │   └── ...
│   └── thermo/
│       ├── peng_robinson.py    ← PR EOS
│       └── flash.py            ← Flash calculator
├── chemsim_gym/                ← RL environment (pure Python)
│   ├── __init__.py
│   ├── env.py                  ← Gym environment wrapper
│   ├── simulator_bridge.py     ← Python simulator interface
│   ├── reward.py               ← Reward function definitions
│   ├── spaces.py               ← Action/observation spaces
│   └── wrappers.py             ← Normalization wrappers
├── training/
│   ├── train.py                ← PPO training script
│   ├── evaluate.py             ← Evaluation + plotting
│   └── config/
│       └── default.yaml        ← Hyperparameters
├── analysis/
│   └── reward_landscape.py     ← Grid scan visualization
├── examples/
│   └── distillation_recycle.json
└── data/
    └── components.json
```

---

## The Environment

### Process: Distillation + Recycle

```
FEED ──→ [COLUMN] ──→ DISTILLATE (high-purity light product)
              │
              └──→ BOTTOMS ──→ [REACTOR] ──→ [RECYCLE SPLITTER]
                                                   │
                                    PURGE ←────────┤
                                                   │
                              FEED ←──────────────┘ (recycle back)
```

The agent controls 4 continuous action variables:
1. Reflux ratio (L/D): range [0.5, 10.0]
2. Distillate flow fraction (D/F): range [0.1, 0.9]
3. Feed temperature (K): range [280, 400]
4. Feed pressure (Pa): range [500k, 3M]

It observes 12 state variables:
- Distillate composition (NC values)
- Bottoms composition (NC values)
- Column temperatures (top, mid, bottom)
- Reboiler duty [W]
- Condenser duty [W]
- Convergence flag (1 = converged, 0 = failed)

---

## Class Design

### Environment

```python
# chemsim_gym/env.py

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .simulator_bridge import SimulatorBridge
from .reward import RewardFunction
from .spaces import ProcessSpaces

class ProcessSimEnv(gym.Env):
    """
    OpenAI Gym environment wrapping the ChemSim C++ process simulator.
    
    Episodic: each episode = one operating point query.
    The agent proposes operating conditions, the simulator evaluates them,
    the reward is returned based on product specs and energy consumption.
    
    Technically this is a bandit (1-step) environment at the simplest level,
    but structured as episodic to allow multi-step refinement mode.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        simulator_path: str,          # path to chemsim binary/lib
        flowsheet_config: str,         # JSON flowsheet definition
        reward_config: dict,           # reward weights and targets
        max_steps: int = 10,           # steps per episode (1 = bandit mode)
        normalize_obs: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.sim = SimulatorBridge(simulator_path, flowsheet_config)
        self.reward_fn = RewardFunction(**reward_config)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._step = 0
        
        spaces = ProcessSpaces(self.sim.nComponents(), self.sim.nStages())
        self.action_space = spaces.action_space()
        self.observation_space = spaces.observation_space()
        
        # Running stats for observation normalization
        self._obs_mean = np.zeros(self.observation_space.shape)
        self._obs_var  = np.ones(self.observation_space.shape)
        self._obs_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step = 0
        
        # Reset simulator to base state
        self.sim.reset()
        
        # Get initial observation (nominal operating point)
        obs = self._get_obs()
        info = {"converged": True, "step": 0}
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        action: numpy array of shape (4,) — [reflux, dist_frac, T_feed, P_feed]
        """
        self._step += 1
        
        # Denormalize action from [-1, 1] to physical units
        action_physical = self._denormalize_action(action)
        
        # Run simulator
        sim_result = self.sim.run(
            reflux_ratio    = action_physical[0],
            distillate_frac = action_physical[1],
            T_feed          = action_physical[2],
            P_feed          = action_physical[3],
        )
        
        # Build observation
        obs = self._build_obs(sim_result)
        
        # Compute reward
        reward, reward_info = self.reward_fn.compute(sim_result, action_physical)
        
        # Episode ends after max_steps or on convergence failure
        terminated = (self._step >= self.max_steps)
        truncated = not sim_result.converged  # penalize and end on divergence
        
        info = {
            "converged": sim_result.converged,
            "distillate_purity": sim_result.distillate_purity,
            "reboiler_duty_kW": sim_result.reboiler_duty / 1000,
            "reward_breakdown": reward_info,
            "action_physical": action_physical,
            "step": self._step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            return self.sim.summary()
    
    def _get_obs(self) -> np.ndarray:
        result = self.sim.currentState()
        return self._build_obs(result)
    
    def _build_obs(self, result) -> np.ndarray:
        obs = np.concatenate([
            result.distillate_z,       # NC values
            result.bottoms_z,          # NC values
            [result.T_top,
             result.T_mid,
             result.T_bottom,
             result.reboiler_duty / 1e6,   # normalize to MW
             result.condenser_duty / 1e6,
             float(result.converged)],
        ])
        return obs.astype(np.float32)
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map from [-1, 1] to physical bounds."""
        low  = np.array([0.5, 0.1, 280.0, 5e5])
        high = np.array([10.0, 0.9, 400.0, 3e6])
        return low + (action + 1.0) / 2.0 * (high - low)
```

---

### Simulator Bridge

```python
# chemsim_gym/simulator_bridge.py
# Thin wrapper around the pybind11 chemsim module

import numpy as np
from dataclasses import dataclass
from typing import Optional
import chemsim  # your pybind11 module

@dataclass
class SimResult:
    converged: bool
    distillate_z: np.ndarray    # mole fractions
    bottoms_z: np.ndarray
    T_top: float                # K
    T_mid: float                # K
    T_bottom: float             # K
    reboiler_duty: float        # W
    condenser_duty: float       # W
    distillate_flow: float      # mol/s
    bottoms_flow: float         # mol/s
    
    @property
    def distillate_purity(self) -> float:
        """Mole fraction of target component in distillate."""
        return float(self.distillate_z[0])  # assume comp 0 is target
    
    @property
    def specific_energy(self) -> float:
        """Reboiler duty per mol of distillate [J/mol]."""
        if self.distillate_flow < 1e-10:
            return 1e10  # penalize zero flow
        return self.reboiler_duty / self.distillate_flow


class SimulatorBridge:
    def __init__(self, simulator_path: str, flowsheet_config: str):
        self.flowsheet = chemsim.Flowsheet.fromJSON(flowsheet_config)
        self._nc = self._get_nc()
        self._current: Optional[SimResult] = None
    
    def reset(self):
        """Reload flowsheet from config (reset to nominal state)."""
        self.flowsheet = chemsim.Flowsheet.fromJSON(self._config_path)
    
    def run(
        self,
        reflux_ratio: float,
        distillate_frac: float,
        T_feed: float,
        P_feed: float,
    ) -> SimResult:
        """
        Set operating conditions and run one steady-state solve.
        Returns SimResult with all relevant stream data.
        """
        try:
            self.flowsheet.setParam("COL1", "refluxRatio", reflux_ratio)
            self.flowsheet.setParam("COL1", "distillateFrac", distillate_frac)
            self.flowsheet.setStream("FEED", T=T_feed, P=P_feed)
            
            converged = self.flowsheet.solve()
            
            if not converged:
                return self._failed_result()
            
            dist = self.flowsheet.getStream("DISTILLATE")
            btms = self.flowsheet.getStream("BOTTOMS")
            col  = self.flowsheet.getUnit("COL1")
            
            result = SimResult(
                converged       = True,
                distillate_z    = np.array(dist.z),
                bottoms_z       = np.array(btms.z),
                T_top           = col.T_top(),
                T_mid           = col.T_mid(),
                T_bottom        = col.T_bottom(),
                reboiler_duty   = col.reboilerDuty(),
                condenser_duty  = col.condenserDuty(),
                distillate_flow = dist.totalFlow,
                bottoms_flow    = btms.totalFlow,
            )
            self._current = result
            return result
        
        except Exception as e:
            return self._failed_result()
    
    def _failed_result(self) -> SimResult:
        nc = self._nc
        return SimResult(
            converged=False,
            distillate_z=np.ones(nc)/nc,
            bottoms_z=np.ones(nc)/nc,
            T_top=300.0, T_mid=300.0, T_bottom=300.0,
            reboiler_duty=1e8,
            condenser_duty=1e8,
            distillate_flow=0.0,
            bottoms_flow=0.0,
        )
    
    def nComponents(self) -> int:
        return self._nc
    
    def _get_nc(self) -> int:
        return len(self.flowsheet.getStream("FEED").z)
    
    def currentState(self) -> SimResult:
        if self._current is None:
            return self._failed_result()
        return self._current
    
    def summary(self) -> str:
        return self.flowsheet.summary()
```

---

### Reward Function

```python
# chemsim_gym/reward.py
# This is the most important design decision in the whole project.
# Reward shaping determines what the agent actually learns.

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class RewardConfig:
    # Target specifications
    target_purity: float = 0.95         # mole fraction of key component in distillate
    min_distillate_flow: float = 10.0   # mol/s minimum acceptable distillate flow
    
    # Reward weights
    w_purity: float    = 10.0   # reward for meeting purity spec
    w_energy: float    = 1.0    # penalty for energy use (per MW reboiler duty)
    w_flow:   float    = 2.0    # reward for maintaining distillate flow
    w_crash:  float    = -20.0  # penalty for convergence failure
    
    # Purity reward shape: linear or step
    purity_reward_type: str = "smooth"  # "step" | "smooth" | "exponential"


class RewardFunction:
    def __init__(self, **kwargs):
        self.cfg = RewardConfig(**kwargs)
    
    def compute(
        self,
        result,
        action: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Returns (total_reward, breakdown_dict).
        
        Reward design notes:
        - Purity reward uses a smooth sigmoid to avoid sharp gradient cliffs
        - Energy penalty scales with reboiler duty to encourage efficiency
        - Flow reward prevents the agent from 'cheating' with tiny distillate flows
        - Convergence penalty is large enough to deter exploration of infeasible regions
        """
        cfg = self.cfg
        
        if not result.converged:
            return cfg.w_crash, {"crash": cfg.w_crash}
        
        # 1. Purity reward
        purity = result.distillate_purity
        r_purity = cfg.w_purity * self._purity_reward(purity, cfg.target_purity)
        
        # 2. Energy penalty (normalized to MW)
        energy_MW = result.reboiler_duty / 1e6
        r_energy = -cfg.w_energy * energy_MW
        
        # 3. Flow reward (log scale to avoid unbounded growth)
        flow = result.distillate_flow
        if flow < cfg.min_distillate_flow:
            r_flow = -cfg.w_flow * (cfg.min_distillate_flow - flow) / cfg.min_distillate_flow
        else:
            r_flow = cfg.w_flow * np.log(flow / cfg.min_distillate_flow + 1)
        
        total = r_purity + r_energy + r_flow
        breakdown = {
            "purity":  round(r_purity, 4),
            "energy":  round(r_energy, 4),
            "flow":    round(r_flow, 4),
            "total":   round(total, 4),
        }
        return total, breakdown
    
    def _purity_reward(self, purity: float, target: float) -> float:
        """
        Smooth sigmoid reward centered on target purity.
        Gives partial credit for near-spec operation.
        Returns value in [0, 1].
        """
        if self.cfg.purity_reward_type == "step":
            return 1.0 if purity >= target else 0.0
        
        elif self.cfg.purity_reward_type == "smooth":
            # Sigmoid centered at target, steep slope
            k = 30.0  # steepness — tune this
            return 1.0 / (1.0 + np.exp(-k * (purity - target)))
        
        elif self.cfg.purity_reward_type == "exponential":
            # Exponential decay from target
            return np.exp(-10.0 * max(0, target - purity))
        
        return 0.0
```

---

### Training Script

```python
# training/train.py

import yaml
import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from chemsim_gym.env import ProcessSimEnv
from chemsim_gym.wrappers import NormalizeObservation, LoggingWrapper


def make_env(config: dict, rank: int = 0):
    def _init():
        env = ProcessSimEnv(
            simulator_path  = config["simulator_path"],
            flowsheet_config= config["flowsheet_config"],
            reward_config   = config["reward"],
            max_steps       = config["max_steps"],
        )
        env = NormalizeObservation(env)
        env = LoggingWrapper(env, log_dir=config["log_dir"])
        env = Monitor(env)
        return env
    return _init


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Vectorized environments for parallelism
    n_envs = cfg.get("n_envs", 4)
    env = make_vec_env(make_env(cfg), n_envs=n_envs)
    eval_env = make_vec_env(make_env(cfg), n_envs=1)
    
    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg["checkpoint_dir"],
        log_path=cfg["log_dir"],
        eval_freq=cfg.get("eval_freq", 5000),
        n_eval_episodes=20,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=cfg.get("save_freq", 10000),
        save_path=cfg["checkpoint_dir"],
        name_prefix="ppo_chemsim",
    )
    
    # PPO agent — good default for continuous action spaces
    model = PPO(
        policy      = "MlpPolicy",
        env         = env,
        learning_rate       = cfg.get("lr", 3e-4),
        n_steps             = cfg.get("n_steps", 2048),
        batch_size          = cfg.get("batch_size", 64),
        n_epochs            = cfg.get("n_epochs", 10),
        gamma               = cfg.get("gamma", 0.99),
        gae_lambda          = cfg.get("gae_lambda", 0.95),
        clip_range          = cfg.get("clip_range", 0.2),
        ent_coef            = cfg.get("ent_coef", 0.01),  # encourage exploration
        verbose             = 1,
        tensorboard_log     = cfg["log_dir"],
        policy_kwargs       = dict(
            net_arch = [dict(pi=[256, 256], vf=[256, 256])]
        ),
    )
    
    print(f"Training PPO on {n_envs} parallel environments")
    print(f"Total timesteps: {cfg['total_timesteps']:,}")
    
    model.learn(
        total_timesteps = cfg["total_timesteps"],
        callback        = CallbackList([eval_cb, ckpt_cb]),
        progress_bar    = True,
    )
    
    model.save(Path(cfg["checkpoint_dir"]) / "final_model")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config/default.yaml")
    args = parser.parse_args()
    train(args.config)
```

---

### Default Config

```yaml
# training/config/default.yaml

simulator_path: "build/libchemsim.so"
flowsheet_config: "examples/distillation_recycle.json"
log_dir: "runs/ppo_baseline"
checkpoint_dir: "checkpoints/ppo_baseline"

# Environment
max_steps: 1          # bandit mode: 1 step per episode

# Reward
reward:
  target_purity: 0.95
  min_distillate_flow: 10.0
  w_purity: 10.0
  w_energy: 1.0
  w_flow: 2.0
  w_crash: -20.0
  purity_reward_type: "smooth"

# Training
total_timesteps: 500_000
n_envs: 4
lr: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01

eval_freq: 5000
save_freq: 20000
```

---

## Implementation Phases

### Phase 1 — Environment Foundation (Week 1–2)
Goal: A working Gym environment that wraps the simulator without crashing.

1. Add `setParam()` and `setStream()` methods to ChemSim's `Flowsheet` class (C++)
2. Rebuild pybind11 bindings to expose these
3. Implement `SimulatorBridge` — test that you can call `run()` and get a valid `SimResult`
4. Implement `ProcessSimEnv` skeleton — `reset()`, `step()`, `action_space`, `observation_space`
5. Sanity check with `gymnasium.utils.check_env(env)` — this will catch most interface bugs

**Test:**
```python
env = ProcessSimEnv(...)
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward={reward:.3f} converged={info['converged']}")
```

---

### Phase 2 — Reward Engineering (Week 3)
Goal: A reward signal that trains correctly — this is harder than it looks.

Critical experiments to run *before* training the agent:

1. **Grid scan** — exhaustive search over a 20×20×10×10 grid of the action space using the simulator. Plot purity vs reflux ratio, energy vs distillate fraction. Understand the landscape.
2. **Reward landscape** — compute reward for every grid point. Check: is the reward smooth? Are there cliffs? Does the optimum sit in a small narrow region or a broad plateau?
3. **Baseline comparisons** — what reward does a random policy get? What does the "textbook" operating point get? These set the floor/ceiling for your training curves.

Key failure modes to avoid:
- Reward cliff at purity target → sigmoid reward, not step
- Agent finds purity by making distillate flow zero → flow reward required
- Agent drives reflux ratio to infinity (infinite energy, perfect purity) → strong energy penalty

---

### Phase 3 — Training PPO (Week 4–5)
Goal: Agent beats random policy and approaches optimal operating point.

1. Install `stable-baselines3` — don't implement PPO from scratch yet
2. Run Phase 2 grid scan to set hyperparameter ranges for curriculum
3. Train baseline model with default config
4. Monitor in TensorBoard: episode reward, purity, energy, convergence rate
5. Tune `ent_coef` first — too low = premature convergence; too high = random forever
6. Tune `w_energy` vs `w_purity` ratio to get the Pareto front you want

**Expected training time:** 500k steps × ~50ms per step (simulator) = ~7 hours on CPU. Reduce by parallelizing with `n_envs=8` and caching simulator states.

---

### Phase 4 — Multi-Objective Extension (Week 6–7)
Goal: Show the Pareto front between purity and energy consumption.

1. Implement **scalarization sweep** — train N agents with different `w_purity/w_energy` ratios
2. Collect the optimal operating point from each agent
3. Plot the Pareto front: x = reboiler duty, y = distillate purity
4. Compare against the grid scan Pareto front — this is your validation

This is a genuinely interesting result: the RL agent should find operating points that are either better than or equal to naive grid search, with less total computation (the agent avoids infeasible regions).

---

### Phase 5 — Analysis + Portfolio Polish (Week 8)
Goal: Turn this into a presentable project.

1. Jupyter notebook: `03_results_analysis.ipynb`
   - Training curves (reward vs timesteps)
   - Pareto front visualization
   - Action distribution at convergence (what did the agent learn?)
   - Comparison vs PID baseline and grid search
   
2. README with:
   - Architecture diagram
   - How to run the environment
   - Key results (purity achieved, energy saved vs baseline)
   
3. GitHub Actions: lint + run environment sanity check on push

---

## Implementation Status

**✅ COMPLETE** — All components implemented in pure Python:

1. **Environment** (`chemsim_gym/`): Full Gymnasium wrapper
2. **Simulator** (`chemsim/`): Pure Python flowsheet engine
3. **Training** (`training/`): PPO training with Stable-Baselines3
4. **Analysis** (`analysis/`): Grid scan and visualization tools

The Python `Flowsheet` class already has all needed methods:
- `set_param()` - Set reflux ratio, distillate fraction
- `set_stream_conditions()` - Set feed T, P
- `get_unit_scalar()` - Get temperatures, duties
- `reset_to_base()` - Reset to nominal state
- `from_json()` - Load flowsheet from JSON

No C++ changes required!

---

## Performance Considerations

The Python simulator is efficient for RL training. With 500k training steps and ~50ms per solve:
- 1 env: ~7 hours
- 4 envs (parallel): ~2 hours
- 8 envs: ~1 hour

To speed up further:
1. **Warm starting** — Pass previous converged state as initial guess (already implemented)
2. **Result caching** — Cache results for similar action queries
3. **Surrogate pretraining** — Train a fast NN approximation first, then fine-tune with simulator

The distillation column Newton solver typically converges in 3–5 iterations from a good guess.

---

## Interesting Extensions (Future Work)

1. **Curriculum learning** — start with easy targets (0.80 purity), gradually increase to 0.98
2. **Constraint satisfaction** — add hard constraints via Lagrangian penalty or SafeRL
3. **Transformer policy** — replace MLP with a small transformer; treat stream variables as tokens
4. **Transfer learning** — train on one feed composition, fine-tune on others
5. **Offline RL** — collect 10k grid-search trajectories, train offline with CQL or IQL

---

## Dependencies

```txt
# requirements.txt
gymnasium>=0.29
stable-baselines3>=2.3
torch>=2.2
tensorboard>=2.16
numpy>=1.26
pyyaml>=6.0
matplotlib>=3.8
seaborn>=0.13
pandas>=2.2
scipy>=1.12
```

---

## First Session Prompt

Paste this at the start:

```
Project: RL Process Optimizer — pure Python Gymnasium environment for ChemSim

Session goal: [e.g. "Run training and evaluate agent performance"]

ChemSim is a pure-Python process simulator at ./chemsim/ with no C++ dependencies.
The RL environment is at ./chemsim_gym/ with full Gymnasium API.

Architecture:
- chemsim/flowsheet/flowsheet.py: Flowsheet class with set_param(), set_stream_conditions()
- chemsim_gym/env.py: ProcessSimEnv wrapping the simulator
- training/train.py: PPO training with Stable-Baselines3

To run training:
  PYTHONPATH=. python training/train.py --config training/config/default.yaml

To analyze reward landscape:
  PYTHONPATH=. python analysis/reward_landscape.py

Ground rules:
- Always check env with gymnasium.utils.check_env() before training
- Reward must be tunable in Python (RewardConfig in reward.py)
- Log every episode: action, reward breakdown, purity, energy
- Simulator failures (non-convergence) must never crash — catch in SimulatorBridge
```
