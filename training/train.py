#!/usr/bin/env python3
"""Train PPO agent on the ChemSim process optimization task."""

import yaml
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor


def make_env(config: dict, rank: int = 0):
    """Factory function for creating environments."""
    def _init():
        from chemsim_gym.env import ProcessSimEnv
        from chemsim_gym.wrappers import NormalizeObservation
        
        env = ProcessSimEnv(
            flowsheet_json=config["flowsheet_config"],
            component_db=config["component_db"],
            reward_config=config["reward"],
            max_steps=config.get("max_steps", 1),
        )
        env = NormalizeObservation(env)
        env = Monitor(env)
        return env
    return _init


def train(config_path: str):
    """Main training function."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Create directories
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Vectorized environments
    n_envs = cfg.get("n_envs", 4)
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(make_env(cfg), n_envs=n_envs, seed=cfg.get("seed", 42))
    eval_env = make_vec_env(make_env(cfg), n_envs=1)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg["checkpoint_dir"],
        log_path=cfg["log_dir"],
        eval_freq=cfg.get("eval_freq", 5000),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=cfg.get("save_freq", 10000),
        save_path=cfg["checkpoint_dir"],
        name_prefix="ppo_chemsim",
        verbose=1,
    )

    # PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.get("lr", 3e-4),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        verbose=1,
        tensorboard_log=cfg["log_dir"],
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    # Train
    total_steps = cfg.get("total_timesteps", 500_000)
    print(f"Training PPO for {total_steps:,} timesteps...")
    print(f"TensorBoard logs: {cfg['log_dir']}")
    
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    # Save final model
    final_path = Path(cfg["checkpoint_dir"]) / "final_model"
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on ChemSim process optimization")
    parser.add_argument("--config", type=str, default="training/config/default.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total training timesteps")
    args = parser.parse_args()
    
    # Load config to potentially override
    if args.total_timesteps is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["total_timesteps"] = args.total_timesteps
        # Write temp config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(cfg, f)
            args.config = f.name
    
    train(args.config)
