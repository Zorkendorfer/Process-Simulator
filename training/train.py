#!/usr/bin/env python3
"""Train PPO agent on the ChemSim process optimization task."""

import yaml
import argparse
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor


class ProgressCallback(BaseCallback):
    """Custom callback to print training progress to terminal."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.start_time = None
        self.last_print_steps = 0
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("\n" + "="*60)
        print(f"{'Step':>10} | {'Reward':>10} | {'Time':>15}")
        print("="*60)
        
    def _on_step(self) -> bool:
        # Print every 500 steps
        if self.num_timesteps - self.last_print_steps >= 500:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Get latest reward from logger
            reward = self.logger.name_to_value.get("rollout/ep_rew_mean", 0)
            
            print(f"{self.num_timesteps:>10,} | {reward:>10.2f} | {elapsed:>6.0f}s ({steps_per_sec:.1f} steps/s)")
            self.last_print_steps = self.num_timesteps
        return True
    
    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        print("="*70)
        print(f"Training complete! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Final steps: {self.num_timesteps:,}")
        print("="*70 + "\n")


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
    progress_cb = ProgressCallback(verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg["checkpoint_dir"],
        log_path=cfg["log_dir"],
        eval_freq=cfg.get("eval_freq", 1000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,  # Silent, we'll show progress in terminal
    )
    ckpt_cb = CheckpointCallback(
        save_freq=cfg.get("save_freq", 5000),
        save_path=cfg["checkpoint_dir"],
        name_prefix="ppo_chemsim",
        verbose=0,  # Silent
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
    print(f"Checkpoints saved to: {cfg['checkpoint_dir']}")
    print(f"Using {n_envs} parallel environments, {cfg.get('n_steps', 2048)} steps per rollout\n")

    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([progress_cb, eval_cb, ckpt_cb]),
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
