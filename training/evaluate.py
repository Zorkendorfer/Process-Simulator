#!/usr/bin/env python3
"""Evaluate trained RL agent and generate analysis plots."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from chemsim_gym.env import ProcessSimEnv
from chemsim_gym.wrappers import NormalizeObservation


def evaluate_agent(model_path: str, env: ProcessSimEnv, n_episodes: int = 100):
    """Evaluate trained agent over multiple episodes."""
    
    # Load model
    model = PPO.load(model_path)
    
    rewards = []
    purities = []
    energies = []
    flows = []
    converged_count = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        rewards.append(total_reward)
        if info["converged"]:
            converged_count += 1
            purities.append(info["distillate_purity"])
            energies.append(info["reboiler_duty_kW"])
            flows.append(info["distillate_flow"])
    
    results = {
        "n_episodes": n_episodes,
        "converged": converged_count,
        "convergence_rate": converged_count / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_purity": float(np.mean(purities)) if purities else 0,
        "std_purity": float(np.std(purities)) if purities else 0,
        "mean_energy": float(np.mean(energies)) if energies else 0,
        "std_energy": float(np.std(energies)) if energies else 0,
        "mean_flow": float(np.mean(flows)) if flows else 0,
    }
    
    return results


def plot_training_progress(log_dir: str, save_path: str = "training_progress.png"):
    """Plot training progress from TensorBoard logs."""
    
    # Try to load eval results
    eval_file = Path(log_dir) / "evaluations.npz"
    if not eval_file.exists():
        print(f"No evaluation data found at {log_dir}")
        return
    
    data = np.load(eval_file)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    std_rewards = data["results"].std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps / 1000, mean_rewards, "b-", linewidth=2, label="Mean reward")
    ax.fill_between(timesteps / 1000, 
                    mean_rewards - std_rewards, 
                    mean_rewards + std_rewards,
                    alpha=0.3, label="±1 std")
    ax.set_xlabel("Timesteps (thousands)")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training progress to {save_path}")
    plt.close()


def plot_agent_performance(results: dict, save_path: str = "agent_performance.png"):
    """Plot agent performance summary."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Reward distribution
    axes[0].bar(["Mean"], [results["mean_reward"]], yerr=[results["std_reward"]], 
                capsize=5, color="steelblue")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Episode Reward")
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Purity distribution
    axes[1].bar(["Mean"], [results["mean_purity"]], yerr=[results["std_purity"]], 
                capsize=5, color="forestgreen")
    axes[1].axhline(y=0.95, color="r", linestyle="--", label="Target (0.95)")
    axes[1].set_ylabel("Purity (mole fraction)")
    axes[1].set_title("Distillate Purity")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Energy
    axes[2].bar(["Mean"], [results["mean_energy"]], yerr=[results["std_energy"]], 
                capsize=5, color="darkorange")
    axes[2].set_ylabel("Reboiler Duty (kW)")
    axes[2].set_title("Energy Consumption")
    axes[2].grid(True, alpha=0.3, axis="y")
    
    plt.suptitle(f"Agent Performance ({results['n_episodes']} episodes, "
                 f"{100*results['convergence_rate']:.0f}% converged)", 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved performance summary to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--flowsheet", type=str, default="examples/distillation_recycle.json")
    parser.add_argument("--component-db", type=str, default="data/components.json")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="evaluation_output")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory for training progress plot")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = ProcessSimEnv(
        flowsheet_json=args.flowsheet,
        component_db=args.component_db,
    )
    env = NormalizeObservation(env)
    
    # Evaluate
    print(f"Evaluating agent over {args.n_episodes} episodes...")
    results = evaluate_agent(args.model, env, n_episodes=args.n_episodes)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {results['n_episodes']}")
    print(f"Convergence rate: {100*results['convergence_rate']:.1f}%")
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean purity: {results['mean_purity']:.4f} ± {results['std_purity']:.4f}")
    print(f"Mean energy: {results['mean_energy']:.1f} ± {results['std_energy']:.1f} kW")
    print(f"Mean flow: {results['mean_flow']:.2f} mol/s")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Generate plots
    plot_agent_performance(results, save_path=output_dir / "agent_performance.png")
    
    if args.log_dir:
        plot_training_progress(args.log_dir, 
                               save_path=output_dir / "training_progress.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
