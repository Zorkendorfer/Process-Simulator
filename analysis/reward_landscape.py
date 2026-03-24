#!/usr/bin/env python3
"""
Reward landscape analysis via grid scan.

Scans the action space (reflux ratio vs distillate fraction) and visualizes:
- Distillate purity
- Reboiler duty (energy)
- Total reward
- Pareto front
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from chemsim_gym.env import ProcessSimEnv


def grid_scan(env, n_reflux=30, n_dist=30, T_feed=None, P_feed=None):
    """
    Perform grid scan over reflux ratio and distillate fraction.
    
    Returns
    -------
    results : dict
        Dictionary with keys: reflux, dist, purity, energy, reward, converged
    """
    reflux_range = np.linspace(0.5, 10.0, n_reflux)
    dist_range = np.linspace(0.1, 0.9, n_dist)
    
    # Default T, P from environment reset
    if T_feed is None or P_feed is None:
        obs, info = env.reset()
        T_feed = 250.0  # Default from flowsheet
        P_feed = 2e6
    
    results = {
        "reflux": [],
        "dist": [],
        "purity": [],
        "energy": [],
        "reward": [],
        "converged": [],
        "flow": [],
    }
    
    print(f"Scanning {n_reflux}x{n_dist} = {n_reflux*n_dist} operating points...")
    
    for i, R in enumerate(reflux_range):
        for j, D in enumerate(dist_range):
            obs, _ = env.reset()
            
            # Normalize action to [-1, 1]
            from chemsim_gym.spaces import ACTION_LOW, ACTION_HIGH
            action_phys = np.array([R, D, T_feed, P_feed])
            action = 2.0 * (action_phys - ACTION_LOW[:4]) / (ACTION_HIGH[:4] - ACTION_LOW[:4]) - 1.0
            action = np.clip(action, -1, 1)
            
            obs, rew, _, _, info = env.step(action)
            
            results["reflux"].append(R)
            results["dist"].append(D)
            results["purity"].append(info["distillate_purity"])
            results["energy"].append(info["reboiler_duty_kW"])
            results["reward"].append(rew)
            results["converged"].append(info["converged"])
            results["flow"].append(info["distillate_flow"])
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{n_reflux} reflux values...")
    
    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k])
    
    # Reshape to 2D
    shape = (n_reflux, n_dist)
    results["purity_2d"] = results["purity"].reshape(shape)
    results["energy_2d"] = results["energy"].reshape(shape)
    results["reward_2d"] = results["reward"].reshape(shape)
    results["converged_2d"] = results["converged"].reshape(shape)
    
    return results, reflux_range, dist_range


def plot_landscape(results, reflux_range, dist_range, save_path="reward_landscape.png"):
    """Create contour plots of the reward landscape."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Purity
    im0 = axes[0, 0].contourf(dist_range, reflux_range, results["purity_2d"], 
                               levels=20, cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_xlabel("Distillate fraction (D/F)")
    axes[0, 0].set_ylabel("Reflux ratio (L/D)")
    axes[0, 0].set_title("Distillate Purity (mole fraction)")
    axes[0, 0].axhline(y=2.0, color="r", linestyle="--", alpha=0.5, label="nominal R")
    axes[0, 0].axvline(x=0.4, color="r", linestyle="--", alpha=0.5, label="nominal D/F")
    plt.colorbar(im0, ax=axes[0, 0], label="Purity")
    axes[0, 0].legend(loc="upper right")
    
    # 2. Energy (reboiler duty)
    im1 = axes[0, 1].contourf(dist_range, reflux_range, results["energy_2d"], 
                               levels=20, cmap="plasma")
    axes[0, 1].set_xlabel("Distillate fraction (D/F)")
    axes[0, 1].set_ylabel("Reflux ratio (L/D)")
    axes[0, 1].set_title("Reboiler Duty (kW)")
    plt.colorbar(im1, ax=axes[0, 1], label="Q_reb [kW]")
    
    # 3. Total reward
    im2 = axes[1, 0].contourf(dist_range, reflux_range, results["reward_2d"], 
                               levels=20, cmap="RdBu_r")
    axes[1, 0].set_xlabel("Distillate fraction (D/F)")
    axes[1, 0].set_ylabel("Reflux ratio (L/D)")
    axes[1, 0].set_title("Total Reward")
    plt.colorbar(im2, ax=axes[1, 0], label="Reward")
    
    # 4. Convergence mask
    im3 = axes[1, 1].contourf(dist_range, reflux_range, results["converged_2d"], 
                               levels=[0, 0.5, 1], cmap="Greys", alpha=0.5)
    axes[1, 1].set_xlabel("Distillate fraction (D/F)")
    axes[1, 1].set_ylabel("Reflux ratio (L/D)")
    axes[1, 1].set_title("Convergence (white=converged, gray=failed)")
    plt.colorbar(im3, ax=axes[1, 1], label="Converged")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved landscape plot to {save_path}")
    plt.close()


def plot_pareto(results, save_path="pareto_front.png"):
    """Plot Pareto front of purity vs energy."""
    
    # Filter converged points
    mask = results["converged"]
    purity = results["purity"][mask]
    energy = results["energy"][mask]
    reward = results["reward"][mask]
    
    # Find Pareto optimal points (maximize purity, minimize energy)
    pareto_mask = []
    for i in range(len(purity)):
        is_pareto = True
        for j in range(len(purity)):
            if i != j:
                # j dominates i if j has higher purity AND lower energy
                if purity[j] >= purity[i] and energy[j] <= energy[i]:
                    if purity[j] > purity[i] or energy[j] < energy[i]:
                        is_pareto = False
                        break
        pareto_mask.append(is_pareto)
    
    pareto_purity = purity[pareto_mask]
    pareto_energy = energy[pareto_mask]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # All points
    scatter = ax.scatter(energy, purity, c=reward, cmap="RdBu_r", 
                         alpha=0.6, s=30, label="Operating points")
    
    # Pareto front
    if len(pareto_purity) > 0:
        # Sort by energy for clean line
        sort_idx = np.argsort(pareto_energy)
        ax.plot(np.array(pareto_energy)[sort_idx], np.array(pareto_purity)[sort_idx], 
                "r-", linewidth=2, label="Pareto front")
        ax.scatter(np.array(pareto_energy)[sort_idx], np.array(pareto_purity)[sort_idx],
                   c="red", s=50, zorder=5)
    
    ax.set_xlabel("Reboiler Duty (kW)")
    ax.set_ylabel("Distillate Purity (mole fraction)")
    ax.set_title("Pareto Front: Purity vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Total Reward")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved Pareto front to {save_path}")
    plt.close()


def save_results(results, reflux_range, dist_range, save_path="grid_scan_results.json"):
    """Save results to JSON for later analysis."""
    data = {
        "reflux_range": reflux_range.tolist(),
        "dist_range": dist_range.tolist(),
        "purity": results["purity_2d"].tolist(),
        "energy": results["energy_2d"].tolist(),
        "reward": results["reward_2d"].tolist(),
        "converged": results["converged_2d"].astype(int).tolist(),
        "flow": results["flow"].reshape(len(reflux_range), len(dist_range)).tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid scan reward landscape analysis")
    parser.add_argument("--flowsheet", type=str, default="examples/distillation_recycle.json")
    parser.add_argument("--component-db", type=str, default="data/components.json")
    parser.add_argument("--n-reflux", type=int, default=30)
    parser.add_argument("--n-dist", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    parser.add_argument("--no-plots", action="store_true")
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
    
    # Run grid scan
    results, reflux_range, dist_range = grid_scan(
        env, n_reflux=args.n_reflux, n_dist=args.n_dist
    )
    
    # Summary statistics
    print("\n=== Grid Scan Summary ===")
    print(f"Total points: {len(results['purity'])}")
    print(f"Converged: {results['converged'].sum()} / {len(results['converged'])} "
          f"({100*results['converged'].mean():.1f}%)")
    print(f"Purity range: [{results['purity'].min():.3f}, {results['purity'].max():.3f}]")
    print(f"Energy range: [{results['energy'].min():.1f}, {results['energy'].max():.1f}] kW")
    print(f"Reward range: [{results['reward'].min():.2f}, {results['reward'].max():.2f}]")
    
    # Find best operating point
    best_idx = np.argmax(results["reward"])
    print(f"\nBest operating point:")
    print(f"  Reflux ratio: {results['reflux'][best_idx]:.2f}")
    print(f"  Distillate frac: {results['dist'][best_idx]:.2f}")
    print(f"  Purity: {results['purity'][best_idx]:.3f}")
    print(f"  Energy: {results['energy'][best_idx]:.1f} kW")
    print(f"  Reward: {results['reward'][best_idx]:.2f}")
    
    if not args.no_plots:
        # Generate plots
        plot_landscape(results, reflux_range, dist_range, 
                       save_path=output_dir / "reward_landscape.png")
        plot_pareto(results, save_path=output_dir / "pareto_front.png")
    
    # Save results
    save_results(results, reflux_range, dist_range, 
                 save_path=output_dir / "grid_scan_results.json")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
