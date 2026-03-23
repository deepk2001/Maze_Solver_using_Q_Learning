"""
Visualize hyperparameter sweep results from hyperparameter_results/ folder.
Grid: 3 learning rates × 3 discount rates for Q-learning (random_negative, performance_based, manhattan).
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 10,
})

RESULTS_ROOT = "hyperparam_sweep/results"
OUTPUT_DIR = "hyperparameter_visualizations"


def load_hyperparam_data(results_root=RESULTS_ROOT):
    """Load all runs from hyperparameter_results, extracting lr/gamma from config."""
    data = []
    for root, _, files in os.walk(results_root):
        if "summary.npy" in files and "rewards.npy" in files and "config.yaml" in files:
            try:
                config_path = os.path.join(root, "config.yaml")
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                hp = config.get("hyperparameters", {})
                lr = hp.get("learning_rate")
                gamma = hp.get("discount_rate")
                if lr is None or gamma is None:
                    continue

                rewards = np.load(os.path.join(root, "rewards.npy"))
                summary = np.load(
                    os.path.join(root, "summary.npy"), allow_pickle=True
                ).item()

                data.append({
                    "path": root,
                    "learning_rate": lr,
                    "discount_rate": gamma,
                    "rewards": rewards,
                    "final_avg": summary.get("avg_reward", 0),
                    "std_reward": summary.get("std_reward", 0),
                    "max_reward": summary.get("max_reward", 0),
                    "avg_steps": summary.get("avg_episode_steps", 0),
                    "total_time": summary.get("total_time", 0),
                    "label": f"lr={lr} γ={gamma}",
                })
            except Exception as e:
                print(f"Skipping {root}: {e}")
    return data


def _smooth(arr, window=None):
    if window is None:
        window = min(51, max(5, len(arr) // 10 * 2 + 1))
    if window % 2 == 0:
        window += 1
    if len(arr) < window:
        return arr
    return savgol_filter(arr, window, 3)


def plot_heatmap(data):
    """Heatmap: rows=learning_rate, cols=discount_rate."""
    lrs = sorted(set(d["learning_rate"] for d in data))
    gammas = sorted(set(d["discount_rate"] for d in data))
    matrix = np.full((len(lrs), len(gammas)), np.nan)

    for d in data:
        i = lrs.index(d["learning_rate"])
        j = gammas.index(d["discount_rate"])
        matrix[i, j] = d["final_avg"]

    fig, ax = plt.subplots(figsize=(7, 5))
    vmin = np.nanmin(matrix) if np.any(~np.isnan(matrix)) else -2
    vmax = np.nanmax(matrix) if np.any(~np.isnan(matrix)) else 3
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(gammas)))
    ax.set_yticks(np.arange(len(lrs)))
    ax.set_xticklabels([f"{g:.3f}" for g in gammas])
    ax.set_yticklabels([f"{lr}" for lr in lrs])
    ax.set_xlabel("Discount rate (γ)")
    ax.set_ylabel("Learning rate")

    for i in range(len(lrs)):
        for j in range(len(gammas)):
            val = matrix[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "-"
            ax.text(j, i, text, ha="center", va="center", fontsize=11)

    ax.set_title("Average Reward: Learning Rate × Discount Rate")
    plt.colorbar(im, ax=ax, label="Avg Reward")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hyperparam_heatmap.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_learning_curves(data):
    """Learning curves for each (lr, gamma) configuration."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for d in sorted(data, key=lambda x: (x["learning_rate"], x["discount_rate"])):
        rewards = d["rewards"]
        if len(rewards) == 0:
            continue
        smoothed = _smooth(rewards)
        ax.plot(smoothed, label=d["label"], alpha=0.85)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward (smoothed)")
    ax.set_title("Learning Curves by Hyperparameters")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlim(left=0)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hyperparam_learning_curves.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_best_configs(data, top_n=9):
    """Bar chart of configurations ranked by final average reward."""
    sorted_data = sorted(data, key=lambda x: x["final_avg"], reverse=True)[:top_n]
    labels = [d["label"] for d in sorted_data]
    scores = [d["final_avg"] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    y_pos = np.arange(len(labels))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(labels))[::-1])
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Average Reward")
    ax.set_title("Hyperparameter Configurations (ranked by final avg reward)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hyperparam_leaderboard.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_and_save_table(data):
    """Print summary table and save to text file."""
    sorted_data = sorted(
        data,
        key=lambda x: (x["learning_rate"], x["discount_rate"]),
    )

    lines = [
        "=" * 80,
        "HYPERPARAMETER SWEEP SUMMARY",
        "Q-learning | random_negative_init | performance_based | manhattan",
        "=" * 80,
        f"{'LR':<8} {'γ':<10} {'Avg R':<10} {'Max R':<10} {'Avg Steps':<12} {'Time(s)':<10}",
        "-" * 80,
    ]
    for d in sorted_data:
        lines.append(
            f"{d['learning_rate']:<8} {d['discount_rate']:<10} "
            f"{d['final_avg']:<10.2f} {d['max_reward']:<10.2f} "
            f"{d['avg_steps']:<12.1f} {d['total_time']:<10.1f}"
        )
    lines.append("=" * 80)

    best = max(data, key=lambda x: x["final_avg"])
    lines.append(f"\nBest: lr={best['learning_rate']}, γ={best['discount_rate']} -> avg_reward={best['final_avg']:.2f}")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)

    out = os.path.join(OUTPUT_DIR, "hyperparam_summary.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"\nSaved: {out}")


def main():
    if not os.path.isdir(RESULTS_ROOT):
        print(f"Error: {RESULTS_ROOT}/ not found. Run ./hyperparam_sweep/run-sweep.zsh first.")
        return

    print(f"Loading data from {RESULTS_ROOT}/...")
    data = load_hyperparam_data()

    if not data:
        print("No results found. Run ./hyperparam_sweep/run-sweep.zsh first.")
        return

    print(f"Loaded {len(data)} runs.\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_heatmap(data)
    plot_learning_curves(data)
    plot_best_configs(data)
    print_and_save_table(data)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
