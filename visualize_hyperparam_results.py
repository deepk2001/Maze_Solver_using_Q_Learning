"""
Visualize hyperparameter sweep results using Steps per Episode (convergence rate).
Grid: learning rates × discount rates. Lower steps = better.
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
    """Load all runs, extracting lr/gamma from config. Requires episode_steps for curves."""
    data = []
    for root, _, files in os.walk(results_root):
        if "summary.npy" not in files:
            continue
        if "config.yaml" not in files:
            continue
        try:
            config_path = os.path.join(root, "config.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            hp = config.get("hyperparameters", {})
            lr = hp.get("learning_rate")
            gamma = hp.get("discount_rate")
            if lr is None or gamma is None:
                continue

            summary = np.load(os.path.join(root, "summary.npy"), allow_pickle=True).item()
            entry = {
                "path": root,
                "learning_rate": lr,
                "discount_rate": gamma,
                "avg_steps": summary.get("avg_episode_steps", np.nan),
                "total_time": summary.get("total_time", 0),
                "label": f"lr={lr} γ={gamma}",
            }
            if "episode_steps.npy" in files:
                entry["steps"] = np.load(os.path.join(root, "episode_steps.npy"))
            else:
                entry["steps"] = None
            data.append(entry)
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
    """Heatmap: LR × γ, avg steps. Lower = better (green)."""
    lrs = sorted(set(d["learning_rate"] for d in data))
    gammas = sorted(set(d["discount_rate"] for d in data))
    matrix = np.full((len(lrs), len(gammas)), np.nan)

    for d in data:
        if not np.isfinite(d["avg_steps"]):
            continue
        i = lrs.index(d["learning_rate"])
        j = gammas.index(d["discount_rate"])
        v = d["avg_steps"]
        if np.isnan(matrix[i, j]) or v < matrix[i, j]:
            matrix[i, j] = v

    if not np.any(np.isfinite(matrix)):
        print("No steps data for heatmap.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    valid = matrix[np.isfinite(matrix)]
    vmin, vmax = np.min(valid), np.max(valid)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(gammas)))
    ax.set_yticks(np.arange(len(lrs)))
    ax.set_xticklabels([f"{g:.3f}" for g in gammas])
    ax.set_yticklabels([f"{lr}" for lr in lrs])
    ax.set_xlabel("Discount rate (γ)")
    ax.set_ylabel("Learning rate")

    for i in range(len(lrs)):
        for j in range(len(gammas)):
            val = matrix[i, j]
            text = f"{val:.1f}" if np.isfinite(val) else "-"
            ax.text(j, i, text, ha="center", va="center", fontsize=11)

    ax.set_title("Avg Steps per Episode: LR × γ (Lower = Better)")
    plt.colorbar(im, ax=ax, label="Avg Steps")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hyperparam_heatmap.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_learning_curves(data):
    """Steps per episode over training for each (lr, γ) configuration."""
    steps_data = [d for d in data if d.get("steps") is not None and len(d["steps"]) > 0]
    if not steps_data:
        print("No episode_steps.npy found. Skipping learning curves.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for d in sorted(steps_data, key=lambda x: (x["learning_rate"], x["discount_rate"])):
        smoothed = _smooth(d["steps"])
        ax.plot(smoothed, label=d["label"], alpha=0.85)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps per Episode (Lower = Better)")
    ax.set_title("Convergence: Steps per Episode by Hyperparameters")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlim(left=0)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hyperparam_learning_curves.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_leaderboard(data, top_n=9):
    """Configurations ranked by avg steps (lowest = best)."""
    sorted_data = sorted(
        data,
        key=lambda x: x["avg_steps"] if np.isfinite(x["avg_steps"]) else 1e9
    )[:top_n]
    labels = [d["label"] for d in sorted_data]
    scores = [d["avg_steps"] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    y_pos = np.arange(len(labels))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Steps per Episode (Lower = Better)")
    ax.set_title("Hyperparameter Configurations (ranked by convergence rate)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hyperparam_leaderboard.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_and_save_table(data):
    """Summary table sorted by avg steps (ascending)."""
    sorted_data = sorted(
        data,
        key=lambda x: (x["learning_rate"], x["discount_rate"]),
    )

    lines = [
        "=" * 75,
        "HYPERPARAMETER SWEEP SUMMARY (Convergence: Steps per Episode)",
        "Q-learning | random_negative_init | performance_based | manhattan",
        "=" * 75,
        f"{'LR':<8} {'γ':<10} {'Avg Steps':<12} {'Time(s)':<10}",
        "-" * 75,
    ]
    for d in sorted_data:
        lines.append(
            f"{d['learning_rate']:<8} {d['discount_rate']:<10} "
            f"{d['avg_steps']:<12.1f} {d['total_time']:<10.1f}"
        )
    lines.append("=" * 75)

    best = min(data, key=lambda x: x["avg_steps"] if np.isfinite(x["avg_steps"]) else 1e9)
    if np.isfinite(best["avg_steps"]):
        lines.append(f"\nBest: lr={best['learning_rate']}, γ={best['discount_rate']} -> avg_steps={best['avg_steps']:.1f}")
    lines.append("=" * 75)

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
    plot_leaderboard(data)
    print_and_save_table(data)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
