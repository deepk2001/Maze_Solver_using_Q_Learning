"""
Additional visualizations for Maze RL experiment results.
Scans the results/ folder and generates aggregate comparison plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 10,
})

OUTPUT_DIR = "additional_visualizations"
COLORS = {
    "q_learning": "#2ecc71",
    "dqn": "#e74c3c",
    "zero_init": "#3498db",
    "random_negative_init": "#9b59b6",
    "decay": "#f39c12",
    "performance_based": "#1abc9c",
    "goal_only": "#e67e22",
    "goal_and_step": "#34495e",
    "manhattan": "#16a085",
}


def parse_result_path(root, results_root):
    """Parse results path into experiment metadata.
    Structure: results/{algo}/{init}/{policy}/{reward_mode}/{maze}/
    """
    rel = os.path.relpath(root, results_root)
    parts = rel.replace("\\", "/").split("/")
    parsed = {
        "algo": "unknown",
        "init": "unknown",
        "policy": "unknown",
        "reward_mode": "unknown",
        "maze": "unknown",
        "path": root,
    }
    if len(parts) >= 1 and parts[0]:
        parsed["algo"] = parts[0].lower()
    if len(parts) >= 2:
        parsed["init"] = parts[1].lower()
    if len(parts) >= 3:
        parsed["policy"] = parts[2].lower()
    if len(parts) >= 4:
        parsed["reward_mode"] = parts[3].lower()
    if len(parts) >= 5:
        parsed["maze"] = parts[4].lower()
    return parsed


def load_all_results(results_root="results"):
    """Load all experiment data from results directory."""
    data = []
    for root, _, files in os.walk(results_root):
        if "summary.npy" in files and "rewards.npy" in files:
            try:
                meta = parse_result_path(root, results_root)

                rewards = np.load(os.path.join(root, "rewards.npy"))
                summary = np.load(
                    os.path.join(root, "summary.npy"), allow_pickle=True
                ).item()

                entry = {
                    **meta,
                    "rewards": rewards,
                    "final_avg": summary.get("avg_reward", 0),
                    "std_reward": summary.get("std_reward", 0),
                    "max_reward": summary.get("max_reward", 0),
                    "min_reward": summary.get("min_reward", 0),
                    "avg_steps": summary.get("avg_episode_steps", 0),
                    "total_time": summary.get("total_time", 0),
                    "label": f"{meta['algo']}/{meta['init']}/{meta['reward_mode']}",
                }
                if "episode_steps.npy" in files:
                    entry["steps"] = np.load(os.path.join(root, "episode_steps.npy"))
                else:
                    entry["steps"] = None

                if "losses.npy" in files:
                    entry["losses"] = np.load(os.path.join(root, "losses.npy"))
                else:
                    entry["losses"] = None

                if "periodic_eval.npy" in files:
                    entry["periodic_eval"] = np.load(
                        os.path.join(root, "periodic_eval.npy"), allow_pickle=True
                    )
                else:
                    entry["periodic_eval"] = None

                data.append(entry)
            except Exception as e:
                print(f"Skipping {root}: {e}")
    return data


def _smooth_curve(arr, window=None):
    """Smooth array with Savitzky-Golay if long enough."""
    if window is None:
        window = min(51, max(5, len(arr) // 10 * 2 + 1))
    if window % 2 == 0:
        window += 1
    if len(arr) < window:
        return arr
    return savgol_filter(arr, window, 3)


def plot_combined_curves(data, group_by, title, metric="rewards", output_name=None):
    """Plot learning curves grouped by a key, with mean and std."""
    categories = {}
    for entry in data:
        cat = entry[group_by]
        if cat not in categories:
            categories[cat] = []
        arr = entry.get(metric)
        if arr is not None and len(arr) > 0:
            categories[cat].append(arr)

    if not categories:
        print(f"No data for {metric} / {group_by}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for cat, arrays in categories.items():
        if not arrays:
            continue
        max_len = max(len(a) for a in arrays)
        padded = [np.pad(a, (0, max_len - len(a)), "edge") for a in arrays]
        mean_arr = np.mean(padded, axis=0)
        std_arr = np.std(padded, axis=0)
        smoothed = _smooth_curve(mean_arr)
        x = np.arange(len(smoothed))
        color = COLORS.get(cat, None)
        ax.plot(x, smoothed, label=cat, color=color, linewidth=2)
        ax.fill_between(x, smoothed - std_arr[:len(smoothed)], smoothed + std_arr[:len(smoothed)], alpha=0.2, color=color)

    ylabel = "Total Reward" if metric == "rewards" else "Steps per Episode" if metric == "steps" else "Loss"
    ax.set_xlabel("Episodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xlim(left=0)
    plt.tight_layout()
    name = output_name or f"comparison_{group_by}_{metric}"
    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_leaderboard(data, top_n=15):
    """Bar chart of top experiments by final average reward."""
    sorted_data = sorted(data, key=lambda x: x["final_avg"], reverse=True)
    top = sorted_data[:top_n]

    labels = [f"{d['algo']}\n{d['init']}\n{d['reward_mode']}" for d in top]
    scores = [d["final_avg"] for d in top]
    colors = [COLORS.get(d["algo"], "#95a5a6") for d in top]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average Reward")
    ax.set_title(f"Top {top_n} Configurations (Final Average Reward)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "leaderboard.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_evaluation_progress(data):
    """Plot periodic evaluation reward and steps over training."""
    runs_with_eval = [d for d in data if d.get("periodic_eval") is not None]
    if not runs_with_eval:
        print("No periodic evaluation data found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for entry in runs_with_eval:
        evals = entry["periodic_eval"]
        if len(evals) == 0:
            continue
        episodes = [e["episode"] for e in evals]
        avg_rewards = [e["avg_reward"] for e in evals]
        avg_steps = [e["avg_steps"] for e in evals]
        label = entry["label"]
        color = COLORS.get(entry["algo"], None)
        ax1.plot(episodes, avg_rewards, "o-", label=label, color=color, markersize=4)
        ax2.plot(episodes, avg_steps, "o-", label=label, color=color, markersize=4)

    ax1.set_ylabel("Avg Eval Reward")
    ax1.set_title("Periodic Evaluation Progress")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax1.set_xlim(left=0)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Eval Steps")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "evaluation_progress.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_summary_heatmap(data):
    """Heatmap of average reward by algo × reward_mode (or similar dimensions)."""
    # Build matrix: rows=algo, cols=reward_mode, aggregated across init/policy
    algos = sorted(set(d["algo"] for d in data))
    reward_modes = sorted(set(d["reward_mode"] for d in data))
    matrix = np.full((len(algos), len(reward_modes)), np.nan)

    for d in data:
        try:
            i = algos.index(d["algo"])
            j = reward_modes.index(d["reward_mode"])
            # Use best so far or average if multiple runs per cell
            if np.isnan(matrix[i, j]) or d["final_avg"] > matrix[i, j]:
                matrix[i, j] = d["final_avg"]
        except ValueError:
            pass

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=matrix[~np.isnan(matrix)].min() if np.any(~np.isnan(matrix)) else -2, vmax=2.5)
    ax.set_xticks(np.arange(len(reward_modes)))
    ax.set_yticks(np.arange(len(algos)))
    ax.set_xticklabels(reward_modes)
    ax.set_yticklabels(algos)
    for i in range(len(algos)):
        for j in range(len(reward_modes)):
            val = matrix[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "-"
            ax.text(j, i, text, ha="center", va="center", fontsize=11)
    ax.set_title("Average Reward by Algorithm × Reward Mode")
    plt.colorbar(im, ax=ax, label="Avg Reward")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "summary_heatmap.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_loss_curves(data):
    """Plot DQN loss curves (DQN only)."""
    dqn_data = [d for d in data if d["algo"] == "dqn" and d.get("losses") is not None]
    if not dqn_data:
        print("No DQN loss data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for entry in dqn_data:
        losses = entry["losses"]
        if len(losses) == 0:
            continue
        label = f"{entry['reward_mode']} ({entry['init']})"
        smoothed = _smooth_curve(losses)
        ax.plot(smoothed, label=label, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("DQN Training Loss (Smoothed)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(left=0)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "dqn_losses.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary_table(data):
    """Print a summary table to console and optionally save as text."""
    sorted_data = sorted(data, key=lambda x: x["final_avg"], reverse=True)
    lines = [
        "=" * 100,
        "EXPERIMENT SUMMARY (sorted by average reward)",
        "=" * 100,
        f"{'Algo':<12} {'Init':<20} {'Policy':<18} {'Reward':<14} {'Avg R':<8} {'Max R':<8} {'Avg Steps':<10} {'Time(s)':<10}",
        "-" * 100,
    ]
    for d in sorted_data:
        lines.append(
            f"{d['algo']:<12} {d['init']:<20} {d['policy']:<18} {d['reward_mode']:<14} "
            f"{d['final_avg']:<8.2f} {d['max_reward']:<8.2f} {d['avg_steps']:<10.1f} {d['total_time']:<10.1f}"
        )
    lines.append("=" * 100)
    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(OUTPUT_DIR, "summary_table.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved: {out_path}")


def main(results_root="results"):
    print("Scanning results folder...")
    data = load_all_results(results_root)

    if not data:
        print("No results found. Run experiments first (run.py).")
        return

    print(f"Loaded {len(data)} experiment runs.\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Algorithm comparison
    plot_combined_curves(
        data, group_by="algo", metric="rewards",
        title="Learning Curves: Q-Learning vs DQN",
        output_name="comparison_algo",
    )

    # 2. Reward mode comparison
    plot_combined_curves(
        data, group_by="reward_mode", metric="rewards",
        title="Impact of Reward Design (goal_only, goal_and_step, manhattan)",
        output_name="comparison_reward_mode",
    )

    # 3. Init strategy comparison
    plot_combined_curves(
        data, group_by="init", metric="rewards",
        title="Q-Table Initialization: zero vs random_negative",
        output_name="comparison_init",
    )

    # 4. Epsilon policy comparison
    plot_combined_curves(
        data, group_by="policy", metric="rewards",
        title="Epsilon Policy: decay vs performance_based",
        output_name="comparison_policy",
    )

    # 5. Steps per episode (convergence efficiency)
    steps_data = [d for d in data if d.get("steps") is not None]
    if steps_data:
        plot_combined_curves(
            steps_data, group_by="algo", metric="steps",
            title="Steps per Episode: Q-Learning vs DQN",
            output_name="comparison_steps",
        )

    # 6. Leaderboard
    plot_leaderboard(data)

    # 7. Evaluation progress (if periodic evals exist)
    plot_evaluation_progress(data)

    # 8. Summary heatmap
    plot_summary_heatmap(data)

    # 9. DQN losses
    plot_loss_curves(data)

    # 10. Summary table
    print_summary_table(data)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
