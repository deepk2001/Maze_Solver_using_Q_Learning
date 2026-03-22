import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def get_safe_window_size(data_length):
    """Determine a safe window size based on data length."""
    if data_length < 3:
        return 1
    elif data_length < 10:
        return 2
    else:
        return min(50, max(3, data_length // 10))

def moving_average(data, window):
    """Compute moving average with proper padding."""
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def compute_confidence_interval(data, window):
    """Safely compute confidence intervals for the moving average."""
    if len(data) < window:
        return np.zeros(1), np.zeros(1)
    
    # Pad the data to handle the edges
    pad_data = np.pad(data, (window//2, window//2), mode='edge')
    rolling_windows = np.array([pad_data[i:i+window] for i in range(len(data))])
    std_err = stats.sem(rolling_windows, axis=1)
    return 1.96 * std_err

def plot_training_metrics(save_dir, window=None):
    """Plots comprehensive training metrics including rewards, epsilon, and time statistics."""
    # Load data
    rewards = np.load(os.path.join(save_dir, 'rewards.npy'))
    epsilons = np.load(os.path.join(save_dir, 'epsilons.npy'))
    episode_times = np.load(os.path.join(save_dir, 'episode_times.npy'))
    episode_steps = np.load(os.path.join(save_dir, 'episode_steps.npy'))
    
    # Determine safe window size if not provided
    if window is None or window >= len(rewards):
        window = get_safe_window_size(len(rewards))
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    episodes = np.arange(len(rewards))
    
    # Plot rewards
    if len(rewards) > 0:
        axes[0].plot(episodes, rewards, alpha=0.3, label='Raw Rewards', color='blue')
        if len(rewards) >= window:
            smoothed_rewards = moving_average(rewards, window)
            smooth_episodes = episodes[window-1:]
            confidence_interval = compute_confidence_interval(rewards, window)
            
            axes[0].plot(smooth_episodes, smoothed_rewards, 
                        label=f'Moving Avg ({window})', color='red')
            axes[0].fill_between(smooth_episodes, 
                               smoothed_rewards - confidence_interval[:len(smoothed_rewards)],
                               smoothed_rewards + confidence_interval[:len(smoothed_rewards)],
                               alpha=0.2, color='red')
    
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Reward Progression")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot steps per episode
    if len(episode_steps) > 0:
        axes[1].plot(episodes, episode_steps, alpha=0.3, label='Raw Steps', color='blue')
        if len(episode_steps) >= window:
            smoothed_steps = moving_average(episode_steps, window)
            smooth_episodes = episodes[window-1:]
            axes[1].plot(smooth_episodes, smoothed_steps, 
                        label=f'Moving Avg ({window})', color='red')
    
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps per Episode")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot epsilon progression
    axes[2].plot(episodes, epsilons, label='Epsilon', color='green')
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_title("Exploration Rate")
    axes[2].grid(True)
    
    # Plot episode duration
    if len(episode_times) > 0:
        axes[3].plot(episodes, episode_times, alpha=0.3, label='Raw Times', color='blue')
        if len(episode_times) >= window:
            smoothed_times = moving_average(episode_times, window)
            smooth_episodes = episodes[window-1:]
            axes[3].plot(smooth_episodes, smoothed_times, 
                        label=f'Moving Avg ({window})', color='red')
    
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Time (seconds)")
    axes[3].set_title("Episode Duration")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def plot_trajectory_heatmaps(save_dir, maze, episodes=None):
    """Plots trajectory heatmaps for specified episodes with enhanced cell clarity."""
    trajectories = np.load(os.path.join(save_dir, 'trajectories.npy'), allow_pickle=True)
    
    if len(trajectories) == 0:
        print("No trajectory data available")
        return
    
    if episodes is None:
        num_plots = min(5, len(trajectories))
        if num_plots == 1:
            episodes = [0]
        else:
            step = (len(trajectories) - 1) // (num_plots - 1)
            episodes = [i * step for i in range(num_plots)]
    else:
        episodes = [ep for ep in episodes if ep < len(trajectories)]
    
    if not episodes:
        print("No valid episodes to plot")
        return
    
    fig, axes = plt.subplots(1, len(episodes), figsize=(4*len(episodes), 4))
    if len(episodes) == 1:
        axes = [axes]
    
    for idx, episode in enumerate(episodes):
        heatmap = np.zeros_like(maze, dtype=float)
        trajectory = trajectories[episode]
        for x, y in trajectory:
            heatmap[x, y] += 1
        
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        axes[idx].imshow(maze, cmap='binary')
        img = axes[idx].imshow(heatmap, cmap='viridis', alpha=0.6)
        plt.colorbar(img, ax=axes[idx], label="Visit Frequency")
        
        # Add grid lines for clarity
        axes[idx].set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        axes[idx].set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        axes[idx].grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        axes[idx].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        axes[idx].set_title(f'Episode {episode + 1}: {len(trajectory)} steps')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_heatmaps.png'))
    plt.close()


def analyze_experiment(save_dir, maze, episodes_to_plot=None, window=None):
    """Comprehensive analysis of experiment results."""
    # Load summary statistics
    summary = np.load(os.path.join(save_dir, 'summary.npy'), allow_pickle=True).item()
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Total training time: {summary['total_time']:.2f} seconds")
    print(f"Average episode time: {summary['avg_episode_time']:.2f} seconds")
    print(f"Average episode steps: {summary['avg_episode_steps']:.2f}")
    print(f"Average reward: {summary['avg_reward']:.2f} (±{summary['std_reward']:.2f})")
    print(f"Best reward: {summary['max_reward']:.2f}")
    print(f"Worst reward: {summary['min_reward']:.2f}")
    
    # Generate plots
    plot_training_metrics(save_dir, window)
    plot_trajectory_heatmaps(save_dir, maze, episodes_to_plot)