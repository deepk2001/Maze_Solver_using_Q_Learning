import os
import time
import numpy as np

class ExperimentLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.rewards = []
        self.losses = []
        self.trajectories = []  # Will store as list, convert to object array when saving
        self.episode_times = []
        self.episode_steps = []
        self.epsilons = []
        
    def log_episode(self, reward, loss, trajectory, time_taken, steps, epsilon):
        self.rewards.append(reward)
        self.losses.append(loss if loss else 0)
        self.trajectories.append(trajectory)  # Store as regular list
        self.episode_times.append(time_taken)
        self.episode_steps.append(steps)
        self.epsilons.append(epsilon)
    
    def save_logs(self):
        # Save fixed-length arrays
        np.save(os.path.join(self.save_dir, 'rewards.npy'), np.array(self.rewards))
        np.save(os.path.join(self.save_dir, 'losses.npy'), np.array(self.losses))
        np.save(os.path.join(self.save_dir, 'episode_times.npy'), np.array(self.episode_times))
        np.save(os.path.join(self.save_dir, 'episode_steps.npy'), np.array(self.episode_steps))
        np.save(os.path.join(self.save_dir, 'epsilons.npy'), np.array(self.epsilons))
        
        # Save trajectories as a pickle file since they're variable length
        np.save(os.path.join(self.save_dir, 'trajectories.npy'), 
                np.array(self.trajectories, dtype=object))
        
        # Save summary statistics
        summary = {
            'total_time': sum(self.episode_times),
            'avg_episode_time': np.mean(self.episode_times),
            'avg_episode_steps': np.mean(self.episode_steps),
            'avg_reward': np.mean(self.rewards),
            'std_reward': np.std(self.rewards),
            'max_reward': max(self.rewards),
            'min_reward': min(self.rewards)
        }
        np.save(os.path.join(self.save_dir, 'summary.npy'), summary)