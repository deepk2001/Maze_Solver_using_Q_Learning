# Project 3: Reinforcement Learning - Maze Solving

## Introduction

This project implements a **maze-solving** environment using **Reinforcement Learning (RL)**. It allows students to compare **Deep Q-Networks (DQN)** and **Q-Learning** in solving a maze by experimenting with different hyperparameters and epsilon strategies (decay, performance-based). The project logs results and generates visualizations to analyze performance.

## Installation

We recommend creating a virtual environment to install the required packages. Specifically, we recommend using `anaconda` or `virtualenv`. To create a virtual environment using `anaconda`, run the following command:

```bash
conda create -n project_3 python=3.10
conda activate project_3
```

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

Finally, you will need to install PyTorch. PyTorch installation will depend on your system configuration. To install PyTorch, follow the instructions on the [official website](https://pytorch.org/get-started/locally/).

## Maze Creation

You can create custom mazes using the Maze Editor. To create an N x N maze, run the following command:

```bash
python rl/maze_editor.py --size 10
```

You should then follow the instructions in the terminal to create the maze and save it to a file. This file can then be used in your experiments.

## Experiment Configs 
Experiments are configured using YAML files. Below is the structure of a sample config (DQN_decay.yaml):

```yaml
experiment_name: "10x10_goalstep_decay"
device: cpu # Use "cuda" if a GPU is available and you would like to use it
agent: "DQN"
maze_file: "mazes/10x10.txt"
reward:
  mode: "goal_and_step"  # goal_and_step | goal_only | manhattan
  goal_reward: 1.0
  step_penalty: -0.01
  other_reward: 0.0
  eta: 0.01
  beta: 0.1

hyperparameters:
  learning_rate: 0.001
  discount_rate: 0.99
  hidden_size: 256
  q_init_strategy: "zero"   # "zero" or "random_negative"
  q_init_random_low: -0.5   # Used only when q_init_strategy is "random_negative"
  q_init_random_high: 0.0   # Must be <= 0.0
  epsilon_policy: "decay"
  epsilon: 1.0
  epsilon_min: 0.1
  epsilon_decay: 0.99
  batch_size: 128
  target_update: 5

training:
  episodes: 500
  max_steps: 200
  render: False
  render_heatmap: True

evaluation:
  enabled: True
  num_videos: 5
  episodes: 1
  max_steps: 50
  save_video: True
  video_heatmap: True
  display: True
  fps: 8

viz:
  num_plots: 4
  window_size: null  # Auto-detect window size based on data length

save_dir: "results/"
```

## Reward Modes

The environment supports three reward modes via `reward.mode`:

1. `goal_and_step`
- If goal reached (`done=True`): reward = `goal_reward`
- Otherwise: reward = `step_penalty`
- Typical use: dense shaping that encourages shorter paths by penalizing every non-terminal step.

2. `goal_only`
- If goal reached (`done=True`): reward = `goal_reward`
- Otherwise: reward = `other_reward` (`0.0` by default)
- Typical use: sparse reward setup; the agent only gets a strong signal at the goal.

3. `manhattan`
- If goal reached (`done=True`): reward = `goal_reward`
- Otherwise:
  - `d_prev = ManhattanDistance(prev_pos, goal)`
  - `d_curr = ManhattanDistance(curr_pos, goal)`
  - reward = `-eta + beta * (d_prev - d_curr)`
- Typical use: potential-based style shaping.
  - Moving closer gives a positive term from `(d_prev - d_curr)`.
  - Moving away gives a negative term.
  - `eta` is a per-step cost; `beta` controls how strongly distance progress is rewarded.

## Running Experiments

Modify configuration files to change parameters such as:
- `maze_file`: Path to the maze file
- `reward.mode`: Reward function (`goal_and_step`, `goal_only`, or `manhattan`)
- `training.render`: Show live training window (useful for Q-learning)
- `training.render_heatmap`: Overlay cumulative visitation heatmap in live training render
- `evaluation.num_videos`: Total number of periodic eval videos, evenly spaced over training
- `evaluation.video_heatmap`: Overlay cumulative visitation heatmap onto saved eval videos
- `evaluation.display`: Show periodic eval rollouts during training (disable with `--no-eval-display`)
- `q_init_strategy`: Q-value initialization strategy. Use `zero` (all initial Q=0) or `random_negative` (small random values below 0).
- `q_init_random_low` / `q_init_random_high`: Random interval for `random_negative`. For Q-learning this is sampled per state-action entry; for DQN this is sampled per output-action bias value.
- `learning_rate`: Update step size (`alpha` in tabular Q-learning, optimizer learning rate in DQN)
- `discount_rate`: Future reward discount factor (`gamma`)
- `epsilon_policy`: Epsilon strategy (decay, performance-based)
- `epsilon`: Initial epsilon value ...

Run folders are created as `experiment_name_YYYYMMDD_XX` (e.g., `_01`, `_02`) to avoid overwriting multiple runs on the same day while keeping path lengths short.

The default YAML files may or may not be optimal or even sufficient for solving a given maze. You may need to experiment with different hyperparameters to achieve desired results.

## Saved Results and Visualization

After training, results are saved in the experiment's `save_dir`, e.g., `results/dqn_decay/`. The following files are stored:

- **`config.yaml`** - The YAML configuration used for the experiment
- **`rewards.npy`** - Rewards per episode
- **`epsilons.npy`** - Exploration rate over time
- **`episode_times.npy`** - Time taken per episode
- **`episode_steps.npy`** - Steps taken per episode
- **`trajectories.npy`** - Agent's movements
- **`summary.npy`** - Experiment summary (average reward, steps, etc.)

### Visualizations:
The project generates the following visualizations:
1. **Reward Progression**: Plots episode rewards with a moving average
2. **Steps Per Episode**: Shows the number of steps taken in each episode
3. **Epsilon Decay**: Tracks the exploration rate over episodes
4. **Episode Duration**: Displays time taken per episode
5. **Trajectory Heatmaps**: Highlights frequently visited locations in the maze

All visualizations are saved as images in the `save_dir`.  

### Important Note:

The saved results can and should be used in your analysis and report. The current visualizations are just there to demonstrate how to load and visualize results. It will be up to you to create the necessary visualizations for your analysis.


## Quickstart

To run an experiment, use the following command:

```bash
python run.py --config configs/q_learning/decay/random_negative/r1_goal_and_step.yaml
```

```bash
python run.py --config configs/q_learning/decay/random_negative/r1_goal_and_step.yaml --no-eval-display
```
