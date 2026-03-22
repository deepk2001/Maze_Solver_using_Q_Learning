import argparse
import numpy as np
import yaml
import os
import time
import torch
import re
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from rl.environment import MazeEnvironment
from rl.DQN_agent import DQNAgent
from rl.Q_learning_agent import QLearningAgent
from rl.logger import ExperimentLogger
from rl.viz import analyze_experiment

# Define parameter filters for different policies
policy_param_keys = {
    "decay": ["epsilon", "epsilon_min", "epsilon_decay"],
    "performance_based": ["epsilon", "epsilon_min", "epsilon_decay"],
}

def format_duration(seconds):
    total = int(max(0, round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def load_experiment_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def _sanitize_name(value):
    return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_") or "unknown"

def _algo_dir_name(agent_name):
    agent_key = _sanitize_name(agent_name)
    if "dqn" in agent_key:
        return "dqn"
    if "q" in agent_key:
        return "q_learning"
    return agent_key

def _init_dir_name(q_init_strategy):
    strategy = _sanitize_name(q_init_strategy)
    if strategy == "zero":
        return "zero_init"
    if strategy == "random_negative":
        return "random_negative_init"
    return f"{strategy}_init"

def build_save_base_dir(config):
    hp = config.get('hyperparameters', {})
    reward_cfg = config.get('reward', {})
    algo_dir = _algo_dir_name(config.get('agent', 'unknown'))
    init_dir = _init_dir_name(hp.get('q_init_strategy', 'zero'))
    policy_dir = _sanitize_name(hp.get('epsilon_policy', 'unknown_policy'))
    reward_dir = _sanitize_name(reward_cfg.get('mode', 'unknown_reward'))
    return os.path.join("results", algo_dir, init_dir, policy_dir, reward_dir)

def build_next_run_dir(save_base_dir, maze_file):
    maze_name = os.path.splitext(os.path.basename(str(maze_file)))[0] or "maze"
    os.makedirs(save_base_dir, exist_ok=True)
    matching_count = 0
    pattern = re.compile(rf"^{re.escape(maze_name)}(?:_(\d+))?$")
    for name in os.listdir(save_base_dir):
        full_path = os.path.join(save_base_dir, name)
        if not os.path.isdir(full_path):
            continue
        if pattern.match(name):
            matching_count += 1
    run_name = maze_name if matching_count == 0 else f"{maze_name}_{matching_count + 1}"
    return os.path.join(save_base_dir, run_name)

def build_eval_config(config):
    eval_cfg = config.get('evaluation', {})
    return {
        'enabled': bool(eval_cfg.get('enabled', True)),
        'num_videos': max(0, int(eval_cfg.get('num_videos', eval_cfg.get('num_recordings', 5)))),
        'episodes': max(1, int(eval_cfg.get('episodes', 1))),
        'max_steps': max(1, int(eval_cfg.get('max_steps', config['training']['max_steps']))),
        'save_video': bool(eval_cfg.get('save_video', True)),
        'video_heatmap': bool(eval_cfg.get('video_heatmap', True)),
        'display': bool(eval_cfg.get('display', True)),
        'fps': max(1, int(eval_cfg.get('fps', 8))),
    }

def overlay_visit_heatmap(frame, visit_counts, cell_size, alpha=0.42, blocked_cells=None):
    """Blend a cumulative visitation heatmap over a maze frame."""
    if frame is None or visit_counts is None:
        return frame
    max_count = float(np.max(visit_counts))
    if max_count <= 0.0:
        return frame

    norm = np.log1p(visit_counts.astype(np.float32)) / np.log1p(max_count)
    mask = norm > 0
    if blocked_cells:
        for x, y in blocked_cells:
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]:
                mask[x, y] = False
    if not np.any(mask):
        return frame

    # Simple hot colormap (blue->green->yellow->red) without extra dependencies.
    r = np.clip(2.5 * norm - 0.5, 0.0, 1.0)
    g = np.clip(3.0 * norm - 1.0, 0.0, 1.0)
    b = np.clip(1.5 - 3.0 * norm, 0.0, 1.0)
    cell_colors = np.stack([r, g, b], axis=-1)

    heatmap = np.kron(cell_colors, np.ones((cell_size, cell_size, 1), dtype=np.float32))
    heatmask = np.kron(mask.astype(np.float32), np.ones((cell_size, cell_size), dtype=np.float32))
    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.float32)

    out = frame.astype(np.float32).copy()
    blend_alpha = alpha * heatmask[..., None]
    out = (1.0 - blend_alpha) * out + blend_alpha * heatmap_u8
    return np.clip(out, 0, 255).astype(np.uint8)

def add_titles_above_frames(frames, titles, header_px=36):
    """Render a title strip above each frame using matplotlib (not pygame)."""
    if not frames or not titles or len(frames) != len(titles):
        return frames

    h, w = frames[0].shape[:2]
    dpi = 100
    fig = Figure(figsize=(w / dpi, (h + header_px) / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 1, height_ratios=[header_px, h], hspace=0.0)
    ax_title = fig.add_subplot(grid[0])
    ax_image = fig.add_subplot(grid[1])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax_title.set_facecolor((0.96, 0.96, 0.96))
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    for spine in ax_title.spines.values():
        spine.set_visible(False)
    title_text = ax_title.text(
        0.5, 0.5, "", ha='center', va='center', fontsize=12, color='black'
    )

    ax_image.set_axis_off()
    image_artist = ax_image.imshow(frames[0])

    output_frames = []
    for frame, title in zip(frames, titles):
        image_artist.set_data(frame)
        title_text.set_text(title)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        output_frames.append(rgba[:, :, :3].copy())

    return output_frames

def save_rollout_video(frames, base_path, fps, titles=None):
    if not frames:
        return None
    frames_to_write = add_titles_above_frames(frames, titles) if titles else frames
    try:
        import imageio.v2 as imageio  # Optional dependency for writing mp4/gif
        mp4_path = f"{base_path}.mp4"
        try:
            with imageio.get_writer(mp4_path, fps=fps) as writer:
                for frame in frames_to_write:
                    writer.append_data(frame)
            return mp4_path
        except Exception:
            gif_path = f"{base_path}.gif"
            imageio.mimsave(gif_path, frames_to_write, duration=1.0 / fps)
            return gif_path
    except Exception:
        fallback_path = f"{base_path}.npy"
        np.save(fallback_path, np.array(frames_to_write, dtype=np.uint8))
        return fallback_path

def run_periodic_evaluation(config, agent, eval_env, episode_idx, video_dir, eval_cfg, display_eval):
    rewards, steps, video_paths = [], [], []
    should_render = bool(display_eval or eval_cfg['save_video'])

    for ep in range(eval_cfg['episodes']):
        state = eval_env.reset()
        total_reward = 0.0
        frames = []
        frame_titles = []
        heatmap_exclusions = {tuple(eval_env.start_pos), tuple(eval_env.goal_pos)}
        visit_counts = np.zeros((eval_env.size, eval_env.size), dtype=np.float32)
        sx, sy = eval_env.agent_pos
        visit_counts[sx, sy] += 1.0

        if should_render:
            title = f"Episode: {episode_idx}"
            frame = eval_env.render(title=title, display=display_eval)
            if eval_cfg['save_video']:
                if eval_cfg['video_heatmap']:
                    frame = overlay_visit_heatmap(
                        frame, visit_counts, eval_env.cell_size, blocked_cells=heatmap_exclusions
                    )
                frames.append(frame)
                frame_titles.append(title)

        for step in range(eval_cfg['max_steps']):
            with torch.no_grad():
                if hasattr(agent, 'greedy_action'):
                    action = agent.greedy_action(state)
                else:
                    action = agent.act(state)
            next_state, reward, done = eval_env.step(action)
            state = next_state
            total_reward += reward
            px, py = eval_env.agent_pos
            visit_counts[px, py] += 1.0

            if should_render:
                title = f"Episode: {episode_idx} | Step: {step + 1}"
                if eval_env.last_move_blocked:
                    bump_frame = eval_env.render(title=title, display=display_eval, show_bump=True)
                    if eval_cfg['save_video']:
                        if eval_cfg['video_heatmap']:
                            bump_frame = overlay_visit_heatmap(
                                bump_frame, visit_counts, eval_env.cell_size, blocked_cells=heatmap_exclusions
                            )
                        frames.append(bump_frame)
                        frame_titles.append(title)
                frame = eval_env.render(title=title, display=display_eval)
                if eval_cfg['save_video']:
                    if eval_cfg['video_heatmap']:
                        frame = overlay_visit_heatmap(
                            frame, visit_counts, eval_env.cell_size, blocked_cells=heatmap_exclusions
                        )
                    frames.append(frame)
                    frame_titles.append(title)

            if done:
                break

        rewards.append(float(total_reward))
        steps.append(step + 1)

        if eval_cfg['save_video']:
            video_base = os.path.join(video_dir, f"episode_{episode_idx:05d}_eval_{ep + 1:02d}")
            video_path = save_rollout_video(frames, video_base, eval_cfg['fps'], titles=frame_titles)
            video_paths.append(video_path)

    return {
        'episode': int(episode_idx),
        'avg_reward': float(np.mean(rewards)),
        'avg_steps': float(np.mean(steps)),
        'rewards': rewards,
        'steps': steps,
        'videos': video_paths,
    }

def train(config, no_eval_display=False):
    train_start_epoch = time.time()
    train_start_iso = datetime.now().isoformat(timespec='seconds')

    save_base_dir = build_save_base_dir(config)
    save_dir = build_next_run_dir(save_base_dir, config.get('maze_file', 'maze'))
    os.makedirs(save_dir, exist_ok=False)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    env = MazeEnvironment(
        size=10,
        maze_file=config['maze_file'],
        reward_config=config.get('reward', {})
    )
    eval_cfg = build_eval_config(config)
    train_cfg = config.get('training', {})
    train_render = bool(train_cfg.get('render', False))
    train_render_heatmap = bool(train_cfg.get('render_heatmap', True))
    eval_display = eval_cfg['display'] and not no_eval_display
    video_dir = os.path.join(save_dir, 'videos')
    if eval_cfg['save_video']:
        os.makedirs(video_dir, exist_ok=True)
    eval_env = MazeEnvironment(
        size=10,
        maze_file=config['maze_file'],
        reward_config=config.get('reward', {})
    ) if eval_cfg['enabled'] else None

    policy = config['hyperparameters']['epsilon_policy']
    if policy not in policy_param_keys:
        valid = ", ".join(sorted(policy_param_keys.keys()))
        raise ValueError(f"Unsupported epsilon_policy '{policy}'. Supported values: {valid}")
    policy_params = {k: v for k, v in config['hyperparameters'].items() if k in policy_param_keys[policy]}
    state_size = env.observation_size
    action_size = env.action_size

    if config['agent'] == 'Q-learning':
        learning_rate = config['hyperparameters'].get('learning_rate', config['hyperparameters'].get('alpha'))
        discount_rate = config['hyperparameters'].get('discount_rate', config['hyperparameters'].get('gamma'))
        if learning_rate is None or discount_rate is None:
            raise ValueError("Q-learning hyperparameters must include learning_rate and discount_rate.")
        agent = QLearningAgent(
            state_size=state_size, action_size=action_size,
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            epsilon_policy=policy,
            q_init_strategy=str(config['hyperparameters'].get('q_init_strategy', 'zero')),
            q_init_random_low=float(config['hyperparameters'].get('q_init_random_low', -0.5)),
            q_init_random_high=float(config['hyperparameters'].get('q_init_random_high', 0.0)),
            **policy_params
        )
    else:
        learning_rate = config['hyperparameters'].get('learning_rate', 1e-3)
        discount_rate = config['hyperparameters'].get('discount_rate', config['hyperparameters'].get('gamma'))
        if discount_rate is None:
            raise ValueError("DQN hyperparameters must include discount_rate.")
        agent = DQNAgent(
            state_size=state_size, action_size=action_size,
            hidden_size=config['hyperparameters']['hidden_size'],
            learning_rate=learning_rate,
            discount_rate=discount_rate,
            batch_size=config['hyperparameters']['batch_size'],
            target_update=config['hyperparameters']['target_update'],
            device=config.get('device', 'cpu'),
            epsilon_policy=policy,
            coord_scale=max(1, env.size - 1),
            coord_dims=env.coord_dims,
            q_init_strategy=str(config['hyperparameters'].get('q_init_strategy', 'zero')),
            q_init_random_low=float(config['hyperparameters'].get('q_init_random_low', -0.5)),
            q_init_random_high=float(config['hyperparameters'].get('q_init_random_high', 0.0)),
            **policy_params
        )

    logger = ExperimentLogger(save_dir)
    best_reward = -np.inf
    periodic_eval_records = []
    total_episodes = config['training']['episodes']
    eval_trigger_episodes = set()
    if eval_cfg['enabled'] and eval_cfg['num_videos'] > 0:
        num_videos = int(eval_cfg['num_videos'])
        for i in range(1, num_videos + 1):
            trigger_episode = int(round((i * total_episodes) / num_videos))
            trigger_episode = max(1, min(total_episodes, trigger_episode))
            if trigger_episode <= total_episodes:
                eval_trigger_episodes.add(trigger_episode)

    for episode in range(config['training']['episodes']):
        episode_start_time = time.time()
        state = env.reset()
        total_reward, episode_trajectory, ep_loss = 0, [], 0
        train_visit_counts = np.zeros((env.size, env.size), dtype=np.float32)
        tx, ty = env.agent_pos
        train_visit_counts[tx, ty] += 1.0
        train_heatmap_exclusions = {tuple(env.start_pos), tuple(env.goal_pos)}

        for step in range(config['training']['max_steps']):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            ep_loss += loss if loss else 0
            state = next_state
            total_reward += reward
            episode_trajectory.append(env.agent_pos)
            tx, ty = env.agent_pos
            train_visit_counts[tx, ty] += 1.0

            if train_render:
                title = f"Train Episode: {episode + 1} | Step: {step + 1}"
                env.render(
                    title=title,
                    display=True,
                    show_bump=env.last_move_blocked,
                    visit_counts=(train_visit_counts if train_render_heatmap else None),
                    heatmap_exclusions=(train_heatmap_exclusions if train_render_heatmap else None),
                )

            if done:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            if hasattr(agent, 'save'):
                agent.save(os.path.join(save_dir, 'best_model.pth'))

        episode_time = time.time() - episode_start_time
        logger.log_episode(total_reward, ep_loss, episode_trajectory,
                           episode_time, step + 1, agent.epsilon)

        agent.update()

        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Epsilon={agent.epsilon:.2f}, Time={episode_time:.2f}s")

        should_run_eval = False
        if eval_cfg['enabled']:
            should_run_eval = (episode + 1) in eval_trigger_episodes

        if should_run_eval:
            eval_record = run_periodic_evaluation(
                config=config,
                agent=agent,
                eval_env=eval_env,
                episode_idx=episode + 1,
                video_dir=video_dir,
                eval_cfg=eval_cfg,
                display_eval=eval_display
            )
            periodic_eval_records.append(eval_record)
            print(
                f"Periodic Eval @ episode {episode + 1}: "
                f"avg_reward={eval_record['avg_reward']:.3f}, "
                f"avg_steps={eval_record['avg_steps']:.1f}"
            )

    logger.save_logs()
    if periodic_eval_records:
        np.save(os.path.join(save_dir, 'periodic_eval.npy'),
                np.array(periodic_eval_records, dtype=object))

    maze = env.maze
    num_episodes = config['training']['episodes']
    num_plots = config['viz']['num_plots']
    episodes_to_plot = np.arange(num_episodes) if num_episodes < num_plots else \
        np.unique(np.linspace(0, num_episodes - 1, num_plots, dtype=int))

    window = max(5, min(50, num_episodes // 10)) if config['viz']['window_size'] is None else config['viz']['window_size']
    analyze_experiment(save_dir, maze, episodes_to_plot, window)
    if eval_env is not None:
        eval_env.close()
    train_end_epoch = time.time()
    train_end_iso = datetime.now().isoformat(timespec='seconds')
    train_elapsed = train_end_epoch - train_start_epoch
    episode_steps = np.array(logger.episode_steps, dtype=np.float32)
    episode_times = np.array(logger.episode_times, dtype=np.float32)
    timing_summary = {
        'phase': 'train',
        'start_time_iso': train_start_iso,
        'end_time_iso': train_end_iso,
        'elapsed_seconds': float(train_elapsed),
        'elapsed_hms': format_duration(train_elapsed),
        'num_episodes': int(len(episode_steps)),
        'episode_steps_avg': float(np.mean(episode_steps)) if episode_steps.size else 0.0,
        'episode_steps_min': int(np.min(episode_steps)) if episode_steps.size else 0,
        'episode_steps_max': int(np.max(episode_steps)) if episode_steps.size else 0,
        'episode_time_avg_seconds': float(np.mean(episode_times)) if episode_times.size else 0.0,
    }
    np.save(os.path.join(save_dir, 'train_timing.npy'), timing_summary)
    print(
        f"Training summary: start={train_start_iso}, end={train_end_iso}, "
        f"elapsed={timing_summary['elapsed_hms']}, "
        f"episode_steps(avg/min/max)="
        f"{timing_summary['episode_steps_avg']:.1f}/"
        f"{timing_summary['episode_steps_min']}/"
        f"{timing_summary['episode_steps_max']}"
    )

    return save_dir, agent, env, timing_summary

def evaluate_agent(config, save_dir, agent, env, episodes=10):
    print("\n🔍 Running final evaluation with best policy...")
    
    if hasattr(agent, 'load'):
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        agent.load(best_model_path)
        agent.policy_net.eval()
        
    total_rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []

        for step in range(config['training']['max_steps']):
            with torch.no_grad():
                if hasattr(agent, 'greedy_action'):
                    action = agent.greedy_action(state)
                else:
                    action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            trajectory.append(env.agent_pos)

            if done:
                break

        total_rewards.append(total_reward)
        print(f"Evaluation {ep+1}: Reward = {total_reward:.2f}")

    avg_eval_reward = np.mean(total_rewards)
    print(f"\n🏁 Average Evaluation Reward over {episodes} episodes: {avg_eval_reward:.2f}")

def main():
    run_start_epoch = time.time()
    run_start_iso = datetime.now().isoformat(timespec='seconds')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment YAML config')
    parser.add_argument('--no-eval-display', action='store_true',
                        help='Disable periodic evaluation display window while still allowing video saving.')
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    save_dir, agent, env, train_timing = train(config, no_eval_display=args.no_eval_display)
    print(f"\nExperiment completed. Results saved to: {save_dir}")

    evaluate_agent(config, save_dir, agent, env, episodes=10)
    env.close()
    run_end_epoch = time.time()
    run_end_iso = datetime.now().isoformat(timespec='seconds')
    run_elapsed = run_end_epoch - run_start_epoch
    run_timing = {
        'phase': 'full_run',
        'start_time_iso': run_start_iso,
        'end_time_iso': run_end_iso,
        'elapsed_seconds': float(run_elapsed),
        'elapsed_hms': format_duration(run_elapsed),
        'training_elapsed_seconds': float(train_timing['elapsed_seconds']),
        'training_elapsed_hms': str(train_timing['elapsed_hms']),
    }
    np.save(os.path.join(save_dir, 'run_timing.npy'), run_timing)
    print(
        f"Run summary: start={run_start_iso}, end={run_end_iso}, "
        f"total_elapsed={run_timing['elapsed_hms']}"
    )

if __name__ == "__main__":
    main()
