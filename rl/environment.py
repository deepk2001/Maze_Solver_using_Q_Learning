import numpy as np
import pygame

class MazeEnvironment:
    def __init__(self, size=10, maze_file=None, reward_config=None):
        self.size = size
        self.agent_pos = (1, 1)  # Default start position
        self.start_pos = self.agent_pos
        self.goal_pos = (size-2, size-2)  # Default goal position
        self.reward_config = reward_config or {}
        self.reward_mode = self.reward_config.get("mode", "goal_and_step")
        self.goal_reward = float(self.reward_config.get("goal_reward", 1.0))
        self.step_penalty = float(self.reward_config.get("step_penalty", -0.01))
        self.other_reward = float(self.reward_config.get("other_reward", 0.0))
        self.eta = float(self.reward_config.get("eta", 0.01))
        self.beta = float(self.reward_config.get("beta", 0.1))
        
        # Initialize pygame for visualization
        pygame.init()
        self.cell_size = 50
        self.screen_size = self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Maze Environment")
        
        # Colors
        self.colors = {
            'wall': (0, 0, 0),
            'empty': (255, 255, 255),
            'goal': (0, 255, 0),
            'start': (255, 0, 0),
        }
        
        # Load and scale the agent sprite
        self.agent_image = pygame.image.load("assets/walle.png")
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        
        # Load maze from file or create random maze
        if maze_file:
            self._load_maze(maze_file)
        else:
            self.maze = np.zeros((self.size, self.size))
            # Add walls randomly
            self.maze[np.random.random((self.size, self.size)) < 0.3] = 1
            # Ensure start and goal positions are empty
            self.start_pos = self.agent_pos
            self.maze[self.agent_pos] = 0
            self.maze[self.goal_pos] = 0
        
        # Store trajectory for tracking movements
        self.trajectory = []
        self.action_size = 4
        self.last_move_blocked = False
        self.last_attempt_dir = (0, 0)
        self.last_attempt_target = self.agent_pos

    def _load_maze(self, maze_file):
        """Load maze configuration from file"""
        with open(maze_file, 'r') as f:
            self.size = int(f.readline().strip())
            self.maze = np.zeros((self.size, self.size))
            for i, line in enumerate(f):
                line = line.strip()
                if i >= self.size:
                    break
                for j, char in enumerate(line[:self.size]):
                    if char == 'S':
                        self.agent_pos = (i, j)
                        self.start_pos = (i, j)
                    elif char == 'G':
                        self.goal_pos = (i, j)
                    elif char == '#':
                        self.maze[i, j] = 1

    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = self.start_pos if hasattr(self, 'start_pos') else (1, 1)
        self.trajectory = [self.agent_pos]
        self.last_move_blocked = False
        self.last_attempt_dir = (0, 0)
        self.last_attempt_target = self.agent_pos
        return self._get_observation()

    def _get_observation(self):
        """Return full maze plus agent/start/goal coordinates."""
        ax, ay = self.agent_pos
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        map_obs = self.maze.flatten().astype(np.float32)
        coord_obs = np.array([ax, ay, sx, sy, gx, gy], dtype=np.float32)
        return np.concatenate([map_obs, coord_obs], axis=0)

    @property
    def observation_size(self):
        return self.size * self.size + 6

    @property
    def coord_dims(self):
        return 6

    def step(self, action):
        """Take action in environment"""
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        x, y = self.agent_pos
        prev_pos = self.agent_pos

        dx, dy = directions[action]
        new_x, new_y = x + dx, y + dy
        self.last_attempt_dir = (dx, dy)
        self.last_attempt_target = (new_x, new_y)

        if self._is_valid_cell(new_x, new_y):
            self.agent_pos = (new_x, new_y)
            self.trajectory.append(self.agent_pos)
            self.last_move_blocked = False
        else:
            # Agent is allowed to try moving into 
            # walls/out-of-bounds and make no progress.
            self.last_move_blocked = True
            
        done = self.agent_pos == self.goal_pos
        reward = self._compute_reward(prev_pos, self.agent_pos, done)
        return self._get_observation(), reward, done

    def _is_valid_cell(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x, y] != 1

    def _distance_to_goal(self, pos):
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

    def _compute_reward(self, prev_pos, current_pos, done):
        mode = str(self.reward_mode).lower()

        if mode == 'goal_and_step':
            return self.goal_reward if done else self.step_penalty

        elif mode == 'goal_only':
            return self.goal_reward if done else self.other_reward

        elif mode == 'manhattan':
            if done:
                return self.goal_reward
            d_prev = self._distance_to_goal(prev_pos)
            d_curr = self._distance_to_goal(current_pos)
            return -self.eta + self.beta * (d_prev - d_curr)

        raise ValueError(
            f"Unsupported reward mode '{self.reward_mode}'. "
            "Use one of: goal_and_step, goal_only, manhattan."
        )

    def render(
        self,
        title=None,
        display=True,
        show_bump=False,
        visit_counts=None,
        heatmap_exclusions=None,
        heatmap_alpha=0.42
    ):
        """Render the maze and agent; optional title is shown in window caption."""
        self.screen.fill((255, 255, 255))
        if title:
            pygame.display.set_caption(title)
        else:
            pygame.display.set_caption("Maze Environment")
        maze_y_offset = 0
        for i in range(self.size):
            for j in range(self.size):
                color = self.colors['wall'] if self.maze[i, j] == 1 else self.colors['empty']
                pygame.draw.rect(self.screen, color, 
                                 (j * self.cell_size, i * self.cell_size + maze_y_offset, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (200, 200, 200), 
                                 (j * self.cell_size, i * self.cell_size + maze_y_offset, self.cell_size, self.cell_size), 1)

        if visit_counts is not None:
            max_count = float(np.max(visit_counts))
            if max_count > 0.0:
                norm = np.log1p(visit_counts.astype(np.float32)) / np.log1p(max_count)
                mask = norm > 0.0
                if heatmap_exclusions:
                    for x, y in heatmap_exclusions:
                        if 0 <= x < self.size and 0 <= y < self.size:
                            mask[x, y] = False

                overlay = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
                for i in range(self.size):
                    for j in range(self.size):
                        if not mask[i, j]:
                            continue
                        v = float(norm[i, j])
                        r = int(np.clip((2.5 * v - 0.5), 0.0, 1.0) * 255)
                        g = int(np.clip((3.0 * v - 1.0), 0.0, 1.0) * 255)
                        b = int(np.clip((1.5 - 3.0 * v), 0.0, 1.0) * 255)
                        a = int(np.clip(heatmap_alpha, 0.0, 1.0) * 255)
                        pygame.draw.rect(
                            overlay,
                            (r, g, b, a),
                            (j * self.cell_size, i * self.cell_size + maze_y_offset, self.cell_size, self.cell_size)
                        )
                self.screen.blit(overlay, (0, 0))

        # Draw start
        pygame.draw.rect(self.screen, self.colors['start'], 
                         (self.start_pos[1] * self.cell_size, self.start_pos[0] * self.cell_size + maze_y_offset, 
                          self.cell_size, self.cell_size))
         
        # Draw goal
        pygame.draw.rect(self.screen, self.colors['goal'], 
                         (self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size + maze_y_offset, 
                          self.cell_size, self.cell_size))

        if (
            show_bump
            and self.last_move_blocked
            and 0 <= self.last_attempt_target[0] < self.size
            and 0 <= self.last_attempt_target[1] < self.size
        ):
            tx, ty = self.last_attempt_target
            pygame.draw.rect(
                self.screen,
                (255, 140, 0),
                (ty * self.cell_size, tx * self.cell_size + maze_y_offset, self.cell_size, self.cell_size),
                width=3
            )

        # Draw agent using sprite (with optional bump offset for blocked moves)
        offset_x, offset_y = 0, 0
        if show_bump and self.last_move_blocked:
            bump_px = max(2, int(self.cell_size * 0.2))
            dx, dy = self.last_attempt_dir
            offset_x = dy * bump_px
            offset_y = dx * bump_px

        self.screen.blit(self.agent_image, 
                         (self.agent_pos[1] * self.cell_size + offset_x,
                          self.agent_pos[0] * self.cell_size + maze_y_offset + offset_y))

        if display:
            pygame.display.flip()

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2)).copy()
    
    def close(self):
        """Close the environment"""
        pygame.quit()
        
