import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size=4, learning_rate=0.1, discount_rate=0.99, epsilon_policy="decay", **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon_policy = epsilon_policy

        if epsilon_policy == "decay":
            self._epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)
        elif epsilon_policy == "performance_based":
            self._epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)
        else:
            raise ValueError("Unsupported epsilon_policy. Use 'decay' or 'performance_based'.")

        self.q_init_strategy = kwargs.get("q_init_strategy", "zero")
        self.q_init_random_low = float(kwargs.get("q_init_random_low", -0.5))
        self.q_init_random_high = float(kwargs.get("q_init_random_high", 0.0))
        self.q_table = {}
        self.recent_rewards = []

    @property
    def epsilon(self):
        return self._epsilon

    def get_state_key(self, state):
        return tuple(np.round(state, decimals=2))

    def _ensure_state(self, state_key):
        if state_key not in self.q_table:
            if self.q_init_strategy == "zero":
                self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)
            elif self.q_init_strategy == "random_negative":
                if self.q_init_random_low >= self.q_init_random_high:
                    raise ValueError("q_init_random_low must be less than q_init_random_high.")
                if self.q_init_random_high > 0.0:
                    raise ValueError("q_init_random_high must be <= 0.0 for random_negative strategy.")
                self.q_table[state_key] = np.random.uniform(
                    low=self.q_init_random_low,
                    high=self.q_init_random_high,
                    size=self.action_size
                ).astype(np.float32)
            else:
                raise ValueError("Unsupported q_init_strategy. Use 'zero' or 'random_negative'.")

    def act(self, state):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))

        self._ensure_state(state_key)
        
        return np.argmax(self.q_table[state_key])

    def greedy_action(self, state):
        state_key = self.get_state_key(state)
        self._ensure_state(state_key)
        return np.argmax(self.q_table[state_key])

    def remember(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        self._ensure_state(state_key)
        self._ensure_state(next_state_key)

        best_next_action = np.argmax(self.q_table[next_state_key])
        target = reward + (1 - done) * self.discount_rate * self.q_table[next_state_key][best_next_action]
        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])

        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)

    def train(self):
        pass  # for compatibility

    def update(self):
        if self.epsilon_policy == "decay":
            self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        elif self.epsilon_policy == "performance_based":
            if len(self.recent_rewards) >= 20:
                recent_mean = np.mean(self.recent_rewards[-10:])
                old_mean = np.mean(self.recent_rewards[:10])
                if recent_mean > old_mean:
                    self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
