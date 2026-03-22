import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from .model import DQN

class DQNAgent:
    def __init__(
        self,
        state_size=6,
        action_size=4,
        hidden_size=128,
        discount_rate=0.99,
        learning_rate=1e-3,
        batch_size=128,
        target_update=5,
        device="cpu",
        epsilon_policy="decay",
        **kwargs
    ):
        self.device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update = max(1, int(target_update))
        self.epsilon_policy = epsilon_policy
        self.coord_scale = max(1.0, float(kwargs.get("coord_scale", 1.0)))
        self.coord_dims = max(0, int(kwargs.get("coord_dims", 2)))
        self._update_calls = 0

        if epsilon_policy == "decay":
            self._epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        elif epsilon_policy == "performance_based":
            self._epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)
        else:
            raise ValueError("Unsupported epsilon_policy. Use 'decay' or 'performance_based'.")
        self.q_init_strategy = kwargs.get("q_init_strategy", "zero")
        self.q_init_random_low = float(kwargs.get("q_init_random_low", -0.5))
        self.q_init_random_high = float(kwargs.get("q_init_random_high", 0.0))

        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self._initialize_q_output_head()
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)
        self.recent_rewards = deque(maxlen=100)

    @property
    def epsilon(self):
        return self._epsilon

    def _initialize_q_output_head(self):
        # Keep initial Q output independent of state; strategy sets per-action bias values.
        with torch.no_grad():
            self.policy_net.fc3.weight.zero_()
            if self.q_init_strategy == "zero":
                self.policy_net.fc3.bias.zero_()
            elif self.q_init_strategy == "random_negative":
                if self.q_init_random_low >= self.q_init_random_high:
                    raise ValueError("q_init_random_low must be less than q_init_random_high.")
                if self.q_init_random_high > 0.0:
                    raise ValueError("q_init_random_high must be <= 0.0 for random_negative strategy.")
                self.policy_net.fc3.bias.uniform_(self.q_init_random_low, self.q_init_random_high)
            else:
                raise ValueError("Unsupported q_init_strategy. Use 'zero' or 'random_negative'.")

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(self._preprocess_state(state)).unsqueeze(0).to(self.device)
            return self.policy_net(state).argmax().item()

    def greedy_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(self._preprocess_state(state)).unsqueeze(0).to(self.device)
            return self.policy_net(state).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)

    def get_recent_rewards(self, n):
        rewards_list = list(self.recent_rewards)
        return rewards_list[-n:] if len(rewards_list) >= n else rewards_list

    def _preprocess_state(self, state):
        state = np.asarray(state, dtype=np.float32).copy()
        if self.coord_dims > 0 and state.shape[-1] >= self.coord_dims:
            state[-self.coord_dims:] = state[-self.coord_dims:] / self.coord_scale
        return state

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array([self._preprocess_state(s) for s in states], dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array([self._preprocess_state(s) for s in next_states], dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.discount_rate * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update(self):
        self._update_calls += 1
        if self._update_calls % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon_policy == "decay":
            self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        elif self.epsilon_policy == "performance_based":
            recent_rewards = self.get_recent_rewards(20)
            if len(recent_rewards) >= 20:
                recent_mean = np.mean(recent_rewards[-10:])
                old_mean = np.mean(recent_rewards[:10])
                if recent_mean > old_mean:
                    self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
