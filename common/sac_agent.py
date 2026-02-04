import parl
from parl.algorithms import SAC
import torch
import torch.nn as nn
from parl.utils import ReplayMemory
import numpy as np

GLOBE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


def to_tensor(item):
    return torch.tensor(item, dtype=torch.float32, device=GLOBE_DEVICE)


class Model(parl.Model):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(Model, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor_model = Actor(obs_dim, action_dim, hidden_size=hidden_size)
        self.critic_model = Critic(obs_dim, action_dim, hidden_size=hidden_size)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

    def forward(self, obs):
        x = torch.relu(self.l1(obs))
        x = torch.relu(self.l2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        return mean, log_std


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()

        # Q1 network
        self.l1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.q1_output = nn.Linear(hidden_size, 1)

        # Q2 network
        self.l3 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.q2_output = nn.Linear(hidden_size, 1)

    def forward(self, obs, action):
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=1)

        # Q1 network
        q1 = torch.relu(self.l1(x))
        q1 = torch.relu(self.l2(q1))
        q1 = self.q1_output(q1)

        # Q2 network
        q2 = torch.relu(self.l3(x))
        q2 = torch.relu(self.l4(q2))
        q2 = self.q2_output(q2)

        return q1, q2


class Agent(parl.Agent):
    def __init__(
        self,
        obs_dim,
        action_dim,
        batch_size,
        max_size,
        lr=0.0001,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(obs_dim, action_dim)
        algorithm = SAC(
            model=model, actor_lr=lr, critic_lr=lr, gamma=gamma, tau=tau, alpha=alpha
        )
        super(Agent, self).__init__(algorithm)

        self.alg.model.to(self.device)
        self.alg.sync_target(decay=0)
        self.replay_buffer = ReplayMemory(max_size, obs_dim, act_dim=action_dim)
        self.batch_size = batch_size

    def predict(self, obs):
        if len(obs.shape) == 1:
            obs = torch.tensor(
                obs.reshape(1, -1), dtype=torch.float32, device=self.device
            )
            action = self.alg.predict(obs)
            action_numpy = action.detach().cpu().numpy()[0]
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action = self.alg.predict(obs)
            action_numpy = action.detach().cpu().numpy()
        return action_numpy

    def sample(self, obs):
        if len(obs.shape) == 1:
            obs = torch.tensor(
                obs.reshape(1, -1), dtype=torch.float32, device=self.device
            )
            action, _ = self.alg.sample(obs)
            action_numpy = action.detach().cpu().numpy()[0]
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action, _ = self.alg.sample(obs)
            action_numpy = action.detach().cpu().numpy()
        return action_numpy

    def learn(self):
        if self.replay_buffer.size() < self.batch_size:
            return 0, 0
        (
            batch_obs,
            batch_action,
            batch_reward,
            batch_next_obs,
            batch_terminal,
        ) = self.replay_buffer.sample_batch(self.batch_size)
        batch_terminal = np.expand_dims(batch_terminal, -1)
        batch_reward = np.expand_dims(batch_reward, -1)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        batch_action = torch.tensor(
            batch_action, dtype=torch.float32, device=self.device
        )
        batch_reward = torch.tensor(
            batch_reward, dtype=torch.float32, device=self.device
        )
        batch_next_obs = torch.tensor(
            batch_next_obs, dtype=torch.float32, device=self.device
        )
        batch_terminal = torch.tensor(
            batch_terminal, dtype=torch.float32, device=self.device
        )
        critic_loss, actor_loss = self.alg.learn(
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal
        )

        return (
            critic_loss.detach().cpu().numpy(),
            actor_loss.detach().cpu().numpy(),
        )

    def append(self, obs, action, reward, next_obs, terminal):
        return self.replay_buffer.append(obs, action, reward, next_obs, terminal)
