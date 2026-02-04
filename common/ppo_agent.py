import numpy as np
import parl
import torch
import torch.nn as nn
from parl.utils.scheduler import LinearDecayScheduler
from parl.algorithms import PPO

GLOBE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(item):
    return torch.tensor(item, dtype=torch.float32, device=GLOBE_DEVICE)


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(Model, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_mean = nn.Linear(self.hidden_size, self.act_dim)
        self.fc_std = nn.Parameter(torch.zeros(self.act_dim))
        self.fc_v = nn.Linear(self.hidden_size, 1)

    def forward(self, obs):
        pass

    def policy(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.fc_std)
        return mean, std

    def value(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        value = self.fc_v(x)
        return value


class PPOAgent(parl.Agent):
    """Agent of PPO env"""

    def __init__(self, config):
        model = Model(
            obs_dim=config["obs_dim"],
            act_dim=config["act_dim"],
            hidden_size=config["hidden_size"],
        )
        algorithm = PPO(
            model,
            clip_param=config["clip_param"],
            entropy_coef=config["entropy_coef"],
            initial_lr=config["initial_lr"],
            continuous_action=True,
        )
        super(PPOAgent, self).__init__(algorithm)
        self.config = config
        if self.config["lr_decay"]:
            self.lr_scheduler = LinearDecayScheduler(
                self.config["initial_lr"], self.config["num_updates"]
            )

    def predict(self, obs):
        """Predict action from current policy given observation

        Args:
            obs (np.array): observation, shape([batch_size] + obs_shape)
        """
        # Add batch dimension if needed
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs = to_tensor(obs)
        action = self.alg.predict(obs)
        action_numpy = action.detach().cpu().numpy()
        return action_numpy

    def sample(self, obs):
        """Sample action from current policy given observation

        Args:
            obs (np.array): observation, shape([batch_size] + obs_shape)
        """
        # Add batch dimension if needed
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs = to_tensor(obs)
        value, action, action_log_probs, action_entropy = self.alg.sample(obs)
        value_numpy = value.detach().cpu().numpy()
        action_numpy = action.detach().cpu().numpy()
        action_log_probs_numpy = action_log_probs.detach().cpu().numpy()
        action_entropy_numpy = action_entropy.detach().cpu().numpy()
        return value_numpy, action_numpy, action_log_probs_numpy, action_entropy_numpy

    def value(self, obs):
        """use the model to predict obs values

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        """
        obs = to_tensor(obs)
        value = self.alg.value(obs)
        value = value.detach().cpu().numpy()
        return value

    def learn(self, rollout):
        """Learn current batch of rollout for ppo_epoch epochs.

        Args:
            rollouts (RolloutStorage): the rollout storage that contains the current rollout
        """
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        if self.config["lr_decay"]:
            lr = self.lr_scheduler.step(step_num=1)
        else:
            lr = None

        minibatch_size = int(
            self.config["batch_size"] // self.config["num_minibatches"]
        )

        indexes = np.arange(self.config["batch_size"])
        for epoch in range(self.config["update_epochs"]):
            np.random.shuffle(indexes)
            for start in range(0, self.config["batch_size"], minibatch_size):
                end = start + minibatch_size
                sample_idx = indexes[start:end]

                (
                    batch_obs,
                    batch_action,
                    batch_logprob,
                    batch_adv,
                    batch_return,
                    batch_value,
                ) = rollout.sample_batch(sample_idx)

                batch_obs = to_tensor(batch_obs)
                batch_action = to_tensor(batch_action)
                batch_logprob = to_tensor(batch_logprob)
                batch_adv = to_tensor(batch_adv)
                batch_return = to_tensor(batch_return)
                batch_value = to_tensor(batch_value)

                value_loss, action_loss, entropy_loss = self.alg.learn(
                    batch_obs,
                    batch_action,
                    batch_value,
                    batch_return,
                    batch_logprob,
                    batch_adv,
                    lr,
                )

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss

        update_steps = self.config["update_epochs"] * self.config["batch_size"]
        value_loss_epoch /= update_steps
        action_loss_epoch /= update_steps
        entropy_loss_epoch /= update_steps

        return (
            value_loss_epoch,
            action_loss_epoch,
            entropy_loss_epoch,
            lr,
        )
