import os
import sys
import argparse
import numpy as np
import torch
import json
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MOPO_ROOT = os.path.join(PROJECT_ROOT, "MOPO")


sys.path.append(PROJECT_ROOT)
sys.path.append(MOPO_ROOT)
from MOPO.algo.mopo import MOPO
from MOPO.algo.sac import SACPolicy as SAC
from MOPO.models.transition_model import TransitionModel as EnsembleDynamicsModel
from MOPO.models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from MOPO.trainer import Trainer
from MOPO.common.buffer import ReplayBuffer
from MOPO.common import util as mopo_util
from common.mopo_utils import load_offline_data_to_mopo_buffer
import gym
from parl.utils import logger


class GymAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_mean = 0.0
        self.obs_std = 1.0

    def set_normalization(self, mean, std):
        self.obs_mean = mean
        self.obs_std = std

    def normalize(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            obs = ret[0]
        else:
            obs = ret
        return self.normalize(obs)

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
            return self.normalize(obs), reward, done, info
        
        # Handle older gym versions
        obs, reward, done, info = ret
        return self.normalize(obs), reward, done, info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random Seed set to: {seed}")


class SimpleLogger:
    """
    A simple wrapper to replace standard heavy logging if needed,
    or just to print to console.
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_path = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = self  

    def get_logdir(self):
        return self.log_dir

    def print(self, msg):
        logger.info(f"[MOPO] {msg}")

    def record(self, key, value, step, printed=False):
        pass


def prob_load_offline_data(data_path, obs_dim, action_dim, ratio):
    """
    Wrapper to load data.
    Ideally this uses mopo_utils.load_offline_data_to_mopo_buffer
    """
    return load_offline_data_to_mopo_buffer(data_path, obs_dim, action_dim, ratio)


def main():
    parser = argparse.ArgumentParser(description="Train MOPO Agent")
    parser.add_argument("--env", default="synthetic", help="Environment to use.")
    parser.add_argument(
        "--train_scenarios",
        nargs="+",
        default=["BS1_right_BS2_right", "BS2_left_BS3_left"],
        help="Name of the train scenarios",
    )
    parser.add_argument(
        "--test_scenario",
        type=str,
        required=True,
        help="Name of the test scenario",
    )
    parser.add_argument("--results_dir", type=str, default="results_mopo")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # MOPO Hyperparameters
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step_per_epoch", type=int, default=100)
    parser.add_argument("--rollout_freq", type=int, default=1000)
    parser.add_argument(
        "--rollout_batch_size", type=int, default=5000
    )
    parser.add_argument("--rollout_length", type=int, default=5)
    parser.add_argument("--model_retain_epochs", type=int, default=5)
    parser.add_argument("--penalty_coef", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)

    env_dir = args.env + "_scenarios"
    scenario_dir = os.path.join(CURRENT_DIR, env_dir, "scenarios", args.test_scenario)
    config_path = None
    if os.path.exists(scenario_dir):
        for f in os.listdir(scenario_dir):
            if f.endswith(".yaml"):
                config_path = os.path.join(scenario_dir, f)
                break

    if not config_path:
        logger.info(
            "Warning: No yaml config found, dimensions inference might fail if Env requires config."
        )
        return

    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        import yaml

        config = yaml.safe_load(f)

    if args.env == "synthetic":
        from env.cio_load_balancing_env import CIOLoadBalancingEnv
    else:
        from env.cio_load_balancing_env_real import CIOLoadBalancingEnv
    env = CIOLoadBalancingEnv(config)
    env = GymAdapter(env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    logger.info(f"Environment: Obs Dim={obs_dim}, Action Dim={action_dim}")

    obs_source_list = []
    act_source_list = []
    rew_source_list = []
    next_obs_source_list = []
    term_source_list = []

    scenarios_list = args.train_scenarios
    if isinstance(scenarios_list, str):
        scenarios_list = [scenarios_list]

    for scenario in scenarios_list:
        path = os.path.join(CURRENT_DIR, env_dir, "offline_data", scenario, "data.npz")

        logger.info(f"Loading Source Data from {path}...")
        source_data = np.load(path)
        obs_source_list.append(source_data["obs"])

        act = (
            source_data["action"] if "action" in source_data else source_data["actions"]
        )
        act_source_list.append(act)

        rew_source_list.append(source_data["reward"])
        next_obs_source_list.append(source_data["next_obs"])

        if "done" in source_data:
            term_source_list.append(source_data["done"])
        elif "terminal" in source_data:
            term_source_list.append(source_data["terminal"])
        else:
            term_source_list.append(np.zeros_like(source_data["reward"], dtype=bool))

    if not obs_source_list:
        raise FileNotFoundError("No training data found in specified scenarios.")

    obs_source = np.concatenate(obs_source_list, axis=0)
    act_source = np.concatenate(act_source_list, axis=0)
    rew_source = np.concatenate(rew_source_list, axis=0)
    next_obs_source = np.concatenate(next_obs_source_list, axis=0)
    term_source = np.concatenate(term_source_list, axis=0)

    target_data_path = os.path.join(
        CURRENT_DIR, env_dir, "offline_data", args.test_scenario, "data.npz"
    )

    if os.path.exists(target_data_path):
        logger.info(
            f"Loading Target Data from {target_data_path} with ratio {args.ratio}..."
        )
        target_data = np.load(target_data_path)
        num_target = int(len(target_data["obs"]) * args.ratio)

        obs_target = target_data["obs"][:num_target]
        act_target = (
            target_data["action"][:num_target]
            if "action" in target_data
            else target_data["actions"][:num_target]
        )
        rew_target = target_data["reward"][:num_target]
        next_obs_target = target_data["next_obs"][:num_target]
        if "done" in target_data:
            term_target = target_data["done"][:num_target]
        elif "terminal" in target_data:
            term_target = target_data["terminal"][:num_target]
        else:
            term_target = np.zeros_like(rew_target, dtype=bool)
        
        obs_all = np.concatenate([obs_source, obs_target], axis=0)
        act_all = np.concatenate([act_source, act_target], axis=0)
        rew_all = np.concatenate([rew_source, rew_target], axis=0)
        next_obs_all = np.concatenate([next_obs_source, next_obs_target], axis=0)
        term_all = np.concatenate([term_source, term_target], axis=0)
    else:
        logger.info(
            f"Warning: Target data not found at {target_data_path}. Using only Source Data."
        )
        obs_all, act_all, rew_all, next_obs_all, term_all = (
            obs_source,
            act_source,
            rew_source,
            next_obs_source,
            term_source,
        )

    if rew_all.ndim == 1:
        rew_all = rew_all[:, None]
    if term_all.ndim == 1:
        term_all = term_all[:, None]

    # Normalize Observations (Same method as in train_cf.py)
    obs_mean = obs_all.mean()
    obs_std = obs_all.std() + 1e-8

    obs_all = (obs_all - obs_mean) / obs_std
    next_obs_all = (next_obs_all - obs_mean) / obs_std

    # Set normalization for the environment (GymAdapter)
    env.set_normalization(obs_mean, obs_std)
    logger.info(f"Observations Normalized: Mean={obs_mean:.4f}, Std={obs_std:.4f}")

    logger.info(f"Total Offline Transitions: {len(obs_all)}")

    offline_buffer = ReplayBuffer(
        buffer_size=len(obs_all),
        obs_shape=(obs_dim,),
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
    )
    offline_buffer.add_batch(obs_all, next_obs_all, act_all, rew_all, term_all)

    class DummySpace:
        def __init__(self, shape):
            self.shape = shape

    class StaticFns:
        def termination_fn(self, obs, act, next_obs):
            return np.zeros((len(obs), 1), dtype=bool)

    dynamics_model = EnsembleDynamicsModel(
        obs_space=DummySpace((obs_dim,)),
        action_space=DummySpace((action_dim,)),
        static_fns=StaticFns(),
        lr=1e-3,
        holdout_ratio=0.1,
        inc_var_loss=True,
        use_weight_decay=True,
        model={
            "hidden_dims": [200, 200, 200, 200],
            "num_networks": 7,
            "num_elites": 5,
            "decay_weights": [
                0.000025,
                0.00005,
                0.000075,
                0.000075,
                0.0001,
            ],
            "decay_weights": [0.000025] * 5,
        },
    )

    # Initialize Policy (SAC)
    # MOPO's SAC implementation expects actor/critic networks construction
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,  # Set to True to match common/sac_agent.py structure (raw mean output)
        conditioned_sigma=True,
    )
    actor = ActorProb(actor_backbone, dist, device)
    critic1 = Critic(critic1_backbone, device)
    critic2 = Critic(critic2_backbone, device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    # Auto-alpha
    target_entropy = -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)

    sac_agent = SAC(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=0.005,
        gamma=0.99,
        alpha=(target_entropy, log_alpha, alpha_optim),
        device=device,
    )

    # Initialize Model Buffer (for synthetic data)
    model_buffer_size = int(1e5)
    model_buffer = ReplayBuffer(
        buffer_size=model_buffer_size,
        obs_shape=(obs_dim,),
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
    )

    # Logger
    save_path = os.path.join(
        CURRENT_DIR,
        env_dir,
        args.results_dir,
        args.test_scenario,
    )
    simple_logger = SimpleLogger(save_path)
    mopo_util.logger = simple_logger

    norm_save_path = os.path.join(save_path, "obs_norm.npz")
    np.savez(norm_save_path, mean=obs_mean, std=obs_std)
    logger.info(f"Saved normalization params to {norm_save_path}")

    # MOPO Algorithm
    mopo = MOPO(
        policy=sac_agent,
        dynamics_model=dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=256,
        rollout_batch_size=args.rollout_batch_size,
        real_ratio=0.05,
        logger=simple_logger,
        device=device,
    )

    # Trainer
    trainer = Trainer(
        algo=mopo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=simple_logger,
        log_freq=100,
        eval_episodes=5,
    )

    logger.info("Starting MOPO Training...")
    logger.info("Stage 1: Learning Dynamics Model...")
    trainer.train_dynamics()

    logger.info("Stage 2: Learning Policy with Penalized Hallucination...")
    trainer.train_policy()

    logger.info(f"Training Complete. Results saved to {save_path}")


if __name__ == "__main__":
    main()
