import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import argparse
import gym
from parl.utils import logger
from common.train_offline_utils import normalize_reward, normalize_obs

# ==========================================
# 1. Path Setup
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MOPO_ROOT = os.path.join(PROJECT_ROOT, "MOPO")

# Add paths for imports
sys.path.append(PROJECT_ROOT)
sys.path.append(MOPO_ROOT)

# Import Project Env
from common.train_offline_utils import load_config

# Import MOPO modules
from MOPO.algo.sac import SACPolicy as SAC
from MOPO.models.policy_models import MLP, ActorProb, Critic, DiagGaussian


# ==========================================
# 2. Gym Adapter (Same as train_mopo.py)
# ==========================================
class GymAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            return ret[0]
        return ret

    def step(self, action):
        ret = self.env.step(action)
        # Handle (obs, reward, terminated, truncated, info) -> (obs, reward, done, info)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
            return obs, reward, done, info
        return ret


# ==========================================
# 3. MOPO Agent Wrapper
# ==========================================
class MOPOAgent:
    def __init__(self, model_path, obs_dim, action_dim, device="cpu"):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize Policy Structure (Must match train_mopo.py)
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
        critic1_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=[256, 256])
        critic2_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=[256, 256])
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True,
        )
        actor = ActorProb(actor_backbone, dist, self.device)
        critic1 = Critic(critic1_backbone, self.device)
        critic2 = Critic(critic2_backbone, self.device)

        # Optimizers (needed for SAC constructor, though not used for eval)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)

        # Dummy Alpha (needed for constructor)
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)

        # Dummy Action Space
        class DummySpace:
            def __init__(self, shape):
                self.shape = shape
                self.high = np.ones(shape)
                self.low = -np.ones(shape)

        # Initialize SAC Policy
        self.policy = SAC(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space=DummySpace((action_dim,)),
            dist=dist,
            tau=0.005,
            gamma=0.99,
            alpha=(target_entropy, log_alpha, alpha_optim),
            device=self.device,
        )

        # Load Weights
        logger.info(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def predict(self, obs, deterministic=True):
        # Prepare observation
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)  # (1, obs_dim)

        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)

        return action.cpu().numpy()[0]  # Return action as 1D numpy array


# ==========================================
# 4. Evaluation Function
# ==========================================
def evaluate_mopo(agent, scenario_files, EnvClass, obs_mean=0.0, obs_std=1.0, num_episodes=None):
    if num_episodes is None:
        num_episodes = len(scenario_files)

    rewards = []
    avg_load_stds = []
    logger.info(f"Evaluating on {len(scenario_files)} scenarios...")

    for i, scenario_file in enumerate(scenario_files):
        config = load_config(scenario_file)
        # Use simple env without GymAdapter because the manual loop here mimics the old gym API style anyway
        # Actually, compare_results_sac.py expects env.step to return 5 vals?
        # Let's check compare_results_sac.py: next_obs, reward, done, _, info = env.step(action)
        # It expects 5 values unpacked.
        env = EnvClass(config)

        obs, _ = env.reset()
        obs = (obs - obs_mean) / obs_std

        episode_reward = 0
        episode_load_stds = []
        truncated = False
        terminated = False
        done = False

        while not done:
            # We assume no normalization for MOPO agent as per train_mopo.py
            action = agent.predict(obs, deterministic=True)

            # Step
            ret = env.step(action)

            if len(ret) == 5:
                next_obs, reward, terminated, truncated, info = ret
                done = terminated or truncated
            elif len(ret) == 4:
                next_obs, reward, done, info = ret

            episode_reward += reward

            if "load_balance" in info:
                episode_load_stds.append(-info["load_balance"])

            next_obs = (next_obs - obs_mean) / obs_std
            obs = next_obs

        rewards.append(episode_reward)

        if episode_load_stds:
            avg_load_stds.append(np.mean(episode_load_stds))
        else:
            avg_load_stds.append(0.0)

        if (i + 1) % 10 == 0:
            logger.info(f"Synced {i + 1}/{len(scenario_files)} scenarios.")

    return (
        np.mean(rewards),
        np.std(rewards),
        rewards,
        np.mean(avg_load_stds),
        np.std(avg_load_stds),
    )


# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="synthetic", help="Environment to use."
    )
    parser.add_argument("--scenario_name", type=str, required=True)
    parser.add_argument(
        "--results_dir",
        default="results_mopo",
        help="Directory where results are saved",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="policy.pth",
        help="Filename of the model to load from results/model/",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    env_dir = args.env + "_scenarios"

    # Locate Model
    results_path = os.path.join(
        CURRENT_DIR, env_dir, args.results_dir, args.scenario_name
    )
    model_path = os.path.join(results_path, args.model_filename)

    if not os.path.exists(model_path):
        logger.info(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # Locate Scenarios
    scenario_dir = os.path.join(CURRENT_DIR, env_dir, "scenarios", args.scenario_name)
    scenario_files = glob.glob(os.path.join(scenario_dir, "*.yaml"))

    if not scenario_files:
        logger.info(f"Error: No scenarios found in {scenario_dir}")
        sys.exit(1)

    if args.env == "synthetic":
        from env.cio_load_balancing_env import CIOLoadBalancingEnv
    else:
        from env.cio_load_balancing_env_real import CIOLoadBalancingEnv

    # Get Dimensions from first scenario
    first_config = load_config(scenario_files[0])
    dummy_env = CIOLoadBalancingEnv(first_config)
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]

    # Initialize Agent
    agent = MOPOAgent(model_path, obs_dim, action_dim, device=args.device)

    # Load Normalization Params
    norm_path = os.path.join(results_path, "obs_norm.npz")
    obs_mean = 0.0
    obs_std = 1.0
    if os.path.exists(norm_path):
        logger.info(f"Loading normalization params from {norm_path}")
        norm_data = np.load(norm_path)
        obs_mean = norm_data["mean"]
        obs_std = norm_data["std"]
    else:
        logger.info(f"Warning: Normalization params not found at {norm_path}. Using identity.")

    # Evaluate
    logger.info(f"Starting evaluation for {args.scenario_name} using MOPO model...")
    mean_rew, std_rew, all_rews, mean_load, std_load = evaluate_mopo(
        agent, scenario_files, CIOLoadBalancingEnv, obs_mean=obs_mean, obs_std=obs_std
    )

    logger.info("-" * 50)
    logger.info(f"Evaluation Results ({len(scenario_files)} episodes):")
    logger.info(f"Mean Reward: {mean_rew:.4f} +/- {std_rew:.4f}")
    logger.info(f"Mean Load Balance Std: {mean_load:.4f} +/- {std_load:.4f}")
    logger.info("-" * 50)

    # Save results to JSON
    import json

    result_data = {
        "scenario": args.scenario_name,
        "model": "mopo",
        "mean_reward": float(mean_rew),
        "std_reward": float(std_rew),
        "mean_load_balance": float(-mean_load),
        "std_load_balance": float(std_load),
        "mean_throughput": 0.0,
        "std_throughput": 0.0,
    }

    # Save to the scenario folder inside results_mopo_base if inferable, or dirname of model_path
    output_dir = os.path.dirname(model_path)
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=4)
    logger.info(f"Results saved to {output_file}")
