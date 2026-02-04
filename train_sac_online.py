import sys
import os
import numpy as np
import argparse
import glob
import random
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym.vector import SyncVectorEnv
from parl.utils import CSVLogger, logger
from common.sac_agent import Agent
import yaml
from common.train_offline_utils import normalize_reward, normalize_obs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random Seed set to: {seed}")


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def train(agent, env, num_episodes, save_path, obs_mean, obs_std):
    """Train SAC agent on multiple CIO load balancing environments in parallel"""
    csv_logger = CSVLogger(os.path.join(save_path, "result.csv"))
    total_steps = 0
    learn_frequency = 4  # Learn every 4 steps
    num_envs = env.num_envs

    # Track metrics for each episode
    episode_rewards_total = np.zeros(num_envs)
    episode_steps_total = np.zeros(num_envs)
    episode_throughputs_total = np.zeros(num_envs)
    episode_non_blockeds_total = np.zeros(num_envs)
    episode_load_balances_total = np.zeros(num_envs)

    # For learning statistics
    critic_losses = []
    actor_losses = []

    for episode in range(num_episodes):
        # Initialize observations and reset episode metrics
        obs, info = env.reset()
        done = np.array([False] * num_envs)

        # Reset episode metrics at the start of each episode
        episode_rewards_total.fill(0)
        episode_steps_total.fill(0)
        episode_throughputs_total.fill(0)
        episode_non_blockeds_total.fill(0)
        episode_load_balances_total.fill(0)

        while not np.all(done):
            normalized_obs = normalize_obs(
                obs, obs_mean, obs_std
            )  # Batch normalization

            # Sample actions for all environments
            action = agent.sample(normalized_obs)  # Batch action

            # Step all environments - handle both old and new Gym API
            step_result = env.step(action)

            next_obs, reward, terminated, truncated, info = step_result
            done = terminated  # Use terminated as done signal

            normalized_next_obs = normalize_obs(next_obs, obs_mean, obs_std)
            normalized_reward = normalize_reward(reward)

            # Store experiences for each environment
            for i in range(num_envs):
                done_flag = done[i] if isinstance(done, (list, np.ndarray)) else done
                agent.append(
                    normalized_obs[i],
                    action[i],
                    normalized_reward[i],
                    normalized_next_obs[i],
                    done_flag,
                )

            # Learn every learn_frequency steps
            if total_steps % learn_frequency == 0:
                critic_loss, actor_loss = agent.learn()
                if critic_loss is not None and actor_loss is not None:
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)

            # Update observations
            obs = next_obs
            total_steps += 1

            # Accumulate episode metrics
            episode_rewards_total += reward
            episode_steps_total += 1

            # Collect metrics from info
            if isinstance(info, dict):
                # Vector environment: info is a dict with arrays for each metric
                if "throughput" in info:
                    throughputs = (
                        info["throughput"]
                        if hasattr(info["throughput"], "shape")
                        else np.array(info["throughput"])
                    )
                    episode_throughputs_total += throughputs
                if "non_blocked" in info:
                    non_blockeds = (
                        info["non_blocked"]
                        if hasattr(info["non_blocked"], "shape")
                        else np.array(info["non_blocked"])
                    )
                    episode_non_blockeds_total += non_blockeds
                if "load_balance" in info:
                    load_balances = (
                        info["load_balance"]
                        if hasattr(info["load_balance"], "shape")
                        else np.array(info["load_balance"])
                    )
                    episode_load_balances_total += load_balances
            else:
                # Fallback: collect from list of dicts
                for i in range(num_envs):
                    if i < len(info):
                        episode_throughputs_total[i] += info[i].get("throughput", 0)
                        episode_non_blockeds_total[i] += info[i].get("non_blocked", 0)
                        episode_load_balances_total[i] += info[i].get("load_balance", 0)

        # Calculate averages for the completed episode
        avg_reward = np.mean(episode_rewards_total)
        avg_throughput = np.mean(episode_throughputs_total / episode_steps_total)
        avg_non_blocked = np.mean(episode_non_blockeds_total / episode_steps_total)
        avg_load_balance = np.mean(episode_load_balances_total / episode_steps_total)
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0

        logger.info(
            f"Episode {episode}, Total Steps: {total_steps}, "
            f"Avg Reward: {avg_reward:.2f}, "
            f"Critic Loss: {avg_critic_loss:.2f}, Actor Loss: {avg_actor_loss:.2f}, "
            f"Throughput: {avg_throughput:.2f}, Non-blocked: {avg_non_blocked:.2f}, Load Balance: {avg_load_balance:.2f}"
        )
        csv_logger.log_dict(
            {
                "episode": episode,
                "total_steps": total_steps,
                "avg_reward": avg_reward,
                "critic_loss": avg_critic_loss,
                "actor_loss": avg_actor_loss,
                "throughput": avg_throughput,
                "non_blocked": avg_non_blocked,
                "load_balance": avg_load_balance,
            }
        )

        # Reset learning statistics for next episode
        critic_losses = []
        actor_losses = []

        # Save model every 100 episodes
        if episode % 100 == 0:
            model_save_path = os.path.join(
                save_path, f"model/model_episode_{episode}.ckpt"
            )
            agent.save(model_save_path)

    # Save final model
    final_model_path = os.path.join(save_path, "model/model_final.ckpt")
    agent.save(final_model_path)
    logger.info(f"Training completed. Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC Agent Online")
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
        help="Name of the test scenario folder",
    )
    parser.add_argument(
        "--results_dir",
        default="results_sac_online",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=300, help="Number of episodes to train"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_data_paths = []
    scenarios_list = args.train_scenarios
    if isinstance(scenarios_list, str):
        scenarios_list = [scenarios_list]

    scenarios_list = sorted(scenarios_list)
    logger.info(f"Training with scenarios (sorted): {scenarios_list}")

    env_dir = args.env + "_scenarios"

    for scenario in scenarios_list:
        path = os.path.join(CURRENT_DIR, env_dir, "offline_data", scenario, "data.npz")
        train_data_paths.append(path)

    test_data_path = os.path.join(
        CURRENT_DIR, env_dir, "offline_data", args.test_scenario, "data.npz"
    )

    obs_list = []
    for path in train_data_paths:
        if not os.path.exists(path):
            logger.info(f"Warning: Train data not found at {path}")
            continue

        logger.info(f"Loading train data from {path}...")
        train_data = np.load(path)
        total_train = len(train_data["obs"])
        obs_train = train_data["obs"]
        dones_train = train_data["terminal"]
        obs_list.append(obs_train)

    if not obs_list:
        raise ValueError("No training data loaded!")

    # Load Test Data (if requested)
    if test_data_path and os.path.exists(test_data_path):
        test_data = np.load(test_data_path)
        obs_test = test_data["obs"]
        obs_list.append(obs_test)
    obs = np.concatenate(obs_list, axis=0)

    rsrp_mean = obs.mean()
    rsrp_std = obs.std() + 1e-8
    print("obs_mean", rsrp_mean, "obs_std", rsrp_std)

    save_path = os.path.join(CURRENT_DIR, env_dir, args.results_dir, args.test_scenario)

    # Create model directory
    model_dir = os.path.join(save_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Created task directory: {save_path}")
    logger.info(f"Created model directory: {model_dir}")

    # Load Scenario Configs
    scenarios_root = os.path.join(CURRENT_DIR, env_dir, "scenarios")
    if not os.path.exists(scenarios_root):
        logger.info(f"Error: Scenarios root directory not found at {scenarios_root}")
        sys.exit(1)

    all_config_paths = []

    # 1. Load Training Scenarios (from args)
    scenarios_list = args.train_scenarios
    if isinstance(scenarios_list, str):
        scenarios_list = [scenarios_list]

    for sc_name in scenarios_list:
        d = os.path.join(scenarios_root, sc_name)
        if os.path.isdir(d):
            yamls = glob.glob(os.path.join(d, "*.yaml"))
            all_config_paths.extend(sorted(yamls))
        else:
            logger.info(f"Warning: Train scenario folder {d} not found.")

    logger.info(f"Loaded {len(all_config_paths)} training scenario files.")
    num_train_scenarios = len(all_config_paths)

    # 2. Load Testing Scenarios (args.test_scenario)
    test_dir = os.path.join(scenarios_root, args.test_scenario)
    if os.path.isdir(test_dir):
        yamls = glob.glob(os.path.join(test_dir, "*.yaml"))
        all_config_paths.extend(sorted(yamls))
        logger.info(
            f"Loaded {len(yamls)} testing scenario files from {args.test_scenario}"
        )
    else:
        logger.info(f"Warning: Test scenario folder {args.test_scenario} not found.")

    logger.info(f"Total scenario files: {len(all_config_paths)}")

    if not all_config_paths:
        logger.info(f"No .yaml files found in {scenarios_root} subdirectories")
        sys.exit(1)

    if args.env == "synthetic":
        from env.cio_load_balancing_env import CIOLoadBalancingEnv
    else:
        from env.cio_load_balancing_env_real import CIOLoadBalancingEnv

    # Create Envs
    def make_env(path):
        def _thunk():
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return CIOLoadBalancingEnv(config)

        return _thunk

    envs = [make_env(p) for p in all_config_paths]

    # Create SyncVectorEnv
    env = SyncVectorEnv(envs)
    logger.info(
        f"CIO load balancing environment created successfully! Env num: {len(envs)}"
    )
    logger.info(
        f"Defined observation space dimension: {env.single_observation_space.shape}"
    )
    logger.info(f"Action space dimension: {env.single_action_space.shape}")

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    num_episodes = args.num_episodes
    max_size = 10000

    agent_config = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "batch_size": 256,
        "max_size": max_size,
        "lr": 0.0001,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
    }

    agent = Agent(**agent_config)

    logger.info("Start training...")
    train(agent, env, num_episodes, save_path, rsrp_mean, rsrp_std)
    norm_save_path = os.path.join(save_path, "normalization_params.npz")
    np.savez(norm_save_path, mean=rsrp_mean, std=rsrp_std)
    logger.info("Training completed!")
