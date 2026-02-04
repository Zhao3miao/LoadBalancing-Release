import numpy as np
import os
import argparse
import glob
import yaml
import sys
from parl.utils import logger
from common.train_offline_utils import normalize_reward, normalize_obs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.sac_agent import Agent


def get_normalization_stats(results_dir=None):
    params_path = os.path.join(results_dir, "normalization_params.npz")

    if os.path.exists(params_path):
        logger.info(f"Loading normalization stats from {params_path}...")
        data = np.load(params_path)
        if "mean" in data and "std" in data:
            return data["mean"], data["std"]
        elif "obs_mean" in data and "obs_std" in data:
            return data["obs_mean"], data["obs_std"]


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval SAC Agent")
    parser.add_argument(
        "--env", default="synthetic", help="Environment to use."
    )
    parser.add_argument(
        "--results_dir",
        default="results_sac_online",
        help="Directory where results are saved",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="test_BS1_right_BS3_left",
        help="Name of the scenario folder to evaluate on",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="model_final.ckpt",
        help="Filename of the model to load from results/model/",
    )
    args = parser.parse_args()

    env_dir = args.env + "_scenarios"

    results_path = os.path.join(
        CURRENT_DIR, env_dir, args.results_dir, args.scenario_name
    )
    rsrp_mean, rsrp_std = get_normalization_stats(results_path)
    model_path = os.path.join(results_path, "model", args.model_filename)

    if not os.path.exists(model_path):
        logger.info(f"Error: Model not found at {model_path}")
        sys.exit(1)

    scenario_dir = os.path.join(CURRENT_DIR, env_dir, "scenarios", args.scenario_name)
    if not os.path.exists(scenario_dir):
        logger.info(f"Error: Scenario directory not found at {scenario_dir}")
        sys.exit(1)

    if args.env == "synthetic":
        from env.cio_load_balancing_env import CIOLoadBalancingEnv
    else:
        from env.cio_load_balancing_env_real import CIOLoadBalancingEnv

    config_paths = glob.glob(os.path.join(scenario_dir, "*.yaml"))

    if not config_paths:
        logger.info(f"No .yaml files found in {scenario_dir}")
        sys.exit(1)

    logger.info(f"Found {len(config_paths)} scenario files.")

    with open(config_paths[0], "r", encoding="utf-8") as f:
        first_config = yaml.safe_load(f)
    dummy_env = CIOLoadBalancingEnv(first_config)
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]

    agent_config = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "batch_size": 256,
        "max_size": 10000,
        "lr": 0.0003, 
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
    }
    agent = Agent(**agent_config)

    total_episodes = len(config_paths)
    logger.info(
        f"Starting evaluation on {total_episodes} scenarios using model: {model_path}"
    )

    agent.restore(model_path)
    logger.info(f"Model loaded successfully.")

    all_rewards = []
    all_throughputs = []
    all_load_balances = []

    for idx, config_path in enumerate(config_paths):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        env = CIOLoadBalancingEnv(config)
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_info = []

        while not done:
            normalized_obs = normalize_obs(obs, rsrp_mean, rsrp_std)
            action = agent.predict(normalized_obs)
            res = env.step(action)
            if len(res) == 5:
                next_obs, reward, terminated, truncated, info = res
                done = terminated or truncated
            else:
                next_obs, reward, done, info = res

            obs = next_obs
            episode_reward += reward
            episode_info.append(info)

        episode_throughput = [info.get("throughput", 0) for info in episode_info]
        episode_load_balance = [info.get("load_balance", 0) for info in episode_info]

        avg_throughput = np.mean(episode_throughput) if episode_throughput else 0
        avg_load_balance = np.mean(episode_load_balance) if episode_load_balance else 0

        all_rewards.append(episode_reward)
        all_throughputs.append(avg_throughput)
        all_load_balances.append(avg_load_balance)

        if (idx + 1) % 10 == 0:
            logger.info(f"Evaluated {idx+1}/{total_episodes} scenarios.")

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_throughput = np.mean(all_throughputs)
    std_throughput = np.std(all_throughputs)
    mean_load_balance = np.mean(all_load_balances)
    std_load_balance = np.std(all_load_balances)

    logger.info("-" * 50)
    logger.info(f"Final Evaluation for {args.scenario_name}:")
    logger.info(f"Mean Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
    logger.info(f"Mean Throughput: {mean_throughput:.4f} +/- {std_throughput:.4f}")
    logger.info(f"Mean Load Balance: {mean_load_balance:.4f} +/- {std_load_balance:.4f}")
    logger.info("-" * 50)

    import json

    result_data = {
        "scenario": args.scenario_name,
        "model": "sac_online",
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_throughput": float(mean_throughput),
        "std_throughput": float(std_throughput),
        "mean_load_balance": float(mean_load_balance),
        "std_load_balance": float(std_load_balance),
    }

    output_file = os.path.join(results_path, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=4)
    logger.info(f"Results saved to {output_file}")
