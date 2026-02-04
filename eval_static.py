import numpy as np
import os
import argparse
import glob
import yaml
import sys
from parl.utils import logger

sys.path.append(os.path.abspath(__file__))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Static Agent (Action=0)")
    parser.add_argument("--env", default="synthetic", help="Environment to use.")
    parser.add_argument(
        "--results_dir",
        default="results_static",
        help="Directory where results are saved",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="test_BS1_right_BS3_left",
        help="Name of the scenario folder to evaluate on",
    )
    args = parser.parse_args()

    env_dir = args.env + "_scenarios"

    scenario_dir = os.path.join(CURRENT_DIR, env_dir, "scenarios", args.scenario_name)
    if not os.path.exists(scenario_dir):
        if os.path.exists(args.scenario_name):
            scenario_dir = args.scenario_name
        else:
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
    action_dim = dummy_env.action_space.shape[0]

    total_episodes = len(config_paths)
    logger.info(
        f"Starting evaluation on {total_episodes} scenarios using Static Policy (Action=0)"
    )

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
            action = np.zeros(action_dim)

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

    logger.info("-" * 50)
    logger.info(f"Final Evaluation for {args.scenario_name}:")
    logger.info(
        f"Mean Reward: {np.mean(all_rewards):.4f} +/- {np.std(all_rewards):.4f}"
    )
    logger.info(
        f"Mean Throughput: {np.mean(all_throughputs):.4f} +/- {np.std(all_throughputs):.4f}"
    )
    logger.info(
        f"Mean Load Balance: {np.mean(all_load_balances):.4f} +/- {np.std(all_load_balances):.4f}"
    )
    logger.info("-" * 50)

    import json

    result_data = {
        "scenario": args.scenario_name,
        "model": "static",
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_throughput": float(np.mean(all_throughputs)),
        "std_throughput": float(np.std(all_throughputs)),
        "mean_load_balance": float(np.mean(all_load_balances)),
        "std_load_balance": float(np.std(all_load_balances)),
    }

    results_dir = os.path.join(
        CURRENT_DIR, env_dir, args.results_dir, args.scenario_name
    )
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=4)
    logger.info(f"Results saved to {output_file}")
