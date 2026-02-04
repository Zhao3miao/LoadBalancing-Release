import sys
import os
import numpy as np
import argparse
import glob
import yaml
import random
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym.vector import SyncVectorEnv
from parl.utils import CSVLogger, logger
from common.ppo_agent import PPOAgent
from common.storage import RolloutStorage
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
    """Train PPO agent on CIO load balancing environment"""
    csv_logger = CSVLogger(os.path.join(save_path, "result.csv"))

    num_envs = env.num_envs
    num_steps = 100  # Rollout length matches episode length

    logger.info(f"Training for {num_episodes} episodes...")

    # Initialize Storage
    rollout = RolloutStorage(
        step_nums=num_steps,
        env_num=num_envs,
        obs_space=env.single_observation_space,
        act_space=env.single_action_space,
    )

    # Reset Environment
    obs, info = env.reset()
    total_steps = 0

    # Metric tracking
    metric_lists = {
        "reward": [],
        "throughput": [],
        "non_blocked": [],
        "load_balance": [],
    }

    for episode in range(1, num_episodes + 1):
        for step in range(200):
            total_steps += num_envs

            normalized_obs = normalize_obs(obs, obs_mean, obs_std)
            value, action, action_log_probs, _ = agent.sample(normalized_obs)

            # Clip action to env bounds
            clipped_action = np.clip(
                action, env.single_action_space.low, env.single_action_space.high
            )

            # Step Env
            result = env.step(clipped_action)

            # Handle Gym/VectorEnv API variations
            if len(result) == 5:
                next_obs, reward, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            else:
                next_obs, reward, done, info = result

            if isinstance(info, dict):
                if "throughput" in info:
                    metric_lists["throughput"].extend(info["throughput"])
                if "non_blocked" in info:
                    metric_lists["non_blocked"].extend(info["non_blocked"])
                if "load_balance" in info:
                    metric_lists["load_balance"].extend(info["load_balance"])
            elif isinstance(info, list):
                for i in range(num_envs):
                    metric_lists["throughput"].append(info[i].get("throughput", 0))
                    metric_lists["non_blocked"].append(info[i].get("non_blocked", 0))
                    metric_lists["load_balance"].append(info[i].get("load_balance", 0))

            metric_lists["reward"].extend(reward)

            # Store
            rollout.append(
                normalized_obs,
                action,
                action_log_probs,
                normalize_reward(reward),
                done,
                value.flatten(),
            )

            obs = next_obs

        next_value = agent.value(normalize_obs(next_obs))
        rollout.compute_returns(next_value, done)

        value_loss, action_loss, entropy_loss, lr = agent.learn(rollout)

        avg_reward = np.mean(metric_lists["reward"]) if metric_lists["reward"] else 0
        avg_throughput = (
            np.mean(metric_lists["throughput"]) if metric_lists["throughput"] else 0
        )
        avg_non_blocked = (
            np.mean(metric_lists["non_blocked"]) if metric_lists["non_blocked"] else 0
        )
        avg_load_balance = (
            np.mean(metric_lists["load_balance"]) if metric_lists["load_balance"] else 0
        )

        logger.info(
            f"Episode {episode}/{num_episodes}, Total Steps: {total_steps}, "
            f"Reward: {avg_reward:.2f}, AVG_Load_Balance: {avg_load_balance:.2f}, "
            f"V_Loss: {value_loss:.2f}, A_Loss: {action_loss:.2f}, Ent: {entropy_loss:.2f}"
        )

        csv_logger.log_dict(
            {
                "episode": episode,
                "total_steps": total_steps,
                "avg_reward": avg_reward,
                "avg_throughput": avg_throughput,
                "avg_non_blocked": avg_non_blocked,
                "avg_load_balance": avg_load_balance,
                "value_loss": value_loss,
                "action_loss": action_loss,
                "entropy_loss": entropy_loss,
                "lr": lr if lr else 0,
            }
        )

        for k in metric_lists:
            metric_lists[k] = []
        if episode % 20 == 0:
            agent.save(os.path.join(save_path, f"model/model_episode_{episode}.ckpt"))

    agent.save(os.path.join(save_path, "model/model_final.ckpt"))
    logger.info(
        f"Training completed. Final model saved to {save_path}/model/model_final.ckpt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent Online")
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
        help="Name of the target/test scenario folder",
    )
    parser.add_argument(
        "--results_dir",
        default="results_ppo_online",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=300,
        help="Approximate number of episodes (converted to steps)",
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
    os.makedirs(os.path.join(save_path, "model"), exist_ok=True)

    # Load Scenario Configs
    scenarios_root = os.path.join(CURRENT_DIR, env_dir, "scenarios")
    all_config_paths = []

    # Train Scenarios
    scenarios_list = args.train_scenarios
    if isinstance(scenarios_list, str):
        scenarios_list = [scenarios_list]

    for sc_name in scenarios_list:
        d = os.path.join(scenarios_root, sc_name)
        if os.path.isdir(d):
            all_config_paths.extend(glob.glob(os.path.join(d, "*.yaml")))
        else:
            logger.info(f"Warning: Train scenario folder {d} not found.")

    # Test/Target Scenario
    if args.test_scenario and os.path.isdir(
        os.path.join(scenarios_root, args.test_scenario)
    ):
        all_config_paths.extend(
            glob.glob(os.path.join(scenarios_root, args.test_scenario, "*.yaml"))
        )

    if not all_config_paths:
        logger.info("No scenario files found!")
        sys.exit(1)

    all_config_paths.sort()

    logger.info(f"Found {len(all_config_paths)} scenarios to train on.")
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
    env = SyncVectorEnv(envs)

    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    agent_config = {
        "obs_dim": obs_dim,
        "act_dim": action_dim,
        "hidden_size": 256,
        "clip_param": 0.1,
        "entropy_coef": 0.01,
        "initial_lr": 0.0001,
        "batch_size": 100 * len(envs),
        "num_minibatches": 20,
        "update_epochs": 4,
        "lr_decay": False,
        "num_updates": args.num_episodes,
    }

    agent = PPOAgent(agent_config)

    train(agent, env, args.num_episodes, save_path, rsrp_mean, rsrp_std)
    norm_save_path = os.path.join(save_path, "normalization_params.npz")
    np.savez(norm_save_path, mean=rsrp_mean, std=rsrp_std)
