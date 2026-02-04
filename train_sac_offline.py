import os
import sys
import numpy as np
import argparse
import random
import torch
from parl.utils import logger
from common.sac_agent import Agent
from common.train_offline_utils import (
    load_offline_data_to_agent,
    train_offline,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hyperparameters
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = "results1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random Seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Train SAC Agent Offline")
    parser.add_argument(
        "--env", default="synthetic", help="Environment to use."
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--train_scenarios",
        nargs="+",
        default=["BS1_right_BS2_right", "BS2_left_BS3_left"],
        help="Name of the train scenarios",
    )
    parser.add_argument(
        "--test_scenario",
        required=True,
        help="Name of the test scenario",
    )
    parser.add_argument(
        "--ratio",
        nargs="+",
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Ratio of data to use (e.g. 1 1.0 0.0)",
    )
    # Training parameters
    parser.add_argument(
        "--num_steps", type=int, default=5000, help="Offline training steps"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--max_buffer_size", type=int, default=10000, help="Replay buffer size"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    env_dir = args.env + "_scenarios"

    train_data_paths = []
    scenarios_list = args.train_scenarios
    if isinstance(scenarios_list, str):
        scenarios_list = [scenarios_list]

    for scenario in scenarios_list:
        path = os.path.join(CURRENT_DIR, env_dir, "offline_data", scenario, "data.npz")
        if os.path.exists(path):
            train_data_paths.append(path)
        else:
            logger.info(
                f"Warning: Train data for scenario '{scenario}' not found at {path}"
            )

    if not train_data_paths:
        logger.info("Error: No valid training data found.")
        return

    test_data_path = os.path.join(
        CURRENT_DIR, env_dir, "offline_data", args.test_scenario, "data.npz"
    )
    save_path = os.path.join(CURRENT_DIR, env_dir, args.results_dir, args.test_scenario)
    if args.ratio[2] > 0:
        cf_data_path = os.path.join(save_path, f"cf_data.npz")

    logger.info(f"Training Configuration:")
    temp_data = np.load(train_data_paths[0])
    obs_dim = temp_data["obs"].shape[1]
    action_dim = temp_data["action"].shape[1]

    all_obs_for_stats = []
    for path in train_data_paths:
        t_data = np.load(path)
        all_obs_for_stats.append(t_data["obs"])

    if args.ratio[1] > 0:
        if os.path.exists(test_data_path):
            test_data = np.load(test_data_path)
            num_fewshot = int(len(test_data["obs"]) * args.ratio[1])
            all_obs_for_stats.append(test_data["obs"][:num_fewshot])
        else:
            logger.info(
                f"Warning: Test data not found at {test_data_path} for normalization stats."
            )

    combined_obs = np.concatenate(all_obs_for_stats, axis=0)
    obs_mean = combined_obs.mean()
    obs_std = combined_obs.std()
    logger.info(f"Detected obs_dim={obs_dim}, action_dim={action_dim}")
    logger.info(f"Normalization params: mean={obs_mean:.4f}, std={obs_std:.4f}")

    # 2. Initialize Agent
    agent = Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        max_size=args.max_buffer_size,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    # 3. Load Train Data into Agent
    logger.info(
        f"Loading train data from {len(train_data_paths)} scenarios with ratio {args.ratio[0]}..."
    )
    load_offline_data_to_agent(
        agent,
        train_data_paths,
        ratio=args.ratio[0],
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

    # 4. Load Test Data (if requested)
    if args.ratio[1] > 0:
        if not os.path.exists(test_data_path):
            logger.info(f"Error: Test data not found at {test_data_path}")
            return
        logger.info(
            f"Loading test data from {test_data_path} with ratio {args.ratio[1]}..."
        )
        load_offline_data_to_agent(
            agent,
            [test_data_path],
            ratio=args.ratio[1],
            obs_mean=obs_mean,
            obs_std=obs_std,
        )

    # 5. Load Counterfactual Data (if provided)
    if args.ratio[2] > 0:
        if not os.path.exists(cf_data_path):
            logger.info(f"Error: CF data not found at {cf_data_path}")
            return
        logger.info(f"Loading counterfactual data from {cf_data_path}...")
        load_offline_data_to_agent(
            agent,
            [cf_data_path],
            ratio=args.ratio[2],
            obs_mean=obs_mean,
            obs_std=obs_std,
        )

    logger.info(f"- Output Dir: {save_path}")

    # 6. Run Offline Training
    train_offline(agent, args.num_steps, save_path)

    # Save normalization params to the same directory
    norm_save_path = os.path.join(save_path, "normalization_params.npz")
    np.savez(norm_save_path, mean=obs_mean, std=obs_std)
    logger.info(f"Saved normalization params to {norm_save_path}")


if __name__ == "__main__":
    main()
