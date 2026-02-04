import os
import sys

# Set environment variables for strict reproducibility BEFORE importing/using torch.cuda
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import numpy as np
import torch
import argparse
import random
from tqdm import tqdm
from parl.utils import logger
from common.cf_model import CounterfactualModel, unflatten_rsrp_tensor
from common.train_offline_utils import load_config, get_scenario_files

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = "results"


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For strict reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
        except Exception as e:
            logger.info(f"Warning: Could not set deterministic algorithms: {e}")
    logger.info(f"Random Seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Counterfactual Data using CF Model"
    )
    parser.add_argument("--env", default="synthetic", help="Environment to use.")
    parser.add_argument(
        "--ratio",
        nargs="+",
        type=float,
        default=[1.0, 0.1, 0.0],
        help="Ratios for [Train, Test, CF]. Uses Test ratio for seed data selection.",
    )
    parser.add_argument(
        "--test_scenario",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
    )
    # Generation parameters
    parser.add_argument(
        "--traj_len", type=int, default=20, help="Length of trajectory window"
    )
    parser.add_argument(
        "--num_actions", type=int, default=1, help="Number of random actions per state"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.0, help="Noise for z sampling"
    )
    parser.add_argument(
        "--rollout_steps", type=int, default=3, help="Number of rollout steps"
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="Stride for sliding window"
    )
    parser.add_argument(
        "--samples_per_window", type=int, default=10, help="Samples per window"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    # Parse Ratios
    if isinstance(args.ratio, list):
        if len(args.ratio) >= 2:
            # [Train, Test, CF] -> Use Test ratio
            test_ratio = args.ratio[1]
        elif len(args.ratio) == 1:
            test_ratio = args.ratio[0]
        else:
            test_ratio = 0.1
    else:
        test_ratio = args.ratio

    env_dir = args.env + "_scenarios"
    scenario_results_dir = os.path.join(
        CURRENT_DIR, env_dir, args.results_dir, args.test_scenario
    )
    os.makedirs(scenario_results_dir, exist_ok=True)

    model_path = os.path.join(scenario_results_dir, "cf_model.pth")
    norm_path = os.path.join(scenario_results_dir, "cf_norm.npz")
    save_path = os.path.join(
        scenario_results_dir, f"cf_data.npz"
    )

    logger.info(f"Configuration:")
    logger.info(f"- Model: {model_path}")
    logger.info(f"- Norm Params: {norm_path}")
    logger.info(f"- Seed Data: {args.test_scenario}")
    logger.info(f"- Test Ratio: {test_ratio}")

    # 1. Load Model and Normalization Params
    if not os.path.exists(model_path):
        logger.info(f"Error: Model not found at {model_path}")
        return

    norm_params = np.load(norm_path)
    rsrp_mean = norm_params["mean"]
    rsrp_std = norm_params["std"]

    model = CounterfactualModel(traj_len=args.traj_len).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    # # Log model checksum to verify model is identical across runs
    # model_sum = sum(p.sum().item() for p in state_dict.values() if isinstance(p, torch.Tensor))
    # logger.info(f"Loaded Model Checksum: {model_sum:.6f}")

    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Seed Data
    test_data_path = os.path.join(
        CURRENT_DIR, env_dir, "offline_data", args.test_scenario, "data.npz"
    )
    logger.info(f"Loading seed data from {test_data_path}")
    test_data = np.load(test_data_path)

    # Apply ratio
    total_count = len(test_data["obs"])
    if test_ratio > 0 and test_ratio < 1.0:
        num_samples = int(total_count * test_ratio)
        obs_seed = test_data["obs"][:num_samples]
        dones_seed = test_data["terminal"][:num_samples]
        logger.info(
            f"Applied ratio {test_ratio}: Selected first {len(obs_seed)}/{total_count} samples."
        )
    else:
        obs_seed = test_data["obs"]
        dones_seed = test_data["terminal"]

    # Normalize Seed Data
    obs_norm = (obs_seed - rsrp_mean) / rsrp_std

    # Calculate Max Delta from Real Data
    valid_mask = ~dones_seed[:-1].astype(bool)
    if np.any(valid_mask):
        diffs = obs_seed[1:] - obs_seed[:-1]
        valid_diffs = diffs[valid_mask]
        max_delta_val = np.max(np.abs(valid_diffs))
    else:
        max_delta_val = 5.0
    logger.info(f"Max Delta constraint: {max_delta_val}")

    # 3. Initialize Environment
    SCENARIO_DIR = os.path.join(CURRENT_DIR, env_dir, "scenarios", args.test_scenario)
    if os.path.exists(SCENARIO_DIR):
        scenario_files = get_scenario_files(SCENARIO_DIR, pattern="*.yaml")
    else:
        # Fallback
        scenario_files = []
        for root, dirs, files in os.walk(
            os.path.join(CURRENT_DIR, "generated_scenarios")
        ):
            for file in files:
                if file.endswith(".yaml"):
                    scenario_files.append(os.path.join(root, file))
                    break

    if not scenario_files:
        logger.info("Error: No scenario YAML files found.")
        return

    config = load_config(scenario_files[0])
    if args.env == "synthetic":
        from env.cio_load_balancing_env import CIOLoadBalancingEnv
    else:
        from env.cio_load_balancing_env_real import CIOLoadBalancingEnv
    env = CIOLoadBalancingEnv(config)
    # Critical: Seed the action space to ensure reproducible random actions
    env.action_space.seed(args.seed)

    cf_obs = []
    cf_actions = []
    cf_rewards = []
    cf_next_obs = []
    cf_terminals = []

    logger.info(f"Generating Counterfactual Data...")

    buffer = []

    for i in tqdm(range(len(obs_seed))):
        buffer.append(obs_norm[i])

        if len(buffer) >= args.traj_len and (i % args.stride == 0):
            # Prepare inputs
            traj_window = np.array(buffer[-args.traj_len :])  # (T, D)

            # To Tensor
            traj_tensor = (
                torch.FloatTensor(traj_window).unsqueeze(0).to(DEVICE)
            )  # (1, T, D)
            # Reshape traj to (1, T, BS, Dev) for model
            traj_tensor = unflatten_rsrp_tensor(traj_tensor)

            # Sample multiple times from the same trajectory window
            for _ in range(args.samples_per_window):
                random_idx = np.random.randint(0, args.traj_len)
                curr_frame = traj_window[random_idx]  # (D,)

                curr_tensor = (
                    torch.FloatTensor(curr_frame).unsqueeze(0).to(DEVICE)
                )  # (1, D)

                for step in range(args.rollout_steps):
                    curr_tensor = (
                        torch.FloatTensor(curr_frame).unsqueeze(0).to(DEVICE)
                    )  # (1, D)

                    # Generate ONE Sample for the next step
                    with torch.no_grad():
                        # counterfactual_sample returns (B, n_samples, D)
                        next_preds_norm = model.counterfactual_sample(
                            traj_tensor,
                            curr_tensor,
                            n_samples=1,
                            noise_std=args.noise_std,
                        )
                        next_pred_norm = next_preds_norm.cpu().numpy()[0][0]  # (D,)

                    # Denormalize
                    curr_obs_denorm = curr_frame * rsrp_std + rsrp_mean
                    next_pred_denorm = next_pred_norm * rsrp_std + rsrp_mean

                    # Apply Max Delta Constraint
                    delta = next_pred_denorm - curr_obs_denorm
                    delta = np.clip(delta, -max_delta_val, max_delta_val)
                    next_pred_denorm = curr_obs_denorm + delta

                    # Re-normalize for next iteration input
                    next_frame = (next_pred_denorm - rsrp_mean) / rsrp_std

                    # Prepare for Env
                    curr_rsrp_matrix = unflatten_rsrp_tensor(
                        torch.FloatTensor(curr_obs_denorm)
                    ).numpy()
                    next_rsrp_matrix = unflatten_rsrp_tensor(
                        torch.FloatTensor(next_pred_denorm)
                    ).numpy()

                    # Sample Action and Get Reward
                    for _ in range(args.num_actions):
                        action = env.action_space.sample()
                        try:
                            (
                                state_t,
                                _,
                                reward,
                                state_t1,
                            ) = CIOLoadBalancingEnv.counterfactual_inference(
                                env.config,
                                curr_rsrp_matrix,
                                action,
                                next_rsrp_matrix,
                            )

                            cf_obs.append(state_t)
                            cf_actions.append(action)
                            cf_rewards.append(reward)
                            cf_next_obs.append(state_t1)
                            cf_terminals.append(False)
                        except AttributeError:
                            pass

                    # Update current frame for next step in rollout
                    curr_frame = next_frame

        if dones_seed[i]:
            buffer = []

    # Save
    logger.info(f"Generated {len(cf_obs)} counterfactual transitions.")
    np.savez(
        save_path,
        obs=np.array(cf_obs),
        action=np.array(cf_actions),
        reward=np.array(cf_rewards),
        next_obs=np.array(cf_next_obs),
        terminal=np.array(cf_terminals),
    )
    logger.info(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
