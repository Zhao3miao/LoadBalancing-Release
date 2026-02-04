import os


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import copy
import random
from torch.utils.data import DataLoader, TensorDataset
from parl.utils import CSVLogger, logger
from common.cf_model import (
    CounterfactualModel,
    gaussian_nll,
    unflatten_rsrp_tensor,
)
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
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


def prepare_data(
        train_data_paths,
        test_data_path,
        train_ratio,
        test_ratio,
        norm_save_path,
        traj_len,
        pred_steps,
        episode_len,
        samples_per_episode,
):
    obs_list = []
    dones_list = []

    for path in train_data_paths:
        if not os.path.exists(path):
            logger.info(f"Warning: Train data not found at {path}")
            continue

        logger.info(f"Loading train data from {path}...")
        train_data = np.load(path)
        total_train = len(train_data["obs"])
        num_train = int(total_train * train_ratio)
        obs_train = train_data["obs"][:num_train]
        dones_train = train_data["terminal"][:num_train]
        obs_list.append(obs_train)
        dones_list.append(dones_train)

    if not obs_list:
        raise ValueError("No training data loaded!")

    # Load Test Data (if requested)
    if test_data_path and test_ratio > 0 and os.path.exists(test_data_path):
        logger.info(
            f"Loading test data from {test_data_path} with ratio {test_ratio}..."
        )
        test_data = np.load(test_data_path)
        total_count = len(test_data["obs"])
        # Strict Few-Shot: Take first N samples
        num_samples = int(total_count * test_ratio)
        logger.info(f"Adding {num_samples}/{total_count} samples from test set.")
        obs_test = test_data["obs"][:num_samples]
        obs_list.append(obs_test)

    elif test_data_path and test_ratio > 0:
        logger.info(f"Warning: Test data not found at {test_data_path}")
    obs = np.concatenate(obs_list, axis=0)

    rsrp_mean = obs.mean()
    rsrp_std = obs.std() + 1e-8

    np.savez(norm_save_path, mean=rsrp_mean, std=rsrp_std)
    logger.info(f"Saved normalization params to {norm_save_path}")

    obs_norm = (obs - rsrp_mean) / rsrp_std

    # Create Trajectories
    X_traj = []
    X_traj_2 = []
    X_curr = []
    Y_next_seq = []  # Sequence of future frames

    num_samples = len(obs_norm)
    num_episodes = num_samples // episode_len
    obs_episodes = obs_norm[: num_episodes * episode_len].reshape(
        num_episodes, episode_len, -1
    )

    logger.info(f"Reshaped data into {num_episodes} episodes of length {episode_len}")

    for ep_idx in range(num_episodes):
        episode = obs_episodes[ep_idx]

        for _ in range(samples_per_episode):
            # 1. Sample Main Trajectory Window (Context + Local History)
            max_start_idx = episode_len - traj_len - pred_steps
            if max_start_idx <= 0:
                continue

            start_idx = np.random.randint(0, max_start_idx + 1)
            traj_window = episode[start_idx: start_idx + traj_len]

            # 2. Sample Current Frame (within traj_window)
            curr_rel_idx = np.random.randint(0, traj_len - pred_steps)
            curr_frame = traj_window[curr_rel_idx]

            # Absolute index of curr in episode
            curr_abs_idx = start_idx + curr_rel_idx

            # 3. Get Ground Truth Future Sequence
            next_seq = episode[curr_abs_idx + 1: curr_abs_idx + 1 + pred_steps]

            # 4. Sample Auxiliary Trajectory Window (Global Context Consistency)
            start_idx_2 = np.random.randint(0, episode_len - traj_len + 1)
            traj_window_2 = episode[start_idx_2: start_idx_2 + traj_len]

            X_traj.append(traj_window)
            X_traj_2.append(traj_window_2)
            X_curr.append(curr_frame)
            Y_next_seq.append(next_seq)

    X_traj = np.array(X_traj)
    X_traj_2 = np.array(X_traj_2)
    X_curr = np.array(X_curr)
    Y_next_seq = np.array(Y_next_seq)  # (B, PRED_STEPS, D)

    # Reshape traj
    B, T, D = X_traj.shape
    X_traj = unflatten_rsrp_tensor(torch.FloatTensor(X_traj)).numpy()
    X_traj_2 = unflatten_rsrp_tensor(torch.FloatTensor(X_traj_2)).numpy()

    logger.info(f"Prepared Dataset: {len(X_traj)} samples.")
    return X_traj, X_traj_2, X_curr, Y_next_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="synthetic", help="Environment to use."
    )
    parser.add_argument(
        "--ratio",
        nargs="+",
        type=float,
        default=[1.0, 0.1, 0.0],
        help="Ratios for [Train, Test, CF] data. Only Train and Test are used here.",
    )
    parser.add_argument(
        "--train_scenarios",
        nargs="+",
        type=str,
        default=["BS1_right_BS2_right", "BS2_left_BS3_left"],
        help="Name of the train scenarios",
    )
    parser.add_argument(
        "--test_scenario", type=str, required=True, help="Name of the test scenario"
    )
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.05,
        help="Standard deviation of Gaussian noise added to latent variables during training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--kl_weight", type=float, default=0.01, help="Weight for KL divergence loss"
    )
    parser.add_argument(
        "--latent_weight",
        type=float,
        default=5.0,
        help="Weight for latent multi-step consistency loss",
    )
    parser.add_argument(
        "--cons_weight",
        type=float,
        default=5.0,
        help="Weight for global latent consistency loss",
    )
    parser.add_argument(
        "--traj_len", type=int, default=20, help="Length of trajectory window"
    )
    parser.add_argument(
        "--pred_steps", type=int, default=3, help="Prediction horizon steps"
    )
    parser.add_argument(
        "--episode_len", type=int, default=100, help="Length of each episode"
    )
    parser.add_argument(
        "--samples_per_episode",
        type=int,
        default=25,
        help="Number of samples to extract per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    assert len(args.ratio) == 3
    train_ratio = args.ratio[0]
    test_ratio = args.ratio[1]

    logger.info(f"Training CF Model with Test Ratio: {test_ratio}")

    # Paths
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

    scenario_results_dir = os.path.join(
        CURRENT_DIR, env_dir, args.results_dir, args.test_scenario
    )
    os.makedirs(scenario_results_dir, exist_ok=True)

    save_path = os.path.join(scenario_results_dir, "cf_model.pth")
    norm_path = os.path.join(scenario_results_dir, "cf_norm.npz")

    X_traj, X_traj_2, X_curr, Y_next_seq = prepare_data(
        train_data_paths,
        test_data_path,
        train_ratio,
        test_ratio,
        norm_path,
        args.traj_len,
        args.pred_steps,
        args.episode_len,
        args.samples_per_episode,
    )

    logger.info(f"Initialized Dataset with {len(X_traj)} samples")
    logger.info(f"Data Checksum X_traj: {np.sum(X_traj):.6f}")
    logger.info(f"Data Checksum Y_next_seq: {np.sum(Y_next_seq):.6f}")

    dataset = TensorDataset(
        torch.FloatTensor(X_traj),
        torch.FloatTensor(X_traj_2),
        torch.FloatTensor(X_curr),
        torch.FloatTensor(Y_next_seq),
    )
    
    # Use a specific generator for DataLoader to ensure reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=g)

    model = CounterfactualModel(traj_len=args.traj_len).to(DEVICE)
    
    # # Log initial model checksum
    # init_sum = sum(p.sum().item() for p in model.parameters())
    # logger.info(f"Initial Model Parameters Checksum: {init_sum:.6f}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger.info(f"Starting training on {DEVICE}...")
    logger.info(f"Noise settings: std={args.noise_std} (fixed)")
    csv_logger = CSVLogger(os.path.join(scenario_results_dir, "train_cf_result.csv"))

    min_loss = float("inf")
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_nll = 0
        if epoch == 0:
            for batch_idx, (traj, traj_2, curr, next_seq_true) in enumerate(dataloader):
                logger.info(f"Epoch 0 Batch 0 Checksum: {traj.sum().item():.6f}")
                break

        total_kl = 0
        total_cons = 0

        current_noise_std = args.noise_std

        for traj, traj_2, curr, next_seq_true in dataloader:
            traj, traj_2, curr, next_seq_true = (
                traj.to(DEVICE),
                traj_2.to(DEVICE),
                curr.to(DEVICE),
                next_seq_true.to(DEVICE),
            )

            optimizer.zero_grad()

            # 1. Encode Global Context
            z_g_mu, z_g_logvar = model.global_enc(traj)
            z_g = model.reparameterize(z_g_mu, z_g_logvar)

            # 2. Encode Initial Local State
            z_l_mu, z_l_logvar = model.local_enc(curr)
            z_l = model.reparameterize(z_l_mu, z_l_logvar)

            loss_nll_accum = 0
            loss_latent_accum = 0

            # Multi-step Prediction Loop
            curr_z_l = z_l

            for step in range(args.pred_steps):
                # Get Ground Truth for this step
                next_true = next_seq_true[:, step, :]

                # Predict Next Latent State
                z_l_next_mu, z_l_next_logvar = model.transition(curr_z_l, z_g)
                z_l_next = model.reparameterize(z_l_next_mu, z_l_next_logvar)

                # Add controllable noise to latent variable
                if current_noise_std > 0:
                    noise = torch.randn_like(z_l_next) * current_noise_std
                    z_l_next = z_l_next + noise

                # Decode to Observation
                next_obs_mean, next_obs_logvar = model.decoder(z_l_next)
                # Reconstruction Loss
                nll = gaussian_nll(next_true, next_obs_mean, next_obs_logvar)
                loss_nll_accum += nll
                # Latent Consistency Loss (Target)
                # Encode the TRUE next state to get target z_l
                with torch.no_grad():
                    z_l_target_mu, _ = model.local_enc(next_true)

                latent_mse = F.mse_loss(z_l_next_mu, z_l_target_mu.detach())
                loss_latent_accum += latent_mse

                # Update current latent state for next step (Auto-regressive)
                curr_z_l = z_l_next

            # Average losses over steps
            avg_nll_step = loss_nll_accum / args.pred_steps
            avg_latent_step = loss_latent_accum / args.pred_steps

            # KL Loss for Global Z
            kl_g = (
                    -0.5
                    * torch.sum(
                1 + z_g_logvar - z_g_mu.pow(2) - z_g_logvar.exp(), dim=1
            ).mean()
            )

            # KL Loss for Initial Local Z
            kl_l = (
                    -0.5
                    * torch.sum(
                1 + z_l_logvar - z_l_mu.pow(2) - z_l_logvar.exp(), dim=1
            ).mean()
            )

            # Global Consistency Loss (Only Positive Consistency)
            z_g_mu_2, _ = model.global_enc(traj_2)

            # Only minimize distance between positive pairs (same episode)
            loss_consistency = F.mse_loss(z_g_mu, z_g_mu_2)

            loss = (
                    avg_nll_step
                    + args.kl_weight * (kl_g + kl_l)
                    + args.latent_weight * avg_latent_step
                    + args.cons_weight * loss_consistency
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_nll += avg_nll_step.item()
            total_kl += (kl_g + kl_l).item()
            total_cons += loss_consistency.item()

        avg_loss = total_loss / len(dataloader)
        avg_nll = total_nll / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_cons = total_cons / len(dataloader)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | NLL: {avg_nll:.4f} | KL: {avg_kl:.4f} | Cons: {avg_cons:.4f}"
            )

        csv_logger.log_dict(
            {
                "episode": epoch,
                "loss": avg_loss,
                "nll": avg_nll,
                "kl": avg_kl,
                "consistency": avg_cons,
            }
        )

        # Check for best model in last 50 epochs
        if epoch >= args.epochs - 50:
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_model_state = copy.deepcopy(model.state_dict())
                # print(f"  New best model found at epoch {epoch+1} with loss {min_loss:.4f}")

    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        logger.info(
            f"Best model (from last 50 epochs) saved to {save_path} with loss {min_loss:.4f}"
        )
    else:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
