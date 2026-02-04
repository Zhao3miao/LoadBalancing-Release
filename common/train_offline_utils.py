import os
import numpy as np
from parl.utils import CSVLogger, logger
import glob
import yaml


def load_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_scenario_files(directory, pattern="**/*.yaml"):
    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    return sorted(files)


def normalize_reward(reward):
    scaled_reward = reward * 10
    return scaled_reward


def normalize_obs(obs, mean=None, std=None):
    if mean is not None and std is not None:
        return (obs - mean) / (std + 1e-8)
    return obs


def load_offline_data_to_agent(
    agent, data_paths, ratio=1.0, obs_mean=None, obs_std=None
):
    """
    Loads .npz data files and populates the agent's replay buffer.
    """
    total_transitions = 0

    for path in data_paths:
        if not os.path.exists(path):
            logger.info(f"Warning: Data file not found: {path}")
            continue

        logger.info(f"Loading data from {path}...")
        data = np.load(path)

        obs = normalize_obs(data["obs"], mean=obs_mean, std=obs_std)
        action = data["action"]
        reward = normalize_reward(data["reward"])
        next_obs = normalize_obs(data["next_obs"], mean=obs_mean, std=obs_std)
        terminal = data["terminal"]

        logger.info(
            f"Reward Stats - Min: {reward.min():.4f}, Max: {reward.max():.4f}, Mean: {reward.mean():.4f}"
        )

        count = len(obs)
        use_count = int(count * ratio)
        logger.info(f"Using {use_count}/{count} transitions (Ratio: {ratio})")

        total_transitions += use_count

        for i in range(use_count):
            agent.append(obs[i], action[i], reward[i], next_obs[i], terminal[i])

    logger.info(f"Loaded {total_transitions} transitions into ReplayBuffer.")
    logger.info(f"Current Buffer Size: {agent.replay_buffer.size()}")


def train_offline(agent, num_steps, save_path, log_interval=1000):
    """
    Trains the agent using data already in the replay buffer.
    """
    os.makedirs(save_path, exist_ok=True)
    model_dir = os.path.join(save_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(save_path, "train_sac_result.csv"))

    logger.info(f"Starting Offline Training for {num_steps} steps...")

    critic_losses = []
    actor_losses = []

    for step in range(num_steps):
        critic_loss, actor_loss = agent.learn()
        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)

        if (step + 1) % log_interval == 0:
            avg_critic = np.mean(critic_losses[-log_interval:])
            avg_actor = np.mean(actor_losses[-log_interval:])
            logger.info(
                f"Step {step+1}, Critic Loss: {avg_critic:.4f}, Actor Loss: {avg_actor:.4f}"
            )
            csv_logger.log_dict(
                {
                    "step": step + 1,
                    "critic_loss": avg_critic,
                    "actor_loss": avg_actor,
                }
            )

        if (step + 1) % 1000 == 0:
            agent.save(os.path.join(model_dir, f"model_step_{step+1}.ckpt"))

    # Save final model
    agent.save(os.path.join(model_dir, "model_final.ckpt"))
    logger.info("Offline Training Completed.")
