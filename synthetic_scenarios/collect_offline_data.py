import os
import sys
import numpy as np
import glob
from tqdm import tqdm

from env.cio_load_balancing_env import CIOLoadBalancingEnv
from common.train_offline_utils import load_config, get_scenario_files


def collect_data_from_scenarios(scenario_files, save_path):
    """
    Collects data using a random policy from the given list of scenario files.
    Saves the data to a .npz file.
    """
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_terminals = []

    print(f"Collecting data from {len(scenario_files)} scenarios...")

    for scenario_file in tqdm(scenario_files):
        config = load_config(scenario_file)
        env = CIOLoadBalancingEnv(config)

        obs, _ = env.reset()
        done = False

        while not done:
            # Random Action
            action = env.action_space.sample()

            next_obs, reward, done, _, info = env.step(action)

            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_terminals.append(done)

            obs = next_obs

    # Convert to numpy arrays
    obs_np = np.array(all_obs, dtype=np.float32)
    action_np = np.array(all_actions, dtype=np.float32)
    reward_np = np.array(all_rewards, dtype=np.float32)
    next_obs_np = np.array(all_next_obs, dtype=np.float32)
    terminal_np = np.array(all_terminals, dtype=np.float32)

    print(f"Collected {len(obs_np)} transitions.")
    print(f"Saving to {save_path}...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        obs=obs_np,
        action=action_np,
        reward=reward_np,
        next_obs=next_obs_np,
        terminal=terminal_np,
    )
    print("Done.")


def main():
    # Determine which directory to use
    if os.path.exists("generated_scenarios"):
        base_dir = "generated_scenarios"
    elif os.path.exists("scenarios"):
        base_dir = "scenarios"
    else:
        print("Error: Neither 'generated_scenarios' nor 'scenarios' directory found.")
        return

    print(f"Using scenario directory: {base_dir}")

    # Find all subdirectories (scenarios)
    all_items = os.listdir(base_dir)
    scenario_ids = [d for d in all_items if os.path.isdir(os.path.join(base_dir, d))]

    if not scenario_ids:
        print("No scenario directories found.")
        return

    print(f"Found {len(scenario_ids)} scenarios to process: {scenario_ids}")

    for scenario_id in scenario_ids:
        scenario_dir = os.path.join(base_dir, scenario_id)
        files = sorted(get_scenario_files(scenario_dir, pattern="*.yaml"))
        
        if not files:
            print(f"Skipping {scenario_id}: No .yaml files found.")
            continue

        print(f"\nProcessing scenario: {scenario_id}")
        
        # Define output path: offline_data/<scenario_id>/data.npz
        output_dir = os.path.join("offline_data", scenario_id)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "data.npz")
        
        collect_data_from_scenarios(files, save_path)

    print("\nAll data collection completed.")


if __name__ == "__main__":
    main()
