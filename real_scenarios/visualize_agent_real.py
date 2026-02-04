import os
from common.sac_agent import Agent
from common.train_offline_utils import normalize_obs, load_config
from env.cio_load_balancing_env_real import CIOLoadBalancingEnv
import numpy as np


def visualize_agent(model_path, scenario_path, osm_file_path=None, save_anim_path=None):
    """
    Visualizes the agent's performance on a specific test scenario.
    Args:
        model (str): Name of the trained model to load.
        scenario_path (str): Path to the scenario file to visualize.
    """

    # 1. Load Scenario
    print(f"Loading scenario: {scenario_path}")
    config = load_config(scenario_path)

    if osm_file_path is None:
        osm_file_path = "../load_balancing/sdnu.osm"
    config["osm_file_path"] = osm_file_path

    if save_anim_path:
        os.makedirs(os.path.dirname(save_anim_path), exist_ok=True)
        config["save_animation_path"] = save_anim_path

    env = CIOLoadBalancingEnv(config)

    # 2. Initialize Agent
    num_bs = len(config["base_stations"])
    num_users = len(config["mobile_devices"])
    obs_dim = num_bs * num_users
    action_dim = num_bs * (num_bs - 1)

    agent = Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        batch_size=256,
        max_size=100000,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    # 3. Load Model
    obs_mean = None
    obs_std = None
    normalization_params_path = os.path.join(model_path, "normalization_params.npz")
    if os.path.exists(normalization_params_path):
        normalization_params = np.load(normalization_params_path)
        obs_mean = normalization_params["mean"]
        obs_std = normalization_params["std"]
    else:
        print(
            f"Error: Normalization parameters not found at {normalization_params_path}"
        )
        return

    model_path = os.path.join(model_path, "model", "model_final.ckpt")
    if os.path.exists(model_path):
        agent.restore(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model not found at {model_path}")
        return

    # 4. Run Episode
    obs, _ = env.reset()
    # obs = extract_rsrp(obs, num_bs, num_users) # Removed as env now returns only RSRP
    env.render(mode="human")  # Initial render

    done = False
    step = 0
    total_reward = 0

    print("\nStarting visualization...")
    print("-" * 50)

    while not done:
        # Normalize observation
        norm_obs = normalize_obs(obs, obs_mean, obs_std)

        # Predict action
        action = agent.predict(norm_obs)

        # Step environment
        next_obs, reward, done, _, info = env.step(action)
        # next_obs = extract_rsrp(next_obs, num_bs, num_users) # Removed as env now returns only RSRP

        # Render
        env.render(mode="")

        total_reward += reward
        obs = next_obs
        step += 1

        print(
            f"Step {step}: Reward={reward:.4f}, LoadBalance={info.get('load_balance', 0):.4f}, Throughput={info.get('throughput', 0):.2f}"
        )

    env.close()
    print("-" * 50)
    print(f"Episode finished. Total Reward: {total_reward:.4f}")
    print(
        f"Animation saved to {save_anim_path}"
        if config.get("save_animation_path")
        else "Visualization finished."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model to visualize")
    parser.add_argument("--scenario_path", type=str, help="Path to the scenario file")
    parser.add_argument("--osm_file_path", type=str, help="Path to the OSM file")
    args = parser.parse_args()


    args.model_path = "./experiment1/results_sac+20+cf/BS1_right_BS3_left"
    args.scenario_path = "./scenarios/BS1_right_BS3_left/osm_controlled_scenario_50users_10.0mps_1_3servers_directional_RLpattern_100steps_0.yaml"
    args.osm_file_path = "./sdnu.osm"
    args.save_anim_path = "./visualizations/animation.gif"

    visualize_agent(args.model_path, args.scenario_path, args.osm_file_path, args.save_anim_path)
