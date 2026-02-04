import yaml
import numpy as np
import random
import argparse
import math
import os
import sys


def generate_random_walk(
    start_pos,
    speed_mps,
    total_time,
    scene_width,
    scene_height,
    movement_pattern="random",
    target_direction=None,
    direction_strength=0.5,
    target_point=None,
    attraction_strength=0.3,
    speed_volatility=0.3,  # Added: Speed volatility level (0-1)
    pause_prob=0.1,  # Added: Probability of pausing per second
    loiter_radius=150.0,  # Added: Loiter radius
):
    """
    Generate a trajectory for a user with controlled movement patterns
    Now supports variable speed, pausing, and loitering.
    """
    # Fixed time interval of 1 second
    time_interval = 1.0
    total_steps = int(total_time / time_interval)

    trajectory = [{"step": 0, "time": 0, "position": start_pos}]
    current_pos = start_pos
    current_time = 0
    current_step = 0
    current_angle = random.uniform(0, 2 * math.pi)  # Initial direction
    current_speed = speed_mps

    for step in range(total_steps):
        # Change direction every second (at integer seconds)
        if current_time % 1.0 < time_interval:
            base_angle = random.uniform(0, 2 * math.pi)

            # Apply movement pattern bias
            if movement_pattern == "directional" and target_direction is not None:
                # Bias towards target direction with specified strength
                if random.random() < direction_strength:
                    # Add some noise to the target direction
                    noise = random.uniform(-math.pi / 4, math.pi / 4)  # Increase noise range
                    base_angle = target_direction + noise

            elif movement_pattern == "attraction" and target_point is not None:
                # Calculate direction towards target
                dx = target_point[0] - current_pos[0]
                dy = target_point[1] - current_pos[1]
                target_dir = math.atan2(dy, dx)

                if random.random() < attraction_strength:
                    # Bias towards target
                    noise = random.uniform(-math.pi / 4, math.pi / 4)
                    base_angle = target_dir + noise

            elif movement_pattern == "repulsion" and target_point is not None:
                # Calculate direction away from target
                dx = target_point[0] - current_pos[0]
                dy = target_point[1] - current_pos[1]
                target_dir = math.atan2(dy, dx)

                if random.random() < attraction_strength:
                    # Bias away from target (opposite direction)
                    noise = random.uniform(-math.pi / 4, math.pi / 4)
                    base_angle = target_dir + math.pi + noise

            elif movement_pattern == "loiter" and target_point is not None:
                # Calculate distance to target
                dx = target_point[0] - current_pos[0]
                dy = target_point[1] - current_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)

                # Logic: If inside radius, random walk. If outside, pull back strongly.
                if dist > loiter_radius:
                    target_dir = math.atan2(dy, dx)  # Direction back to center
                    # Strong bias to return
                    if random.random() < 0.8:
                        noise = random.uniform(-math.pi / 4, math.pi / 4)
                        base_angle = target_dir + noise
                        # Also potentially increase speed if far out? (Optional, kept simple for now)
                else:
                    # Inside radius: mostly random, maybe slight bias to stay inside?
                    # Pure random walk is usually fine for loitering
                    pass

            current_angle = base_angle

        # --- Dynamic Speed Logic ---
        # 1. Pause logic
        if random.random() < pause_prob:
            step_speed = 0.0
        else:
            # 2. Speed fluctuation: Brownian motion on speed
            # Change speed by a random factor centered at 1.0
            speed_factor = random.gauss(1.0, speed_volatility * 0.5)
            current_speed = current_speed * speed_factor

            # Clip speed to reasonable bounds [0.1 * avg, 2.0 * avg]
            current_speed = max(speed_mps * 0.1, min(speed_mps * 2.0, current_speed))
            step_speed = current_speed

        current_time += time_interval
        current_step += 1
        # Distance traveled in this time interval
        distance = step_speed * time_interval

        # Calculate new position using current direction
        dx = distance * math.cos(current_angle)
        dy = distance * math.sin(current_angle)
        new_x = max(0, min(scene_width, current_pos[0] + dx))
        new_y = max(0, min(scene_height, current_pos[1] + dy))

        current_pos = [new_x, new_y]
        trajectory.append(
            {"step": current_step, "time": current_time, "position": current_pos}
        )

    return trajectory


def generate_randomized_scenario_config(
    num_users_range,
    speed_range,
    server_configs,  # List of base station configurations, each element is a dict, must be provided
    total_time=20.0,
    output_file=None,
    movement_pattern="random",  # Global default movement pattern (used when base station config not specified)
    target_direction=None,  # Global default direction
    direction_strength=0.5,
    target_point=None,
    attraction_strength=0.3,
):
    """
    Generate a scenario configuration with randomized user counts, speeds, and initial distributions.
    Each base station can be configured with independent movement parameters.
    """
    # Randomize number of users
    num_users = random.randint(num_users_range[0], num_users_range[1])

    # Fixed time interval
    time_interval = 1.0
    total_steps = int(total_time / time_interval)

    # Base configuration
    config = {
        "scene": {
            "width": 1500,
            "height": 500,
            "total_time": total_time,
            "total_steps": total_steps,
        },
        "base_stations": [
            {
                "id": 1,
                "position": [250, 250],
                "tx_power": 46,
                "frequency": 1800,
                "bandwidth": 20,
            },
            {
                "id": 2,
                "position": [750, 250],
                "tx_power": 46,
                "frequency": 1800,
                "bandwidth": 20,
            },
            {
                "id": 3,
                "position": [1250, 250],
                "tx_power": 46,
                "frequency": 1800,
                "bandwidth": 20,
            },
        ],
        "hysteresis_config": {"enabled": True, "value": 3.0},
        "reward_config": {"type": "load_balance"},
        "cio_config": {"min_value": -6, "max_value": 6, "step_size": 0.5},
        "network_config": {
            "noise_floor": -104,
            "shadow_fading_std": 8,
            "penetration_loss": 10,
            "receiver_noise_figure": 7,
            "thermal_noise_density": -174,
            "constant_bit_rate": 1,
            "mobility_speed": "variable",  # Indicating variable speed
        },
    }

    # Define regions around each base station
    server_regions = {
        1: {"x_range": [100, 400], "y_range": [100, 400]},
        2: {"x_range": [600, 900], "y_range": [100, 400]},
        3: {"x_range": [1100, 1400], "y_range": [100, 400]},
    }

    # Validate base station configurations
    if server_configs is None:
        raise ValueError("server_configs must be provided")

    active_servers = []
    server_params = {}
    for server_config in server_configs:
        server_id = server_config.get("id")
        if server_id not in [1, 2, 3]:
            raise ValueError(f"Invalid server ID: {server_id}")
        active_servers.append(server_id)
        # Extract movement parameters for this base station, use global parameters if not specified
        server_params[server_id] = {
            "movement_pattern": server_config.get("movement_pattern", movement_pattern),
            "target_direction": server_config.get("target_direction", target_direction),
            "direction_strength": server_config.get(
                "direction_strength", direction_strength
            ),
            "target_point": server_config.get("target_point", target_point),
            "attraction_strength": server_config.get(
                "attraction_strength", attraction_strength
            ),
            "loiter_radius": server_config.get(
                "loiter_radius", 150.0
            ),  # Default loiter radius
            "speed_range": server_config.get("speed_range", speed_range),
            "user_count": server_config.get("user_count", None),  # Optional: specify user count for this base station
        }
    server_ids = active_servers

    # Determine user count allocation for each base station
    # If user_count is specified in base station config, use the specified value
    user_counts_specified = any(
        params.get("user_count") is not None for params in server_params.values()
    )
    if user_counts_specified:
        # Allocate according to specified user counts
        initial_server_assignments = []
        for server_id in server_ids:
            count = server_params[server_id].get("user_count", 0)
            initial_server_assignments.extend([server_id] * count)
        # Ensure total user count matches
        if len(initial_server_assignments) != num_users:
            # Adjust to match total user count
            if len(initial_server_assignments) < num_users:
                # Insufficient: randomly add to other base stations
                additional = num_users - len(initial_server_assignments)
                additional_assignments = np.random.choice(server_ids, size=additional)
                initial_server_assignments.extend(additional_assignments)
            else:
                # Exceeds: truncate (but should be avoided)
                initial_server_assignments = initial_server_assignments[:num_users]
    else:
        # User counts not specified, use random distribution, ensure each base station has at least 5 users
        min_users_per_server = 5
        total_min_users = len(server_ids) * min_users_per_server

        # Ensure sufficient user count (according to calling convention, num_users should be >= total_min_users)
        if num_users < total_min_users:
            # If insufficient, adjust user count (but according to requirements, this should not happen)
            num_users = total_min_users
            print(
                f"Warning: num_users increased to {num_users} to meet minimum requirement"
            )

        if len(server_ids) > 1:
            weights = np.random.dirichlet(np.ones(len(server_ids)), size=1)[0]
        else:
            weights = [1.0]

        # Assign at least 5 users to each base station
        base_assignments = []
        for server_id in server_ids:
            base_assignments.extend([server_id] * min_users_per_server)

        # Allocate remaining users by weights
        remaining_users = num_users - total_min_users
        if remaining_users > 0:
            remaining_assignments = np.random.choice(
                server_ids, size=remaining_users, p=weights
            )
            base_assignments.extend(remaining_assignments)

        initial_server_assignments = base_assignments

    mobile_devices = []
    initial_connections = {}

    for i in range(1, num_users + 1):
        selected_server = initial_server_assignments[i - 1]
        server_param = server_params[selected_server]

        # Use the speed_range of this base station
        speed_range_server = server_param["speed_range"]
        user_speed = random.uniform(speed_range_server[0], speed_range_server[1])

        # Get the region for the selected server
        region = server_regions[selected_server]
        x_range = region["x_range"]
        y_range = region["y_range"]

        # Generate random position within the server's region
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])

        start_pos = [x, y]
        trajectory = generate_random_walk(
            start_pos,
            user_speed,
            config["scene"]["total_time"],
            config["scene"]["width"],
            config["scene"]["height"],
            movement_pattern=server_param["movement_pattern"],
            target_direction=server_param["target_direction"],
            direction_strength=server_param["direction_strength"],
            target_point=server_param["target_point"],
            attraction_strength=server_param["attraction_strength"],
            # Added dynamic parameters
            speed_volatility=0.4,  # Moderate speed volatility
            pause_prob=0.15,  # 15% probability of pausing
            loiter_radius=server_param["loiter_radius"],
        )

        mobile_devices.append(
            {
                "id": i,
                "trajectory": trajectory,
                "speed": user_speed,  # Store initial speed for reference
            }
        )

        # Set initial connection to the selected server
        initial_connections[i] = int(selected_server)  # Ensure int type

    config["mobile_devices"] = mobile_devices
    config["initial_connections"] = initial_connections
    config["users"] = num_users  # Store actual num users

    # Save to file
    if output_file is None:
        server_str = "_".join(map(str, server_ids))
        # Generate descriptive string
        dir_str = ""
        for server_id in server_ids:
            param = server_params[server_id]
            if param["movement_pattern"] == "directional":
                if param["target_direction"] is not None:
                    dir_str += "R" if abs(param["target_direction"] - 0) < 0.1 else "L"
                else:
                    dir_str += "U"
            else:
                dir_str += param["movement_pattern"][0].upper()
        pattern_str = f"indep_{dir_str}"
        output_file = f"randomized_scenario_{num_users}users_{server_str}servers_{pattern_str}pattern.yaml"

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return config


def generate_scenarios_from_config(config_dict):
    # Set default values
    config = {
        "users_range": [35, 50],  # Random user count between 35 and 50
        "speed_range": [5.0, 15.0],  # Random speed between 5 and 15 m/s
        "total_time": 100.0,
        "num_scenarios": 1,
        "movement_pattern": "random",
        "direction_strength": 0.5,
        "attraction_strength": 0.3,
        **config_dict,
    }

    # Check if server_configs is provided
    if "server_configs" not in config:
        raise ValueError("'server_configs' parameter is required")

    scenario_id = config.get("scenario_id", "default_randomized_scenario")

    # Create output directory
    scenario_dir = os.path.join("generated_scenarios/", scenario_id)
    os.makedirs(scenario_dir, exist_ok=True)

    for i in range(config["num_scenarios"]):
        server_configs = config["server_configs"]
        # Generate descriptive string
        server_ids = [s.get("id") for s in server_configs]
        server_str = "_".join(map(str, server_ids))
        pattern_str = "indep"
        # Call function with server_configs
        # Note: need to pass speed_range, but each base station config can override
        output_path = os.path.join(
            scenario_dir,
            f"randomized_scenario_{i}.yaml",
        )
        random.seed(i + 1000)
        np.random.seed(i + 1000)
        generate_randomized_scenario_config(
            config["users_range"],
            config["speed_range"],
            server_configs=server_configs,
            total_time=config["total_time"],
            output_file=output_path,
            movement_pattern=config["movement_pattern"],  # As global default value
            target_direction=config.get("target_direction"),
            direction_strength=config["direction_strength"],
            target_point=config.get("target_point"),
            attraction_strength=config["attraction_strength"],
        )

    print(f"Generated {config['num_scenarios']} scenarios in {scenario_dir}")


if __name__ == "__main__":
    # Global setting: Fixed number of users for all scenarios
    FIXED_NUM_USERS = 50

    training_scenarios = [
        # Training scenario 1: Base station 1 right, base station 2 right (same direction)
        {
            "server_configs": [
                {
                    "id": 1,
                    "movement_pattern": "directional",
                    "target_direction": 0,
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                    # "user_count": 25,  # Specify user count
                },
                {
                    "id": 2,
                    "movement_pattern": "directional",
                    "target_direction": 0,
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                    # "user_count": 25,
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "train_BS1_right_BS2_right",  # Corrected scenario_id
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
        # Training scenario 2: Base station 2 left, base station 3 left (same direction)
        {
            "server_configs": [
                {
                    "id": 2,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,  # Left
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                    # "user_count": 25,  # Specify user count
                },
                {
                    "id": 3,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                    # "user_count": 25,
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "train_BS2_left_BS3_left",  # Corrected scenario_id
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
    ]

    # Define Testing Scenarios
    # Design principle: Test direction combinations not seen in training, especially direction conflicts
    testing_scenarios = [
        # Testing scenario 1: Three base stations mixed directions (including conflicts)
        {
            "server_configs": [
                {
                    "id": 1,
                    "movement_pattern": "directional",
                    "target_direction": 0,  # Right
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
                {
                    "id": 2,
                    "movement_pattern": "directional",
                    "target_direction": 0,  # Left
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
                {
                    "id": 3,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,  # Right
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "test_BS1_right_BS2_right_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
        {
            "server_configs": [
                {
                    "id": 1,
                    "movement_pattern": "directional",
                    "target_direction": 0,
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
                {
                    "id": 2,
                    "movement_pattern": "loiter",
                    "target_point": [750, 250],  # BS2 Position
                    "loiter_radius": 150.0,
                    "speed_range": [5.0, 15.0],
                },
                {
                    "id": 3,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,  # Left
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "test_BS1_right_BS2_loiter_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
        # Testing scenario 3: Base station 1 and base station 3 combination (not appeared in training)
        {
            "server_configs": [
                {
                    "id": 1,
                    "movement_pattern": "directional",
                    "target_direction": 0,  # Right
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                    # "user_count": 25,
                },
                {
                    "id": 3,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,  # Left
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                    # "user_count": 25,
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "test_BS1_right_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
    ]

    print(f"Generating randomized scenarios (Fixed Users: {FIXED_NUM_USERS})...")

    print("--- Generating Training Set ---")
    for scenario_config in training_scenarios:
        generate_scenarios_from_config(scenario_config)

    print("--- Generating Testing Set ---")
    for scenario_config in testing_scenarios:
        generate_scenarios_from_config(scenario_config)