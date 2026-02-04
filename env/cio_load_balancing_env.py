import gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import math
import os

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"


class CIOLoadBalancingEnv(gym.Env):
    def __init__(self, config):
        super(CIOLoadBalancingEnv, self).__init__()

        self.config = config

        self.scene_width = self.config["scene"]["width"]
        self.scene_height = self.config["scene"]["height"]
        self.total_time = self.config["scene"]["total_steps"]
        self.current_time = 0

        self.base_stations = self.config["base_stations"]
        self.mobile_devices = self.config["mobile_devices"]

        # COST Hata model parameters
        self.f_c = 2000  # Carrier frequency (MHz)
        self.h_b = 30  # Base station antenna height (m)
        self.h_m = 1.5  # UE antenna height (m)
        self.P_tx_dBm = 20  # eNodeB transmit power (dBm)
        self.G_tx = 14  # Base station antenna gain (dBi) - typical value
        self.G_rx = 0  # UE antenna gain (dBi)
        self.cable_loss = 2  # Cable loss (dB) - typical value

        # Initialize CIO matrix (CIO values between base stations)
        self.cio_matrix = {}
        for i, bs_i in enumerate(self.base_stations):
            for j, bs_j in enumerate(self.base_stations):
                if i != j:
                    self.cio_matrix[(bs_i["id"], bs_j["id"])] = 0  # Initial CIO is 0

        # State space: only contains RSRP values
        self.num_base_stations = len(self.base_stations)
        self.num_devices = len(self.mobile_devices)
        self.state_dim = (
            self.num_base_stations * self.num_devices
        )  # Only contains RSRP features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.action_dim = len(self.base_stations) * (len(self.base_stations) - 1)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        self.fig = None
        self.ax = None
        self.rendering = False

        self.save_animation_path = self.config.get("save_animation_path", None)
        self.frames = []  # Store animation frames

        self.throughput_history = deque(maxlen=100)
        self.fairness_history = deque(maxlen=100)
        self.blocking_history = deque(maxlen=100)
        self.average_load_history = deque(
            maxlen=100
        )  # New: average load history recording

        self.device_trajectories = {}
        for device in self.mobile_devices:
            self.device_trajectories[device["id"]] = deque(maxlen=50)

        self.reset()

    def reset(self):
        self.current_time = 0
        self.throughput_history.clear()
        self.fairness_history.clear()
        self.blocking_history.clear()

        for device in self.mobile_devices:
            device["current_position"] = device["trajectory"][0]["position"]
            device["connected_bs"] = None
            device["rsrp_measurements"] = {}

        for bs in self.base_stations:
            bs["connected_devices"] = []
            bs["load"] = 0
            bs["throughput"] = 0
            bs["resource_utilization"] = 0
            bs["mcs_distribution"] = [0] * 10  # Only record first 10 MCS indices

        for key in self.cio_matrix:
            self.cio_matrix[key] = 0

        initial_connections = self.config.get("initial_connections", {})

        if initial_connections:
            for device_id, bs_id in initial_connections.items():
                device = next(
                    (d for d in self.mobile_devices if d["id"] == device_id), None
                )
                bs = next((b for b in self.base_stations if b["id"] == bs_id), None)

                if device and bs:
                    device["connected_bs"] = bs
                    bs["connected_devices"].append(device)
        else:
            scene_center = (self.scene_width / 2, self.scene_height / 2)
            middle_bs = None
            min_distance = float("inf")

            for bs in self.base_stations:
                bs_position = bs["position"]
                distance = math.sqrt(
                    (bs_position[0] - scene_center[0]) ** 2
                    + (bs_position[1] - scene_center[1]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    middle_bs = bs

            if middle_bs is not None:
                for device in self.mobile_devices:
                    device["connected_bs"] = middle_bs
                    middle_bs["connected_devices"].append(device)

        self._measure_rsrp()
        # self._update_connections()

        self._calculate_load()
        throughput = self._calculate_throughput()
        fairness = self._calculate_fairness()
        blocking = self._calculate_blocking()

        current_average_load = np.mean(
            [bs["resource_utilization"] for bs in self.base_stations]
        )
        self.average_load_history.append(current_average_load)

        self.throughput_history.append(throughput)
        self.fairness_history.append(fairness)
        self.blocking_history.append(blocking)

        for device in self.mobile_devices:
            self.device_trajectories[device["id"]].append(device["current_position"])
        return self._get_observation(), {}

    def step(self, action):
        action_index = 0
        for i, bs_i in enumerate(self.base_stations):
            for j, bs_j in enumerate(self.base_stations):
                if i != j:
                    cio = action[action_index] * 6
                    cio = max(-6, min(6, cio))  # Limit CIO to [-6, 6]
                    self.cio_matrix[(bs_i["id"], bs_j["id"])] = cio

                    action_index += 1

        self._update_connections()

        self._calculate_load()
        throughput = self._calculate_throughput()
        fairness = self._calculate_fairness()
        blocking = self._calculate_blocking()

        loads = [bs["resource_utilization"] for bs in self.base_stations]
        load_balance = -np.std(loads) if loads else 0

        reward = self._calculate_reward(throughput, fairness, blocking)

        self.current_time += 1
        self._update_device_positions()

        self._measure_rsrp()

        done = self.current_time >= self.total_time

        self.throughput_history.append(throughput)
        self.fairness_history.append(fairness)
        self.blocking_history.append(blocking)
        self.average_load_history.append(-load_balance)

        for device in self.mobile_devices:
            self.device_trajectories[device["id"]].append(device["current_position"])

        rsrp_matrix = np.zeros((self.num_base_stations, self.num_devices))
        for i, bs in enumerate(self.base_stations):
            for j, device in enumerate(self.mobile_devices):
                rsrp_matrix[i, j] = device["rsrp_measurements"].get(bs["id"], 0.0)

        info = {
            "throughput": throughput,
            "fairness": fairness,
            "blocking": blocking,
            "load_balance": load_balance,
            "non_blocked": 1 - blocking,
            "rsrp_matrix": rsrp_matrix,
        }

        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        state = []

        for bs in self.base_stations:
            for device in self.mobile_devices:
                rsrp_value = device["rsrp_measurements"].get(bs["id"], 0.0)
                state.append(rsrp_value)

        return np.array(state, dtype=np.float32)

    def _update_device_positions(self):
        for device in self.mobile_devices:
            for point in device["trajectory"]:
                if point["step"] == self.current_time:
                    device["current_position"] = point["position"]
                    break

    def _calculate_path_loss_cost_hata(self, distance_m):
        """
        Calculate COST231-Hata path loss (applicable to 1500-2000 MHz)
        L = 46.3 + 33.9*log10(f) - 13.82*log10(h_b) - a(h_m) + (44.9 - 6.55*log10(h_b))*log10(d) + C_m

        Where:
        f: frequency (MHz)
        h_b: base station height (m)
        h_m: mobile station height (m)
        d: distance (km)
        C_m: city center correction factor (0 dB for medium city, 3 dB for metropolitan)
        a(h_m): mobile station antenna height correction factor
        """
        d_km = max(distance_m / 1000.0, 0.02)  # Minimum 20m

        f = self.f_c

        a_hm = (1.1 * math.log10(f) - 0.7) * self.h_m - (1.56 * math.log10(f) - 0.8)

        C_m = 0  # Assume medium city/suburban environment

        loss = (
            46.3
            + 33.9 * math.log10(f)
            - 13.82 * math.log10(self.h_b)
            - a_hm
            + (44.9 - 6.55 * math.log10(self.h_b)) * math.log10(d_km)
            + C_m
        )

        return loss

    def _measure_rsrp(self):
        for device in self.mobile_devices:
            device["rsrp_measurements"] = {}
            for bs in self.base_stations:
                distance = math.sqrt(
                    (device["current_position"][0] - bs["position"][0]) ** 2
                    + (device["current_position"][1] - bs["position"][1]) ** 2
                )

                path_loss = self._calculate_path_loss_cost_hata(distance)

                rsrp = (
                    self.P_tx_dBm + self.G_tx + self.G_rx - path_loss - self.cable_loss
                )

                device["rsrp_measurements"][bs["id"]] = rsrp

    def _update_connections(self):
        for device in self.mobile_devices:
            best_bs = None
            best_score = -np.inf

            current_bs = device["connected_bs"]

            hysteresis_enabled = self.config.get("hysteresis_config", {}).get(
                "enabled", True
            )
            hysteresis_value = self.config.get("hysteresis_config", {}).get(
                "value", 2.0
            )

            for bs in self.base_stations:
                rsrp = device["rsrp_measurements"][bs["id"]]

                if current_bs is not None and current_bs["id"] != bs["id"]:
                    cio = (
                        self.cio_matrix[(current_bs["id"], bs["id"])]
                        - self.cio_matrix[(bs["id"], current_bs["id"])]
                    )
                else:
                    cio = 0

                score = rsrp + cio

                if (
                    hysteresis_enabled
                    and current_bs is not None
                    and bs["id"] == current_bs["id"]
                ):
                    score += hysteresis_value

                if score > best_score:
                    best_score = score
                    best_bs = bs

            if device["connected_bs"]["id"] != best_bs["id"]:
                if device["connected_bs"] is not None:
                    device["connected_bs"]["connected_devices"].remove(device)
                device["connected_bs"] = best_bs
                best_bs["connected_devices"].append(device)

    def _calculate_load(self):
        total_prbs = 25
        user_demand_bps = self.config.get("user_demand", 1024 * 1000)

        for bs in self.base_stations:
            bs["throughput"] = 0
            bs["mcs_distribution"] = [0] * 10

            current_bs_prb_usage = 0

            for device in bs["connected_devices"]:
                rsrp = device["rsrp_measurements"].get(bs["id"], -120)
                min_rsrp = -110
                max_rsrp = -65

                if rsrp >= max_rsrp:
                    mcs_index = 9
                elif rsrp <= min_rsrp:
                    mcs_index = 0
                else:
                    mcs_index = int(9 * (rsrp - min_rsrp) / (max_rsrp - min_rsrp))

                bs["mcs_distribution"][mcs_index] += 1

                spectral_efficiency = 0.5 + mcs_index * 0.5

                prb_bandwidth = 180 * 1000

                rate_per_prb = spectral_efficiency * prb_bandwidth

                if rate_per_prb > 0:
                    required_prbs_user = math.ceil(user_demand_bps / rate_per_prb)
                else:
                    required_prbs_user = total_prbs

                current_bs_prb_usage += required_prbs_user

                device["required_prbs"] = required_prbs_user
                device["rate_per_prb"] = rate_per_prb

            bs["resource_utilization"] = min(current_bs_prb_usage / total_prbs, 1.0)

            if current_bs_prb_usage <= total_prbs:
                for device in bs["connected_devices"]:
                    bs["throughput"] += user_demand_bps
            else:
                scaling_factor = total_prbs / current_bs_prb_usage
                for device in bs["connected_devices"]:
                    allocated_prbs = device["required_prbs"] * scaling_factor
                    achieved_rate = allocated_prbs * device["rate_per_prb"]
                    bs["throughput"] += achieved_rate

    def _calculate_throughput(self):
        throughput = 0
        for bs in self.base_stations:
            throughput += bs["throughput"]
        return throughput

    def _calculate_fairness(self):
        throughputs = [
            bs["throughput"] for bs in self.base_stations if bs["throughput"] > 0
        ]
        if len(throughputs) == 0:
            return 1
        return sum(throughputs) ** 2 / (
            len(throughputs) * sum(t**2 for t in throughputs)
        )

    def _calculate_blocking(self):
        blocked = 0
        blocked_bps = 512 * 1000
        user_demand_bps = self.config.get("user_demand", 1024 * 1000)
        total_prbs = 25  # 5 MHz

        for device in self.mobile_devices:
            if device["connected_bs"] is None:
                blocked += 1
            else:
                bs = device["connected_bs"]

                if bs["resource_utilization"] >= 1.0:
                    rsrp = device["rsrp_measurements"].get(bs["id"], -120)
                    min_rsrp = -110
                    max_rsrp = -80
                    if rsrp >= max_rsrp:
                        mcs_index = 9
                    elif rsrp <= min_rsrp:
                        mcs_index = 0
                    else:
                        mcs_index = int(9 * (rsrp - min_rsrp) / (max_rsrp - min_rsrp))

                    spectral_efficiency = 0.5 + mcs_index * 0.5
                    rate_per_prb = spectral_efficiency * 180 * 1000

                    if rate_per_prb > 0:
                        required_prbs_user = math.ceil(user_demand_bps / rate_per_prb)
                    else:
                        required_prbs_user = total_prbs

                    total_required_prbs = 0
                    for d in bs["connected_devices"]:
                        r = d["rsrp_measurements"].get(bs["id"], -120)
                        if r >= max_rsrp:
                            m = 9
                        elif r <= min_rsrp:
                            m = 0
                        else:
                            m = int(9 * (r - min_rsrp) / (max_rsrp - min_rsrp))
                        se = 0.5 + m * 0.5
                        rp = se * 180 * 1000
                        if rp > 0:
                            total_required_prbs += math.ceil(user_demand_bps / rp)
                        else:
                            total_required_prbs += total_prbs

                    if total_required_prbs > total_prbs:
                        scaling_factor = total_prbs / total_required_prbs
                        achieved_rate = (
                            required_prbs_user * scaling_factor * rate_per_prb
                        )
                    else:
                        achieved_rate = user_demand_bps
                else:
                    achieved_rate = user_demand_bps

                if achieved_rate < blocked_bps:
                    blocked += 1

        return blocked / len(self.mobile_devices) if self.mobile_devices else 0

    def _calculate_reward(self, throughput, fairness, blocking):
        reward_type = self.config.get("reward_config", {}).get("type", "throughput")

        if reward_type == "throughput":
            max_throughput_per_bs = 25 * 5.0 * 180 * 1000  # bps
            max_possible_throughput = self.num_base_stations * max_throughput_per_bs

        elif reward_type == "non_blocked":
            return 1 - blocking
        elif reward_type == "load_balance":
            loads = [bs["resource_utilization"] for bs in self.base_stations]
            return -np.std(loads)
        elif reward_type == "throughput_and_balance":
            max_throughput_per_bs = 25 * 5.0 * 180 * 1000  # bps
            max_possible_throughput = self.num_base_stations * max_throughput_per_bs
            if max_possible_throughput > 0:
                normalized_throughput = throughput / max_possible_throughput
            else:
                normalized_throughput = 0

            loads = [bs["resource_utilization"] for bs in self.base_stations]
            load_std = np.std(loads) if loads else 0
            load_balance_reward = 1 - load_std

            return normalized_throughput + load_balance_reward
        else:
            alpha = self.config["reward_config"].get("alpha", 0.5)
            beta = self.config["reward_config"].get("beta", 0.3)
            gamma = self.config["reward_config"].get("gamma", 0.2)
            return alpha * throughput + beta * fairness - gamma * blocking

    def render(self, mode="human"):
        if self.fig is None:
            aspect_ratio = self.scene_width / self.scene_height
            base_height = 10
            calculated_width = (base_height * aspect_ratio) / 0.75
            max_width = 20
            min_width = 8
            fig_width = max(min(calculated_width, max_width), min_width)

            self.fig, self.ax = plt.subplots(figsize=(fig_width, base_height))
            self.rendering = True

        self.ax.clear()

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "stix"

        self.ax.set_xlim(0, self.scene_width)
        self.ax.set_ylim(0, self.scene_height)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"CIO Load Balancing Environment - Time: {self.current_time}", fontsize=16
        )

        for bs in self.base_stations:
            self.ax.plot(
                bs["position"][0],
                bs["position"][1],
                "s",
                color="red",
                markersize=12,
                markeredgecolor="darkred",
                markeredgewidth=2,
            )
            self.ax.text(
                bs["position"][0],
                bs["position"][1] + 0.5,
                f"BS{bs['id']}\nLoad:{bs['resource_utilization']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
            )

        colors = [
            "blue",
            "orange",
            "purple",
            "brown",
            "pink",
            "olive",
            "cyan",
            "magenta",
        ]
        for i, device in enumerate(self.mobile_devices):
            device_id = device["id"]
            if len(self.device_trajectories[device_id]) > 1:
                trajectory = list(self.device_trajectories[device_id])
                x_coords = [pos[0] for pos in trajectory]
                y_coords = [pos[1] for pos in trajectory]
                color = colors[i % len(colors)]
                self.ax.plot(
                    x_coords, y_coords, "-", color=color, alpha=0.7, linewidth=2
                )

        for device in self.mobile_devices:
            color = "green" if device["connected_bs"] is not None else "gray"
            self.ax.plot(
                device["current_position"][0],
                device["current_position"][1],
                "o",
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            self.ax.text(
                device["current_position"][0],
                device["current_position"][1] + 0.5,
                f"D{device['id']}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.2"),
            )

        for device in self.mobile_devices:
            if device["connected_bs"] is not None:
                bs_pos = device["connected_bs"]["position"]
                dev_pos = device["current_position"]
                self.ax.plot(
                    [bs_pos[0], dev_pos[0]],
                    [bs_pos[1], dev_pos[1]],
                    color="blue",
                    alpha=0.4,
                    linewidth=1,
                )

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Base Station",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=8,
                label="User Devices",
            ),
        ]

        plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.75)

        current_throughput = (
            self.throughput_history[-1] if self.throughput_history else 0
        )
        current_fairness = self.fairness_history[-1] if self.fairness_history else 0
        current_blocking = self.blocking_history[-1] if self.blocking_history else 0
        current_average_load = (
            self.average_load_history[-1] if self.average_load_history else 0
        )

        performance_text = (
            f"Throughput: {current_throughput:.2f} kbps\n"
            f"Fairness: {current_fairness:.2f}\n"
            f"Blocking: {current_blocking:.2f}\n"
            f"Load Std: {current_average_load:.2f}\n"
        )

        self.fig.text(
            0.76,
            0.75,
            performance_text,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"),
            verticalalignment="top",
            horizontalalignment="left",
            linespacing=1.4,
        )

        bs_ids = sorted([bs["id"] for bs in self.base_stations])
        cio_matrix_str = "CIO Matrix:\n"

        header = "     " + " ".join([f"BS{id:>4}" for id in bs_ids]) + "\n"
        cio_matrix_str += header

        for i, id_i in enumerate(bs_ids):
            row = f"BS{id_i} "
            for j, id_j in enumerate(bs_ids):
                if i == j:
                    value = 0.0
                else:
                    value = self.cio_matrix.get((id_i, id_j), 0.0)
                row += f"{value:6.2f} "
            cio_matrix_str += row + "\n"

        self.fig.text(
            0.76,
            0.60,
            cio_matrix_str,
            fontsize=9,
            bbox=dict(facecolor="lightblue", alpha=0.8, boxstyle="round,pad=0.5"),
            verticalalignment="top",
            horizontalalignment="left",
            linespacing=1.2,
            fontfamily="monospace",
        )

        self.ax.legend(
            handles=legend_elements,
            loc="upper left",
            fancybox=True,
            shadow=True,
            fontsize=10,
            ncol=1,
        )

        plt.draw()
        plt.pause(0.01)

        if self.save_animation_path:
            self.fig.canvas.draw()
            frame = np.array(self.fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)

        if mode == "human":
            try:
                input("Press Enter to continue...")
            except (EOFError, KeyboardInterrupt):
                pass

    def close(self):
        if self.save_animation_path and self.frames:
            self._save_animation()

        if self.rendering:
            plt.close(self.fig)
            self.rendering = False

    def _save_animation(self):
        """Save animation as GIF file"""
        if not self.frames:
            print("No frames to save")
            return

        try:
            os.makedirs(os.path.dirname(self.save_animation_path), exist_ok=True)

            from PIL import Image

            pil_images = []
            for frame in self.frames:
                img = Image.fromarray(frame).convert("RGB")
                pil_images.append(img)

            if pil_images:
                pil_images[0].save(
                    self.save_animation_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=200,
                    loop=0,
                )
                print(f"Animation saved to: {self.save_animation_path}")

        except Exception as e:
            print(f"Error saving animation: {e}")

    def save_animation_at_end(self):
        """Call this method at the end of the environment to save animation"""
        if self.save_animation_path and self.frames:
            self._save_animation()

    @staticmethod
    def counterfactual_inference(
        config, rsrp_matrix_t, counterfactual_action, rsrp_matrix_t1
    ):
        """
        Counterfactual inference method (static method)
        Receives RSRP matrix at time t, counterfactual action a_t, RSRP matrix at time t+1
        Calculates device connection states to base stations (CIO all 0), then calculates reward based on t+1 RSRP matrix and counterfactual action at
        Returns complete s_t, a_t, r_t, s_t+1

        Args:
            config: Environment configuration
            rsrp_matrix_t: RSRP matrix at time t [num_base_stations, num_devices]
            counterfactual_action: Counterfactual action [action_dim]
            rsrp_matrix_t1: RSRP matrix at time t+1 [num_base_stations, num_devices]

        Returns:
            state_t: State at time t [state_dim]
            action_t: Counterfactual action [action_dim]
            reward_t: Counterfactual reward [1]
            state_t1: State at time t+1 [state_dim]
        """
        # Create a temporary environment instance for calculation, do not modify the original environment state
        temp_env = CIOLoadBalancingEnv(config)

        # Step 1: Set RSRP at time t and calculate connection states (CIO all 0)
        temp_env._set_rsrp_from_matrix(rsrp_matrix_t)

        # Reset CIO matrix to all zeros
        for key in temp_env.cio_matrix:
            temp_env.cio_matrix[key] = 0

        # Update connection states (based on RSRP at time t and CIO=0)
        temp_env._update_connections()

        # Calculate load and performance metrics at time t
        temp_env._calculate_load()

        # Get state at time t
        state_t = temp_env._get_observation()

        # Step 2: Apply counterfactual action
        action_index = 0
        for i, bs_i in enumerate(temp_env.base_stations):
            for j, bs_j in enumerate(temp_env.base_stations):
                if i != j:
                    cio = counterfactual_action[action_index] * 6
                    cio = max(-6, min(6, cio))
                    temp_env.cio_matrix[(bs_i["id"], bs_j["id"])] = cio
                    action_index += 1

        # Step 3: Set RSRP at time t+1
        temp_env._set_rsrp_from_matrix(rsrp_matrix_t1)

        # Step 4: Update connection states at time t+1
        temp_env._update_connections()

        # Calculate load and performance metrics at time t+1
        temp_env._calculate_load()
        throughput = temp_env._calculate_throughput()
        fairness = temp_env._calculate_fairness()
        blocking = temp_env._calculate_blocking()

        # Calculate counterfactual reward
        reward_t = temp_env._calculate_reward(throughput, fairness, blocking)

        # Get state at time t+1
        state_t1 = temp_env._get_observation()

        # Clean up temporary environment
        del temp_env

        return state_t, counterfactual_action, reward_t, state_t1

    def _set_rsrp_from_matrix(self, rsrp_matrix):
        """
        Set device RSRP measurements from RSRP matrix

        Args:
            rsrp_matrix: RSRP matrix [num_base_stations, num_devices]
        """
        for i, bs in enumerate(self.base_stations):
            for j, device in enumerate(self.mobile_devices):
                device["rsrp_measurements"][bs["id"]] = rsrp_matrix[i, j]

    def __call__(self):
        return self
