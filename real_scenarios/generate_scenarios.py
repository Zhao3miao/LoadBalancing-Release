import os
import yaml
import osmnx as ox
from typing import List, Dict, Tuple
import random
import math
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


class OSMScenarioGenerator:
    """
    Real scenario generator based on OSM map data
    Ensures initial positions are legal and movement follows legal paths
    """

    def __init__(self, osm_file_path: str):
        """
        Initialize the scenario generator

        Args:
            osm_file_path: OSM map file path
        """
        self.osm_file_path = osm_file_path
        self.osm_graph = None
        self.road_nodes = None
        self.road_edges = None
        self.node_positions_cache = {}  # Node position cache
        self.adjacent_nodes_cache = {}  # Adjacent nodes cache

    def load_osm_data(self):
        """Load and parse OSM map data"""
        print("Loading OSM map data...")
        start_time = time.time()

        try:
            # Use osmnx to load OSM file
            self.osm_graph = ox.graph_from_xml(self.osm_file_path)

            # Extract road nodes and edges
            self.road_nodes, self.road_edges = ox.graph_to_gdfs(self.osm_graph)

            # Precompute node position cache
            self._precompute_node_positions()

            # Precompute adjacent node cache
            self._precompute_adjacent_nodes()

            load_time = time.time() - start_time
            print(f"OSM data loading completed: {len(self.road_nodes)} road nodes, {len(self.road_edges)} road edges")
            print(f"Data loading time: {load_time:.2f} seconds")

        except Exception as e:
            print(f"Error loading OSM data: {e}")
            raise

    def _precompute_node_positions(self):
        """Precompute all node positions in meters"""
        print("Precomputing node positions...")
        scene_bounds = self._get_scene_bounds_degrees()
        min_lon, min_lat, max_lon, max_lat = scene_bounds
        center_lat = (min_lat + max_lat) / 2

        for node_id, node_data in self.road_nodes.iterrows():
            lon = float(node_data.geometry.x)
            lat = float(node_data.geometry.y)

            x_meters = self._lon_to_meters(lon, min_lon, center_lat)
            y_meters = self._lat_to_meters(lat, min_lat)

            self.node_positions_cache[node_id] = (x_meters, y_meters)

    def _precompute_adjacent_nodes(self):
        """Precompute adjacent nodes for each node"""
        print("Precomputing adjacent nodes...")
        for node in self.osm_graph.nodes():
            try:
                neighbors = list(self.osm_graph.neighbors(node))
                self.adjacent_nodes_cache[node] = neighbors
            except Exception:
                self.adjacent_nodes_cache[node] = []

    def generate_scenario(
        self,
        base_stations: List[Dict],
        user_distribution: List[int],
        total_steps: int = 100,
        scenario_name: str = "real_scenario",
        user_speed: float = 1.5,
        speed_range: Tuple[float, float] = None,
        target_direction: float = None,
        direction_strength: float = 0.5,
        base_station_directions: Dict[int, float] = None,
        base_station_direction_strengths: Dict[int, float] = None,
        server_configs_raw: List[Dict] = None,
    ) -> Dict:
        """
        Generate complete scenario configuration (using metric coordinate system)
        Ensures initial positions are legal and movement follows legal paths

        Args:
            base_stations: Base station configuration list [{"id": int, "position": [lon, lat]}, ...]
            user_distribution: User distribution around each base station [num_users_bs1, num_users_bs2, ...]
            total_steps: Total simulation steps
            scenario_name: Scenario name
            user_speed: User movement speed (meters/step), default 1.5 m/step (used when speed_range is None)
            speed_range: User movement speed range [min, max] (meters/step), if provided each user's speed is randomly selected within this range
            target_direction: Target direction (radians) (global default, used if base station does not specify separately)
            direction_strength: Direction strength (0-1) (global default)
            base_station_directions: Base station specific target direction dict {base_station_id: direction_radians}
            base_station_direction_strengths: Base station specific direction strength dict {base_station_id: strength}
            server_configs_raw: Raw server configuration list (containing movement_pattern etc. detailed information)

        Returns:
            Scenario configuration dict (all coordinates converted to metric)
        """
        if self.osm_graph is None:
            self.load_osm_data()

        # Validate input parameters
        self._validate_inputs(base_stations, user_distribution)

        # Process speed parameters
        if speed_range is not None:
            min_speed, max_speed = speed_range
            if min_speed <= 0 or max_speed <= 0 or min_speed > max_speed:
                raise ValueError("Speed range must be positive and min_speed <= max_speed")
            # print(f"User movement speed range set to: [{min_speed:.1f}, {max_speed:.1f}] meters/step")
            # Generate random speed for each user
            total_users = sum(user_distribution)
            user_speeds = [
                random.uniform(min_speed, max_speed) for _ in range(total_users)
            ]
        else:
            if user_speed <= 0:
                raise ValueError("User movement speed must be greater than 0")
            # print(f"User movement speed set to: {user_speed} meters/step")
            user_speeds = None

        # Display direction information
        if base_station_directions is not None:
            for bs_id, direction in base_station_directions.items():
                strength = (
                    base_station_direction_strengths.get(bs_id, direction_strength)
                    if base_station_direction_strengths
                    else direction_strength
                )
                # print(f"Base station {bs_id} target direction: {direction:.2f} radians, direction strength: {strength}")
        elif target_direction is not None:
            pass
        # print(f"Global target direction: {target_direction:.2f} radians, direction strength: {direction_strength}")

        # Convert base station coordinates to meters
        base_stations_meters = self._convert_base_stations_to_meters(base_stations)

        # Generate legal initial positions (on road nodes), recording the base station each device belongs to
        initial_nodes_with_bs = self._generate_legal_initial_positions_with_bs(
            base_stations_meters, user_distribution
        )

        # Generate legal movement trajectories (along road network)
        print("Generating movement trajectories (Parallel)...")
        start_time = time.time()

        # Use parallel processing to accelerate trajectory generation, passing base station specific direction parameters and user speeds, and raw configurations
        mobile_devices = self._generate_trajectories_parallel_with_bs_directions(
            initial_nodes_with_bs,
            total_steps,
            user_speed,
            user_speeds,
            target_direction,
            direction_strength,
            base_station_directions,
            base_station_direction_strengths,
            server_configs_raw,  # Pass
        )

        trajectory_time = time.time() - start_time
        print(f"Trajectory generation completed: Total {len(mobile_devices)} devices, time taken {trajectory_time:.2f} seconds")

        # Build scenario configuration
        scenario_config = self._build_scenario_config(
            base_stations_meters, mobile_devices, total_steps, scenario_name
        )

        return scenario_config

    def _validate_inputs(self, base_stations: List[Dict], user_distribution: List[int]):
        """Validate input parameters"""
        if len(base_stations) != len(user_distribution):
            raise ValueError("Number of base stations must match the length of user distribution list")

        if sum(user_distribution) == 0:
            raise ValueError("Sum of user distribution list cannot be 0")

    def _convert_base_stations_to_meters(self, base_stations: List[Dict]) -> List[Dict]:
        """
        Convert base station coordinates from lat/lon to metric coordinates

        Args:
            base_stations: Base station configuration list (lat/lon coordinates)

        Returns:
            Converted base station configuration list (metric coordinates)
        """
        # Get scene bounds (lat/lon)
        scene_bounds = self._get_scene_bounds_degrees()
        min_lon, min_lat, max_lon, max_lat = scene_bounds

        # Calculate center latitude of the scene (for longitude to meters conversion)
        center_lat = (min_lat + max_lat) / 2

        base_stations_meters = []

        for bs in base_stations:
            lon, lat = bs["position"]

            # Convert lat/lon to metric coordinates
            x_meters = self._lon_to_meters(lon, min_lon, center_lat)
            y_meters = self._lat_to_meters(lat, min_lat)

            base_stations_meters.append(
                {"id": bs["id"], "position": [x_meters, y_meters]}
            )

            print(
                f"Base station {bs['id']}: lat/lon ({lon:.6f}, {lat:.6f}) -> metric coordinates ({x_meters:.2f}, {y_meters:.2f})"
            )

        return base_stations_meters

    def _lon_to_meters(
        self, lon: float, reference_lon: float, center_lat: float
    ) -> float:
        """
        Convert longitude to metric coordinates

        Args:
            lon: Longitude
            reference_lon: Reference longitude (scene minimum longitude)
            center_lat: Center latitude

        Returns:
            Metric X coordinate
        """
        # Earth radius (meters)
        earth_radius = 6371000.0

        # Longitude difference converted to meters (considering latitude influence)
        lon_diff_rad = math.radians(lon - reference_lon)
        x_meters = lon_diff_rad * earth_radius * math.cos(math.radians(center_lat))

        return x_meters

    def _lat_to_meters(self, lat: float, reference_lat: float) -> float:
        """
        Convert latitude to metric coordinates

        Args:
            lat: Latitude
            reference_lat: Reference latitude (scene minimum latitude)

        Returns:
            Metric Y coordinate
        """
        # Earth radius (meters)
        earth_radius = 6371000.0

        # Latitude difference converted to meters
        lat_diff_rad = math.radians(lat - reference_lat)
        y_meters = lat_diff_rad * earth_radius

        return y_meters

    def _get_scene_bounds_degrees(self) -> Tuple[float, float, float, float]:
        """Get the lat/lon bounds of the road network"""
        min_lon, min_lat = (
            self.road_nodes.geometry.x.min(),
            self.road_nodes.geometry.y.min(),
        )
        max_lon, max_lat = (
            self.road_nodes.geometry.x.max(),
            self.road_nodes.geometry.y.max(),
        )
        return (min_lon, min_lat, max_lon, max_lat)

    def _get_scene_bounds_meters(self) -> Tuple[float, float, float, float]:
        """Get the metric bounds of the road network"""
        min_lon, min_lat, max_lon, max_lat = self._get_scene_bounds_degrees()
        center_lat = (min_lat + max_lat) / 2

        min_x = self._lon_to_meters(min_lon, min_lon, center_lat)
        min_y = self._lat_to_meters(min_lat, min_lat)
        max_x = self._lon_to_meters(max_lon, min_lon, center_lat)
        max_y = self._lat_to_meters(max_lat, min_lat)

        return (min_x, min_y, max_x, max_y)

    def _generate_legal_initial_positions(
        self, base_stations: List[Dict], user_distribution: List[int]
    ) -> List:
        """
        Generate legal initial positions around base stations (on road nodes)

        Args:
            base_stations: Base station configurations (metric coordinates)
            user_distribution: User distribution

        Returns:
            List of initial node IDs
        """
        print("Generating legal initial positions...")
        start_time = time.time()

        all_initial_nodes = []
        device_id = 0

        for bs_index, (bs, num_users) in enumerate(
            zip(base_stations, user_distribution)
        ):
            bs_position = bs["position"]
            bs_id = bs["id"]

            print(f"Generating legal initial positions for {num_users} users at base station {bs_id}...")

            # Find all road nodes within 500 meters around the base station
            nearby_nodes = self._find_nearby_road_nodes(bs_position, max_distance=100)

            if not nearby_nodes:
                print(f"Warning: No road nodes found within 500 meters of base station {bs_id}, using all nodes in the scene")
                nearby_nodes = list(self.node_positions_cache.keys())

            # Ensure there are enough nodes
            if len(nearby_nodes) < num_users:
                print(
                    f"Warning: Only {len(nearby_nodes)} nodes found around base station {bs_id}, but {num_users} users are needed"
                )
                # Reuse nodes or supplement from the entire scene
                if len(nearby_nodes) == 0:
                    nearby_nodes = list(self.node_positions_cache.keys())
                # Repeat nodes until the quantity requirement is met
                while len(nearby_nodes) < num_users:
                    nearby_nodes.extend(nearby_nodes)
                nearby_nodes = nearby_nodes[:num_users]

            # Randomly select nodes as initial positions
            selected_nodes = random.sample(nearby_nodes, num_users)
            all_initial_nodes.extend(selected_nodes)

            print(f"Base station {bs_id}: Selected {num_users} nodes from {len(nearby_nodes)} candidate nodes")

        initial_time = time.time() - start_time
        print(f"Initial position generation completed: Total {len(all_initial_nodes)} legal positions, time taken {initial_time:.2f} seconds")
        return all_initial_nodes

    def _generate_legal_initial_positions_with_bs(
        self, base_stations: List[Dict], user_distribution: List[int]
    ) -> List[Tuple]:
        """
        Generate legal initial positions (on road nodes) around base stations, recording the base station each device belongs to

        Args:
            base_stations: Base station configurations (metric coordinates)
            user_distribution: User distribution

        Returns:
            List of tuples with initial node IDs and base station IDs [(node_id, base_station_id), ...]
        """
        print("Generating legal initial positions...")
        start_time = time.time()

        all_initial_nodes_with_bs = []
        device_id = 0

        for bs_index, (bs, num_users) in enumerate(
            zip(base_stations, user_distribution)
        ):
            bs_position = bs["position"]
            bs_id = bs["id"]

            print(f"Generating {num_users} users' legal initial positions for base station {bs_id}...")

            # Find all road nodes within 100 meters of the base station
            nearby_nodes = self._find_nearby_road_nodes(bs_position, max_distance=200)

            if not nearby_nodes:
                print(f"Warning: No road nodes found within 100 meters of base station {bs_id}, using entire scene nodes")
                nearby_nodes = list(self.node_positions_cache.keys())

            # Ensure there are enough nodes
            if len(nearby_nodes) < num_users:
                print(
                    f"Warning: Base station {bs_id} has only {len(nearby_nodes)} nodes, but needs {num_users} users"
                )
                # Repeat nodes or supplement from entire scene
                if len(nearby_nodes) == 0:
                    nearby_nodes = list(self.node_positions_cache.keys())
                # Repeat nodes until sufficient quantity
                while len(nearby_nodes) < num_users:
                    nearby_nodes.extend(nearby_nodes)
                nearby_nodes = nearby_nodes[:num_users]

            # Randomly select nodes as initial positions
            selected_nodes = random.sample(nearby_nodes, num_users)
            # Add base station ID info for each node
            for node_id in selected_nodes:
                all_initial_nodes_with_bs.append((node_id, bs_id))

            print(f"Base station {bs_id}: Selected {num_users} nodes from {len(nearby_nodes)} candidate nodes")

        initial_time = time.time() - start_time
        print(
            f"Initial position generation completed: Total {len(all_initial_nodes_with_bs)} legal positions, time taken {initial_time:.2f} seconds"
        )
        return all_initial_nodes_with_bs

    def _find_nearby_road_nodes(
        self, position: Tuple[float, float], max_distance: float = 500
    ):
        """Find all road nodes within a specified distance from the given position"""
        pos_x, pos_y = position
        nearby_nodes = []

        for node_id, node_pos in self.node_positions_cache.items():
            node_x, node_y = node_pos
            distance = math.sqrt((node_x - pos_x) ** 2 + (node_y - pos_y) ** 2)

            if distance <= max_distance:
                nearby_nodes.append(node_id)

        return nearby_nodes

    def _generate_trajectories_parallel(
        self,
        initial_nodes: List,
        total_steps: int,
        user_speed: float = 1.5,
        target_direction: float = None,
        direction_strength: float = 0.5,
    ) -> List[Dict]:
        """
        Use parallel processing to generate movement trajectories for each initial node

        Args:
            initial_nodes: List of initial nodes
            total_steps: Total number of steps
            user_speed: User movement speed (meters/step)
            target_direction: Target direction (radians)
            direction_strength: Direction strength (0-1)

        Returns:
            List of mobile device configurations
        """
        mobile_devices = []

        # Use thread pool for parallel trajectory generation
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_device = {
                executor.submit(
                    self._generate_single_trajectory,
                    node,
                    total_steps,
                    device_id,
                    user_speed,
                    target_direction,
                    direction_strength,
                ): device_id
                for device_id, node in enumerate(initial_nodes)
            }

            completed = 0
            total = len(initial_nodes)

            for future in as_completed(future_to_device):
                device_id = future_to_device[future]
                try:
                    trajectory = future.result()
                    device_config = {"id": device_id, "trajectory": trajectory}
                    mobile_devices.append(device_config)
                    completed += 1

                    # Display progress
                    if completed % 10 == 0 or completed == total:
                        print(
                            f"Trajectory generation progress: {completed}/{total} ({completed/total*100:.1f}%)"
                        )

                except Exception as e:
                    print(f"Error generating trajectory for device {device_id}: {e}")
                    # Create default trajectory as fallback
                    default_trajectory = self._generate_default_trajectory(
                        device_id, total_steps
                    )
                    device_config = {"id": device_id, "trajectory": default_trajectory}
                    mobile_devices.append(device_config)

        return mobile_devices

    def _generate_trajectories_parallel_with_bs_directions(
        self,
        initial_nodes_with_bs: List[Tuple],
        total_steps: int,
        user_speed: float = 1.5,
        user_speeds: List[float] = None,
        target_direction: float = None,
        direction_strength: float = 0.5,
        base_station_directions: Dict[int, float] = None,
        base_station_direction_strengths: Dict[int, float] = None,
        server_configs_raw: List[Dict] = None,
    ) -> List[Dict]:
        """
        Use parallel processing to generate movement trajectories for each initial node, supporting independent direction control per base station and individual speed per user
        Supports Loiter mode
        """
        mobile_devices = []

        # Use thread pool for parallel trajectory generation
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_device = {}
            for device_id, (node, bs_id) in enumerate(initial_nodes_with_bs):
                # Determine movement pattern parameters
                device_target_direction = target_direction
                device_direction_strength = direction_strength
                device_movement_pattern = "directional"  # Default movement pattern
                device_target_point = None
                device_loiter_radius = 150.0

                # Try to get more detailed base station level configurations from raw configs
                if server_configs_raw:
                    # server_configs_raw is list of dict, each dict has id
                    for sc in server_configs_raw:
                        if sc["id"] == bs_id:
                            device_movement_pattern = sc.get(
                                "movement_pattern", "directional"
                            )
                            if device_movement_pattern == "loiter":
                                # Note: target_point here is lat/lon, needs to be converted to meters for internal calculation
                                # If real scenario, need to convert lat/lon to meters ourselves
                                # We cheat here, if target_point is lat/lon, handle in _generate_single_trajectory?
                                # Or preprocess?
                                # More robust approach is to convert lat/lon to meter coordinates here
                                raw_target = sc.get("target_point")
                                if raw_target:
                                    # Assume raw config has [lon, lat]
                                    # Use generator's conversion function
                                    scene_bounds = self._get_scene_bounds_degrees()
                                    min_lon, min_lat, max_lon, max_lat = scene_bounds
                                    center_lat = (min_lat + max_lat) / 2

                                    x_meters = self._lon_to_meters(
                                        raw_target[0], min_lon, center_lat
                                    )
                                    y_meters = self._lat_to_meters(
                                        raw_target[1], min_lat
                                    )
                                    device_target_point = [x_meters, y_meters]

                                device_loiter_radius = sc.get("loiter_radius", 150.0)
                            else:
                                # Directional
                                device_target_direction = sc.get(
                                    "target_direction", target_direction
                                )
                                device_direction_strength = sc.get(
                                    "direction_strength", direction_strength
                                )
                            break
                # Fallback to simple dicts if raw config not matching or provided
                elif (
                    base_station_directions is not None
                    and bs_id in base_station_directions
                ):
                    device_target_direction = base_station_directions[bs_id]
                    if (
                        base_station_direction_strengths is not None
                        and bs_id in base_station_direction_strengths
                    ):
                        device_direction_strength = base_station_direction_strengths[
                            bs_id
                        ]

                # Determine the speed for this device
                if user_speeds is not None and device_id < len(user_speeds):
                    device_speed = user_speeds[device_id]
                else:
                    device_speed = user_speed

                future = executor.submit(
                    self._generate_single_trajectory,
                    node,
                    total_steps,
                    device_id,
                    device_speed,
                    device_target_direction,
                    device_direction_strength,
                    device_movement_pattern,  # New arg
                    device_target_point,  # New arg
                    device_loiter_radius,  # New arg
                )
                future_to_device[future] = (device_id, bs_id, device_speed)

            completed = 0
            total = len(initial_nodes_with_bs)

            for future in as_completed(future_to_device):
                device_id, bs_id, device_speed = future_to_device[future]
                try:
                    trajectory = future.result()
                    device_config = {
                        "id": device_id,
                        "trajectory": trajectory,
                        "base_station_id": bs_id,
                        "speed": device_speed,  # Save speed for each device
                    }
                    mobile_devices.append(device_config)
                    completed += 1

                    # Show progress
                    if completed % 10 == 0 or completed == total:
                        print(
                            f"Trajectory generation progress: {completed}/{total} ({completed/total*100:.1f}%)"
                        )

                except Exception as e:
                    print(f"Error generating trajectory for device {device_id} (BS {bs_id}): {e}")
                    # Create default trajectory as fallback
                    default_trajectory = self._generate_default_trajectory(
                        device_id, total_steps
                    )
                    device_config = {
                        "id": device_id,
                        "trajectory": default_trajectory,
                        "base_station_id": bs_id,
                        "speed": device_speed
                        if "device_speed" in locals()
                        else user_speed,
                    }
                    mobile_devices.append(device_config)

        return mobile_devices

    def _generate_single_trajectory(
        self,
        start_node,
        total_steps: int,
        device_id: int = 0,
        user_speed: float = 1.5,
        target_direction: float = None,
        direction_strength: float = 0.5,
        movement_pattern: str = "directional",
        target_point: List[float] = None,
        loiter_radius: float = 150.0,
    ) -> List[Dict]:
        """
        Generate movement trajectory for a single device (legal movement based on road network)
        Enhanced version: Supports direction-aware intersection turning logic, considering long-term directional consistency and dead-end handling
        """
        trajectory = []
        current_node = start_node

        # Add initial position
        current_pos = self.node_positions_cache.get(current_node, [0, 0])
        trajectory.append({"step": 0, "position": list(current_pos)})

        # Movement state variables
        current_direction = None  # Current movement direction
        target_node = None  # Current target node
        remaining_steps_in_segment = 0  # Remaining steps in current road segment
        previous_node = None  # Previous node, used to calculate current movement direction
        stuck_counter = 0  # Stuck counter, used to detect infinite loops
        max_stuck_count = 5  # Maximum stuck count, stop moving if exceeded
        direction_history = []  # Direction history record, used to maintain long-term directional consistency

        # Get scene bounds for boundary detection
        scene_bounds = self._get_scene_bounds_meters()
        min_x, min_y, max_x, max_y = scene_bounds
        boundary_margin = 50  # Boundary detection margin (meters)

        # Generate trajectory based on road network
        for step in range(1, total_steps):
            # --- Direction decision logic ---
            use_direction = target_direction

            if movement_pattern == "loiter" and target_point is not None:
                # Calculate distance from current position to loiter center
                dx = target_point[0] - current_pos[0]
                dy = target_point[1] - current_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > loiter_radius:
                    # If beyond radius, target direction points to center
                    use_direction = math.atan2(dy, dx)
                    # Enhance regression strength
                    effective_strength = 0.7
                else:
                    # Within radius, random walk (no forced direction, or weak direction maintenance)
                    use_direction = None  # Random
                    effective_strength = 0.0  # Pure random
            else:
                effective_strength = direction_strength

            # Check if approaching boundary (Common Logic)
            if use_direction is not None:
                if (
                    current_pos[0] <= min_x + boundary_margin
                    or current_pos[0] >= max_x - boundary_margin
                    or current_pos[1] <= min_y + boundary_margin
                    or current_pos[1] >= max_y - boundary_margin
                ):
                    adjusted = self._adjust_direction_at_boundary(
                        current_pos, use_direction, scene_bounds
                    )
                    if adjusted is not None:
                        use_direction = adjusted

            # If no current movement direction or reached target node, choose new movement direction
            if current_direction is None or remaining_steps_in_segment <= 0:
                # Get adjacent nodes of current node
                adjacent_nodes = self.adjacent_nodes_cache.get(current_node, [])

                if not adjacent_nodes:
                    # If no adjacent nodes, maintain current position
                    trajectory.append({"step": step, "position": list(current_pos)})
                    stuck_counter += 1
                    if stuck_counter >= max_stuck_count:
                        # print(f"Device {device_id} stopped moving in dead end")
                        break
                    continue

                # Filter out previous node (avoid immediate return)
                available_nodes = [
                    node for node in adjacent_nodes if node != previous_node
                ]

                # If no available nodes (dead end), consider U-turn or stop
                if not available_nodes:
                    if len(adjacent_nodes) == 1 and adjacent_nodes[0] == previous_node:
                        # Dead end situation, only return path
                        # print(f"Device {device_id} encountered dead end, attempting U-turn")
                        available_nodes = [previous_node]  # Allow return
                        stuck_counter += 1
                    else:
                        available_nodes = adjacent_nodes

                # Intelligently select next node (considering target direction)
                target_node = self._select_next_node_with_direction(
                    current_node,
                    available_nodes,
                    previous_node,
                    use_direction,
                    effective_strength,  # Use calculated direction and strength
                )

                if target_node is None:
                    # If no suitable node, check if dead end
                    if len(available_nodes) == 0:
                        # Complete dead end, stop moving
                        trajectory.append({"step": step, "position": list(current_pos)})
                        stuck_counter += 1
                        if stuck_counter >= max_stuck_count:
                            break
                        continue
                    else:
                        # Randomly select a node
                        target_node = random.choice(available_nodes)

                target_pos = self.node_positions_cache.get(target_node, [0, 0])

                # Calculate distance between two points (Rest of the loop logic is unchanged)
                distance = math.sqrt(
                    (target_pos[0] - current_pos[0]) ** 2
                    + (target_pos[1] - current_pos[1]) ** 2
                )

                if distance <= 0:
                    current_pos = target_pos
                    previous_node = current_node
                    current_node = target_node
                    trajectory.append({"step": step, "position": list(current_pos)})
                    stuck_counter = 0
                    continue

                remaining_steps_in_segment = max(1, int(distance / user_speed))

                current_direction = [
                    (target_pos[0] - current_pos[0]) / distance,
                    (target_pos[1] - current_pos[1]) / distance,
                ]

                # ... (rest of logic same as before)
                if len(direction_history) >= 10:
                    direction_history.pop(0)
                direction_history.append(current_direction)
                stuck_counter = 0

            # Move one step in current direction
            step_size = user_speed

            new_pos = [
                current_pos[0] + current_direction[0] * step_size,
                current_pos[1] + current_direction[1] * step_size,
            ]

            if (
                new_pos[0] < min_x
                or new_pos[0] > max_x
                or new_pos[1] < min_y
                or new_pos[1] > max_y
            ):
                new_pos[0] = max(min_x, min(new_pos[0], max_x))
                new_pos[1] = max(min_y, min(new_pos[1], max_y))
                remaining_steps_in_segment = 0
                current_direction = None

            current_pos = new_pos

            target_pos = self.node_positions_cache.get(target_node, [0, 0])
            distance_to_target = math.sqrt(
                (target_pos[0] - current_pos[0]) ** 2
                + (target_pos[1] - current_pos[1]) ** 2
            )

            if distance_to_target <= step_size:
                previous_node = current_node
                current_node = target_node
                current_pos = target_pos
                remaining_steps_in_segment = 0

            trajectory.append({"step": step, "position": list(current_pos)})

        return trajectory

    def _adjust_direction_at_boundary(
        self,
        current_pos: List[float],
        target_direction: float,
        scene_bounds: Tuple[float, float, float, float],
    ) -> float:
        """
        Intelligently adjust movement direction at boundaries to avoid moving in opposite directions

        Args:
            current_pos: Current position [x, y]
            target_direction: Target direction (radians)
            scene_bounds: Scene bounds (min_x, min_y, max_x, max_y)

        Returns:
            Adjusted direction, or None if no adjustment needed
        """
        min_x, min_y, max_x, max_y = scene_bounds
        boundary_margin = 50  # Boundary detection margin (meters)

        # Check if near boundary
        near_left = current_pos[0] <= min_x + boundary_margin
        near_right = current_pos[0] >= max_x - boundary_margin
        near_bottom = current_pos[1] <= min_y + boundary_margin
        near_top = current_pos[1] >= max_y - boundary_margin

        if not (near_left or near_right or near_bottom or near_top):
            return None  # Not near boundary, no adjustment needed

        # Intelligently adjust based on boundary position and target direction
        if near_left and target_direction is not None:
            # At left boundary, avoid moving left
            if target_direction < -math.pi / 2 or target_direction > math.pi / 2:
                # If target direction is left, adjust to right
                return 0.0  # Right
        elif near_right and target_direction is not None:
            # At right boundary, avoid moving right
            if -math.pi / 2 < target_direction < math.pi / 2:
                # If target direction is right, adjust to left
                return math.pi  # Left
        elif near_bottom and target_direction is not None:
            # At bottom boundary, avoid moving down
            if target_direction < 0:
                # If target direction is down, adjust to up
                return math.pi / 2  # Up
        elif near_top and target_direction is not None:
            # At top boundary, avoid moving up
            if target_direction > 0:
                # If target direction is up, adjust to down
                return -math.pi / 2  # Down

        return target_direction  # No adjustment needed

    def _select_next_node_with_direction(
        self,
        current_node,
        adjacent_nodes,
        previous_node,
        target_direction: float,
        direction_strength: float,
    ):
        """
        Intelligently select next node based on target direction, implementing strict directional control
        When target direction is right, prohibit left movement, but allow up/down movement

        Args:
            current_node: Current node
            adjacent_nodes: List of adjacent nodes
            previous_node: Previous node (used to determine current movement direction)
            target_direction: Target direction (radians)
            direction_strength: Direction strength (0-1)

        Returns:
            Selected node, or None if no suitable node
        """
        if target_direction is None or len(adjacent_nodes) <= 1:
            # If no direction specified or only one adjacent node, choose randomly
            return random.choice(adjacent_nodes)

        # Calculate current movement direction (direction from previous node to current node)
        if previous_node is not None:
            prev_pos = self.node_positions_cache.get(previous_node, [0, 0])
            curr_pos = self.node_positions_cache.get(current_node, [0, 0])
            current_direction = math.atan2(
                curr_pos[1] - prev_pos[1], curr_pos[0] - prev_pos[0]
            )
        else:
            # If no previous node, assume initial direction is 0 (right)
            current_direction = 0

        # Calculate relative turning angle for each adjacent node
        node_scores = []
        curr_pos = self.node_positions_cache.get(current_node, [0, 0])

        for node in adjacent_nodes:
            node_pos = self.node_positions_cache.get(node, [0, 0])

            # Calculate direction to this node
            node_direction = math.atan2(
                node_pos[1] - curr_pos[1], node_pos[0] - curr_pos[0]
            )

            # Calculate turning angle relative to current direction
            turn_angle = self._normalize_angle(node_direction - current_direction)

            # Calculate turning angle relative to target direction
            target_turn_angle = self._normalize_angle(node_direction - target_direction)

            # Initialize base score
            base_score = 0.0

            # Strict directional control: prohibit opposite direction sectors based on target direction
            if target_direction is not None:
                # Define direction sectors
                # Right sector: -45° to 45° (i.e., -π/4 to π/4)
                # Up sector: 45° to 135° (i.e., π/4 to 3π/4)
                # Left sector: 135° to 225° (i.e., 3π/4 to 5π/4)
                # Down sector: -135° to -45° (i.e., -3π/4 to -π/4)

                # Normalize direction to [0, 2π) range
                normalized_direction = node_direction % (2 * math.pi)

                # Determine prohibited sectors based on target direction
                # If target direction is right (0), prohibit left sector (135° to 225°)
                # If target direction is left (π), prohibit right sector (-45° to 45°)
                # If target direction is up (π/2), prohibit down sector (-135° to -45°)
                # If target direction is down (-π/2), prohibit up sector (45° to 135°)
                # By default, prohibit left sector (consistent with original logic)

                # Initialize prohibition flag
                is_forbidden = False

                # Determine target direction and set prohibited sectors
                if abs(target_direction - 0) < 0.1:  # Right
                    # Prohibit left sector (135° to 225°)
                    if (
                        normalized_direction > math.pi * 0.75
                        and normalized_direction < math.pi * 1.25
                    ):
                        is_forbidden = True
                elif abs(target_direction - math.pi) < 0.1:  # Left
                    # Prohibit right sector (-45° to 45°), corresponding to normalized [315°, 360°] and [0°, 45°]
                    if (
                        normalized_direction >= math.pi * 1.75
                        or normalized_direction <= math.pi * 0.25
                    ):
                        is_forbidden = True
                elif abs(target_direction - math.pi / 2) < 0.1:  # Upward
                    # Prohibit lower sector (-135° to -45°), corresponding to normalized [225°, 315°]
                    if (
                        normalized_direction > math.pi * 1.25
                        and normalized_direction < math.pi * 1.75
                    ):
                        is_forbidden = True
                elif abs(target_direction + math.pi / 2) < 0.1:  # Downward
                    # Prohibit upper sector (45° to 135°)
                    if (
                        normalized_direction > math.pi * 0.25
                        and normalized_direction <= math.pi * 0.75
                    ):
                        is_forbidden = True
                else:
                    # For other target directions, prohibit left sector (keep original logic)
                    if (
                        normalized_direction > math.pi * 0.75
                        and normalized_direction < math.pi * 1.25
                    ):
                        is_forbidden = True

                if is_forbidden:
                    # Completely prohibit this direction
                    base_score = 0.0
                else:
                    # Calculate smooth turning score
                    turn_score = 1.0 - abs(turn_angle) / math.pi

                    # Calculate target direction matching degree
                    target_score = 1.0 - abs(target_turn_angle) / math.pi

                    # Adjust weights based on direction sector
                    # Right sector: highest priority
                    if (
                        normalized_direction >= math.pi * 1.75
                        or normalized_direction <= math.pi * 0.25
                    ):
                        # Right sector: weight biased towards target direction
                        base_score = 0.3 * turn_score + 0.7 * target_score
                    # Up-down sectors: medium priority
                    elif (
                        normalized_direction > math.pi * 0.25
                        and normalized_direction <= math.pi * 0.75
                    ) or (
                        normalized_direction >= math.pi * 1.25
                        and normalized_direction < math.pi * 1.75
                    ):
                        # Up-down sectors: weight biased towards smooth turning
                        base_score = 0.7 * turn_score + 0.3 * target_score
                    else:
                        # Use default weights for other cases
                        base_score = 0.6 * turn_score + 0.4 * target_score
            else:
                # If no target direction, use default scoring
                turn_score = 1.0 - abs(turn_angle) / math.pi
                target_score = 1.0 - abs(target_turn_angle) / math.pi
                base_score = 0.6 * turn_score + 0.4 * target_score

            # Ensure score is not negative
            total_score = max(0.0, base_score)

            node_scores.append((node, total_score))

        # Filter out nodes with score 0 (forbidden directions)
        valid_nodes = [(node, score) for node, score in node_scores if score > 0]

        if not valid_nodes:
            # If no valid nodes, return None
            return None

        # Select nodes based on direction strength
        if direction_strength >= 1.0:
            # Select completely by direction: choose the node with highest score
            best_node = max(valid_nodes, key=lambda x: x[1])[0]
            return best_node
        elif direction_strength <= 0.0:
            # Select completely randomly (from valid nodes)
            return random.choice([node for node, _ in valid_nodes])
        else:
            # Select by probability: nodes with higher scores have higher probability of being selected
            scores = [score for _, score in valid_nodes]
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                # All nodes have same score, select randomly
                return random.choice([node for node, _ in valid_nodes])

            # Normalize scores and apply direction strength
            normalized_scores = [
                (
                    node,
                    min_score
                    + (score - min_score)
                    * direction_strength
                    / (max_score - min_score),
                )
                for node, score in valid_nodes
            ]

            # Select by probability
            total_weight = sum(score for _, score in normalized_scores)
            if total_weight <= 0:
                return random.choice([node for node, _ in valid_nodes])

            rand_val = random.random() * total_weight
            cumulative = 0
            for node, score in normalized_scores:
                cumulative += score
                if rand_val <= cumulative:
                    return node

            # If not selected due to floating point error, return the node with highest score
            return max(normalized_scores, key=lambda x: x[1])[0]

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-π, π] range

        Args:
            angle: Input angle

        Returns:
            Normalized angle
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _generate_default_trajectory(
        self, device_id: int, total_steps: int
    ) -> List[Dict]:
        """
        Generate default trajectory (used when unable to generate legal trajectory)

        Args:
            device_id: Device ID
            total_steps: Total steps

        Returns:
            List of trajectory points
        """
        # Use scene center as default position
        scene_bounds = self._get_scene_bounds_meters()
        min_x, min_y, max_x, max_y = scene_bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        trajectory = []
        for step in range(total_steps):
            trajectory.append({"step": step, "position": [center_x, center_y]})

        return trajectory

    def _build_scenario_config(
        self,
        base_stations: List[Dict],
        mobile_devices: List[Dict],
        total_steps: int,
        scenario_name: str,
    ) -> Dict:
        """
        Build complete scenario configuration (metric coordinate system)

        Args:
            base_stations: Base station configurations (metric coordinates)
            mobile_devices: Mobile device configurations (metric coordinates)
            total_steps: Total steps
            scenario_name: Scenario name

        Returns:
            Scenario configuration dict
        """
        # Get scene metric bounds
        min_x, min_y, max_x, max_y = self._get_scene_bounds_meters()
        scene_width = max_x - min_x
        scene_height = max_y - min_y

        print(f"Scene range: width={scene_width:.2f}m, height={scene_height:.2f}m")

        scenario_config = {
            "scene": {
                "width": float(scene_width),
                "height": float(scene_height),
                "total_steps": total_steps,
            },
            "base_stations": base_stations,
            "mobile_devices": mobile_devices,
            "reward_config": {"type": "load_balance"},
            "hysteresis_config": {"enabled": True, "value": 3.0},
        }

        return scenario_config

    def save_scenario_to_file(self, scenario_config: Dict, output_path: str):
        """
        Save scenario configuration to YAML file

        Args:
            scenario_config: Scenario configuration dict
            output_path: Output file path
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(scenario_config, f, default_flow_style=False, allow_unicode=True)

        print(f"Scenario configuration saved to: {output_path}")


def generate_controlled_osm_scenario(
    osm_file_path: str,
    base_stations: List[Dict],
    server_ids: List[int],
    num_users: int,
    total_steps: int = 100,
    output_file: str = None,
    user_speed: float = 30.0,
    speed_range: Tuple[float, float] = None,
    movement_pattern: str = "random",
    target_direction: float = None,
    direction_strength: float = 0.5,
    target_point: List[float] = None,
    attraction_strength: float = 0.3,
    scenario_name: str = "controlled_osm_scenario",
    base_station_directions: Dict[int, float] = None,
    base_station_direction_strengths: Dict[int, float] = None,
    server_configs_raw: List[Dict] = None,
):
    """
    Generate controlled scenario configuration based on OSM map, supporting crowd initial position and movement pattern control, supporting independent direction control per base station

    Args:
        osm_file_path: OSM map file path
        base_stations: Base station configuration list [{"id": int, "position": [lon, lat]}, ...]
        server_ids: Specified server ID list, users will be generated around these base stations
        num_users: Total number of users
        total_steps: Total simulation steps
        output_file: Output file path
        user_speed: User movement speed (meters/step) (used when speed_range is None)
        speed_range: User movement speed range [min, max] (meters/step), if provided each user's speed is randomly selected within this range
        movement_pattern: Movement pattern ("random", "directional", "attraction", "repulsion")
        target_direction: Target direction (radians, for directional mode) (global default)
        direction_strength: Direction strength (0-1) (global default)
        target_point: Target point [x, y] (for attraction/repulsion mode)
        attraction_strength: Attraction/repulsion strength (0-1)
        scenario_name: Scenario name
        base_station_directions: Base station specific target direction dict {base_station_id: direction_radians}
        server_configs_raw: Raw server configuration list, containing movement_pattern etc. detailed information
        base_station_direction_strengths: Base station specific direction strength dict {base_station_id: strength}
    """
    # Create generator instance
    generator = OSMScenarioGenerator(osm_file_path)

    # Calculate user distribution (randomly distributed around specified servers)
    user_distribution = [0] * len(base_stations)

    # Ensure each specified base station has at least one user
    for server_id in server_ids:
        if server_id <= len(base_stations):
            user_distribution[server_id - 1] = 1
            num_users -= 1

    # Randomly assign remaining users
    for _ in range(num_users):
        server_id = random.choice(server_ids)
        if server_id <= len(base_stations):
            user_distribution[server_id - 1] += 1

    print(f"User distribution: {user_distribution} (Base station IDs: {[bs['id'] for bs in base_stations]})")

    # Generate scenario (if direction specified, use direction-aware trajectory generation)
    scenario_config = generator.generate_scenario(
        base_stations=base_stations,
        user_distribution=user_distribution,
        total_steps=total_steps,
        scenario_name=scenario_name,
        user_speed=user_speed,
        speed_range=speed_range,
        target_direction=target_direction,
        direction_strength=direction_strength,
        server_configs_raw=server_configs_raw,
        base_station_directions=base_station_directions,
        base_station_direction_strengths=base_station_direction_strengths,
    )

    # Add initial connection configuration
    initial_connections = {}
    device_id = 0
    for bs_index, num_bs_users in enumerate(user_distribution):
        bs_id = base_stations[bs_index]["id"]
        for _ in range(num_bs_users):
            initial_connections[device_id] = bs_id
            device_id += 1

    scenario_config["initial_connections"] = initial_connections

    # Add movement pattern information to scenario configuration
    scenario_config["movement_config"] = {
        "pattern": movement_pattern,
        "speed": user_speed,
        "target_direction": target_direction,
        "direction_strength": direction_strength,
        "target_point": target_point,
        "attraction_strength": attraction_strength,
        "base_station_directions": base_station_directions,
        "base_station_direction_strengths": base_station_direction_strengths,
    }

    # Save scenario configuration
    if output_file is None:
        server_str = "_".join(map(str, server_ids))
        pattern_str = movement_pattern
        # If base station independent directions specified, add direction info to filename
        dir_str = ""
        if base_station_directions is not None:
            for server_id in server_ids:
                if server_id in base_station_directions:
                    dir_val = base_station_directions[server_id]
                    dir_str += "R" if abs(dir_val - 0) < 0.1 else "L"
                else:
                    dir_str += "U"
        if dir_str:
            pattern_str = f"{pattern_str}_{dir_str}"
        output_file = f"generated_scenarios/osm_controlled_scenario_{num_users}users_{user_speed}mps_{server_str}servers_{pattern_str}pattern_{total_steps}steps.yaml"

    generator.save_scenario_to_file(scenario_config, output_file)

    print(f"Controlled OSM scenario generation completed:")
    print(f"- Number of users: {num_users}")
    print(f"- Movement speed: {user_speed} meters/step")
    print(f"- Specified servers: {server_ids}")
    print(f"- Movement pattern: {movement_pattern}")
    if base_station_directions is not None:
        for bs_id in server_ids:
            if bs_id in base_station_directions:
                strength = (
                    base_station_direction_strengths.get(bs_id, direction_strength)
                    if base_station_direction_strengths
                    else direction_strength
                )
                dir_val = base_station_directions[bs_id]
                dir_str = "right" if abs(dir_val - 0) < 0.1 else "left"
                print(
                    f"- Base station {bs_id} target direction: {dir_str} ({dir_val:.2f} radians), strength: {strength}"
                )
    elif target_direction is not None:
        print(f"- Target direction: {target_direction:.2f} radians, strength: {direction_strength}")
    if target_point is not None:
        print(f"- Target point: {target_point}, strength: {attraction_strength}")
    print(f"- Scenario file: {output_file}")

    return scenario_config


def generate_osm_scenarios_from_config(config_dict: Dict):
    """
    Generate multiple OSM scenarios from configuration dict, supporting independent direction control per base station

    Args:
        config_dict: Configuration dict containing the following keys:
            - osm_file: OSM file path (required)
            - base_stations: Base station configuration list (required)
            - servers: Specified server ID list (required)
            - users: Number of users (default: 41)
            - speed: Movement speed (default: 10.0)
            - total_steps: Total steps (default: 100)
            - num_scenarios: Number of scenarios (default: 1)
            - movement_pattern: Movement pattern (default: "random")
            - target_direction: Global target direction (optional)
            - direction_strength: Global direction strength (default: 0.5)
            - target_point: Target point (optional)
            - attraction_strength: Attraction strength (default: 0.3)
            - scenario_id: Scenario ID (for directory naming)
            - output: Output filename (used for single scenario)
            - base_station_directions: Base station specific target direction dict {base_station_id: direction_radians} (optional)
            - base_station_direction_strengths: Base station specific direction strength dict {base_station_id: strength} (optional)
            - loiter_configs: Loiter configurations {base_station_id: {target_point, radius}} (optional)
            - server_configs_raw: Raw server configuration list (optional)
    """
    # Set default values
    config = {
        "users": 41,
        "speed": 10.0,
        "total_steps": 100,
        "num_scenarios": 1,
        "movement_pattern": "random",
        "direction_strength": 0.5,
        "attraction_strength": 0.3,
        **config_dict,  # Override provided values
    }

    # Validate required parameters
    required_params = ["osm_file", "base_stations", "servers"]
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' is missing")

    # Get scenario ID for directory name
    scenario_id = config.get("scenario_id", "default_osm_scenario")

    # Create output directory
    scenario_dir = os.path.join(
        "generated_scenarios/osm_controlled_scenarios", scenario_id
    )
    os.makedirs(scenario_dir, exist_ok=True)

    # Generate multiple scenarios
    for i in range(config["num_scenarios"]):
        if config.get("output") and config["num_scenarios"] == 1:
            # Single scenario with specified output filename
            output_path = os.path.join(scenario_dir, config["output"])
        else:
            # Multiple scenarios or default naming
            server_str = "_".join(map(str, config["servers"]))
            pattern_str = config["movement_pattern"]

            # If base station independent directions are specified, add direction info to filename
            dir_str = ""
            if config.get("base_station_directions") is not None:
                for server_id in config["servers"]:
                    if server_id in config["base_station_directions"]:
                        dir_val = config["base_station_directions"][server_id]
                        dir_str += "R" if abs(dir_val - 0) < 0.1 else "L"
                    else:
                        dir_str += "U"
                if dir_str:
                    pattern_str = f"{pattern_str}_{dir_str}"

            if config["num_scenarios"] > 1:
                # Add index for multiple scenarios
                output_path = os.path.join(
                    scenario_dir,
                    f"osm_controlled_scenario_{config['users']}users_{config['speed']}mps_{server_str}servers_{pattern_str}pattern_{config['total_steps']}steps_{i}.yaml",
                )
            else:
                # Single scenario
                output_path = os.path.join(
                    scenario_dir,
                    f"osm_controlled_scenario_{config['users']}users_{config['speed']}mps_{server_str}servers_{pattern_str}pattern_{config['total_steps']}steps.yaml",
                )

        # Set random seed for reproducible but different scenarios
        random.seed(i)

        generate_controlled_osm_scenario(
            osm_file_path=config["osm_file"],
            base_stations=config["base_stations"],
            server_ids=config["servers"],
            num_users=config["users"],
            total_steps=config["total_steps"],
            output_file=output_path,
            user_speed=config["speed"],
            speed_range=config.get("speed_range"),  # Pass speed range
            movement_pattern=config["movement_pattern"],
            target_direction=config.get("target_direction"),
            direction_strength=config["direction_strength"],
            target_point=config.get("target_point"),
            attraction_strength=config["attraction_strength"],
            scenario_name=f"{scenario_id}_{i}",
            server_configs_raw=config.get("server_configs_raw"),  # Pass detailed configuration
            base_station_directions=config.get("base_station_directions"),
            base_station_direction_strengths=config.get(
                "base_station_direction_strengths"
            ),
        )

        print(f"Generating scenario {i+1}/{config['num_scenarios']}")


def generate_real_scenarios_from_synthetic_config(config_dict: Dict):
    """
    Generate real OSM scenarios based on synthetic scenario configurations, making real scenarios use the same scenario configurations as synthetic scenarios
    Supports independent direction control per base station

    Args:
        config_dict: Synthetic scenario configuration dict, containing the following keys (same as new_synthetic_scenarios/generate_scenario.py):
            - server_configs: Server configuration list, each element is a dict containing id, movement_pattern, target_direction, etc.
            - users_range: User number range [min, max]
            - total_time: Total time (default: 100.0)
            - num_scenarios: Number of scenarios (default: 1)
            - scenario_id: Scenario ID (for directory naming)
            - movement_pattern: Global movement pattern (default: "random")
            - target_direction: Global target direction (optional)
            - direction_strength: Global direction strength (default: 0.5)
            - target_point: Global target point (optional)
            - attraction_strength: Global attraction strength (default: 0.3)
    """
    # Set default values
    config = {
        "users_range": [35, 50],
        "total_time": 100.0,
        "num_scenarios": 1,
        "movement_pattern": "random",
        "direction_strength": 0.5,
        "attraction_strength": 0.3,
        **config_dict,
    }

    # Validate required parameters
    if "server_configs" not in config:
        raise ValueError("'server_configs' parameter is required")

    # Fixed parameters for real scenarios
    osm_file = "sdnu.osm"
    # Real base station latitude/longitude coordinates (corresponding to base station IDs in synthetic scenarios)
    base_stations = [
        {"id": 1, "position": [116.8239735, 36.5429144]},
        {"id": 2, "position": [116.8274521, 36.5456314]},
        {"id": 3, "position": [116.8306442, 36.5443304]},
    ]

    # Extract servers list and movement parameters from server_configs
    server_configs = config["server_configs"]
    servers = [s.get("id") for s in server_configs]

    # Extract independent direction parameters for each base station
    base_station_directions = {}
    base_station_direction_strengths = {}
    speed_range = None

    for server_config in server_configs:
        server_id = server_config.get("id")
        if server_id is not None:
            # Extract target direction
            target_direction = server_config.get("target_direction")
            if target_direction is not None:
                base_station_directions[server_id] = target_direction

            # Extract direction strength
            direction_strength = server_config.get("direction_strength")
            if direction_strength is not None:
                base_station_direction_strengths[server_id] = direction_strength

        # Extract speed range (assuming all base stations have the same speed range, take the first non-empty value)
        if speed_range is None and server_config.get("speed_range") is not None:
            speed_range = server_config.get("speed_range")

    # If no base station independent directions specified, use global parameters
    # Note: Global parameters will be used as defaults in subsequent configurations

    # Use the movement_pattern of the first base station as the movement_pattern for the entire scene
    # (Assuming all base stations use the same movement_pattern, but can have independent directions)
    # first_server = server_configs[0] if server_configs else {}
    # movement_pattern = first_server.get("movement_pattern", config["movement_pattern"])

    # Complex scenario: Allow mixed modes. If different patterns found, mark as "mixed" and need generate_scenario to support mixed
    patterns = set(s.get("movement_pattern", "random") for s in server_configs)
    if len(patterns) > 1:
        movement_pattern = "mixed"
    else:
        movement_pattern = list(patterns)[0] if patterns else config["movement_pattern"]

    # Extract loiter configurations (if exist)
    loiter_configs = {}
    for sc in server_configs:
        if sc.get("movement_pattern") == "loiter":
            # Real scenario needs to convert loiter target_point to metric coordinates or lat/lon?
            # Here input is lat/lon (if from main_synthetic)
            # Actually, main_synthetic defines BS2 loiter target point as lat/lon
            loiter_configs[sc["id"]] = {
                "target_point": sc.get("target_point"),
                "loiter_radius": sc.get("loiter_radius", 150),
            }

    # If base station independent directions not specified, use global target_direction and direction_strength as fallback
    global_target_direction = config.get("target_direction")
    global_direction_strength = config["direction_strength"]

    # If base station independent direction dict is empty, use global direction
    if not base_station_directions and global_target_direction is not None:
        # Set same global direction for all specified servers
        for server_id in servers:
            base_station_directions[server_id] = global_target_direction

    # If base station independent strength dict is empty, use global strength
    if not base_station_direction_strengths:
        # Set same global strength for all specified servers
        for server_id in servers:
            base_station_direction_strengths[server_id] = global_direction_strength

    # Number of users (use middle value of users_range, or take first value)
    users_min, users_max = config["users_range"]
    users = (users_min + users_max) // 2

    # Total steps (assuming one step per second, total_time seconds)
    total_steps = int(config["total_time"])

    # Movement speed (if speed_range not extracted, use fixed value 10.0 meters/step)
    if speed_range is None:
        speed = 10.0
        speed_range = None
    else:
        speed = (speed_range[0] + speed_range[1]) / 2  # Use average speed as default

    # Scenario ID
    scenario_id = config.get("scenario_id", "real_from_synthetic")

    # Create real scenario configuration, including base station independent direction parameters
    real_config = {
        "osm_file": osm_file,
        "base_stations": base_stations,
        "servers": servers,
        "users": users,
        "speed": speed,
        "speed_range": speed_range,  # Add speed range
        "total_steps": total_steps,
        "num_scenarios": config["num_scenarios"],
        "movement_pattern": movement_pattern,
        "target_direction": global_target_direction,  # As fallback
        "direction_strength": global_direction_strength,  # As fallback
        "target_point": config.get("target_point"),
        "attraction_strength": config["attraction_strength"],
        "scenario_id": scenario_id,
        "base_station_directions": base_station_directions
        if base_station_directions
        else None,
        "base_station_direction_strengths": base_station_direction_strengths
        if base_station_direction_strengths
        else None,
        "loiter_configs": loiter_configs if loiter_configs else None,  # Pass mixed mode configuration
        "server_configs_raw": server_configs,  # Pass raw configuration for finer-grained control
    }

    # Call existing OSM scenario generation function, which now supports base station independent directions
    generate_osm_scenarios_from_config(real_config)

    print(f"Generated real scenario from synthetic configuration: {scenario_id}")
    print(f"- Servers: {servers}")
    print(f"- User count: {users}")
    print(f"- Movement pattern: {movement_pattern}")

    # Display direction configuration for each base station
    if base_station_directions:
        for server_id in servers:
            if server_id in base_station_directions:
                direction = base_station_directions[server_id]
                strength = base_station_direction_strengths.get(
                    server_id, global_direction_strength
                )
                dir_str = "right" if abs(direction - 0) < 0.1 else "left"
                print(
                    f"- Base station {server_id} target direction: {dir_str} ({direction:.2f} radians), strength: {strength}"
                )
    elif global_target_direction is not None:
        print(
            f"- Global target direction: {global_target_direction:.2f} radians, strength: {global_direction_strength}"
        )

    print(f"- Movement parameters: {movement_pattern}")
    if movement_pattern == "loiter" and config.get("target_point"):
        print(f"  - Loiter center: {config.get('target_point')}")


def main_synthetic():
    """
    Generate real scenario configurations that are exactly the same as synthetic scenarios
    Use the same scenario configurations as new_synthetic_scenarios/generate_scenario.py
    """
    # OSM file path
    osm_file_path = "sdnu.osm"

    # Real base station lat/lon coordinates (corresponding to base station IDs in synthetic scenarios)
    base_stations = [
        {"id": 1, "position": [116.8239735, 36.5429144]},
        {"id": 2, "position": [116.8274521, 36.5456314]},
        {"id": 3, "position": [116.8306442, 36.5443304]},
    ]

    # Fixed number of users (same as synthetic scenarios)
    FIXED_NUM_USERS = 50

    # Define training scenario configurations (exactly the same as synthetic scenarios)
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
                },
                {
                    "id": 2,
                    "movement_pattern": "directional",
                    "target_direction": 0,
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "BS1_right_BS2_right",
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
                },
                {
                    "id": 3,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "BS2_left_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
    ]

    # Define testing scenario configurations (identical to synthetic scenarios)
    testing_scenarios = [
        # Testing scenario 1: Three base stations mixed directions (right-right-left)
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
                    "target_direction": 0,  # Right
                    "direction_strength": 0.7,
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
            "scenario_id": "BS1_right_BS2_right_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
        # Test scenario 2: Base station 2 loiter
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
                    "movement_pattern": "loiter",
                    "target_point": [116.8274521, 36.5456314],  # BS2 Position
                    "direction_strength": 0.7,
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
            "scenario_id": "BS1_right_BS2_loiter_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
        # Test scenario 3: Dual base station direction conflict (base station 1 right, base station 3 left)
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
                    "id": 3,
                    "movement_pattern": "directional",
                    "target_direction": math.pi,  # Left
                    "direction_strength": 0.7,
                    "speed_range": [5.0, 15.0],
                },
            ],
            "num_scenarios": 10,
            "scenario_id": "BS1_right_BS3_left",
            "users_range": [FIXED_NUM_USERS, FIXED_NUM_USERS],
            "total_time": 100.0,
        },
    ]

    print("=" * 60)
    print("Generate real scenario configurations that are exactly the same as synthetic scenarios")
    print(f"Fixed number of users: {FIXED_NUM_USERS}")
    print("=" * 60)

    print("\n--- Generate training scenarios ---")
    for scenario_config in training_scenarios:
        print(f"Generate training scenario: {scenario_config['scenario_id']}")
        generate_real_scenarios_from_synthetic_config(scenario_config)

    print("\n--- Generate testing scenarios ---")
    for scenario_config in testing_scenarios:
        print(f"Generate testing scenario: {scenario_config['scenario_id']}")
        generate_real_scenarios_from_synthetic_config(scenario_config)

    print("\n=== Scenario generation completed ===")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate controlled scenarios based on OSM map")
    parser.add_argument(
        "--mode",
        choices=["synthetic"],
        default="synthetic",
        help="Run mode: synthetic (generate real scenarios same as synthetic scenarios)",
    )

    args = parser.parse_args()

    if args.mode == "synthetic":
        main_synthetic()
    else:
        print("Please use --mode synthetic to generate real scenarios same as synthetic scenarios")
