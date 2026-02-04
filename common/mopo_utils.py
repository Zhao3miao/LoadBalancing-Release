import numpy as np
import torch
import sys
import os

# Add MOPO root to path to import its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
mopo_root = os.path.abspath(os.path.join(current_dir, "..", "MOPO"))
if mopo_root not in sys.path:
    sys.path.append(mopo_root)

from MOPO.common.buffer import ReplayBuffer

def load_offline_data_to_mopo_buffer(data_path, obs_dim, action_dim, ratio=1.0):
    """
    Load data from .npz file and convert to MOPO ReplayBuffer
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    total_samples = len(data["obs"])
    num_samples = int(total_samples * ratio)
    
    # Create MOPO ReplayBuffer
    # Note: MOPO buffer assumes max_size is fixed
    buffer = ReplayBuffer(
        buffer_size=num_samples,
        obs_shape=(obs_dim,),
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32
    )
    
    observations = data["obs"][:num_samples]
    # Adjust actions: Ensure they are consistent with the saved data
    # Some datasets key is 'action', some might be 'actions'
    if "action" in data:
        actions = data["action"][:num_samples]
    elif "actions" in data:
        actions = data["actions"][:num_samples]
    else:
        raise KeyError("Cannot find 'action' or 'actions' in npz file")

    rewards = data["reward"][:num_samples]
    # Ensure reward is (N, 1)
    if rewards.ndim == 1:
        rewards = rewards.reshape(-1, 1)
        
    terminals = data["terminal"][:num_samples]
    # Ensure terminal is (N, 1)
    if terminals.ndim == 1:
        terminals = terminals.reshape(-1, 1)
        
    next_observations = data["next_obs"][:num_samples]

    # Batch add to buffer
    # The MOPO ReplayBuffer usually has an add_batch or similar, or we can iterate
    # Checking MOPO ReplayBuffer implementation (inferred):
    # It likely has .observations, .actions etc. arrays directly accessible or an add function.
    # To be safe and fast, we can directly assign if allow, or loop.
    # Assuming standard implementation:
    
    buffer.observations[:num_samples] = observations
    buffer.actions[:num_samples] = actions
    buffer.rewards[:num_samples] = rewards
    buffer.terminals[:num_samples] = terminals
    buffer.next_observations[:num_samples] = next_observations
    
    # Update pointer and size
    buffer.ptr = num_samples % buffer.max_size  # Potentially full
    buffer.size = num_samples
    
    print(f"Loaded {num_samples} transitions into MOPO buffer.")
    return buffer
