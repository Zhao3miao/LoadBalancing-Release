import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def load_data(path, label, num_samples=1000):
    """Loads state data from a .npz file."""
    if not os.path.exists(path):
        print(f"File not found: {path}, skipping...")
        return None
        
    try:
        data = np.load(path)
        
        states = None
        # Explicit priority given to 'observations' as requested by user ("only draw states")
        if 'observations' in data:
            states = data['observations']
        elif 'obs' in data:
            states = data['obs']
        elif 'states' in data:
            states = data['states']
        
        if states is None:
            # Fallback for old npz files if necessary, but printing keys helps debug
            # print(f"Could not find state key in {path}. Keys: {list(data.keys())}")
            return None
            
        # Randomly sample to avoid overcrowding plots if dataset is large
        if len(states) > num_samples:
            indices = np.random.choice(len(states), num_samples, replace=False)
            states = states[indices]
            
        return states
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def visualize_distribution_shift():
    try:
        # File paths
        cf_data_path = "../synthetic_scenarios/experiment1/results_sac+20+cf/BS1_right_BS3_left/cf_data.npz"
        test_data_path = "../synthetic_scenarios/offline_data/BS1_right_BS3_left/data.npz"
        train_paths = [
            "../synthetic_scenarios/offline_data/BS2_left_BS3_left/data.npz",
            "../synthetic_scenarios/offline_data/BS1_right_BS2_right/data.npz"
        ]
        
        print("--- Starting Visualization (4-Layer) ---")
        
        # 1. Load Data
        cf_states = load_data(cf_data_path, "CF Data", num_samples=1000)
        target_states_full = load_data(test_data_path, "Target/Test Data", num_samples=2000)
        
        train_states_list = []
        for path in train_paths:
            s = load_data(path, "Source/Train Data", num_samples=1000)
            if s is not None:
                train_states_list.append(s)
        train_states = np.concatenate(train_states_list, axis=0) if train_states_list else None

        if target_states_full is None or cf_states is None or train_states is None:
            print("Insufficient data. Exiting.")
            return

        # 2. Simulate the Few-Shot Split on Target Data
        np.random.seed(42)
        indices = np.arange(len(target_states_full))
        # np.random.shuffle(indices)
        
        split_point = int(len(target_states_full) * 0.2)
        target_seeds = target_states_full[indices[:split_point]]      # The 20% "Green" Seeds
        target_unseen = target_states_full[indices[split_point:]]     # The 80% "Gray" Unseen
        
        print(f"Source samples: {len(train_states)} (Blue)")
        print(f"Target Seeds: {len(target_seeds)} (Green)")
        print(f"Target Unseen: {len(target_unseen)} (Gray)")
        print(f"Generated CF: {len(cf_states)} (Red)")

        # --- Generate Contrast Data (Gaussian Noise) ---
        # Purpose: Show that simple noise augmentation creates unstructured blobs/spheres
        print("Generating Gaussian Noise Contrast...")
        # Use feature-wise std to scale noise, factor tunable to match CF spread magnitude roughly
        feature_std = np.std(target_seeds, axis=0)
        feature_std[feature_std == 0] = 1e-6 # Avoid div by zero
        
        # Resample seeds to create a noise dataset of same size as CF
        idx_noise = np.random.choice(len(target_seeds), len(cf_states), replace=True)
        # Add noise: Scale factor 1.5 approximates natural data variance to make it visible
        noise_data = target_seeds[idx_noise] + np.random.normal(0, feature_std, (len(cf_states), target_seeds.shape[1]))

        # 3. PCA Projection
        # Refined Plan: Fit PCA on Seeds + CF + Noise so all are in the same view
        combined_local = np.concatenate([target_seeds, cf_states, noise_data], axis=0)
        
        print("Computing PCA projection (Local)...")
        pca = PCA(n_components=2)
        pca.fit(combined_local)
        
        # Transform 
        p_seeds = pca.transform(target_seeds)
        p_cf = pca.transform(cf_states)
        p_noise = pca.transform(noise_data)
        
        # 4. Plotting
        plt.figure(figsize=(10, 8))
        # Use default style and turn off grid explicitly
        plt.style.use('default')
        
        # Layer 1: Gaussian Noise (Contrast)
        # "Dumb" spherical expansion - Faint Blue Circles
        plt.scatter(p_noise[:, 0], p_noise[:, 1], 
                   c='deepskyblue', alpha=0.15, s=30, label='Gaussian Noise (Baseline)', 
                   marker='o', edgecolors='none', zorder=1)
        
        # Layer 2: Generated CF (The "Smart Expansion")
        # Physics-informed manifold - Red Circles
        plt.scatter(p_cf[:, 0], p_cf[:, 1], 
                   c='blue', alpha=0.4, s=30, label='Counterfactual Augmentation (Ours)', 
                   marker='o', edgecolors='none', zorder=2)

        # Layer 3: Target Seeds (The "Anchors")
        # The ground truth visible data - Green Circles (No Edge)
        plt.scatter(p_seeds[:, 0], p_seeds[:, 1], 
                   c='green', alpha=1.0, s=60, label='Few-Shot Seeds', 
                   marker='o', edgecolors='none', zorder=3)

        plt.title("Augmentation Quality: Physics-Informed Manifold vs. Random Noise", fontsize=15)
        plt.xlabel(f"Principal Component 1", fontsize=12)
        plt.ylabel(f"Principal Component 2", fontsize=12)
        plt.grid(False) # Explicitly remove grid
        
        plt.legend(frameon=True, fontsize=12, loc='best', fancybox=True, framealpha=0.9, facecolor='white', edgecolor='lightgray')
        plt.tight_layout()
        
        output_file = "augmentation_mechanism.png"
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        plt.xlabel(f"Principal Component 1", fontsize=12)
        plt.ylabel(f"Principal Component 2", fontsize=12)
        
        plt.legend(frameon=True, fontsize=12, loc='best', fancybox=True, framealpha=0.9)
        plt.tight_layout()
        
        output_file = "augmentation_mechanism.png"
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_distribution_shift()