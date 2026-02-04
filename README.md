# A Counterfactual Framework for Data-Scarce Offline RL in Mobility Load Balancing

> **⚠️ Disclaimer**: 
1.This project and its corresponding paper have not yet undergone peer review. The code is provided for research purposes only. Please do __NOT__ download or use this repo.
2.The content of this repo may be modified, added, or removed depending on the review status.

## 📖 Project Overview

This repository implements a **Counterfactual Framework** for addressing data scarcity and distributional shift challenges in **Offline Reinforcement Learning (RL)** for **Mobility Load Balancing (MLB)**.

Mobility Load Balancing is critical for 5G/6G networks but is difficult to optimize using traditional Online RL due to safety risks and service interruptions. Offline RL offers a safe alternative but suffers from data scarcity in real-world deployments. 

Our framework solves this by combining **Counterfactual** with **Deep Reinforcement Learning**:
1.  **Causal Generative Model**: Learns disentangled representations of global mobility contexts and local micro-dynamics.
2.  **Counterfactual Augmentation**: Generates physically consistent "what-if" scenarios (counterfactual states) via latent space interventions.
3.  **Physics-Informed Reward**: Uses wireless propagation laws to provide ground-truth rewards for synthetic states, ensuring reliability.

By augmenting limited real-world data with these counterfactual transitions, we train a **Soft Actor-Critic (SAC)** agent that achieves robust generalization in both synthetic and realistic tracking scenarios.

## 🖼️ Visualizations

Below are the visualizations of the agent's load balancing performance in both real-world (OSM-based) and synthetic environments.

<p align="center">
  <strong>Real-World Scenario (OSM)</strong><br>
  <img src="real_scenarios/visualizations/animation.gif" width="80%" alt="Real Scenario Animation">
</p>

<p align="center">
  <strong>Synthetic Scenario</strong><br>
  <img src="synthetic_scenarios/visualizations/animation.gif" width="80%" alt="Synthetic Scenario Animation">
</p>

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/LoadBalancing.git
    cd LoadBalancing
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n cf_mlb python=3.7
    conda activate cf_mlb
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start & Reproduction

### 1. View Existing Results
*   **Main Results**: Run `python aggregate_experiments.py` to see the summarized performance table.
*   **Visualizations**: Check the `./plot` directory for PCA visualizations of state space coverage.
*   **Parameter Analysis**: Detailed parameter studies can be found in `./parameter_discussion/aggregated_experiment_results.csv`.

### 2. Reproduce Experiments

Follow these steps to train and evaluate the models from scratch.

#### Step 1: Data Generation (Optional)
> **Note**: This step involves randomness. Regenerating data may lead to slight variations in results compared to the paper. If you wish to strictly benchmark against our pre-calculated models, skip this step.

```bash
# Generate scenarios and collect offline training data
python synthetic_scenarios/generate_scenario.py
python real_scenarios/generate_scenarios.py

python synthetic_scenarios/collect_offline_data.py
python real_scenarios/collect_offline_data.py
```

#### Step 2: Training
Train the Counterfactual (CF) framework and baselines (SAC, MOPO, etc.) across all scenarios.
*   `experiments1`, `experiments2`, `experiments3`: Independent experimental runs initialized with different random seeds to ensure statistical reliability.

```bash
# Train models for Synthetic Scenarios
python run_synthetic_experiments1.sh
python run_synthetic_experiments2.sh
python run_synthetic_experiments3.sh

# Train models for Real-World Scenarios
python run_real_experiments1.sh
python run_real_experiments2.sh
python run_real_experiments3.sh
```

#### Step 3: Evaluation
Evaluate the trained policies on the test scenarios.

```bash
# Evaluate Synthetic Models
python run_synthetic_evals1.sh
python run_synthetic_evals2.sh
python run_synthetic_evals3.sh

# Evaluate Real-World Models
python run_real_evals1.sh
python run_real_evals2.sh
python run_real_evals3.sh
```

#### Step 4: Aggregate Results
Collect all evaluation metrics into a summary CSV file.

```bash
python aggregate_experiments.py
```

## 📄 License and Contributing

This project uses multiple open-source components. 
*   **Code**: Released under the license specified in [LICENSE](LICENSE).
*   **Data**: Scenario data is generated based on OpenStreetMap (ODbL).

Issues and Pull Requests are welcome! By submitting code, you agree that your contributions will be released under the same license as the project.

## 🔗 Citation

If you use this project in your research, please cite:

```bibtex
[Add your citation information here]
```
