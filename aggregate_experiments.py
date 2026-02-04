import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import glob


def get_ci_95(data):
    """
    Returns the half-width of the 95% confidence interval.
    CI = t * (std / sqrt(n))
    """
    n = len(data)
    if n <= 1:
        return 0.0
    # standard error
    se = stats.sem(data)
    # t-score for 95% CI
    t_score = stats.t.ppf(0.975, df=n - 1)
    return t_score * se


def find_json_files(root_dir, filename="evaluation_results.json"):
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            json_files.append(os.path.join(root, filename))
    return json_files


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define experiment structure
    experiments = [
        "experiment1",
        "experiment2",
        "experiment3",
    ]  # , "experiment4", "experiment5"]
    env_types = {"Synthetic": "synthetic_scenarios", "Real": "real_scenarios"}

    # Define methods to look for (matching summarize_results_json.py)
    target_methods = [
        "results_static",
        "results_random",
        "results_mopo",
        "results_ppo",
        "results_sac_online",
        "results_sac+0",
        "results_sac+20",
        "results_sac+20+cf",
        "results_sac+100",
    ]

    all_data = []

    print("Scanning for evaluation results...")

    # 1. Load Data from JSONs
    for exp_name in experiments:
        for pretty_type, folder_name in env_types.items():
            exp_path = os.path.join(base_dir, folder_name, exp_name)

            if not os.path.exists(exp_path):
                print(f"Skipping missing directory: {exp_path}")
                continue

            for method in target_methods:
                method_path = os.path.join(exp_path, method)
                if not os.path.exists(method_path):
                    # print(f"  Missing method folder: {method} in {exp_path}")
                    continue

                json_paths = find_json_files(method_path)

                for json_path in json_paths:
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Extract metrics
                        row = {
                            "Type": pretty_type,
                            "Experiment": exp_name,
                            "Method": method,
                            "Scenario": data.get("scenario", "unknown"),
                            "Reward": data.get("mean_reward", np.nan),
                            "Throughput": data.get("mean_throughput", np.nan),
                            "Load Balance": data.get("mean_load_balance", np.nan),
                        }

                        # Sometimes Load Balance is stored as negative in JSON (to maximize)
                        # or positive (std dev). summarize_results_json.py treated it as:
                        # "mean_load_balance": float(-mean_load)  <-- in eval_mopo it was negative of std
                        # Let's trust the JSON value is the metric we want to analyze (usually higher is better or consistent)
                        # If the JSON has it as negative, we might want to flip it?
                        # eval_mopo writes: "mean_load_balance": float(-mean_load) which is negative of std.
                        # So closer to 0 is better (larger negative number).
                        # Let's just keep raw values for aggregation.

                        all_data.append(row)
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}")

    if not all_data:
        print("No evaluation data found.")
        return

    df = pd.DataFrame(all_data)

    # 2. Aggregation across Experiments
    # Group by [Type, Scenario, Method]
    # Calculate Mean and CI/Std for the metrics

    # We want to aggregate "Reward", "Throughput", "Load Balance"
    metrics = ["Reward", "Throughput", "Load Balance"]

    # Convert to numeric
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    grouped = df.groupby(["Type", "Scenario", "Method"])

    agg_funcs = {m: ["mean", "std", get_ci_95] for m in metrics}
    agg_df = grouped.agg(agg_funcs).reset_index()

    # Flatten columns
    new_cols = []
    for col in agg_df.columns.values:
        if col[0] in metrics:
            new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col[0])
    agg_df.columns = new_cols

    # 3. Calculate Improvement vs Static (Aggregated)
    # We need to compute improvement on the AGGREGATED means

    final_output_rows = []

    # Iterate through each (Type, Scenario) group
    for (type_val, scenario_val), group in agg_df.groupby(["Type", "Scenario"]):
        # Find Static baseline for this group
        static_row = group[group["Method"] == "results_static"]

        base_reward = None
        base_throughput = None
        base_lb = None

        if not static_row.empty:
            base_reward = static_row.iloc[0]["Reward_mean"]
            base_throughput = static_row.iloc[0]["Throughput_mean"]
            base_lb = static_row.iloc[0]["Load Balance_mean"]

        for idx, row in group.iterrows():
            res = row.to_dict()

            # Calculate improvements
            if base_reward and base_reward != 0:
                res["Reward_Imp"] = (
                    (row["Reward_mean"] - base_reward) / abs(base_reward) * 100
                )
            else:
                res["Reward_Imp"] = 0.0

            if base_throughput and base_throughput != 0:
                res["Throughput_Imp"] = (
                    (row["Throughput_mean"] - base_throughput)
                    / abs(base_throughput)
                    * 100
                )
            else:
                res["Throughput_Imp"] = 0.0

            # For Load Balance?
            # If closer to 0 is better (and values are negative), then (New - Old) / abs(Old)
            # If -0.1 vs -0.5 (baseline). (-0.1 - (-0.5)) / 0.5 = 0.4 / 0.5 = 80% improvement.
            if base_lb and base_lb != 0:
                res["Load_Balance_Imp"] = (
                    (row["Load Balance_mean"] - base_lb) / abs(base_lb) * 100
                )
            else:
                res["Load_Balance_Imp"] = 0.0

            final_output_rows.append(res)

    final_df = pd.DataFrame(final_output_rows)

    # Reorder columns slightly for readability
    cols_order = ["Type", "Scenario", "Method"]
    # Filter only Load Balance and its relative improvement
    m = "Load Balance"
    cols_order.extend([f"{m}_mean", f"{m}_std", f"{m}_get_ci_95"])
    cols_order.extend(["Load_Balance_Imp"])

    # Filter only columns that exist
    cols_order = [c for c in cols_order if c in final_df.columns]
    final_df = final_df[cols_order]

    # Formatting
    # Round float columns
    float_cols = final_df.select_dtypes(include=["float"]).columns
    final_df[float_cols] = final_df[float_cols].round(4)

    output_path = os.path.join(base_dir, "aggregated_experiment_results.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")


if __name__ == "__main__":
    main()
