import sys
from pathlib import Path
import pandas as pd

# Directory containing hyper‑parameter sub‑folders
BASE_DIR = Path("/home/zxy/project/TAPD/Demo/outcome/graph_trainn/MTLGridSearch")

# Helper to read metrics from a CSV, tolerant to column name variations
def read_metrics(csv_path: Path, metric_names):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    result = {}
    for name in metric_names:
        # Find column that contains the metric name (case‑insensitive), allowing extra characters like parentheses or %
        col = next((c for c in df.columns if name.lower() in c.lower()), None)
        if col is None:
            raise KeyError(f"Column containing '{name}' not found in {csv_path}")
        # Convert to numeric (ignore non‑numeric rows) and compute mean
        result[name] = pd.to_numeric(df[col], errors='coerce').mean()
    return result

def main():
    # Collect results from each sub‑folder
    records = []
    for subdir in sorted(BASE_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        k_path = subdir / "parity_plot_kstar_stats.csv"
        try:
            k_metrics = read_metrics(k_path, ["MAPE", "RMSE", "R2"])
        except Exception as e:
            print(f"[WARN] Skipping {subdir.name}: {e}")
            continue

        # Old‑formula score for k*: lower is better (MAPE + RMSE - R2)
        old_score = k_metrics["MAPE"] + k_metrics["RMSE"] - k_metrics["R2"]

        records.append({
            "model": subdir.name,
            "kstar_MAPE": k_metrics["MAPE"],
            "kstar_RMSE": k_metrics["RMSE"],
            "kstar_R2":   k_metrics["R2"],
            "old_score":  old_score,
        })

    if not records:
        print("No valid sub‑folders found. Exiting.")
        sys.exit(0)

    df = pd.DataFrame(records)
    # Normalized new score (0‑1 range)
    min_score = df["old_score"].min()
    max_score = df["old_score"].max()
    if max_score == min_score:
        df["new_norm_score"] = 0.0
    else:
        df["new_norm_score"] = (df["old_score"] - min_score) / (max_score - min_score)

    # Rank by the new normalized score (ascending → better performance)
    df = df.sort_values(by="new_norm_score", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Output columns order
    output_cols = ["rank", "model", "kstar_MAPE", "kstar_RMSE", "kstar_R2", "old_score", "new_norm_score"]
    output_df = df[output_cols]

    output_path = BASE_DIR / "k_ranking_summary.csv"
    output_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"K‑star ranking saved to {output_path}")
    print(output_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
