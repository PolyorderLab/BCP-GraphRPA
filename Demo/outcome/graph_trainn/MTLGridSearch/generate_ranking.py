import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project src is on sys.path if needed
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # /home/zxy/project/TAPD
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Directory containing all hyper‑parameter sub‑folders
BASE_DIR = Path("/home/zxy/project/TAPD/Demo/outcome/graph_trainn/MTLGridSearch")

# Columns we expect in the parity plot CSVs – adjust if the real files differ
CHI_COLUMNS = ["MAPE", "RMSE", "R2"]  # chiN metrics
K_COLUMNS   = ["MAPE", "RMSE", "R2"]  # k* metrics

# Helper to safely read a CSV and extract the needed metrics
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
        # Convert to numeric if needed and compute mean
        result[name] = pd.to_numeric(df[col], errors='coerce').mean()
    return result

# Collect results from each sub‑folder
records = []
for subdir in sorted(BASE_DIR.iterdir()):
    if not subdir.is_dir():
        continue
    chi_path = subdir / "parity_plot_chiN_stats.csv"
    k_path   = subdir / "parity_plot_kstar_stats.csv"
    try:
        chi_metrics = read_metrics(chi_path, CHI_COLUMNS)
        k_metrics   = read_metrics(k_path, K_COLUMNS)
    except Exception as e:
        print(f"[WARN] Skipping {subdir.name}: {e}")
        continue

    # Simple old‑formula score: sum of chiN and k* errors (lower is better)
    # We treat R2 as a benefit, so we subtract it (the higher the better).
    old_score = (
        chi_metrics["MAPE"] + chi_metrics["RMSE"] - chi_metrics["R2"] +
        k_metrics["MAPE"]   + k_metrics["RMSE"]   - k_metrics["R2"]
    )

    records.append({
        "model": subdir.name,
        "chiN_MAPE": chi_metrics["MAPE"],
        "chiN_RMSE": chi_metrics["RMSE"],
        "chiN_R2":   chi_metrics["R2"],
        "kstar_MAPE": k_metrics["MAPE"],
        "kstar_RMSE": k_metrics["RMSE"],
        "kstar_R2":   k_metrics["R2"],
        "old_score": old_score,
    })

if not records:
    print("No valid sub‑folders found. Exiting.")
    sys.exit(0)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(records)

# Normalized new score: (old_score - min) / (max - min)
min_score = df["old_score"].min()
max_score = df["old_score"].max()
if max_score == min_score:
    df["new_norm_score"] = 0.0
else:
    df["new_norm_score"] = (df["old_score"] - min_score) / (max_score - min_score)

# Rank by the new normalized score (ascending – lower is better)
df = df.sort_values(by="new_norm_score", ascending=True).reset_index(drop=True)
df.insert(0, "rank", df.index + 1)

# Select columns for final output – you can adjust the order as needed
output_cols = [
    "rank",
    "model",
    "chiN_MAPE",
    "chiN_RMSE",
    "chiN_R2",
    "old_score",
    "new_norm_score",
]
output_df = df[output_cols]

# Save ranking to CSV (you may also print it)
output_path = BASE_DIR / "ranking_summary.csv"
output_df.to_csv(output_path, index=False, float_format="%.4f")
print(f"Ranking saved to {output_path}")
print(output_df.head(10).to_string(index=False))
