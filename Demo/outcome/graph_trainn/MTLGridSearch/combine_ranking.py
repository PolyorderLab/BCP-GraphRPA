import sys
from pathlib import Path
import pandas as pd

# Paths to the two ranking CSVs
BASE_DIR = Path("/home/zxy/project/TAPD/Demo/outcome/graph_trainn/MTLGridSearch")
CHI_CSV = BASE_DIR / "ranking_summary.csv"
K_CSV   = BASE_DIR / "k_ranking_summary.csv"

def main():
    # Verify files exist
    for p in (CHI_CSV, K_CSV):
        if not p.is_file():
            print(f"[ERROR] Missing file: {p}")
            sys.exit(1)

    # Load CSVs
    chi_df = pd.read_csv(CHI_CSV)
    k_df   = pd.read_csv(K_CSV)

    # Ensure the model column exists and is the same name in both files
    if "model" not in chi_df.columns or "model" not in k_df.columns:
        print("[ERROR] 'model' column not found in one of the CSVs")
        sys.exit(1)

    # Keep only the columns we need for merging
    chi_keep = ["model", "chiN_MAPE", "chiN_RMSE", "chiN_R2", "new_norm_score"]
    k_keep   = ["model", "kstar_MAPE", "kstar_RMSE", "kstar_R2", "new_norm_score"]
    chi_df = chi_df[chi_keep].rename(columns={"new_norm_score": "chi_norm"})
    k_df   = k_df[k_keep].rename(columns={"new_norm_score": "k_norm"})

    # Merge on model name (inner join – only models present in both rankings)
    merged = pd.merge(chi_df, k_df, on="model", how="inner")

    # Compute combined score with equal weight (1:1)
    merged["combined_score"] = (merged["chi_norm"] + merged["k_norm"]) / 2.0

    # Rank by combined_score (lower is better)
    merged = merged.sort_values(by="combined_score", ascending=True).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)

    # Reorder columns for a tidy output
    output_cols = [
        "rank",
        "model",
        "chiN_MAPE",
        "chiN_RMSE",
        "chiN_R2",
        "kstar_MAPE",
        "kstar_RMSE",
        "kstar_R2",
        "combined_score",
    ]
    out_df = merged[output_cols]

    # Save to CSV and Excel
    csv_path = BASE_DIR / "combined_ranking.csv"
    xlsx_path = BASE_DIR / "combined_ranking.xlsx"
    out_df.to_csv(csv_path, index=False, float_format="%.4f")
    out_df.to_excel(xlsx_path, index=False, engine="openpyxl", float_format="%.4f")
    print(f"[OK] Combined ranking saved to {csv_path}")
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
