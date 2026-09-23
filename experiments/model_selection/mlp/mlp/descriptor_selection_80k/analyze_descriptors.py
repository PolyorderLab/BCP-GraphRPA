#!/usr/bin/env python3
"""Leakage-free descriptor screening for the polymer MLP feature set.

All fitting decisions (imputation, scaling, variance/correlation/VIF filtering,
PCA, and feature selection) are fitted on the training YML files only.  The
test set is transformed and reported only after those choices are fixed.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from polymer_descriptors import DESCRIPTOR_NAMES, compute_descriptors


def _record_from_yml(path_string: str) -> dict:
    path = Path(path_string)
    with path.open(encoding="utf-8") as handle:
        system = yaml.safe_load(handle)["system"]
    blocks = system["components"][0]["blocks"]
    node_labels = {node for block in blocks for node in block["ends"]}
    nodes = {node: index for index, node in enumerate(sorted(node_labels, key=str))}
    edge_index = np.asarray([[nodes[a], nodes[b]] for a, b in (block["ends"] for block in blocks)])
    edge_attr = np.asarray(
        [[float(block["length"]), float(str(block["segment"]) == "A")] for block in blocks]
    )
    values = compute_descriptors(edge_index, edge_attr)
    values["sample_id"] = path.stem
    values["chiN_star"] = float(system["χN_star"])
    values["log10_chiN_star"] = float(np.log10(max(values["chiN_star"], 1e-12)))
    return values


def extract_split(input_dir: Path, split_name: str, workers: int, limit: int | None) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.yml"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No YML files in {input_dir}")
    print(f"Extracting {split_name}: {len(paths)} YML files with {workers} workers", flush=True)
    path_strings = [str(path) for path in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(_record_from_yml, path_strings, chunksize=100), start=1):
            rows.append(row)
            if index % 5000 == 0 or index == len(path_strings):
                print(f"  {split_name}: {index}/{len(path_strings)}", flush=True)
    return pd.DataFrame(rows, columns=["sample_id", *DESCRIPTOR_NAMES, "chiN_star", "log10_chiN_star"])


def target_correlations(matrix: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows = []
    for name in matrix.columns:
        x = matrix[name].to_numpy(dtype=float)
        if np.std(x) < 1e-12 or np.std(target) < 1e-12:
            pearson_r = spearman_r = 0.0
            pearson_p = spearman_p = 1.0
        else:
            pearson_r, pearson_p = pearsonr(x, target)
            spearman_r, spearman_p = spearmanr(x, target)
        rows.append({
            "descriptor": name, "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "abs_spearman_r": abs(spearman_r),
        })
    return pd.DataFrame(rows).sort_values("abs_spearman_r", ascending=False, ignore_index=True)


def greedy_correlation_filter(matrix: pd.DataFrame, relevance: pd.Series, threshold: float):
    """Keep the target-more-relevant feature from each highly correlated pair."""
    corr = matrix.corr(method="spearman").abs().fillna(0.0)
    ordered = sorted(matrix.columns, key=lambda name: (-float(relevance[name]), name))
    kept, removed_rows = [], []
    for name in ordered:
        conflicts = [kept_name for kept_name in kept if corr.loc[name, kept_name] >= threshold]
        if conflicts:
            representative = max(conflicts, key=lambda kept_name: (relevance[kept_name], kept_name))
            removed_rows.append({
                "removed": name, "kept": representative,
                "abs_spearman_between_descriptors": corr.loc[name, representative],
                "removed_abs_spearman_to_target": relevance[name],
                "kept_abs_spearman_to_target": relevance[representative],
            })
        else:
            kept.append(name)
    return kept, pd.DataFrame(removed_rows)


def calculate_vif(matrix: pd.DataFrame) -> pd.Series:
    """VIF from R² of each descriptor regressed on all other descriptors."""
    values = matrix.to_numpy(dtype=float)
    result = {}
    for index, name in enumerate(matrix.columns):
        other = np.delete(values, index, axis=1)
        if other.shape[1] == 0:
            result[name] = 1.0
            continue
        r2 = Ridge(alpha=1e-10).fit(other, values[:, index]).score(other, values[:, index])
        result[name] = float(1.0 / max(1.0 - r2, 1e-12))
    return pd.Series(result, name="VIF").sort_values(ascending=False)


def iterative_vif_filter(matrix: pd.DataFrame, relevance: pd.Series, threshold: float):
    active = list(matrix.columns)
    log = []
    while len(active) > 1:
        vif = calculate_vif(matrix[active])
        largest_name, largest_value = vif.index[0], float(vif.iloc[0])
        if largest_value <= threshold:
            break
        log.append({
            "removed": largest_name, "VIF_at_removal": largest_value,
            "abs_spearman_to_target": float(relevance[largest_name]),
        })
        active.remove(largest_name)
    return active, pd.DataFrame(log), calculate_vif(matrix[active])


def choose_mutual_information_k(matrix: pd.DataFrame, target: np.ndarray, random_state: int):
    max_k = matrix.shape[1]
    candidate_k = sorted({min(max_k, k) for k in (3, 5, 8, 10, 15, 20, 25, 30, max_k)})
    mi = pd.Series(
        mutual_info_regression(matrix, target, random_state=random_state), index=matrix.columns, name="mutual_information"
    ).sort_values(ascending=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    for k in candidate_k:
        selected = mi.index[:k].tolist()
        score = -cross_val_score(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)), matrix[selected], target,
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1,
        ).mean()
        rows.append({"k": k, "cv_rmse_log10_chiN": score})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["cv_rmse_log10_chiN"].idxmin(), "k"])
    return mi, scores, best_k, mi.index[:best_k].tolist()


def run_analysis(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train_descriptor_matrix.csv", index=False)
    test.to_csv(output_dir / "test_descriptor_matrix.csv", index=False)

    x_train_raw = train.loc[:, DESCRIPTOR_NAMES].replace([np.inf, -np.inf], np.nan)
    x_test_raw = test.loc[:, DESCRIPTOR_NAMES].replace([np.inf, -np.inf], np.nan)
    y_train = train["log10_chiN_star"].to_numpy(dtype=float)
    y_test = test["log10_chiN_star"].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(imputer.fit_transform(x_train_raw), columns=DESCRIPTOR_NAMES)
    x_test = pd.DataFrame(imputer.transform(x_test_raw), columns=DESCRIPTOR_NAMES)

    pearson = x_train.corr(method="pearson")
    spearman = x_train.corr(method="spearman")
    pearson.to_csv(output_dir / "descriptor_pearson_correlation_train.csv")
    spearman.to_csv(output_dir / "descriptor_spearman_correlation_train.csv")
    # Both are saved: raw chiN is the RPA-derived physical value, while log10
    # chiN is the regression target used by the MLP baseline.
    target_corr_raw = target_correlations(x_train, train["chiN_star"].to_numpy(dtype=float))
    target_corr_raw.to_csv(output_dir / "descriptor_chiN_star_correlation_train.csv", index=False)
    target_corr = target_correlations(x_train, y_train)
    target_corr.to_csv(output_dir / "descriptor_log10_chiN_star_correlation_train.csv", index=False)

    variance = pd.DataFrame({"descriptor": DESCRIPTOR_NAMES, "variance_train": x_train.var(axis=0, ddof=0).to_numpy()})
    variance["kept_by_variance"] = variance["variance_train"] > args.variance_threshold
    variance.to_csv(output_dir / "variance_filter_train.csv", index=False)
    variance_kept = variance.loc[variance["kept_by_variance"], "descriptor"].tolist()
    x_variance = x_train[variance_kept]
    relevance = target_corr.set_index("descriptor")["abs_spearman_r"]

    corr_kept, corr_removed = greedy_correlation_filter(x_variance, relevance, args.correlation_threshold)
    corr_removed.to_csv(output_dir / "correlation_pruning_train.csv", index=False)
    vif_kept, vif_removed, final_vif = iterative_vif_filter(x_train[corr_kept], relevance, args.vif_threshold)
    vif_removed.to_csv(output_dir / "vif_pruning_train.csv", index=False)
    final_vif.rename_axis("descriptor").reset_index().to_csv(output_dir / "vif_final_train.csv", index=False)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    pca = PCA().fit(x_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_summary = pd.DataFrame({
        "component": np.arange(1, len(DESCRIPTOR_NAMES) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": cumulative,
    })
    pca_summary.to_csv(output_dir / "pca_explained_variance_train.csv", index=False)
    pd.DataFrame(pca.components_, columns=DESCRIPTOR_NAMES, index=[f"PC{i}" for i in range(1, len(DESCRIPTOR_NAMES) + 1)]).to_csv(
        output_dir / "pca_loadings_train.csv"
    )

    selection_matrix = x_train[vif_kept]
    mi, mi_cv_scores, best_k, mi_selected = choose_mutual_information_k(selection_matrix, y_train, args.random_state)
    mi.rename_axis("descriptor").reset_index().to_csv(output_dir / "mutual_information_train.csv", index=False)
    mi_cv_scores.to_csv(output_dir / "mutual_information_k_selection_cv.csv", index=False)

    lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=args.random_state, n_jobs=1, max_iter=20000))
    lasso.fit(selection_matrix, y_train)
    lasso_coef = pd.Series(lasso[-1].coef_, index=selection_matrix.columns, name="lasso_coefficient")
    lasso_coef.rename_axis("descriptor").reset_index().assign(abs_coefficient=lambda df: df.lasso_coefficient.abs()).sort_values(
        "abs_coefficient", ascending=False
    ).to_csv(output_dir / "lasso_coefficients_train.csv", index=False)
    lasso_selected = lasso_coef.index[lasso_coef.abs() > 1e-12].tolist()

    forest = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=args.random_state, n_jobs=1)
    forest.fit(selection_matrix, y_train)
    importance = pd.Series(forest.feature_importances_, index=selection_matrix.columns, name="extra_trees_importance").sort_values(ascending=False)
    importance.rename_axis("descriptor").reset_index().to_csv(output_dir / "extra_trees_importance_train.csv", index=False)

    selected_frame = pd.DataFrame({
        "descriptor": selection_matrix.columns,
        "kept_after_variance_correlation_vif": True,
        "selected_by_mutual_information": selection_matrix.columns.isin(mi_selected),
        "selected_by_lasso": selection_matrix.columns.isin(lasso_selected),
    })
    selected_frame["recommended_for_first_mlp"] = selected_frame["selected_by_mutual_information"] & selected_frame["selected_by_lasso"]
    if not selected_frame["recommended_for_first_mlp"].any():
        selected_frame["recommended_for_first_mlp"] = selected_frame["selected_by_mutual_information"]
    selected_frame.to_csv(output_dir / "feature_selection_summary_train.csv", index=False)

    # Held-out evaluation of the training-only selected linear baseline.
    recommended = selected_frame.loc[selected_frame["recommended_for_first_mlp"], "descriptor"].tolist()
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train[recommended], y_train)
    test_rmse = float(np.sqrt(np.mean((model.predict(x_test[recommended]) - y_test) ** 2)))
    summary = {
        "n_train": int(len(train)), "n_test": int(len(test)), "candidate_descriptors": len(DESCRIPTOR_NAMES),
        "after_variance": len(variance_kept), "after_correlation": len(corr_kept), "after_vif": len(vif_kept),
        "pca_components_for_95pct_variance": int(np.searchsorted(cumulative, 0.95) + 1),
        "mutual_information_best_k": best_k, "mi_selected": mi_selected, "lasso_selected": lasso_selected,
        "recommended_for_first_mlp": recommended, "held_out_ridge_rmse_log10_chiN": test_rmse,
        "correlation_threshold": args.correlation_threshold, "vif_threshold": args.vif_threshold,
        "variance_threshold": args.variance_threshold,
        "random_state": args.random_state,
        "split_indices": str(args.split_indices) if args.split_indices is not None else None,
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--train-matrix-csv", type=Path,
                        help="Reuse a previously computed descriptor matrix instead of extracting YML files")
    parser.add_argument("--test-matrix-csv", type=Path,
                        help="Reuse a previously computed held-out descriptor matrix")
    parser.add_argument("--split-indices", type=Path,
                        help="Optional NPZ whose train_index selects a training-only fold from the train matrix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="debug-only cap applied independently to each split")
    parser.add_argument("--variance-threshold", type=float, default=1e-12)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=20260805)
    args = parser.parse_args()
    if args.train_matrix_csv is not None or args.test_matrix_csv is not None:
        if args.train_matrix_csv is None or args.test_matrix_csv is None:
            parser.error("--train-matrix-csv and --test-matrix-csv must be supplied together")
        train = pd.read_csv(args.train_matrix_csv)
        test = pd.read_csv(args.test_matrix_csv)
    else:
        if args.train_dir is None or args.test_dir is None:
            parser.error("provide either both data directories or both precomputed matrix CSV files")
        train = extract_split(args.train_dir, "train", args.workers, args.limit)
        test = extract_split(args.test_dir, "test", args.workers, args.limit)
    if args.split_indices is not None:
        split = np.load(args.split_indices)
        if "train_index" not in split:
            parser.error(f"{args.split_indices} does not contain train_index")
        train_index = split["train_index"]
        if len(np.unique(train_index)) != len(train_index) or train_index.min() < 0 or train_index.max() >= len(train):
            parser.error("train_index contains duplicates or out-of-range values")
        train = train.iloc[train_index].reset_index(drop=True)
    run_analysis(train, test, args.output_dir, args)


if __name__ == "__main__":
    main()
