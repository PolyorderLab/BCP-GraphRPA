#!/usr/bin/env python3
"""Draw the two 9x5 ABTAB2 diagnostics from the final-C2 rerun."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/home/zxy/project/TAPD")
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "mainmodel/interpretability/abtab2_rpa_gnn/plot_abtab2_grid_heatmaps_mpltex.py"

spec = importlib.util.spec_from_file_location("abtab2_grid_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load plot source: {SOURCE}")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)
plots.RESULTS = HERE / "results"
figure_dir = HERE.parent / "figure" / "c2_model" / "interpretability_ablation"
figure_dir.mkdir(parents=True, exist_ok=True)
original_save_dual = plots.save_dual
plots.save_dual = lambda fig, stem, **kwargs: original_save_dual(fig, figure_dir / Path(stem).name, **kwargs)

if __name__ == "__main__":
    metrics = plots.build_grid_metrics()
    plots.plot_heatmap(metrics, "RPA_deletion", "abtab2_rpa_deletion_grid")
    plots.plot_heatmap(metrics, "matched_path_L5", "abtab2_matched_path_grid")
    print(f"saved two 9x5 grids to {figure_dir}")
