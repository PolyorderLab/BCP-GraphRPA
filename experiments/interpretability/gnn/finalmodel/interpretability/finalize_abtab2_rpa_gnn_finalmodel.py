#!/usr/bin/env python3
"""Merge parallel C2 ABTAB2 mask shards and finish probes/path diagnostics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_abtab2_rpa_gnn_finalmodel.py"
OUT = HERE / "results"
SHARDS = HERE / "shards"

spec = importlib.util.spec_from_file_location("final_rerun", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load runner: {RUNNER}")
rerun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerun)


def main() -> None:
    pieces = sorted(SHARDS.glob("*/abtab2_gnnexplainer_edges.csv"))
    if len(pieces) != 4:
        raise RuntimeError(f"Expected four explainer shards, found {len(pieces)}")
    edges = pd.concat([pd.read_csv(path) for path in pieces], ignore_index=True)
    if len(edges) != 45 * 3 * 10 * 5:
        raise RuntimeError(f"Unexpected explainer edge-row count: {len(edges)}")
    expected = {(float(fa), float(rho)) for fa in np.round(np.linspace(0.1, 0.9, 9), 2)
                for rho in (0.5, 1.0, 2.0, 5.0, 10.0)}
    if set(zip(edges.f_A, edges.rho)) != expected:
        raise RuntimeError("Merged explainer shards do not cover the exact 9x5 grid")
    OUT.mkdir(parents=True, exist_ok=True)
    edges.to_csv(OUT / "abtab2_gnnexplainer_edges.csv", index=False)
    agg = edges.groupby(["f_A", "rho", "edge_index", "edge"], as_index=False).agg(
        mask_mean=("mask", "mean"), mask_sd=("mask", "std"), deletion_mean=("deletion_effect", "mean"),
        rpa_I_self=("rpa_I_self", "first"), rpa_I_junction=("rpa_I_junction", "first"),
        rpa_I_separated=("rpa_I_separated", "first"), rpa_I_total=("rpa_I_total", "first"),
    )
    agg.to_csv(OUT / "abtab2_gnnexplainer_edge_summary.csv", index=False)
    pairs = list(zip(np.abs(agg.rpa_I_total), agg.deletion_mean))
    explainer = {"n_sample_edge": len(pairs), "spearman_abs_rpa_vs_deletion": float(spearmanr(*zip(*pairs)).statistic)}

    device = torch.device("cpu")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    model = rerun.build_final_model(device)
    model.load_state_dict(torch.load(rerun.DEFAULT_RUN / "best_model.pth", map_location=device, weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rerun.study.make_graph = rerun.make_finalmodel_graph
    rerun.study.layer_embeddings = rerun.layer_embeddings_final
    rpa_by_key = {f"{fa:.6f}_{rho:.6f}": rerun.study.rpa_sample(float(fa), rho)
                  for fa in np.round(np.linspace(0.1, 0.9, 9), 2)
                  for rho in (0.5, 1.0, 2.0, 5.0, 10.0)}
    probe = rerun.study.run_probe(model, device, rpa_by_key, OUT)
    matched_path = rerun.study.run_matched_path(model, device, OUT)
    config = json.loads((pieces[0].parent / "experiment_config.json").read_text())
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "abtab2_summary.json").write_text(json.dumps(
        {**config, "explainer": explainer, "probe": probe, "matched_path": matched_path}, indent=2), encoding="utf-8")
    print(json.dumps({"explainer": explainer, "probe_rows": len(probe), "path_rows": len(matched_path)}))


if __name__ == "__main__":
    main()
