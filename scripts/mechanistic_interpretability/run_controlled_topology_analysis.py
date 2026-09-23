#!/usr/bin/env python3
"""Recompute the 45-condition controlled-topology RPA--GNN analysis."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ORIGINAL = HERE / "original"
DEFAULT_RUN = CODE / "checkpoints/gnn/seed202601"
DEFAULT_OUTPUT = CODE / "data/check_run/mechanistic_controlled"

sys.path[:0] = [str(HERE), str(CODE)]
from experiment_support import build_model  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load_module("abtab2_protocol", ORIGINAL / "protocol/run_abtab2_rpa_gnn.py")
probe_protocol = load_module(
    "abtab2_pca_ridge_probe",
    ORIGINAL / "finalmodel/validate_abtab2_layerwise_leave_one_rho.py",
)


def make_graph(f_a: float, rho: float, assignment=None) -> Data:
    blocks = study.blocks_for(f_a, rho, assignment=assignment)
    node_order = list(dict.fromkeys(node for block in blocks for node in (block["u"], block["v"])))
    node_id = {name: index for index, name in enumerate(node_order)}
    edges, attrs = [], []
    for block in blocks:
        left, right = node_id[block["u"]], node_id[block["v"]]
        attr = (block["f"], 1.0 if block["species"] == "A" else 0.0)
        edges.extend(((left, right), (right, left)))
        attrs.extend((attr, attr))
    virtual = 32
    for node in range(32):
        edges.extend(((virtual, node), (node, virtual)))
        attrs.extend(((0.0, 0.0), (0.0, 0.0)))
    data = Data(
        x=torch.zeros((33, 3), dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        batch=torch.zeros(33, dtype=torch.long),
        ptr=torch.tensor([0, 33], dtype=torch.long),
    )
    data.num_physical_undirected = len(blocks)
    data.num_physical_directed = 2 * len(blocks)
    data.edge_roles = [block["label"] for block in blocks]
    return data


def layer_embeddings(model, data, device):
    data = data.to(device)
    encoder = model.encoder
    x = encoder.node_embedding[: data.num_nodes]
    pooled = [global_mean_pool(x, data.batch).detach().cpu().numpy().reshape(-1)]
    with torch.no_grad():
        for index, convolution in enumerate(encoder.gine.convs):
            x = convolution(x, data.edge_index, edge_attr=encoder.gine.edge_lin[index](data.edge_attr))
            if index != len(encoder.gine.convs) - 1:
                x = encoder.gine.relu(x)
            pooled.append(global_mean_pool(x, data.batch).cpu().numpy().reshape(-1))
    return pooled


def permute_nodes(data, seed):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(data.num_nodes - 1)
    old_to_new = np.arange(data.num_nodes)
    old_to_new[:-1] = permutation
    output = data.clone()
    output.edge_index = torch.as_tensor(old_to_new[data.edge_index.cpu().numpy()], dtype=torch.long)
    return output


def write_rpa_grid(output: Path):
    rpa_rows, edge_rows, by_key = [], [], {}
    for f_a in np.round(np.linspace(0.1, 0.9, 9), 2):
        for rho in (0.5, 1.0, 2.0, 5.0, 10.0):
            result = study.rpa_sample(float(f_a), rho)
            by_key[f"{f_a:.6f}_{rho:.6f}"] = result
            rpa_rows.append({k: v for k, v in result.items() if k not in {"edge_influence", "separated_pairs"}})
            edge_rows.extend({"f_A": f_a, "rho": rho, **row} for row in result["edge_influence"])
    pd.DataFrame(rpa_rows).to_csv(output / "abtab2_rpa_scan.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output / "abtab2_rpa_edge_influence.csv", index=False)
    (output / "abtab2_rpa_pair_manifest.json").write_text(
        json.dumps({
            "topology": study.TOPOLOGY,
            "separated_pairs": by_key["0.500000_1.000000"]["separated_pairs"],
        }, indent=2),
        encoding="utf-8",
    )
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rpa-only", action="store_true", help="Generate the 45-condition RPA grid and stop.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rpa_by_key = write_rpa_grid(args.output_dir)
    if args.rpa_only:
        print(json.dumps({"topology": study.TOPOLOGY, "conditions": len(rpa_by_key)}))
        return

    device = torch.device(args.device)
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    model = build_model(metrics, device)
    model.load_state_dict(
        torch.load(args.run_dir / "best_model.pth", map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    study.make_graph = make_graph
    study.layer_embeddings = layer_embeddings
    study.permute_nodes = permute_nodes
    study.TOPOLOGY = "A1B1T(A2,B2,B3)"
    explainer = study.run_explainer(model, device, rpa_by_key, args.output_dir)
    study.run_probe(model, device, rpa_by_key, args.output_dir)
    probe = probe_protocol.fit_probe(args.output_dir)
    matched_path = study.run_matched_path(model, device, args.output_dir)
    summary = {
        "topology": study.TOPOLOGY,
        "checkpoint": str(args.run_dir.resolve()),
        "device": str(device),
        "f_A": [float(value) for value in np.round(np.linspace(0.1, 0.9, 9), 2)],
        "rho": [0.5, 1.0, 2.0, 5.0, 10.0],
        "explainer": explainer,
        "probe": probe,
        "matched_path": matched_path,
    }
    (args.output_dir / "abtab2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"conditions": len(rpa_by_key), "output": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
