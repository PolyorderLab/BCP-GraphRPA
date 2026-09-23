#!/usr/bin/env python3
"""Re-run the six ABTAB2 RPA--GNN diagnostics with the final C2 checkpoint.

The RPA calculations, explainer protocol (45 conditions, three node relabelings,
and ten mask initialisations), probes, and matched-path counterfactuals are
identical to the boss-presentation study.  Only the frozen GNN is replaced by
the final C2 learnable-node virtual-node model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

ROOT = Path("/home/zxy/project/TAPD")
FINALMODEL = ROOT / "baselinemodels/pub0818/gnn/finalmodel"
SOURCE = ROOT / "mainmodel/interpretability/abtab2_rpa_gnn/run_abtab2_rpa_gnn.py"
DEFAULT_RUN = FINALMODEL / "C2_nested_train__80k__seed202601"
DEFAULT_OUTPUT = FINALMODEL / "interpretability/results"
PROBE_SOURCE = Path(__file__).resolve().parent / "validate_abtab2_layerwise_leave_one_rho.py"


def _load_source():
    spec = importlib.util.spec_from_file_location("abtab2_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source protocol: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_source()


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("abtab2_pca_ridge_probe", PROBE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probe implementation: {PROBE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe_protocol = _load_probe_module()


def build_final_model(device: torch.device):
    from baselinemodels.pub0818.gnn.finalmodel.transfer.scripts.train_c2_kstar_transfer import build_model

    return build_model(device)


def make_finalmodel_graph(f_a: float, rho: float, assignment=None) -> Data:
    """Build C2's exact zero-node/full-edge/one-virtual-node graph format."""
    blocks = study.blocks_for(f_a, rho, assignment=assignment)
    node_order: list[str] = []
    for block in blocks:
        for node in (block["u"], block["v"]):
            if node not in node_order:
                node_order.append(node)
    node_id = {name: idx for idx, name in enumerate(node_order)}
    edges, attrs = [], []
    for block in blocks:
        u, v = node_id[block["u"]], node_id[block["v"]]
        attr = (block["f"], 1.0 if block["species"] == "A" else 0.0)
        edges.extend(((u, v), (v, u)))
        attrs.extend((attr, attr))
    # The source dataset keeps a fixed 32-row ``F_pad`` tensor.  C2 was trained
    # after virtual-node wrapping, hence its positional embedding always sees
    # the 32 padded rows plus virtual node 32, even for a six-junction graph.
    padded_nodes = 32
    virtual = padded_nodes
    for node in range(padded_nodes):
        edges.extend(((virtual, node), (node, virtual)))
        attrs.extend(((0.0, 0.0), (0.0, 0.0)))
    data = Data(
        x=torch.zeros((padded_nodes + 1, 3), dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        batch=torch.zeros(padded_nodes + 1, dtype=torch.long),
        ptr=torch.tensor([0, padded_nodes + 1], dtype=torch.long),
    )
    data.num_physical_undirected = len(blocks)
    data.num_physical_directed = 2 * len(blocks)
    data.edge_roles = [block["label"] for block in blocks]
    return data


def layer_embeddings_final(model, data, device):
    """Pool the positional embedding and each C2 GINE layer's node states."""
    data = data.to(device)
    encoder = model.encoder
    node_count = int((data.ptr[1:] - data.ptr[:-1]).item())
    x = encoder.node_embedding[:node_count]
    pooled = [global_mean_pool(x, data.batch).detach().cpu().numpy().reshape(-1)]
    for idx, conv in enumerate(encoder.gine.convs):
        x = conv(x, data.edge_index, edge_attr=encoder.gine.edge_lin[idx](data.edge_attr))
        if idx != len(encoder.gine.convs) - 1:
            x = encoder.gine.relu(x)
        pooled.append(global_mean_pool(x, data.batch).detach().cpu().numpy().reshape(-1))
    return pooled


def permute_finalmodel_nodes(data, seed):
    """Relabel the six non-padding junctions; preserve C2's padded positions."""
    rng = np.random.default_rng(seed)
    physical = np.arange(data.num_nodes - 1)
    perm = rng.permutation(physical)
    old_to_new = np.arange(data.num_nodes)
    old_to_new[physical] = perm
    out = data.clone()
    out.edge_index = torch.as_tensor(old_to_new[data.edge_index.cpu().numpy()], dtype=torch.long)
    return out


def run_explainer_subset(model, device, rpa_by_key, out: Path, condition_indices: set[int]):
    """Run a deterministic slice of the 45-condition mask protocol."""
    rows = []
    sample_grid = [(fa, rho) for fa in np.round(np.linspace(0.1, 0.9, 9), 2)
                   for rho in (0.5, 1.0, 2.0, 5.0, 10.0)]
    for sample_idx, (fa, rho) in enumerate(sample_grid):
        if sample_idx not in condition_indices:
            continue
        print(f"GNNExplainer condition {sample_idx + 1}/45: f_A={fa}, rho={rho}", flush=True)
        rpa = rpa_by_key[f"{fa:.6f}_{rho:.6f}"]
        base_data = make_finalmodel_graph(float(fa), rho)
        for permutation in range(3):
            data = permute_finalmodel_nodes(base_data, 9100 + permutation * 101 + int(100 * rho))
            for seed in range(10):
                masks, deletion, base = study.explain_one(model, data, device, 830000 + seed, epochs=100)
                for idx, edge in enumerate(rpa["edge_influence"]):
                    rows.append({"f_A": fa, "rho": rho, "permutation": permutation, "explainer_seed": seed,
                                 "edge_index": idx, "edge": edge["edge"], "rpa_I_self": edge["I_self"],
                                 "rpa_I_junction": edge["I_junction"], "rpa_I_separated": edge["I_separated"],
                                 "rpa_I_total": edge["I_total"], "mask": float(masks[idx]),
                                 "deletion_effect": float(deletion[idx]), "base_prediction": base})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "abtab2_gnnexplainer_edges.csv", index=False)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--condition-indices", default=None,
                        help="Comma-separated zero-based indices from the fixed 45-condition grid; enables explainer-only shard mode.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cpu"):
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
    device = torch.device(args.device)
    model = build_final_model(device)
    model.load_state_dict(torch.load(args.run_dir / "best_model.pth", map_location=device, weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Reuse the established experimental protocol while substituting C2-specific
    # graph construction and layer accessors.
    study.make_graph = make_finalmodel_graph
    study.layer_embeddings = layer_embeddings_final
    study.permute_nodes = permute_finalmodel_nodes
    study.TOPOLOGY = "A1B1T(A2,B2,B3)"

    rpa_rows, edge_rows, rpa_by_key = [], [], {}
    for fa in np.round(np.linspace(0.1, 0.9, 9), 2):
        for rho in (0.5, 1.0, 2.0, 5.0, 10.0):
            result = study.rpa_sample(float(fa), rho)
            rpa_by_key[f"{fa:.6f}_{rho:.6f}"] = result
            rpa_rows.append({k: v for k, v in result.items() if k not in {"edge_influence", "separated_pairs"}})
            edge_rows.extend({"f_A": fa, "rho": rho, **row} for row in result["edge_influence"])
    import pandas as pd
    pd.DataFrame(rpa_rows).to_csv(args.output_dir / "abtab2_rpa_scan.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(args.output_dir / "abtab2_rpa_edge_influence.csv", index=False)
    (args.output_dir / "abtab2_rpa_pair_manifest.json").write_text(json.dumps({
        "topology": study.TOPOLOGY,
        "separated_pairs": rpa_by_key["0.500000_1.000000"]["separated_pairs"],
    }, indent=2), encoding="utf-8")
    config = {
        "topology": study.TOPOLOGY, "checkpoint": str(args.run_dir.resolve()), "device": str(device),
        "model_condition": "C2_virtualnode_learnable_node_full_edge",
        "architecture": "LearnableNodeGINEEncoder(hidden_dim=512, num_layers=5, virtual node)",
        "selection": "seed202601: lowest leakage-free development-validation MAPE among finalmodel checkpoints (0.36665%)",
        "f_A": [float(v) for v in np.round(np.linspace(0.1, 0.9, 9), 2)], "rho": [0.5, 1, 2, 5, 10],
        "explainer": "3 physical-node relabelings x 10 mask seeds x 100 optimisation epochs",
    }
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.condition_indices is not None:
        indices = {int(value) for value in args.condition_indices.split(",") if value.strip()}
        n_rows = run_explainer_subset(model, device, rpa_by_key, args.output_dir, indices)
        print(json.dumps({"mode": "explainer_shard", "condition_indices": sorted(indices), "edge_rows": n_rows}))
        return
    explainer = study.run_explainer(model, device, rpa_by_key, args.output_dir)
    # Generate all frozen-C2 layer embeddings, then replace the legacy 512-D
    # direct Ridge scores with the fixed-capacity, leakage-free PCA--Ridge probe.
    study.run_probe(model, device, rpa_by_key, args.output_dir)
    probe = probe_protocol.fit_probe(args.output_dir)
    matched_path = study.run_matched_path(model, device, args.output_dir)
    summary = {**config, "explainer": explainer, "probe": probe, "matched_path": matched_path}
    (args.output_dir / "abtab2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"explainer": explainer, "probe_rows": len(probe), "path_rows": len(matched_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
