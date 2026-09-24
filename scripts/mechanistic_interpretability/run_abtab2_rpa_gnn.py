#!/usr/bin/env python3
"""RPA--GNN bridge for A1-B1-T(A2,B2,B3).

This experiment intentionally keeps the trained GNN frozen.  It combines:
1) block/pair-resolved RPA existing-correlation influence;
2) a constrained GNNExplainer-style physical-edge mask;
3) layerwise RPA concept probes;
4) fixed-edge-multiset matched counterfactuals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mainmodel.scripts.train_single_virtualnode_undirected_freeend_branch_degree import build_model  # noqa: E402

RUN03 = ROOT / "checkpoints/gnn/seed202603"
OUT_DEFAULT = ROOT / "data/mechanistic_interpretability/rpa"

# A1-B1-T(A2,B2,B3): A2, B2 and B3 all attach to the B1-side junction.
TOPOLOGY = "A1B1T(A2,B2,B3)"
BLUEPRINT = [
    ("A1", "A", "n0", "n1"),
    ("B1", "B", "n1", "T"),
    ("A2", "A", "T", "n2"),
    ("B2", "B", "T", "n3"),
    ("B3", "B", "T", "n4"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def blocks_for(f_a: float, rho: float, assignment=None):
    b_total = 1.0 - f_a
    b1 = b_total * rho / (rho + 2.0)
    bs = b_total / (rho + 2.0)
    base = [
        {"label": "A1", "species": "A", "u": "n0", "v": "n1", "f": f_a / 2.0},
        {"label": "B1", "species": "B", "u": "n1", "v": "T", "f": b1},
        {"label": "A2", "species": "A", "u": "T", "v": "n2", "f": f_a / 2.0},
        {"label": "B2", "species": "B", "u": "T", "v": "n3", "f": bs},
        {"label": "B3", "species": "B", "u": "T", "v": "n4", "f": bs},
    ]
    if assignment is None:
        return base
    result = []
    for idx, ((species, length), template) in enumerate(zip(assignment, base)):
        item = dict(template)
        item["species"] = species
        item["f"] = float(length)
        item["label"] = f"e{idx+1}_{species}"
        result.append(item)
    return result


def g_debye(x: float, fraction: float) -> float:
    z = x * fraction
    if abs(z) < 1e-4:
        # g_D = f^2 [1 - z/3 + z^2/12 - ...]
        return fraction * fraction * (1.0 - z / 3.0 + z * z / 12.0 - z**3 / 60.0)
    return 2.0 * (math.exp(-z) + z - 1.0) / (x * x)


def h_func(x: float, fraction: float) -> float:
    return -math.expm1(-x * fraction) / x


def block_adjacency(blocks):
    adjacency = {i: [] for i in range(len(blocks))}
    for i, left in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            if {left["u"], left["v"]} & {blocks[j]["u"], blocks[j]["v"]}:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def shortest_block_path(adjacency, start, goal):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return path
        for nxt in adjacency[path[-1]]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    raise RuntimeError("Disconnected block graph")


def phi_and_components(q: float, blocks):
    """Return Phi(q), S components, and pair records with edge allocation paths."""
    x = q * q
    adjacency = block_adjacency(blocks)
    comp = {f"S_{pair}_{rel}": 0.0 for pair in ("AA", "BB", "AB")
            for rel in ("self", "adjacent", "separated")}
    pairs = []
    for i, left in enumerate(blocks):
        # self terms
        pair = left["species"] * 2
        s = g_debye(x, left["f"])
        comp[f"S_{pair}_self"] += s
        pairs.append({"i": i, "j": i, "pair": pair, "relation": "self",
                      "s": s, "multiplicity": 1, "path": [i], "d": 0.0})
        for j in range(i + 1, len(blocks)):
            right = blocks[j]
            if left["species"] == right["species"]:
                pair = left["species"] * 2
                multiplicity = 2  # ordered pair (i,j) and (j,i) in S_ii
            else:
                pair = "AB"
                multiplicity = 1
            if j in adjacency[i]:
                relation = "adjacent"
                path = [i, j]
                d_value = 0.0
                s = h_func(x, left["f"]) * h_func(x, right["f"])
            else:
                relation = "separated"
                path = shortest_block_path(adjacency, i, j)
                d_value = sum(blocks[k]["f"] for k in path[1:-1])
                s = h_func(x, left["f"]) * math.exp(-x * d_value) * h_func(x, right["f"])
            comp[f"S_{pair}_{relation}"] += multiplicity * s
            pairs.append({"i": i, "j": j, "pair": pair, "relation": relation,
                          "s": s, "multiplicity": multiplicity, "path": path, "d": d_value})
    for pair in ("AA", "BB", "AB"):
        comp[f"S_{pair}"] = sum(comp[f"S_{pair}_{rel}"] for rel in ("self", "adjacent", "separated"))
    saa, sbb, sab = comp["S_AA"], comp["S_BB"], comp["S_AB"]
    t = saa + sbb + 2.0 * sab
    w = saa * sbb - sab * sab
    phi = t / (2.0 * w)
    comp.update({"T": t, "W": w, "Phi": phi})
    return phi, comp, pairs


def rpa_sample(f_a: float, rho: float):
    blocks = blocks_for(f_a, rho)
    objective = lambda q: phi_and_components(float(q), blocks)[0]
    coarse = np.linspace(0.2, 7.0, 500)
    values = np.asarray([objective(q) for q in coarse])
    idx = int(np.argmin(values))
    lo = float(coarse[max(0, idx - 2)])
    hi = float(coarse[min(len(coarse) - 1, idx + 2)])
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
    q_star = float(result.x)
    phi, comp, pairs = phi_and_components(q_star, blocks)
    saa, sbb, sab = comp["S_AA"], comp["S_BB"], comp["S_AB"]
    t, w = comp["T"], comp["W"]
    dphi_dbb = (w - t * saa) / (2.0 * w * w)
    dphi_dab = (w + t * sab) / (w * w)
    # Allocate each pair's existing correlation equally over the blocks in its path.
    edge = {i: {"self": 0.0, "adjacent": 0.0, "separated": 0.0} for i in range(len(blocks))}
    for item in pairs:
        grad = dphi_dab if item["pair"] == "AB" else (dphi_dbb if item["pair"] == "BB" else (w - t * sbb) / (2.0 * w * w))
        allocated = item["multiplicity"] * item["s"] / len(item["path"])
        for block_idx in item["path"]:
            edge[block_idx][item["relation"]] += grad * allocated
    rows = []
    for idx, block in enumerate(blocks):
        vals = edge[idx]
        rows.append({"edge_index": idx, "edge": block["label"], "species": block["species"],
                     "f": block["f"], "I_self": vals["self"], "I_junction": vals["adjacent"],
                     "I_separated": vals["separated"], "I_total": sum(vals.values())})
    return {"f_A": f_a, "rho": rho, "q_star": q_star, "x_star": q_star * q_star,
            "chiN_star": phi, "S_AA": saa, "S_BB": sbb, "S_AB": sab,
            "S_AA_self": comp["S_AA_self"], "S_AA_adjacent": comp["S_AA_adjacent"],
            "S_AA_separated": comp["S_AA_separated"], "S_BB_self": comp["S_BB_self"],
            "S_BB_adjacent": comp["S_BB_adjacent"], "S_BB_separated": comp["S_BB_separated"],
            "S_AB_self": comp["S_AB_self"], "S_AB_adjacent": comp["S_AB_adjacent"],
            "S_AB_separated": comp["S_AB_separated"], "dPhi_dSBB": dphi_dbb,
            "dPhi_dSAB": dphi_dab, "edge_influence": rows,
            "separated_pairs": [{"left": blocks[p["i"]]["label"], "right": blocks[p["j"]]["label"],
                                 "d": p["d"]} for p in pairs if p["relation"] == "separated"]}


def make_graph(f_a, rho, assignment=None):
    blocks = blocks_for(f_a, rho, assignment=assignment)
    node_order = []
    for block in blocks:
        for node in (block["u"], block["v"]):
            if node not in node_order:
                node_order.append(node)
    node_id = {name: idx for idx, name in enumerate(node_order)}
    degree = np.zeros(32, dtype=np.float32)
    for block in blocks:
        degree[node_id[block["u"]]] += 1.0
        degree[node_id[block["v"]]] += 1.0
    x = np.stack([degree == 1, degree > 1, degree], axis=1).astype(np.float32)
    edges, attrs = [], []
    for block in blocks:
        u, v = node_id[block["u"]], node_id[block["v"]]
        attr = (block["f"], 1.0 if block["species"] == "A" else 0.0)
        edges.extend([(u, v), (v, u)])
        attrs.extend([attr, attr])
    virtual = 32
    for node in range(32):
        edges.extend([(virtual, node), (node, virtual)])
        attrs.extend([(0.0, 0.0), (0.0, 0.0)])
    data = Data(x=torch.tensor(np.vstack([x, np.zeros((1, 3), np.float32)])),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(attrs, dtype=torch.float32),
                batch=torch.zeros(33, dtype=torch.long))
    data.num_physical_undirected = len(blocks)
    data.num_physical_directed = 2 * len(blocks)
    data.edge_roles = [b["label"] for b in blocks]
    return data


def model_prediction(model, data, device, embedding=False):
    data = data.to(device)
    with torch.no_grad():
        z = model.encoder(data)
        log_y = model.decoder(z).reshape(())
        y = torch.pow(torch.tensor(10.0, device=device), log_y)
    if embedding:
        return float(y.cpu()), z.cpu().numpy().reshape(-1)
    return float(y.cpu())


def layer_embeddings(model, data, device):
    data = data.to(device)
    x = data.x
    pooled = [global_mean_pool(x, data.batch).detach().cpu().numpy().reshape(-1)]
    with torch.no_grad():
        for idx, conv in enumerate(model.encoder.convs):
            x = conv(x, data.edge_index, edge_attr=model.encoder.edge_lin[idx](data.edge_attr))
            if idx != len(model.encoder.convs) - 1:
                x = model.encoder.relu(x)
            pooled.append(global_mean_pool(x, data.batch).cpu().numpy().reshape(-1))
    return pooled


def permute_nodes(data, seed):
    rng = np.random.default_rng(seed)
    real = np.arange(6)
    perm = rng.permutation(real)
    old_to_new = np.arange(33)
    old_to_new[real] = perm
    out = data.clone()
    out.x = data.x.clone()
    out.x[real] = data.x[perm]
    out.edge_index = torch.as_tensor(old_to_new[data.edge_index.cpu().numpy()], dtype=torch.long)
    return out


def masked_prediction(model, data, arm_values, device):
    data = data.to(device)
    physical = torch.as_tensor(arm_values, dtype=torch.float32, device=device).repeat_interleave(2)
    fixed = torch.ones(data.edge_index.size(1) - physical.numel(), device=device)
    mask = torch.cat([physical, fixed])
    set_masks(model, mask, data.edge_index, apply_sigmoid=False)
    try:
        z = model.encoder(data)
        log_y = model.decoder(z).reshape(())
        return torch.pow(torch.tensor(10.0, device=device), log_y)
    finally:
        clear_masks(model)


def explain_one(model, data, device, seed, epochs=100):
    set_seed(seed)
    base = model_prediction(model, data, device)
    logits = torch.nn.Parameter(torch.randn(data.num_physical_undirected, device=device) * 0.1)
    optimizer = torch.optim.Adam([logits], lr=0.05)
    target = torch.tensor(base, dtype=torch.float32, device=device)
    for _ in range(epochs):
        optimizer.zero_grad()
        values = logits.sigmoid()
        pred = masked_prediction(model, data, values, device)
        entropy = -values * torch.log(values + 1e-8) - (1 - values) * torch.log(1 - values + 1e-8)
        loss = ((pred - target) / torch.clamp(target.abs(), min=1.0)) ** 2 + 0.015 * values.mean() + 0.10 * entropy.mean()
        loss.backward()
        optimizer.step()
    values = logits.sigmoid().detach().cpu().numpy()
    deletion = []
    for idx in range(len(values)):
        leave = np.ones(len(values), dtype=np.float32)
        leave[idx] = 0.0
        deletion.append(abs(base - float(masked_prediction(model, data, leave, device).detach().cpu())))
    return values, np.asarray(deletion), base


def run_explainer(model, device, rpa_by_key, out):
    rows = []
    sample_grid = [(fa, rho) for fa in np.round(np.linspace(0.1, 0.9, 9), 2)
                   for rho in (0.5, 1.0, 2.0, 5.0, 10.0)]
    for sample_idx, (fa, rho) in enumerate(sample_grid, start=1):
        print(f"GNNExplainer sample {sample_idx}/{len(sample_grid)}: f_A={fa}, rho={rho}", flush=True)
        key = f"{fa:.6f}_{rho:.6f}"
        rpa = rpa_by_key[key]
        base_data = make_graph(fa, rho)
        for permutation in range(3):
            data = permute_nodes(base_data, 9100 + permutation * 101 + int(100 * rho))
            for seed in range(10):
                masks, deletion, base = explain_one(model, data, device, 830000 + seed, epochs=100)
                for idx, edge in enumerate(rpa["edge_influence"]):
                    rows.append({"f_A": fa, "rho": rho, "permutation": permutation, "explainer_seed": seed,
                                 "edge_index": idx, "edge": edge["edge"], "rpa_I_self": edge["I_self"],
                                 "rpa_I_junction": edge["I_junction"], "rpa_I_separated": edge["I_separated"],
                                 "rpa_I_total": edge["I_total"], "mask": float(masks[idx]),
                                 "deletion_effect": float(deletion[idx]), "base_prediction": base})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "abtab2_gnnexplainer_edges.csv", index=False)
    agg = frame.groupby(["f_A", "rho", "edge_index", "edge"], as_index=False).agg(
        mask_mean=("mask", "mean"), mask_sd=("mask", "std"), deletion_mean=("deletion_effect", "mean"),
        rpa_I_self=("rpa_I_self", "first"), rpa_I_junction=("rpa_I_junction", "first"),
        rpa_I_separated=("rpa_I_separated", "first"), rpa_I_total=("rpa_I_total", "first"))
    agg.to_csv(out / "abtab2_gnnexplainer_edge_summary.csv", index=False)
    pairs = []
    for group in agg.groupby(["f_A", "rho"]):
        g = group[1]
        pairs.extend(zip(np.abs(g.rpa_I_total), g.deletion_mean))
    summary = {"n_sample_edge": len(pairs), "spearman_abs_rpa_vs_deletion": float(spearmanr(*zip(*pairs)).statistic)}
    with (out / "abtab2_gnnexplainer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def run_probe(model, device, rpa_by_key, out):
    records = []
    rho_values = (0.5, 1.0, 2.0, 5.0, 10.0)
    fa_values = tuple(np.round(np.linspace(0.1, 0.9, 9), 2))
    for fa in fa_values:
        for rho in rho_values:
            key = f"{fa:.6f}_{rho:.6f}"
            data = make_graph(fa, rho)
            layers = layer_embeddings(model, data, device)
            row = {"f_A": fa, "rho": rho, "q_star": rpa_by_key[key]["q_star"],
                   "chiN_star": rpa_by_key[key]["chiN_star"],
                   "S_AA": rpa_by_key[key]["S_AA"], "S_BB": rpa_by_key[key]["S_BB"],
                   "S_AB": rpa_by_key[key]["S_AB"]}
            for idx, value in enumerate(layers):
                row[f"embedding_L{idx}"] = json.dumps(value.tolist(), separators=(",", ":"))
            records.append(row)
    frame = pd.DataFrame(records)
    concepts = ["S_AA", "S_BB", "S_AB", "q_star", "chiN_star"]
    groups = frame.rho.to_numpy()
    rows = []
    for layer in range(6):
        x = np.stack(frame[f"embedding_L{layer}"].map(json.loads))
        for concept in concepts:
            y = frame[concept].to_numpy()
            pred = np.zeros_like(y)
            for train_idx, test_idx in GroupKFold(n_splits=5).split(x, y, groups=groups):
                pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                pipe.fit(x[train_idx], y[train_idx])
                pred[test_idx] = pipe.predict(x[test_idx])
            rows.append({"layer": layer, "concept": concept, "n": len(y), "R2_grouped_by_rho": float(r2_score(y, pred))})
    pd.DataFrame(rows).to_csv(out / "abtab2_layerwise_probe.csv", index=False)
    frame.to_csv(out / "abtab2_layerwise_embeddings.csv", index=False)
    return rows


def unique_assignments(fa, rho):
    base = blocks_for(fa, rho)
    labels = [(b["species"], round(b["f"], 12)) for b in base]
    return sorted(set(itertools.permutations(labels)))


def path_concepts(fa, rho, assignment):
    blocks = blocks_for(fa, rho, assignment=assignment)
    q = rpa_sample(fa, rho) if assignment is None else None
    # RPA for an assigned edge multiset and fixed topology.
    objective = lambda value: phi_and_components(float(value), blocks)[0]
    grid = np.linspace(0.2, 7.0, 350)
    vals = np.asarray([objective(v) for v in grid])
    idx = int(np.argmin(vals))
    res = minimize_scalar(objective, bounds=(float(grid[max(0, idx - 2)]), float(grid[min(len(grid)-1, idx + 2)])), method="bounded")
    _, comp, _ = phi_and_components(float(res.x), blocks)
    return np.array([comp["S_AA_separated"], comp["S_BB_separated"], comp["S_AB_separated"], float(res.x)], dtype=float)


def run_matched_path(model, device, out):
    rows = []
    group_pairs = []
    rng = np.random.default_rng(202603)
    group_id = 0
    for fa in tuple(np.round(np.linspace(0.1, 0.9, 9), 2)):
        for rho in (0.5, 1.0, 2.0, 5.0, 10.0):
            assignments = unique_assignments(fa, rho)
            embeddings = []
            concepts = []
            for assignment_idx, assignment in enumerate(assignments):
                data = make_graph(fa, rho, assignment=assignment)
                layers = layer_embeddings(model, data, device)
                embeddings.append(layers)
                concepts.append(path_concepts(fa, rho, assignment))
                rows.append({"group": group_id, "f_A": fa, "rho": rho, "assignment": assignment_idx,
                             **{f"embedding_L{l}": json.dumps(layers[l].tolist(), separators=(",", ":")) for l in range(6)},
                             "S_AA_separated": concepts[-1][0], "S_BB_separated": concepts[-1][1],
                             "S_AB_separated": concepts[-1][2], "q_star": concepts[-1][3]})
            n = len(assignments)
            all_pairs = list(itertools.combinations(range(n), 2))
            if len(all_pairs) > 300:
                all_pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), 300, replace=False)]
            for i, j in all_pairs:
                group_pairs.append({"group": group_id, "i": i, "j": j,
                                    "concept": np.linalg.norm(concepts[i] - concepts[j]),
                                    **{f"embedding_L{l}": np.linalg.norm(embeddings[i][l] - embeddings[j][l]) for l in range(6)}})
            group_id += 1
    pair_frame = pd.DataFrame(group_pairs)
    concept = pair_frame.concept.to_numpy()
    summary = []
    for layer in range(6):
        summary.append({"layer": layer, "n_pairs": len(pair_frame),
                        "spearman_embedding_vs_RPA_path": float(spearmanr(pair_frame[f"embedding_L{layer}"], concept).statistic)})
    pd.DataFrame(rows).to_csv(out / "abtab2_matched_path_embeddings.csv", index=False)
    pair_frame.to_csv(out / "abtab2_matched_path_pairs.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "abtab2_matched_path_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN03)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid excessive BLAS/OpenMP oversubscription on CPU-only workers.
    if (args.device or "").startswith("cpu") or not torch.cuda.is_available():
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = build_model(device)
    model.load_state_dict(torch.load(args.run_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"topology": TOPOLOGY, "checkpoint": str(args.run_dir), "device": str(device),
                   "f_A_explainer": list(np.round(np.linspace(0.1, 0.9, 9), 2)),
                   "rho_explainer": [0.5, 1, 2, 5, 10],
                   "f_A_probe": list(np.round(np.linspace(0.1, 0.9, 9), 2)),
                   "rho_probe": [0.5, 1, 2, 5, 10]}, handle, indent=2, ensure_ascii=False)
    # RPA scan covers the probe/explainer grids and the path family.
    all_fa = sorted(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    all_rho = [0.5, 1.0, 2.0, 5.0, 10.0]
    rpa_rows, edge_rows, rpa_by_key = [], [], {}
    for fa in all_fa:
        for rho in all_rho:
            result = rpa_sample(fa, rho)
            key = f"{fa:.6f}_{rho:.6f}"
            rpa_by_key[key] = result
            rpa_rows.append({k: v for k, v in result.items() if k not in {"edge_influence", "separated_pairs"}})
            for row in result["edge_influence"]:
                edge_rows.append({"f_A": fa, "rho": rho, **row})
    pd.DataFrame(rpa_rows).to_csv(args.output_dir / "abtab2_rpa_scan.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(args.output_dir / "abtab2_rpa_edge_influence.csv", index=False)
    with (args.output_dir / "abtab2_rpa_pair_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"topology": TOPOLOGY, "separated_pairs": rpa_by_key["0.500000_1.000000"]["separated_pairs"]}, handle, indent=2, ensure_ascii=False)
    explainer_summary = run_explainer(model, device, rpa_by_key, args.output_dir)
    probe_summary = run_probe(model, device, rpa_by_key, args.output_dir)
    path_summary = run_matched_path(model, device, args.output_dir)
    with (args.output_dir / "abtab2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"topology": TOPOLOGY, "checkpoint": str(args.run_dir), "device": str(device),
                   "explainer": explainer_summary, "probe": probe_summary, "matched_path": path_summary},
                  handle, indent=2, ensure_ascii=False)
    print(json.dumps({"topology": TOPOLOGY, "device": str(device), "explainer": explainer_summary,
                      "probe_rows": len(probe_summary), "path_rows": len(path_summary)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
