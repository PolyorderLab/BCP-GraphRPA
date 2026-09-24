#!/usr/bin/env python3
"""Generate numerical RPA/GNN composition data without drawing figures.

The supplied RPA grids define every published composition coordinate. This
entry point loads the five frozen C2 GNN checkpoints, predicts each coordinate,
and writes the source CSV/JSON used by the composition-curves notebook.
Use recompute_rpa.py first when the RPA values themselves must be freshly
recalculated with Julia.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
SOURCE_DIR = HERE / "data" / "rpa_inputs"
RESULT_DIR = CODE_ROOT / "results" / "source_data" / "composition_curves"
OUTPUT_JSON = RESULT_DIR / "five_seed_phase_diagram_predictions.json"
OUTPUT_CSV = RESULT_DIR / "five_seed_phase_diagram_predictions.csv"
SEEDS = [202601, 202602, 202603, 202604, 202605]
CHECKPOINTS = {
    seed: CODE_ROOT / "checkpoints" / "gnn" / f"seed{seed}" / "best_model.pth"
    for seed in SEEDS
}
GLOBAL_SOURCE = SOURCE_DIR / "global_phase_diagrams.json"
LOCAL_SOURCE = SOURCE_DIR / "m1_m2_local.json"
BTABAT_SOURCE = SOURCE_DIR / "btabat.json"
SHI_SOURCE = SOURCE_DIR / "shi_b2a2.json"
ABABAB_SOURCE = SOURCE_DIR / "ababab.json"

sys.path.insert(0, str(HERE))
from graph_builder import build_copolymer_graph  # noqa: E402
from model_architecture import GINEEncoder, SingleTaskRegressor  # noqa: E402

class LearnableNodeEncoder(nn.Module):
    """C2 encoder: local node-position embeddings followed by a GINE encoder."""

    def __init__(self, max_nodes: int = 80, hidden_dim: int = 512, num_layers: int = 5):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.empty(max_nodes, hidden_dim))
        self.gine = GINEEncoder(hidden_dim, edge_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, data):
        counts = data.ptr[1:] - data.ptr[:-1]
        local_ids = torch.cat([torch.arange(int(n), device=data.x.device) for n in counts])
        if local_ids.numel() and int(local_ids.max()) >= self.node_embedding.shape[0]:
            raise ValueError("Graph exceeds the checkpoint's 80-node embedding table")
        embedded = data.clone()
        embedded.x = self.node_embedding[local_ids]
        return self.gine(embedded)


def add_model_features_and_virtual_node(data: Data) -> Data:
    """Match the graph preprocessing used when the five C2 models were trained.

    The original degree column becomes three indicators/features
    ``[is_free_end, is_branch_point, degree]``.  A zero-feature virtual node is
    then connected bidirectionally to every padded graph node with zero-valued
    edge attributes.
    """
    degree = data.x[:, 0:1].clone()
    is_free_end = (degree == 1).to(data.x.dtype)
    is_branch_point = (degree > 1).to(data.x.dtype)
    x = torch.cat([is_free_end, is_branch_point, degree], dim=1)

    num_nodes = x.size(0)
    virtual_idx = num_nodes
    node_ids = torch.arange(num_nodes, dtype=torch.long)
    virtual_edges = torch.cat([
        torch.stack([torch.full((num_nodes,), virtual_idx), node_ids]),
        torch.stack([node_ids, torch.full((num_nodes,), virtual_idx)]),
    ], dim=1)
    virtual_edge_attr = torch.zeros(
        (virtual_edges.size(1), data.edge_attr.size(1)), dtype=data.edge_attr.dtype
    )
    out = Data(
        x=torch.cat([x, torch.zeros((1, x.size(1)), dtype=x.dtype)], dim=0),
        edge_index=torch.cat([data.edge_index, virtual_edges], dim=1),
        edge_attr=torch.cat([data.edge_attr, virtual_edge_attr], dim=0),
    )
    for key, value in data:
        if key not in {"x", "edge_index", "edge_attr"}:
            setattr(out, key, value.clone() if torch.is_tensor(value) else value)
    return out


class C2Predictor:
    def __init__(self, checkpoint: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = LearnableNodeEncoder()
        self.model = SingleTaskRegressor(encoder, hidden_dim=512, num_dec_layers=1)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def predict(self, topology: str, f_a: np.ndarray) -> np.ndarray:
        graphs = [
            add_model_features_and_virtual_node(
                build_copolymer_graph(topology, float(value), N_max=32)
            )
            for value in f_a
        ]
        batch = Batch.from_data_list(graphs).to(self.device)
        with torch.inference_mode():
            pred_log10 = self.model(batch).cpu().numpy().reshape(-1)
        return np.power(10.0, pred_log10)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_rpa(source: dict, topologies: list[str]) -> dict:
    return {topology: source[topology]["rpa"] for topology in topologies}


def assemble_cases(recomputed_rpa: Path | None = None) -> dict[str, dict]:
    """Load published grids, or freshly recomputed values in recompute_rpa format."""
    if recomputed_rpa is not None:
        payload = load_json(recomputed_rpa)
        if "cases" not in payload:
            raise ValueError("Expected a recompute_rpa JSON object containing 'cases'.")
        return payload["cases"]
    global_data = load_json(GLOBAL_SOURCE)
    local_data = load_json(LOCAL_SOURCE)
    btabat_data = load_json(BTABAT_SOURCE)
    shi_data = load_json(SHI_SOURCE)
    ababab_data = load_json(ABABAB_SOURCE)
    global_rpa = extract_rpa(global_data, ["AB", "ABA", "ABAB", "ABABA", "AB2", "AB3", "M1", "M2"])
    global_rpa.update(extract_rpa(ababab_data, ["ABABAB"]))
    return {
        "global": global_rpa,
        "m1_m2_local": extract_rpa(local_data, ["M1", "M2"]),
        "btabat": extract_rpa(btabat_data, ["BTABAT"]),
        "shi": extract_rpa(shi_data, ["AB", "B2A2", "(B2A2)2"]),
    }


def infer_all(cases: dict[str, dict]) -> dict:
    output = {}
    for case, topology_data in cases.items():
        output[case] = {}
        for topology, rpa in topology_data.items():
            f_a = np.asarray(rpa["fA"], dtype=float)
            output[case][topology] = {
                "rpa": {"fA": f_a.tolist(), "chiN": list(map(float, rpa["chiN"]))},
                "seeds": {},
            }

    for seed, checkpoint in CHECKPOINTS.items():
        print(f"Loading seed {seed}: {checkpoint}", flush=True)
        predictor = C2Predictor(checkpoint)
        for case, topology_data in output.items():
            for topology, values in topology_data.items():
                f_a = np.asarray(values["rpa"]["fA"], dtype=float)
                pred = predictor.predict(topology, f_a)
                values["seeds"][str(seed)] = pred.tolist()
                print(f"  {case}/{topology}: {len(pred)} points", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for topology_data in output.values():
        for values in topology_data.values():
            stack = np.asarray([values["seeds"][str(seed)] for seed in SEEDS])
            values["ml_mean"] = stack.mean(axis=0).tolist()
            values["ml_std"] = stack.std(axis=0, ddof=1).tolist()
    return output


def update_one_seed(data: dict, seed: int) -> dict:
    """Replace one seed in saved predictions and refresh mean/sample SD."""
    if seed not in CHECKPOINTS:
        raise ValueError(f"Unknown seed {seed}; choose one of {SEEDS}")
    predictor = C2Predictor(CHECKPOINTS[seed])
    for case, topology_data in data.items():
        for topology, values in topology_data.items():
            f_a = np.asarray(values["rpa"]["fA"], dtype=float)
            values["seeds"][str(seed)] = predictor.predict(topology, f_a).tolist()
            stack = np.asarray([values["seeds"][str(item)] for item in SEEDS])
            values["ml_mean"] = stack.mean(axis=0).tolist()
            values["ml_std"] = stack.std(axis=0, ddof=1).tolist()
            print(f"Updated {case}/{topology} for seed {seed}: {len(f_a)} points", flush=True)
    return data


def save_predictions(data: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump({"seeds": SEEDS, "cases": data}, handle, indent=2)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = ["case", "topology", "point_index", "fA", "rpa_chiN", *[f"seed_{s}" for s in SEEDS], "ml_mean", "ml_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, topology_data in data.items():
            for topology, values in topology_data.items():
                for idx, (f_a, rpa, mean, std) in enumerate(zip(values["rpa"]["fA"], values["rpa"]["chiN"], values["ml_mean"], values["ml_std"]), start=1):
                    row = {"case": case, "topology": topology, "point_index": idx, "fA": f_a, "rpa_chiN": rpa, "ml_mean": mean, "ml_std": std}
                    row.update({f"seed_{seed}": values["seeds"][str(seed)][idx - 1] for seed in SEEDS})
                    writer.writerow(row)

def main() -> None:
    global RESULT_DIR, OUTPUT_JSON, OUTPUT_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RESULT_DIR,
        help="Directory receiving five_seed_phase_diagram_predictions.csv/json.",
    )
    parser.add_argument(
        "--seed", type=int, choices=SEEDS,
        help="Update one seed in an existing output JSON instead of recomputing all seeds.",
    )
    parser.add_argument(
        "--rpa-json", type=Path,
        help="Optional rpa_recomputed.json from recompute_rpa.py; otherwise use published grids.",
    )
    args = parser.parse_args()
    RESULT_DIR = args.output_dir
    OUTPUT_JSON = RESULT_DIR / "five_seed_phase_diagram_predictions.json"
    OUTPUT_CSV = RESULT_DIR / "five_seed_phase_diagram_predictions.csv"
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    if args.seed is None:
        data = infer_all(assemble_cases(args.rpa_json))
    else:
        if not OUTPUT_JSON.is_file():
            raise FileNotFoundError("--seed requires an existing output JSON: " + str(OUTPUT_JSON))
        data = update_one_seed(load_json(OUTPUT_JSON)["cases"], args.seed)
    save_predictions(data)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
