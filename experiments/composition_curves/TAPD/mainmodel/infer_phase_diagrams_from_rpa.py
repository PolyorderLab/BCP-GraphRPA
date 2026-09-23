"""Infer ODT values with the mainmodel checkpoint and redraw mpltex figures."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from torch_geometric.data import Batch, Data

ROOT = Path("/home/zxy/project/TAPD")
SRC_DIR = ROOT / "src"
DEMO_PHASE_DIR = ROOT / "Demo" / "outcome" / "phase_diagrams"
MAINMODEL_DIR = ROOT / "mainmodel"
MAINMODEL_PHASE_DIR = MAINMODEL_DIR / "phase_diagrams"
CHECKPOINT = (
    MAINMODEL_DIR
    / "five_seeds"
    / "vn_free_branch_degree_tuned_from_ablation_rerun_5runs_20260628_134001"
    / "VN_FreeBranchDegree"
    / "VN_FreeBranchDegree_run03_seed202603"
    / "best_model.pth"
)

RPA_INPUT = DEMO_PHASE_DIR / "plot_data.json"
M1_M2_RPA_INPUT = DEMO_PHASE_DIR / "m1_m2_comparison_data.json"
MAINMODEL_RPA_COPY = MAINMODEL_PHASE_DIR / "rpa_from_demo_phase_diagrams.json"
MAINMODEL_M1_M2_RPA_COPY = MAINMODEL_PHASE_DIR / "m1_m2_rpa_from_demo_phase_diagrams.json"
MAINMODEL_INFERRED = MAINMODEL_PHASE_DIR / "mainmodel_inferred_plot_data.json"
MAINMODEL_M1_M2_INFERRED = MAINMODEL_PHASE_DIR / "mainmodel_inferred_m1_m2_comparison_data.json"

TOPOLOGIES = ["AB", "ABA", "ABAB", "ABABA", "AB2", "AB3", "M1", "M2"]

sys.path.append(str(ROOT))
sys.path.append(str(SRC_DIR))
from src.graph_builder import build_copolymer_graph  # noqa: E402
from src.ira_graph_common import GINEEncoder, SingleTaskRegressor  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rpa_only(data: dict) -> dict:
    out = {}
    for topo, values in data.items():
        if "rpa" in values:
            out[topo] = {"rpa": values["rpa"]}
    return out


def add_mainmodel_features_and_virtual_node(data: Data) -> Data:
    degree = data.x[:, 0:1].clone()
    is_free_end = (degree == 1).to(data.x.dtype)
    is_branch_point = (degree > 1).to(data.x.dtype)
    x = torch.cat([is_free_end, is_branch_point, degree], dim=1)

    num_nodes = x.size(0)
    virtual_idx = num_nodes
    virtual_x = torch.zeros((1, x.size(1)), dtype=x.dtype)
    node_ids = torch.arange(num_nodes, dtype=torch.long)
    virtual_to_nodes = torch.stack(
        [torch.full((num_nodes,), virtual_idx, dtype=torch.long), node_ids],
        dim=0,
    )
    nodes_to_virtual = torch.stack(
        [node_ids, torch.full((num_nodes,), virtual_idx, dtype=torch.long)],
        dim=0,
    )
    virtual_edges = torch.cat([virtual_to_nodes, nodes_to_virtual], dim=1)
    virtual_edge_attr = torch.zeros((virtual_edges.size(1), data.edge_attr.size(1)), dtype=data.edge_attr.dtype)

    out = Data(
        x=torch.cat([x, virtual_x], dim=0),
        edge_index=torch.cat([data.edge_index, virtual_edges], dim=1),
        edge_attr=torch.cat([data.edge_attr, virtual_edge_attr], dim=0),
    )
    for key, value in data:
        if key not in {"x", "edge_index", "edge_attr"}:
            setattr(out, key, value.clone() if torch.is_tensor(value) else value)
    return out


class MainModelChiNPredictor:
    def __init__(self, checkpoint: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = GINEEncoder(in_feats=3, edge_dim=2, hidden_dim=256, num_layers=5)
        self.model = SingleTaskRegressor(encoder, hidden_dim=256, num_dec_layers=1)
        state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, topo: str, f_a_values: np.ndarray) -> np.ndarray:
        graphs = [
            add_mainmodel_features_and_virtual_node(build_copolymer_graph(topo, float(f_a), N_max=32))
            for f_a in f_a_values
        ]
        batch = Batch.from_data_list(graphs).to(self.device)
        with torch.no_grad():
            log_chi_n = self.model(batch).detach().cpu().numpy().reshape(-1)
        return np.power(10.0, log_chi_n)


def infer_from_rpa(rpa_data: dict, predictor: MainModelChiNPredictor) -> dict:
    out = rpa_only(rpa_data)
    for topo in out:
        f_a = np.asarray(out[topo]["rpa"]["fA"], dtype=float)
        chi_n = predictor.predict_batch(topo, f_a)
        out[topo]["ml"] = {"fA": f_a.tolist(), "chiN": chi_n.tolist()}
        print(f"Inferred {topo}: {len(f_a)} points")
    return out


def main() -> None:
    MAINMODEL_PHASE_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(RPA_INPUT, MAINMODEL_RPA_COPY)
    shutil.copy2(M1_M2_RPA_INPUT, MAINMODEL_M1_M2_RPA_COPY)
    rpa_data = rpa_only(load_json(MAINMODEL_RPA_COPY))
    m1_m2_rpa_data = rpa_only(load_json(MAINMODEL_M1_M2_RPA_COPY))
    write_json(MAINMODEL_RPA_COPY, rpa_data)
    write_json(MAINMODEL_M1_M2_RPA_COPY, m1_m2_rpa_data)
    print(f"Copied RPA data to {MAINMODEL_PHASE_DIR}")

    predictor = MainModelChiNPredictor(CHECKPOINT)
    inferred = infer_from_rpa(rpa_data, predictor)
    inferred_m1_m2 = infer_from_rpa(m1_m2_rpa_data, predictor)

    write_json(MAINMODEL_INFERRED, inferred)
    write_json(MAINMODEL_M1_M2_INFERRED, inferred_m1_m2)
    sys.path.append(str(ROOT / "Demo"))
    from Demo.redraw_phase_diagrams_mpltex import redraw_from_data

    redraw_from_data(MAINMODEL_INFERRED, MAINMODEL_M1_M2_INFERRED, MAINMODEL_PHASE_DIR)


if __name__ == "__main__":
    main()
