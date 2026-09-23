"""Frozen model, graph wrappers, data loading, and five-block RPA helpers."""
from __future__ import annotations

import glob
import itertools
import math
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import yaml
from scipy.optimize import minimize_scalar
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool


class GINEEncoder(nn.Module):
    def __init__(self, in_feats: int, edge_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.edge_lin = nn.ModuleList()
        self.num_layers = num_layers
        for layer in range(num_layers):
            input_dim = in_feats if layer == 0 else hidden_dim
            self.edge_lin.append(nn.Linear(edge_dim, input_dim))
            network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            self.convs.append(GINEConv(network))
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x
        for layer, convolution in enumerate(self.convs):
            x = convolution(x, data.edge_index, edge_attr=self.edge_lin[layer](data.edge_attr))
            if layer != self.num_layers - 1:
                x = self.relu(x)
        return global_mean_pool(x, data.batch)


class LearnableNodeGINEEncoder(nn.Module):
    def __init__(self, max_nodes: int, edge_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(max_nodes, hidden_dim) * 0.01)
        self.gine = GINEEncoder(hidden_dim, edge_dim, hidden_dim, num_layers)

    def forward(self, data):
        counts = (data.ptr[1:] - data.ptr[:-1]).tolist()
        data = data.clone()
        data.x = torch.cat([self.node_embedding[:count] for count in counts], dim=0)
        return self.gine(data)


class SingleTaskRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, decoder_layers: int):
        super().__init__()
        self.encoder = encoder
        layers: list[nn.Module] = []
        for _ in range(decoder_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
        layers.append(nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, data):
        return self.decoder(self.encoder(data))


def build_model(config: dict, device: torch.device) -> SingleTaskRegressor:
    encoder = LearnableNodeGINEEncoder(
        max_nodes=80,
        edge_dim=2,
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_layers=int(config.get("num_layers", 5)),
    )
    return SingleTaskRegressor(
        encoder,
        hidden_dim=int(config.get("hidden_dim", 512)),
        decoder_layers=int(config.get("dec_layers", 1)),
    ).to(device)


def clone_graph_attrs(source, target):
    for key, value in source:
        if key not in {"x", "edge_index", "edge_attr"}:
            setattr(target, key, value.clone() if torch.is_tensor(value) else value)
    return target


class VirtualNodeWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].clone()
        data.x = torch.zeros_like(data.x)
        count = data.x.size(0)
        virtual = count
        nodes = torch.arange(count, dtype=torch.long)
        virtual_edges = torch.cat(
            (
                torch.stack((torch.full((count,), virtual), nodes)),
                torch.stack((nodes, torch.full((count,), virtual))),
            ),
            dim=1,
        )
        output = Data(
            x=torch.cat((data.x, torch.zeros((1, data.x.size(1)), dtype=data.x.dtype))),
            edge_index=torch.cat((data.edge_index, virtual_edges), dim=1),
            edge_attr=torch.cat(
                (data.edge_attr, torch.zeros((virtual_edges.size(1), data.edge_attr.size(1)))),
                dim=0,
            ),
        )
        return clone_graph_attrs(data, output)


def load_seed43_dataset(data_dir: Path) -> list[Data]:
    """Parse sorted MAT/YML pairs exactly as the training data loader does."""
    output: list[Data] = []
    for mat_name in sorted(glob.glob(str(data_dir / "*.mat"))):
        mat_path = Path(mat_name)
        yaml_path = mat_path.with_suffix(".yml")
        if not yaml_path.is_file():
            continue
        try:
            payload = sio.loadmat(mat_path)
            node_features = np.asarray(payload["F_pad"], dtype=np.float32)
            edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(payload["edge_attr"], dtype=np.float32)
        except (NotImplementedError, ValueError):
            with h5py.File(mat_path, "r") as handle:
                node_features = np.asarray(handle["F_pad"], dtype=np.float32).T
                edge_index = np.asarray(handle["edge_index"], dtype=np.int64).T
                edge_attr = np.asarray(handle["edge_attr"], dtype=np.float32).T
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        system = config["system"]
        edges = torch.tensor(edge_index - 1, dtype=torch.long)
        if edges.ndim == 2 and edges.shape[0] != 2:
            edges = edges.t().contiguous()
        output.append(Data(
            x=torch.tensor(node_features[:, :3], dtype=torch.float32),
            edge_index=edges,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            num_blocks=torch.tensor([len(system["components"][0]["blocks"])], dtype=torch.long),
            y=torch.tensor([[np.log10(float(system["χN_star"]))]], dtype=torch.float32),
        ))
    return output


def blocks_for(f_a: float, rho: float, assignment=None):
    b_total = 1.0 - f_a
    b1 = b_total * rho / (rho + 2.0)
    b_small = b_total / (rho + 2.0)
    base = [
        {"label": "A1", "species": "A", "u": "n0", "v": "n1", "f": f_a / 2.0},
        {"label": "B1", "species": "B", "u": "n1", "v": "T", "f": b1},
        {"label": "A2", "species": "A", "u": "T", "v": "n2", "f": f_a / 2.0},
        {"label": "B2", "species": "B", "u": "T", "v": "n3", "f": b_small},
        {"label": "B3", "species": "B", "u": "T", "v": "n4", "f": b_small},
    ]
    if assignment is None:
        return base
    result = []
    for index, ((species, length), template) in enumerate(zip(assignment, base, strict=True)):
        item = dict(template)
        item.update(species=species, f=float(length), label=f"e{index + 1}_{species}")
        result.append(item)
    return result


def unique_assignments(f_a: float, rho: float):
    labels = [(block["species"], round(block["f"], 12)) for block in blocks_for(f_a, rho)]
    return sorted(set(itertools.permutations(labels)))


def legal_growth_orders(blocks) -> list[tuple[str, ...]]:
    adjacency: dict[str, set[str]] = {}
    for block in blocks:
        adjacency.setdefault(block["u"], set()).add(block["v"])
        adjacency.setdefault(block["v"], set()).add(block["u"])
    orders: list[tuple[str, ...]] = []

    def extend(order: list[str], seen: set[str]) -> None:
        if len(order) == len(adjacency):
            orders.append(tuple(order))
            return
        for node in sorted({v for u in seen for v in adjacency[u]} - seen):
            extend(order + [node], seen | {node})

    for root in sorted(adjacency):
        extend([root], {root})
    return orders


def make_five_block_graph(f_a: float, rho: float, assignment, node_order) -> Data:
    blocks = blocks_for(f_a, rho, assignment)
    node_id = {name: index for index, name in enumerate(node_order)}
    edges = []
    attributes = []
    for block in blocks:
        left, right = node_id[block["u"]], node_id[block["v"]]
        attribute = (block["f"], 1.0 if block["species"] == "A" else 0.0)
        edges.extend(((left, right), (right, left)))
        attributes.extend((attribute, attribute))
    padded_nodes = 32
    virtual = padded_nodes
    for node in range(padded_nodes):
        edges.extend(((virtual, node), (node, virtual)))
        attributes.extend(((0.0, 0.0), (0.0, 0.0)))
    return Data(
        x=torch.zeros((padded_nodes + 1, 3), dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(attributes, dtype=torch.float32),
        batch=torch.zeros(padded_nodes + 1, dtype=torch.long),
        ptr=torch.tensor([0, padded_nodes + 1], dtype=torch.long),
    )


def g_debye(x: float, fraction: float) -> float:
    z = x * fraction
    if abs(z) < 1e-4:
        return fraction * fraction * (1.0 - z / 3.0 + z * z / 12.0 - z**3 / 60.0)
    return 2.0 * (math.exp(-z) + z - 1.0) / (x * x)


def h_func(x: float, fraction: float) -> float:
    return -math.expm1(-x * fraction) / x


def block_adjacency(blocks):
    adjacency = {index: [] for index in range(len(blocks))}
    for left_index, left in enumerate(blocks):
        for right_index in range(left_index + 1, len(blocks)):
            if {left["u"], left["v"]} & {blocks[right_index]["u"], blocks[right_index]["v"]}:
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
    return adjacency


def shortest_block_path(adjacency, start: int, goal: int):
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
    x = q * q
    adjacency = block_adjacency(blocks)
    components = {
        f"S_{pair}_{relation}": 0.0
        for pair in ("AA", "BB", "AB")
        for relation in ("self", "adjacent", "separated")
    }
    for left_index, left in enumerate(blocks):
        pair = left["species"] * 2
        components[f"S_{pair}_self"] += g_debye(x, left["f"])
        for right_index in range(left_index + 1, len(blocks)):
            right = blocks[right_index]
            if left["species"] == right["species"]:
                pair = left["species"] * 2
                multiplicity = 2
            else:
                pair = "AB"
                multiplicity = 1
            if right_index in adjacency[left_index]:
                relation = "adjacent"
                value = h_func(x, left["f"]) * h_func(x, right["f"])
            else:
                relation = "separated"
                path = shortest_block_path(adjacency, left_index, right_index)
                distance = sum(blocks[index]["f"] for index in path[1:-1])
                value = h_func(x, left["f"]) * math.exp(-x * distance) * h_func(x, right["f"])
            components[f"S_{pair}_{relation}"] += multiplicity * value
    totals = {
        pair: sum(components[f"S_{pair}_{relation}"] for relation in ("self", "adjacent", "separated"))
        for pair in ("AA", "BB", "AB")
    }
    determinant = totals["AA"] * totals["BB"] - totals["AB"] ** 2
    phi = (totals["AA"] + totals["BB"] + 2.0 * totals["AB"]) / determinant
    return phi, components


def path_concepts(f_a: float, rho: float, assignment) -> np.ndarray:
    blocks = blocks_for(f_a, rho, assignment)
    objective = lambda value: phi_and_components(float(value), blocks)[0]
    grid = np.linspace(0.2, 7.0, 350)
    values = np.asarray([objective(value) for value in grid])
    index = int(np.argmin(values))
    result = minimize_scalar(
        objective,
        bounds=(float(grid[max(0, index - 2)]), float(grid[min(len(grid) - 1, index + 2)])),
        method="bounded",
    )
    _, components = phi_and_components(float(result.x), blocks)
    return np.asarray([
        components["S_AA_separated"],
        components["S_BB_separated"],
        components["S_AB_separated"],
        float(result.x),
    ])
