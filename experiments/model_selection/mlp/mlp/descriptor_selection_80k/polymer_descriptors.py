#!/usr/bin/env python3
"""Compute the mathematically de-duplicated polymer-graph descriptor set.

Input edge attributes have two columns: ``[block_fraction, block_type]``, where
block_type is 1/A for an A block and 0/B for a B block.  The public API accepts
either (2, E) or (E, 2) ``edge_index`` arrays.  The CLI reads the project's YML
records and writes one CSV row per polymer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np
import yaml


DESCRIPTOR_NAMES = (
    "Nnodes", "Nedges", "density", "lambda_alg", "r_deg", "d_nb_avg",
    "C_bet", "phi_diam", "phi_rad", "wiener", "branch_count", "Gini_f",
    "f_A", "J_AB", "M_AB", "IR", "ISC", "CAF", "tau_mean", "CV_arm",
    "D_f", "SLE", "shortest_path_std", "global_efficiency", "n_leaves",
    "max_degree", "degree_std", "degree_entropy", "laplacian_largest",
    "adjacency_spectral_radius", "adjacency_spectral_gap", "C_bet_max",
    "C_bet_std", "C_close_mean", "C_close_std",
)


def _as_2xe(edge_index: np.ndarray) -> np.ndarray:
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError("edge_index must be a 2-D array")
    if edge_index.shape[0] == 2:
        return edge_index
    if edge_index.shape[1] == 2:
        return edge_index.T
    raise ValueError("edge_index must have shape (2, E) or (E, 2)")


def _build_graph(edge_index, edge_attr):
    edges = _as_2xe(edge_index)
    attrs = np.asarray(edge_attr, dtype=float)
    if attrs.ndim != 2:
        raise ValueError("edge_attr must be a 2-D array with [fraction, type]")
    if attrs.shape[0] != edges.shape[1] and attrs.shape[1] == edges.shape[1]:
        attrs = attrs.T
    if attrs.shape != (edges.shape[1], 2):
        raise ValueError("edge_attr must have one [fraction, type] row per edge")

    graph = nx.Graph()
    fractions: Dict[Tuple[int, int], float] = {}
    types: Dict[Tuple[int, int], str] = {}
    for col in range(edges.shape[1]):
        u, v = int(edges[0, col]), int(edges[1, col])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in fractions:  # input may contain both directed copies
            continue
        graph.add_edge(*key)
        fractions[key] = max(float(attrs[col, 0]), 0.0)
        types[key] = "A" if float(attrs[col, 1]) > 0.5 else "B"
    return graph, fractions, types


def _safe_assortativity(graph: nx.Graph) -> float:
    value = nx.degree_assortativity_coefficient(graph)
    return float(value) if np.isfinite(value) else 0.0


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _interfaces(graph, types):
    return [
        node for node in graph.nodes
        if {types[_edge_key(node, nb)] for nb in graph.neighbors(node)} == {"A", "B"}
    ]


def _arm_edges(graph: nx.Graph):
    """Return leaf-to-first-branch arms, as in the original 24-descriptor code."""
    arms = []
    for leaf in (node for node, degree in graph.degree if degree == 1):
        previous, node, path = None, leaf, [leaf]
        while graph.degree(node) < 3:
            next_nodes = [nb for nb in graph.neighbors(node) if nb != previous]
            if not next_nodes:
                break
            previous, node = node, next_nodes[0]
            path.append(node)
        if graph.degree(node) >= 3:
            arms.append([_edge_key(u, v) for u, v in zip(path[:-1], path[1:])])
    return arms


def _tau_mean(graph, fractions, types) -> float:
    values = []
    for node, degree in graph.degree:
        if degree < 2:
            continue
        branch_a = []
        for neighbor in graph.neighbors(node):
            copy = graph.copy()
            copy.remove_node(node)
            component = nx.node_connected_component(copy, neighbor)
            amount = sum(
                fractions[_edge_key(u, v)]
                for u, v in graph.subgraph(component).edges
                if types[_edge_key(u, v)] == "A"
            )
            connecting_edge = _edge_key(node, neighbor)
            if types[connecting_edge] == "A":
                amount += fractions[connecting_edge]
            branch_a.append(amount)
        total = sum(branch_a)
        values.append(max(branch_a) / total if total > 0 else 1.0 / len(branch_a))
    return float(np.mean(values)) if values else 0.5


def compute_descriptors(edge_index, edge_attr) -> Dict[str, float]:
    """Return the 35 de-duplicated descriptors for one connected polymer graph."""
    graph, fractions, types = _build_graph(edge_index, edge_attr)
    if graph.number_of_nodes() == 0:
        return {name: 0.0 for name in DESCRIPTOR_NAMES}
    if not nx.is_connected(graph):
        raise ValueError("polymer graph must be connected")

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = np.array([degree for _, degree in graph.degree], dtype=float)
    path_values = np.array(
        [distance for source, paths in nx.all_pairs_shortest_path_length(graph)
         for target, distance in paths.items() if source < target], dtype=float
    )
    betweenness = np.array(list(nx.betweenness_centrality(graph).values()), dtype=float)
    closeness = np.array(list(nx.closeness_centrality(graph).values()), dtype=float)
    interfaces = _interfaces(graph, types)

    # Composition and local A/B-interface quantities.
    f_a = sum(fractions[key] for key, block_type in types.items() if block_type == "A")
    jab = mab = 0.0
    isc_terms, isc_weights = [], []
    ir_numerator = ir_denominator = 0.0
    center = min(nx.eccentricity(graph), key=nx.eccentricity(graph).get)
    center_distances = nx.single_source_shortest_path_length(graph, center)
    for node in interfaces:
        a_edges = [_edge_key(node, nb) for nb in graph.neighbors(node)
                   if types[_edge_key(node, nb)] == "A"]
        b_edges = [_edge_key(node, nb) for nb in graph.neighbors(node)
                   if types[_edge_key(node, nb)] == "B"]
        jab += 1.0
        mab += len(a_edges) * len(b_edges)
        # Definition in the supplied txt: every local A/B edge pair, weighted f_a*f_b.
        for a_edge in a_edges:
            for b_edge in b_edges:
                fa, fb = fractions[a_edge], fractions[b_edge]
                weight = fa * fb
                if weight > 0:
                    isc_terms.append(abs(np.log(fa / fb)))
                    isc_weights.append(weight)
        # IR retains the original MLP descriptor definition.
        local_weight = (max(fractions[e] for e in a_edges) + max(fractions[e] for e in b_edges)) / 2.0
        ir_numerator += local_weight * center_distances[node]
        ir_denominator += local_weight

    arms = _arm_edges(graph)
    arm_volumes = np.array([sum(fractions[e] for e in arm) for arm in arms], dtype=float)
    composite_volume = sum(
        sum(fractions[e] for e in arm)
        for arm in arms if len({types[e] for e in arm}) > 1
    )
    all_arm_volume = float(arm_volumes.sum())
    leaves = [node for node, degree in graph.degree if degree == 1]
    leaf_pair_spans = [
        sum(fractions[_edge_key(u, v)] for u, v in zip(path[:-1], path[1:]))
        for i, source in enumerate(leaves)
        for target in leaves[i + 1:]
        for path in [nx.shortest_path(graph, source, target)]
    ]
    fraction_values = np.array(list(fractions.values()), dtype=float)
    total_fraction = float(fraction_values.sum())
    probs = fraction_values / total_fraction if total_fraction > 0 else np.array([])
    nonzero_probs = probs[probs > 0]
    sorted_fractions = np.sort(fraction_values)
    gini = 0.0 if total_fraction == 0 else float(
        (2 * np.dot(np.arange(1, m + 1), sorted_fractions) - (m + 1) * total_fraction)
        / (m * total_fraction)
    )

    adjacency = nx.to_numpy_array(graph, dtype=float)
    # L = D - A; its largest eigenvalue is desired, not that of -A.
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    laplacian_largest = float(np.linalg.eigvalsh(laplacian)[-1])
    adjacency_eigs = np.sort(np.linalg.eigvalsh(adjacency))[::-1]

    return {
        "Nnodes": float(n), "Nedges": float(m), "density": float(nx.density(graph)),
        "lambda_alg": float(nx.algebraic_connectivity(graph)) if n > 1 else 0.0,
        "r_deg": _safe_assortativity(graph),
        "d_nb_avg": float(np.mean(list(nx.average_neighbor_degree(graph).values()))),
        "C_bet": float(betweenness.mean()), "phi_diam": float(nx.diameter(graph)),
        "phi_rad": float(nx.radius(graph)), "wiener": float(path_values.sum()),
        "branch_count": float(np.sum(degrees >= 3)), "Gini_f": gini, "f_A": float(f_a),
        "J_AB": jab, "M_AB": mab,
        "IR": float(ir_numerator / ir_denominator) if ir_denominator > 0 else 0.0,
        "ISC": float(np.average(isc_terms, weights=isc_weights)) if isc_weights else 0.0,
        "CAF": float(composite_volume / all_arm_volume) if all_arm_volume > 0 else 0.0,
        "tau_mean": _tau_mean(graph, fractions, types),
        "CV_arm": float(arm_volumes.std() / arm_volumes.mean()) if arm_volumes.size > 1 and arm_volumes.mean() > 0 else 0.0,
        "D_f": float(max(leaf_pair_spans)) if leaf_pair_spans else 0.0,
        "SLE": float(-(nonzero_probs * np.log(nonzero_probs)).sum() / np.log(len(nonzero_probs))) if len(nonzero_probs) > 1 else 0.0,
        "shortest_path_std": float(path_values.std()) if path_values.size else 0.0,
        "global_efficiency": float(nx.global_efficiency(graph)), "n_leaves": float(len(leaves)),
        "max_degree": float(degrees.max()), "degree_std": float(degrees.std()),
        "degree_entropy": float(-sum(p * np.log(p) for p in np.unique(degrees, return_counts=True)[1] / n)),
        "laplacian_largest": laplacian_largest,
        "adjacency_spectral_radius": float(adjacency_eigs[0]),
        "adjacency_spectral_gap": float(adjacency_eigs[0] - adjacency_eigs[1]) if n > 1 else 0.0,
        "C_bet_max": float(betweenness.max()), "C_bet_std": float(betweenness.std()),
        "C_close_mean": float(closeness.mean()), "C_close_std": float(closeness.std()),
    }


def _from_yml(path: Path):
    with path.open(encoding="utf-8") as handle:
        blocks = yaml.safe_load(handle)["system"]["components"][0]["blocks"]
    node_ids = {node for block in blocks for node in block["ends"]}
    node_index = {node: i for i, node in enumerate(sorted(node_ids, key=str))}
    edge_index = np.array([[node_index[a], node_index[b]] for a, b in (block["ends"] for block in blocks)])
    edge_attr = np.array([[float(block["length"]), float(str(block["segment"]) == "A")] for block in blocks])
    return compute_descriptors(edge_index, edge_attr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing polymer .yml records")
    parser.add_argument("output_csv", type=Path, help="destination CSV")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.yml")):
        row = {"sample_id": path.stem}
        row.update(_from_yml(path))
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", *DESCRIPTOR_NAMES))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} samples x {len(DESCRIPTOR_NAMES)} descriptors to {args.output_csv}")


if __name__ == "__main__":
    main()
