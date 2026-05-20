"""
描述符计算模块

整合配置、图构建、结构描述符与 BCP 描述符计算，以及主调用接口。
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, Union

# ===================== 1. 配置 =====================

DESCRIPTOR_NAMES = [
    "C_bet", "lambda_alg", "phi_diam",
    "f_A", "delta_f",
]

DESCRIPTOR_DIM = len(DESCRIPTOR_NAMES)

ALL_AVAILABLE_NAMES = [
    # 结构描述符 (11)
    "Nnodes", "Nedges", "lambda_alg", "phi_diam", "phi_rad",
    "d_avg", "d_nb_avg", "density", "C_deg", "C_bet", "r_deg",
    # BCP 物理描述符 (7)
    "f_A", "delta_f", "J_AB", "M_AB", "D_f", "CV_arm", "Gini_f",
    # 拓扑不对称性 (3)
    "tau_max", "tau_mean", "tau_var",
]


# ===================== 2. 私有图构建工具 =====================

def _build_graph(
    edge_index: Union[np.ndarray, torch.Tensor],
    edge_attr: Union[np.ndarray, torch.Tensor] = None,
):
    """将 edge_index / edge_attr 转成 nx.Graph。

    edge_attr=None 时仅返回 nx.Graph；
    否则返回 (G, edge_types, edge_fracs)，其中 edge_types 与 edge_fracs
    以无向边 (min(u,v), max(u,v)) 为键。
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if edge_attr is not None and isinstance(edge_attr, torch.Tensor):
        edge_attr = edge_attr.cpu().numpy()

    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_attr is not None and edge_attr.shape[1] != 2 and edge_attr.shape[0] == 2:
        edge_attr = edge_attr.T

    G = nx.Graph()
    if edge_attr is None:
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            G.add_edge(u, v)
        return G

    edge_types: Dict[Tuple[int, int], str] = {}
    edge_fracs: Dict[Tuple[int, int], float] = {}
    seen = set()
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            G.add_edge(u, v)
            edge_types[key] = 'A' if float(edge_attr[i, 1]) > 0.5 else 'B'
            edge_fracs[key] = float(edge_attr[i, 0])

    return G, edge_types, edge_fracs


# ===================== 3. 描述符计算 =====================

def get_struct_descriptors(
    edge_index: Union[np.ndarray, torch.Tensor],
    edge_attr: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """计算 11 个结构拓扑描述符，返回 {name: value} 字典。

    edge_attr 仅用于图构建（与 edge_index 成对出现），不参与结构描述符计算。
    """
    G, *_ = _build_graph(edge_index, edge_attr)

    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    lambda_alg = nx.algebraic_connectivity(G)
    phi_diam = nx.diameter(G)
    phi_rad = nx.radius(G)

    degrees = [d for _, d in G.degree()]
    d_avg = sum(degrees) / len(G.nodes()) if len(G.nodes()) > 0 else 0.0
    d_nb_avg = (
        np.mean(list(nx.average_neighbor_degree(G).values()))
        if len(G.nodes()) > 0 else 0.0
    )
    density = nx.density(G)
    C_deg = (
        np.mean(list(nx.degree_centrality(G).values()))
        if len(G.nodes()) > 0 else 0.0
    )
    C_bet = (
        np.mean(list(nx.betweenness_centrality(G).values()))
        if len(G.nodes()) > 0 else 0.0
    )
    r_deg = nx.degree_assortativity_coefficient(G)

    return {
        "Nnodes": float(n_nodes),
        "Nedges": float(n_edges),
        "lambda_alg": float(lambda_alg),
        "phi_diam": float(phi_diam),
        "phi_rad": float(phi_rad),
        "d_avg": float(d_avg),
        "d_nb_avg": float(d_nb_avg),
        "density": float(density),
        "C_deg": float(C_deg),
        "C_bet": float(C_bet),
        "r_deg": float(r_deg),
    }

def get_bcp_descriptors(
    edge_index: Union[np.ndarray, torch.Tensor],
    edge_attr: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """计算 7 个 BCP 物理描述符，返回 {name: value} 字典。"""
    G, edge_types, edge_fracs = _build_graph(edge_index, edge_attr)

    if G.number_of_edges() == 0:
        return {n: 0.0 for n in ["f_A", "delta_f", "J_AB", "M_AB",
                                  "D_f", "CV_arm", "Gini_f",
                                  "tau_max", "tau_mean", "tau_var"]}

    f_A = sum(fv for key, fv in edge_fracs.items() if edge_types[key] == 'A')
    delta_f = abs(2.0 * f_A - 1.0)
    J_AB, M_AB = _compute_J_AB_M_AB(G, edge_types)
    D_f = _compute_D_f(G, edge_fracs)
    CV_arm = _compute_CV_arm(G, edge_fracs)
    Gini_f = _compute_Gini_f(edge_fracs)
    tau_max, tau_mean, tau_var = _compute_tau(G, edge_types, edge_fracs)

    return {
        "f_A": float(f_A),
        "delta_f": float(delta_f),
        "J_AB": float(J_AB),
        "M_AB": float(M_AB),
        "D_f": float(D_f),
        "CV_arm": float(CV_arm),
        "Gini_f": float(Gini_f),
        "tau_max": float(tau_max),
        "tau_mean": float(tau_mean),
        "tau_var": float(tau_var),
    }


def _compute_J_AB_M_AB(G, edge_types):
    J_AB = 0
    M_AB = 0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        nA = sum(1 for nb in neighbors
                 if edge_types.get((min(node, nb), max(node, nb)), 'B') == 'A')
        nB = len(neighbors) - nA
        if nA > 0 and nB > 0:
            J_AB += 1
            M_AB += nA * nB
    return float(J_AB), float(M_AB)


def _compute_D_f(G, edge_fracs):
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if len(leaves) < 2:
        return 0.0
    max_span = 0.0
    for i, u in enumerate(leaves):
        for v in leaves[i + 1:]:
            path = nx.shortest_path(G, u, v)
            span = sum(
                edge_fracs[(min(a, b), max(a, b))]
                for a, b in zip(path[:-1], path[1:])
            )
            max_span = max(max_span, span)
    return max_span


def _compute_CV_arm(G, edge_fracs):
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if len(leaves) < 2:
        return 0.0
    ecc = nx.eccentricity(G)
    center = min(ecc, key=ecc.get)
    arm_lengths = []
    for leaf in leaves:
        path = nx.shortest_path(G, center, leaf)
        length = sum(
            edge_fracs[(min(a, b), max(a, b))]
            for a, b in zip(path[:-1], path[1:])
        )
        arm_lengths.append(length)
    arr = np.array(arm_lengths)
    mean_val = arr.mean()
    if abs(mean_val) < 1e-12:
        return 0.0
    return float(np.std(arr) / (mean_val + 1e-8))


def _compute_Gini_f(edge_fracs):
    f_vals = np.array(list(edge_fracs.values()))
    if len(f_vals) == 0:
        return 0.0
    f_sorted = np.sort(f_vals)
    n = len(f_sorted)
    total = f_sorted.sum()
    if total < 1e-12:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * f_sorted) - (n + 1) * total) / (n * total)
    return float(gini)


def _get_subtree_nodes(G, start, exclude):
    """从 start 出发在 G 中做 BFS，不经过 exclude 节点，返回可达的所有节点。"""
    visited = {exclude}
    queue = [start]
    subtree = set()
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            subtree.add(node)
            for nb in G.neighbors(node):
                if nb not in visited:
                    queue.append(nb)
    return subtree


def _compute_tau(G, edge_types, edge_fracs):
    """广义拓扑不对称性：tau_max, tau_mean, tau_var。"""
    tau_values = []
    for node in G.nodes():
        if G.degree(node) >= 2:  # junction
            neighbors = list(G.neighbors(node))
            subtree_fA = []
            for neighbor in neighbors:
                subtree = _get_subtree_nodes(G, neighbor, node)

                # 子树内 A 边的体积分数之和（不重复计数）
                fA_sum = 0.0
                for u in subtree:
                    for v in G.neighbors(u):
                        if v in subtree and u < v:
                            key = (u, v)
                            if edge_types.get(key) == 'A':
                                fA_sum += edge_fracs.get(key, 0.0)

                # 加上连接 junction 到该分支的边
                conn_key = (min(node, neighbor), max(node, neighbor))
                if edge_types.get(conn_key) == 'A':
                    fA_sum += edge_fracs.get(conn_key, 0.0)

                subtree_fA.append(fA_sum)

            total_fA = sum(subtree_fA)
            if total_fA > 0:
                tau_j = max(subtree_fA) / total_fA
            else:
                tau_j = 1.0 / len(subtree_fA)
            tau_values.append(tau_j)

    if not tau_values:
        return 0.5, 0.5, 0.0  # 线性链默认

    return max(tau_values), np.mean(tau_values), np.var(tau_values)


# ===================== 4. 主接口 =====================

def get_descriptors(
    edge_index: Union[np.ndarray, torch.Tensor],
    edge_attr: Union[np.ndarray, torch.Tensor],
    desc_names: list = None,
) -> np.ndarray:
    """根据 desc_names 或 DESCRIPTOR_NAMES 返回描述符向量。"""
    names = desc_names if desc_names is not None else DESCRIPTOR_NAMES

    struct = get_struct_descriptors(edge_index, edge_attr)
    bcp = get_bcp_descriptors(edge_index, edge_attr)
    all_desc = {**struct, **bcp}

    return np.array([all_desc[n] for n in names], dtype=np.float32)


# ===================== 5. 工具类 =====================

class DescriptorScaler:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, descriptors: np.ndarray):
        self.scaler.fit(descriptors)
        self.fitted = True

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler not fitted yet")
        return self.scaler.transform(descriptors)

    def fit_transform(self, descriptors: np.ndarray) -> np.ndarray:
        self.fitted = True
        return self.scaler.fit_transform(descriptors)
