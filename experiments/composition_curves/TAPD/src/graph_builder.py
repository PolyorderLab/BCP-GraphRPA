import torch
import numpy as np
from torch_geometric.data import Data


def build_ab_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for AB diblock copolymer matching training data format.

    Structure (0-indexed): 0 --(A)-- 1 --(B)-- 2
    Nodes: 3, Edges: 2 blocks (bidirectional = 4 directed edges)
    Degrees: [1, 2, 1]
    """
    fB = 1.0 - fA
    x = torch.zeros(N_max, 3, dtype=torch.float)

    # Node features: [degree, avg_neighbor_degree, padding]
    # Node 0: FreeEnd (degree=1), neighbor is Node 1 (degree=2)
    # Node 1: BranchPoint (degree=2), neighbors are Node 0,2 (degree=1)
    # Node 2: FreeEnd (degree=1), neighbor is Node 1 (degree=2)
    x[0, 0] = 1.0  # Node 0: degree=1
    x[1, 0] = 2.0  # Node 1: degree=2
    x[2, 0] = 1.0  # Node 2: degree=1

    x[0, 1] = 2.0  # avg_neighbor_degree for Node 0: Node 1 has degree 2
    x[1, 1] = 1.0  # avg_neighbor_degree for Node 1: (1+1)/2 = 1
    x[2, 1] = 2.0  # avg_neighbor_degree for Node 2: Node 1 has degree 2

    # Edge index: (2, num_directed_edges) format, 0-indexed
    # Block 1 (A): 0 -> 1, reverse 1 -> 0
    # Block 2 (B): 1 -> 2, reverse 2 -> 1
    edge_index = torch.tensor([
        [0, 1, 1, 2],  # src
        [1, 0, 2, 1]   # dst
    ], dtype=torch.long)

    edge_attr = torch.zeros(4, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fA, 1.0])  # A block forward
    edge_attr[1] = torch.tensor([fA, 1.0])  # A block reverse
    edge_attr[2] = torch.tensor([fB, 0.0])  # B block forward
    edge_attr[3] = torch.tensor([fB, 0.0])  # B block reverse

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([2]), topo_type=torch.tensor([0]))


def build_aba_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for ABA triblock copolymer matching training data format.

    Structure (0-indexed): 0 --(A1)-- 1 --(B)-- 2 --(A2)-- 3
    Nodes: 4, Edges: 3 blocks (6 directed edges)
    Degrees: [1, 2, 2, 1]
    """
    fa = fA / 2.0
    fB = 1.0 - fA
    x = torch.zeros(N_max, 3, dtype=torch.float)

    x[0, 0] = 1.0; x[1, 0] = 2.0; x[2, 0] = 2.0; x[3, 0] = 1.0
    x[0, 1] = 2.0; x[1, 1] = 1.0; x[2, 1] = 1.0; x[3, 1] = 2.0

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)

    edge_attr = torch.zeros(6, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fa, 1.0]); edge_attr[1] = torch.tensor([fa, 1.0])
    edge_attr[2] = torch.tensor([fB, 0.0]); edge_attr[3] = torch.tensor([fB, 0.0])
    edge_attr[4] = torch.tensor([fa, 1.0]); edge_attr[5] = torch.tensor([fa, 1.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([3]), topo_type=torch.tensor([1]))


def build_abab_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for ABAB tetrablock copolymer matching training data format.

    Structure (0-indexed): 0 --(A1)-- 1 --(B1)-- 2 --(A2)-- 3 --(B2)-- 4
    Nodes: 5, Edges: 4 blocks (8 directed edges)
    Degrees: [1, 2, 2, 2, 1]
    """
    fA_half = fA / 2.0
    fB_half = (1.0 - fA) / 2.0
    x = torch.zeros(N_max, 3, dtype=torch.float)

    x[0, 0] = 1.0; x[1, 0] = 2.0; x[2, 0] = 2.0; x[3, 0] = 2.0; x[4, 0] = 1.0
    x[0, 1] = 2.0; x[1, 1] = 1.0; x[2, 1] = 2.0; x[3, 1] = 1.0; x[4, 1] = 2.0

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)

    edge_attr = torch.zeros(8, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fA_half, 1.0]); edge_attr[1] = torch.tensor([fA_half, 1.0])
    edge_attr[2] = torch.tensor([fB_half, 0.0]); edge_attr[3] = torch.tensor([fB_half, 0.0])
    edge_attr[4] = torch.tensor([fA_half, 1.0]); edge_attr[5] = torch.tensor([fA_half, 1.0])
    edge_attr[6] = torch.tensor([fB_half, 0.0]); edge_attr[7] = torch.tensor([fB_half, 0.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([4]), topo_type=torch.tensor([2]))


def build_ababa_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for ABABA pentablock copolymer matching training data format.

    Structure (0-indexed): 0 --(A1)-- 1 --(B1)-- 2 --(A2)-- 3 --(B2)-- 4 --(A3)-- 5
    Nodes: 6, Edges: 5 blocks (10 directed edges)
    Degrees: [1, 2, 2, 2, 2, 1]
    """
    fA_third = fA / 3.0
    fB_half = (1.0 - fA) / 2.0
    x = torch.zeros(N_max, 3, dtype=torch.float)

    x[0, 0] = 1.0; x[1, 0] = 2.0; x[2, 0] = 2.0; x[3, 0] = 2.0; x[4, 0] = 2.0; x[5, 0] = 1.0
    x[0, 1] = 2.0; x[1, 1] = 1.0; x[2, 1] = 2.0; x[3, 1] = 2.0; x[4, 1] = 1.0; x[5, 1] = 2.0

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    ], dtype=torch.long)

    edge_attr = torch.zeros(10, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fA_third, 1.0]); edge_attr[1] = torch.tensor([fA_third, 1.0])
    edge_attr[2] = torch.tensor([fB_half, 0.0]); edge_attr[3] = torch.tensor([fB_half, 0.0])
    edge_attr[4] = torch.tensor([fA_third, 1.0]); edge_attr[5] = torch.tensor([fA_third, 1.0])
    edge_attr[6] = torch.tensor([fB_half, 0.0]); edge_attr[7] = torch.tensor([fB_half, 0.0])
    edge_attr[8] = torch.tensor([fA_third, 1.0]); edge_attr[9] = torch.tensor([fA_third, 1.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([5]), topo_type=torch.tensor([3]))


def build_ababab_graph(fA: float, N_max: int = 32) -> Data:
    """Build a linear ABABAB alternating hexablock copolymer graph."""
    fa = fA / 3.0
    fb = (1.0 - fA) / 3.0
    x = torch.zeros(N_max, 3, dtype=torch.float)
    x[:7, 0] = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])

    forward = [(index, index + 1) for index in range(6)]
    directed = [edge for pair in forward for edge in (pair, pair[::-1])]
    edge_index = torch.tensor(directed, dtype=torch.long).T.contiguous()

    edge_attr = torch.zeros(12, 2, dtype=torch.float)
    for block_index, fraction in enumerate([fa, fb, fa, fb, fa, fb]):
        block_type = 1.0 if block_index % 2 == 0 else 0.0
        edge_attr[2 * block_index:2 * block_index + 2] = torch.tensor(
            [fraction, block_type], dtype=torch.float
        )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([6]), topo_type=torch.tensor([12]))


def build_ab2_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for AB2 star copolymer matching training data format.

    Structure (0-indexed):
        1 --(B1)-- 0 --(A)-- 2
                  |
                 3 --(B2)--

    Nodes: 4, Edges: 3 blocks
    Degrees: [3, 1, 1, 1] (branch point + 3 free ends)
    """
    fB = (1.0 - fA) / 2.0
    x = torch.zeros(N_max, 3, dtype=torch.float)

    x[0, 0] = 3.0; x[1, 0] = 1.0; x[2, 0] = 1.0; x[3, 0] = 1.0
    x[0, 1] = 1.0; x[1, 1] = 3.0; x[2, 1] = 3.0; x[3, 1] = 3.0

    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3],
        [1, 0, 2, 0, 3, 0]
    ], dtype=torch.long)

    edge_attr = torch.zeros(6, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fA, 1.0]); edge_attr[1] = torch.tensor([fA, 1.0])
    edge_attr[2] = torch.tensor([fB, 0.0]); edge_attr[3] = torch.tensor([fB, 0.0])
    edge_attr[4] = torch.tensor([fB, 0.0]); edge_attr[5] = torch.tensor([fB, 0.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([3]), topo_type=torch.tensor([4]))


def build_ab3_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for AB3 star copolymer matching training data format.

    Structure (0-indexed):
        1 --(B1)-- 0 --(A)-- 2
                  |
                 3 --(B2)--
                  |
                 4 --(B3)--

    Nodes: 5, Edges: 4 blocks
    Degrees: [4, 1, 1, 1, 1] (branch point + 4 free ends)
    """
    fB = (1.0 - fA) / 3.0
    x = torch.zeros(N_max, 3, dtype=torch.float)

    x[0, 0] = 4.0; x[1, 0] = 1.0; x[2, 0] = 1.0; x[3, 0] = 1.0; x[4, 0] = 1.0
    x[0, 1] = 1.0; x[1, 1] = 4.0; x[2, 1] = 4.0; x[3, 1] = 4.0; x[4, 1] = 4.0

    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 0, 4],
        [1, 0, 2, 0, 3, 0, 4, 0]
    ], dtype=torch.long)

    edge_attr = torch.zeros(8, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fA, 1.0]); edge_attr[1] = torch.tensor([fA, 1.0])
    edge_attr[2] = torch.tensor([fB, 0.0]); edge_attr[3] = torch.tensor([fB, 0.0])
    edge_attr[4] = torch.tensor([fB, 0.0]); edge_attr[5] = torch.tensor([fB, 0.0])
    edge_attr[6] = torch.tensor([fB, 0.0]); edge_attr[7] = torch.tensor([fB, 0.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([4]), topo_type=torch.tensor([5]))


def build_m1_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for M1 matching gen_m1_m2_gnn.jl structure.

    Structure:
        A1 = (FreeEnd, eb0)
        B1 = (eb0, eb1)
        B2 = (eb0, eb2)
        B3 = (eb0, eb3)
        A2 = (eb2, FreeEnd)

    Nodes (0-indexed):
      0: FreeEnd(A1) - degree=1
      1: eb0 - degree=4 (connected to A1, B1, B2, B3)
      2: eb1 - degree=1 (B1 endpoint)
      3: eb2 - degree=2 (B2/A2 junction)
      4: eb3 - degree=1 (B3 endpoint)
      5: FreeEnd(A2) - degree=1

    5 blocks = 10 directed edges
    """
    fa = fA / 2.0
    fb1 = 0.05 * (1.0 - fA)
    fb2 = 0.25 * (1.0 - fA)
    fb3 = 0.70 * (1.0 - fA)

    x = torch.zeros(N_max, 3, dtype=torch.float)
    # Node degrees: [1, 4, 1, 2, 1, 1]
    for i, deg in enumerate([1, 4, 1, 2, 1, 1]):
        x[i, 0] = float(deg)

    # Average neighbor degrees
    # Node 0: neighbor eb0(4) -> avg=4.0
    # Node 1: neighbors eb1(1), eb2(2), eb3(1), A1(1) -> avg=(1+2+1+1)/4=1.25
    # Node 2: neighbor eb0(4) -> avg=4.0
    # Node 3: neighbors eb0(4), A2(1) -> avg=(4+1)/2=2.5
    # Node 4: neighbor eb0(4) -> avg=4.0
    # Node 5: neighbor eb2(2) -> avg=2.0
    avg_nbr = [4.0, 1.25, 4.0, 2.5, 4.0, 2.0]
    for i, v in enumerate(avg_nbr):
        x[i, 1] = v

    # Edge index: 5 blocks = 10 directed edges
    # A1: 0->1, 1->0
    # B1: 1->2, 2->1
    # B2: 1->3, 3->1
    # B3: 1->4, 4->1
    # A2: 3->5, 5->3
    edge_index = torch.tensor([
        [0, 1, 1, 2, 1, 3, 1, 4, 3, 5],
        [1, 0, 2, 1, 3, 1, 4, 1, 5, 3]
    ], dtype=torch.long)

    edge_attr = torch.zeros(10, 2, dtype=torch.float)
    # A1: fa
    edge_attr[0] = torch.tensor([fa, 1.0]); edge_attr[1] = torch.tensor([fa, 1.0])
    # B1: fb1
    edge_attr[2] = torch.tensor([fb1, 0.0]); edge_attr[3] = torch.tensor([fb1, 0.0])
    # B2: fb2
    edge_attr[4] = torch.tensor([fb2, 0.0]); edge_attr[5] = torch.tensor([fb2, 0.0])
    # B3: fb3
    edge_attr[6] = torch.tensor([fb3, 0.0]); edge_attr[7] = torch.tensor([fb3, 0.0])
    # A2: fa
    edge_attr[8] = torch.tensor([fa, 1.0]); edge_attr[9] = torch.tensor([fa, 1.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([5]), topo_type=torch.tensor([6]))


def build_m2_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for M2 matching gen_m1_m2_gnn.jl structure.

    Structure:
        A1 = (FreeEnd, eb0)
        B1 = (eb0, eb1)
        B2 = (eb0, eb2)
        A2 = (eb2, FreeEnd)
        (no B3)

    Nodes (0-indexed):
      0: FreeEnd(A1) - degree=1
      1: eb0 - degree=3 (connected to A1, B1, B2)
      2: eb1 - degree=1 (B1 endpoint)
      3: eb2 - degree=2 (B2/A2 junction)
      4: FreeEnd(A2) - degree=1

    4 blocks = 8 directed edges
    """
    fa = fA / 2.0
    fb1 = 0.74 * (1.0 - fA)  # τB1 = 1 - τB2 = 0.74
    fb2 = 0.26 * (1.0 - fA)  # τB2 = 0.26

    x = torch.zeros(N_max, 3, dtype=torch.float)
    # Node degrees: [1, 3, 1, 2, 1]
    for i, deg in enumerate([1, 3, 1, 2, 1]):
        x[i, 0] = float(deg)

    # Average neighbor degrees
    # Node 0: neighbor eb0(3) -> avg=3.0
    # Node 1: neighbors eb1(1), eb2(2), A1(1) -> avg=(1+2+1)/3=1.333
    # Node 2: neighbor eb0(3) -> avg=3.0
    # Node 3: neighbors eb0(3), A2(1) -> avg=(3+1)/2=2.0
    # Node 4: neighbor eb2(2) -> avg=2.0
    avg_nbr = [3.0, 1.333, 3.0, 2.0, 2.0]
    for i, v in enumerate(avg_nbr):
        x[i, 1] = v

    # Edge index: 4 blocks = 8 directed edges
    # A1: 0->1, 1->0
    # B1: 1->2, 2->1
    # B2: 1->3, 3->1
    # A2: 3->4, 4->3
    edge_index = torch.tensor([
        [0, 1, 1, 2, 1, 3, 3, 4],
        [1, 0, 2, 1, 3, 1, 4, 3]
    ], dtype=torch.long)

    edge_attr = torch.zeros(8, 2, dtype=torch.float)
    # A1: fa
    edge_attr[0] = torch.tensor([fa, 1.0]); edge_attr[1] = torch.tensor([fa, 1.0])
    # B1: fb1
    edge_attr[2] = torch.tensor([fb1, 0.0]); edge_attr[3] = torch.tensor([fb1, 0.0])
    # B2: fb2
    edge_attr[4] = torch.tensor([fb2, 0.0]); edge_attr[5] = torch.tensor([fb2, 0.0])
    # A2: fa
    edge_attr[6] = torch.tensor([fa, 1.0]); edge_attr[7] = torch.tensor([fa, 1.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([4]), topo_type=torch.tensor([7]))


def build_btabat_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for the (BT)AB(AT) copolymer in Xie et al. Figure 4.

    Structure:
        A1 = (N_A1, tA)
        A2 = (tA, AB)
        B1 = (AB, tB)
        B2 = (tB, N_B2)
        B3 = (tA, N_B3)
        A3 = (tB, N_A3)

    Fractions:
        A1=A2=fA/4, A3=fA/2
        B1=B2=0.15(1-fA), B3=0.70(1-fA)
    """
    fa1 = 0.25 * fA
    fa2 = 0.25 * fA
    fa3 = 0.50 * fA
    fb1 = 0.15 * (1.0 - fA)
    fb2 = 0.15 * (1.0 - fA)
    fb3 = 0.70 * (1.0 - fA)

    x = torch.zeros(N_max, 3, dtype=torch.float)
    degrees = [1, 3, 2, 3, 1, 1, 1]
    avg_nbr = [3.0, 4.0 / 3.0, 3.0, 4.0 / 3.0, 3.0, 3.0, 3.0]
    for i, deg in enumerate(degrees):
        x[i, 0] = float(deg)
    for i, v in enumerate(avg_nbr):
        x[i, 1] = float(v)

    # Nodes: 0=N_A1, 1=tA, 2=AB, 3=tB, 4=N_B2, 5=N_B3, 6=N_A3
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 3, 6],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 3],
    ], dtype=torch.long)

    edge_attr = torch.zeros(12, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fa1, 1.0]); edge_attr[1] = torch.tensor([fa1, 1.0])
    edge_attr[2] = torch.tensor([fa2, 1.0]); edge_attr[3] = torch.tensor([fa2, 1.0])
    edge_attr[4] = torch.tensor([fb1, 0.0]); edge_attr[5] = torch.tensor([fb1, 0.0])
    edge_attr[6] = torch.tensor([fb2, 0.0]); edge_attr[7] = torch.tensor([fb2, 0.0])
    edge_attr[8] = torch.tensor([fb3, 0.0]); edge_attr[9] = torch.tensor([fb3, 0.0])
    edge_attr[10] = torch.tensor([fa3, 1.0]); edge_attr[11] = torch.tensor([fa3, 1.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([6]), topo_type=torch.tensor([8]))


def build_b2a2_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for Shi 2021 B2A2 with two A and two B arms at one J1.

    Nodes:
      0: J1, degree=4
      1,2: B free ends
      3,4: A free ends
    Fractions:
      A1=A2=fA/2; B1=B2=(1-fA)/2
    """
    fa = fA / 2.0
    fb = (1.0 - fA) / 2.0

    x = torch.zeros(N_max, 3, dtype=torch.float)
    degrees = [4, 1, 1, 1, 1]
    avg_nbr = [1.0, 4.0, 4.0, 4.0, 4.0]
    for i, deg in enumerate(degrees):
        x[i, 0] = float(deg)
    for i, value in enumerate(avg_nbr):
        x[i, 1] = float(value)

    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 0, 4],
        [1, 0, 2, 0, 3, 0, 4, 0],
    ], dtype=torch.long)

    edge_attr = torch.zeros(8, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fb, 0.0]); edge_attr[1] = torch.tensor([fb, 0.0])
    edge_attr[2] = torch.tensor([fb, 0.0]); edge_attr[3] = torch.tensor([fb, 0.0])
    edge_attr[4] = torch.tensor([fa, 1.0]); edge_attr[5] = torch.tensor([fa, 1.0])
    edge_attr[6] = torch.tensor([fa, 1.0]); edge_attr[7] = torch.tensor([fa, 1.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([4]), topo_type=torch.tensor([9]))


def build_b2a2_2_graph(fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for Shi 2021 (B2A2)2 comb-like topology.

    The two B2A2 units are connected through an A-side J2 junction:
      J1a has B1, B2, A1, A2; J1b has B3, B4, A3, A4; A2-J2-A3 bridges the units.
    Fractions:
      A1=A2=A3=A4=fA/4; B1=B2=B3=B4=(1-fA)/4
    """
    fa = fA / 4.0
    fb = (1.0 - fA) / 4.0

    x = torch.zeros(N_max, 3, dtype=torch.float)
    degrees = [4, 1, 1, 1, 2, 4, 1, 1, 1]
    avg_nbr = [1.25, 4.0, 4.0, 4.0, 4.0, 1.25, 4.0, 4.0, 4.0]
    for i, deg in enumerate(degrees):
        x[i, 0] = float(deg)
    for i, value in enumerate(avg_nbr):
        x[i, 1] = float(value)

    # Nodes: 0=J1a, 1=N_B1, 2=N_B2, 3=N_A1, 4=J2, 5=J1b, 6=N_A4, 7=N_B3, 8=N_B4
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 0, 4, 4, 5, 5, 6, 5, 7, 5, 8],
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 4, 6, 5, 7, 5, 8, 5],
    ], dtype=torch.long)

    edge_attr = torch.zeros(16, 2, dtype=torch.float)
    edge_attr[0] = torch.tensor([fb, 0.0]); edge_attr[1] = torch.tensor([fb, 0.0])
    edge_attr[2] = torch.tensor([fb, 0.0]); edge_attr[3] = torch.tensor([fb, 0.0])
    edge_attr[4] = torch.tensor([fa, 1.0]); edge_attr[5] = torch.tensor([fa, 1.0])
    edge_attr[6] = torch.tensor([fa, 1.0]); edge_attr[7] = torch.tensor([fa, 1.0])
    edge_attr[8] = torch.tensor([fa, 1.0]); edge_attr[9] = torch.tensor([fa, 1.0])
    edge_attr[10] = torch.tensor([fa, 1.0]); edge_attr[11] = torch.tensor([fa, 1.0])
    edge_attr[12] = torch.tensor([fb, 0.0]); edge_attr[13] = torch.tensor([fb, 0.0])
    edge_attr[14] = torch.tensor([fb, 0.0]); edge_attr[15] = torch.tensor([fb, 0.0])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_blocks=torch.tensor([8]), topo_type=torch.tensor([10]))


def build_copolymer_graph(topo_type: str, fA: float, N_max: int = 32) -> Data:
    """Build PyG graph for a given copolymer topology using degree features."""
    if topo_type == 'AB':
        return build_ab_graph(fA, N_max)
    elif topo_type == 'ABA':
        return build_aba_graph(fA, N_max)
    elif topo_type == 'ABAB':
        return build_abab_graph(fA, N_max)
    elif topo_type == 'ABABA':
        return build_ababa_graph(fA, N_max)
    elif topo_type == 'ABABAB':
        return build_ababab_graph(fA, N_max)
    elif topo_type == 'AB2':
        return build_ab2_graph(fA, N_max)
    elif topo_type == 'AB3':
        return build_ab3_graph(fA, N_max)
    elif topo_type == 'M1':
        return build_m1_graph(fA, N_max)
    elif topo_type == 'M2':
        return build_m2_graph(fA, N_max)
    elif topo_type == 'BTABAT':
        return build_btabat_graph(fA, N_max)
    elif topo_type == 'B2A2':
        return build_b2a2_graph(fA, N_max)
    elif topo_type in {'(B2A2)2', 'B2A2_2'}:
        return build_b2a2_2_graph(fA, N_max)
    else:
        raise ValueError(f"Unknown topology: {topo_type}")
