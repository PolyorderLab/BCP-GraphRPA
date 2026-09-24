"""
RPA Calculator for block copolymers using Julia's PolymerArchitecture package.

Optimized version that batches calculations in a single Julia subprocess.
"""

import os
import subprocess
import numpy as np
from typing import Tuple, Dict, List
import re
from pathlib import Path


JULIA_PROJECT = os.environ.get(
    "TAPD_JULIA_PROJECT",
    str(Path(__file__).resolve().parents[2] / "data" / "generation" / "rpa"),
)


def compute_rpa_batch_julia(topology: str, fA_values: List[float], chiN0: float = 15.0,
                            julia_threads: int = 112) -> np.ndarray:
    """
    Batch compute RPA stability limits in a single Julia call.

    Args:
        topology: Topology type ('AB', 'ABA', 'ABAB', 'ABABA')
        fA_values: List of fA values
        chiN0: Initial guess for chiN

    Returns:
        Array of chiN* values
    """
    topo_code = {
        'AB': 2,
        'ABA': 3,
        'ABAB': 4,
        'ABABA': 5,
        'AB2': 6,
        'AB3': 7,
        'M1': 8,
        'M2': 9,
        'BTABAT': 10,
        'B2A2': 11,
        '(B2A2)2': 12,
        'B2A2_2': 12,
        'ABABAB': 13,
    }.get(topology, 2)

    n_points = len(fA_values)
    fA_str = "[" + ",".join(str(f) for f in fA_values) + "]"

    julia_script = f'''
using Pkg
Pkg.activate(raw"{JULIA_PROJECT}")
using LinearAlgebra
using Roots
using Optim
using Polymer
using PolymerArchitecture
using Polyorder: RPA

LinearAlgebra.BLAS.set_num_threads({julia_threads})

chiN0 = {chiN0}
k2_max = 100.0
topo_code = {topo_code}
fA_arr = {fA_str}
n = {n_points}

results = Vector{{Float64}}(undef, n)

for i in 1:n
    fA = fA_arr[i]

    if topo_code == 2
        chain = BlockCopolymer(
            Symbol("AB"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:A), fA, FreeEnd(Symbol("N1")), BranchPoint(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), 1.0-fA, BranchPoint(Symbol("N2")), FreeEnd(Symbol("N3")))
            ];
            check=false
        )
    elseif topo_code == 3
        fa = fA / 2.0
        fb = 1.0 - fA
        chain = BlockCopolymer(
            Symbol("ABA"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:A), fa, FreeEnd(Symbol("N1")), BranchPoint(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("N2")), BranchPoint(Symbol("N3"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:A), fa, BranchPoint(Symbol("N3")), FreeEnd(Symbol("N4")))
            ];
            check=false
        )
    elseif topo_code == 4
        fa = fA / 2.0
        fb = (1.0 - fA) / 2.0
        chain = BlockCopolymer(
            Symbol("ABAB"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:A), fa, FreeEnd(Symbol("N1")), BranchPoint(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("N2")), BranchPoint(Symbol("N3"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:A), fa, BranchPoint(Symbol("N3")), BranchPoint(Symbol("N4"))),
                PolymerBlock(Symbol("b4"), KuhnSegment(:B), fb, BranchPoint(Symbol("N4")), FreeEnd(Symbol("N5")))
            ];
            check=false
        )
    elseif topo_code == 5
        fa = fA / 3.0
        fb = (1.0 - fA) / 2.0
        chain = BlockCopolymer(
            Symbol("ABABA"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:A), fa, FreeEnd(Symbol("N1")), BranchPoint(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("N2")), BranchPoint(Symbol("N3"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:A), fa, BranchPoint(Symbol("N3")), BranchPoint(Symbol("N4"))),
                PolymerBlock(Symbol("b4"), KuhnSegment(:B), fb, BranchPoint(Symbol("N4")), BranchPoint(Symbol("N5"))),
                PolymerBlock(Symbol("b5"), KuhnSegment(:A), fa, BranchPoint(Symbol("N5")), FreeEnd(Symbol("N6")))
            ];
            check=false
        )
    elseif topo_code == 6  # AB2: Star - A at one end, 2 B branches
        fa = fA
        fb = (1.0 - fA) / 2.0
        chain = BlockCopolymer(
            Symbol("AB2"),
            [
                PolymerBlock(Symbol("a1"), KuhnSegment(:A), fa, FreeEnd(Symbol("N1")), BranchPoint(Symbol("bp"))),
                PolymerBlock(Symbol("b1"), KuhnSegment(:B), fb, BranchPoint(Symbol("bp")), FreeEnd(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("bp")), FreeEnd(Symbol("N3")))
            ];
            check=false
        )
    elseif topo_code == 7  # AB3: Star - A at one end, 3 B branches
        fa = fA
        fb = (1.0 - fA) / 3.0
        chain = BlockCopolymer(
            Symbol("AB3"),
            [
                PolymerBlock(Symbol("a1"), KuhnSegment(:A), fa, FreeEnd(Symbol("N1")), BranchPoint(Symbol("bp"))),
                PolymerBlock(Symbol("b1"), KuhnSegment(:B), fb, BranchPoint(Symbol("bp")), FreeEnd(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("bp")), FreeEnd(Symbol("N3"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:B), fb, BranchPoint(Symbol("bp")), FreeEnd(Symbol("N4")))
            ];
            check=false
        )
    elseif topo_code == 8  # M1
        # 缁撴瀯:
        #   A1 = (FreeEnd, eb0)
        #   B1 = (eb0, eb1)
        #   B2 = (eb0, eb2)
        #   B3 = (eb0, eb3)
        #   A2 = (eb2, FreeEnd)
        # 蟿B1=0.05, 蟿B2=0.25, 蟿B3=0.70
        fa  = fA / 2.0
        fb1 = 0.05 * (1.0 - fA)
        fb2 = 0.25 * (1.0 - fA)
        fb3 = 0.70 * (1.0 - fA)
        chain = BlockCopolymer(
            Symbol("M1"),
            [
                PolymerBlock(Symbol("a1"),  KuhnSegment(:A), fa,  FreeEnd(Symbol("N_A1")),       BranchPoint(Symbol("eb0"))),
                PolymerBlock(Symbol("b1"),  KuhnSegment(:B), fb1, BranchPoint(Symbol("eb0")),   BranchPoint(Symbol("eb1"))),
                PolymerBlock(Symbol("b2"),  KuhnSegment(:B), fb2, BranchPoint(Symbol("eb0")),   BranchPoint(Symbol("eb2"))),
                PolymerBlock(Symbol("b3"),  KuhnSegment(:B), fb3, BranchPoint(Symbol("eb0")),   BranchPoint(Symbol("eb3"))),
                PolymerBlock(Symbol("a2"),  KuhnSegment(:A), fa,  BranchPoint(Symbol("eb2")),   FreeEnd(Symbol("N_A2"))),
            ];
            check=false
        )
    elseif topo_code == 9  # M2
        # 缁撴瀯:
        #   A1 = (FreeEnd, eb0)
        #   B1 = (eb0, eb1)
        #   B2 = (eb0, eb2)
        #   A2 = (eb2, FreeEnd)
        #   (娌℃湁B3)
        # 蟿B2=0.26, 蟿B1=1-蟿B2=0.74
        fa  = fA / 2.0
        fb1 = 0.74 * (1.0 - fA)
        fb2 = 0.26 * (1.0 - fA)
        chain = BlockCopolymer(
            Symbol("M2"),
            [
                PolymerBlock(Symbol("a1"),  KuhnSegment(:A), fa,  FreeEnd(Symbol("N_A1")),       BranchPoint(Symbol("eb0"))),
                PolymerBlock(Symbol("b1"),  KuhnSegment(:B), fb1, BranchPoint(Symbol("eb0")),   BranchPoint(Symbol("eb1"))),
                PolymerBlock(Symbol("b2"),  KuhnSegment(:B), fb2, BranchPoint(Symbol("eb0")),   BranchPoint(Symbol("eb2"))),
                PolymerBlock(Symbol("a2"),  KuhnSegment(:A), fa,  BranchPoint(Symbol("eb2")),   FreeEnd(Symbol("N_A2"))),
            ];
            check=false
        )
    elseif topo_code == 10  # BTABAT: (BT)AB(AT) from Xie et al. Figure 4
        # Main chain: A1-A2-B1-B2.
        # B3 is tethered at the A1/A2 junction; A3 is tethered at the B1/B2 junction.
        fa1 = 0.25 * fA
        fa2 = 0.25 * fA
        fa3 = 0.50 * fA
        fb1 = 0.15 * (1.0 - fA)
        fb2 = 0.15 * (1.0 - fA)
        fb3 = 0.70 * (1.0 - fA)
        chain = BlockCopolymer(
            Symbol("BTABAT"),
            [
                PolymerBlock(Symbol("a1"), KuhnSegment(:A), fa1, FreeEnd(Symbol("N_A1")),       BranchPoint(Symbol("tA"))),
                PolymerBlock(Symbol("a2"), KuhnSegment(:A), fa2, BranchPoint(Symbol("tA")),     BranchPoint(Symbol("AB"))),
                PolymerBlock(Symbol("b1"), KuhnSegment(:B), fb1, BranchPoint(Symbol("AB")),     BranchPoint(Symbol("tB"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb2, BranchPoint(Symbol("tB")),     FreeEnd(Symbol("N_B2"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:B), fb3, BranchPoint(Symbol("tA")),     FreeEnd(Symbol("N_B3"))),
                PolymerBlock(Symbol("a3"), KuhnSegment(:A), fa3, BranchPoint(Symbol("tB")),     FreeEnd(Symbol("N_A3"))),
            ];
            check=false
        )
    elseif topo_code == 11  # B2A2: two B arms and two A arms sharing one J1 junction
        fa = fA / 2.0
        fb = (1.0 - fA) / 2.0
        chain = BlockCopolymer(
            Symbol("B2A2"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:B), fb, BranchPoint(Symbol("J1")), FreeEnd(Symbol("N_B1"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("J1")), FreeEnd(Symbol("N_B2"))),
                PolymerBlock(Symbol("a1"), KuhnSegment(:A), fa, BranchPoint(Symbol("J1")), FreeEnd(Symbol("N_A1"))),
                PolymerBlock(Symbol("a2"), KuhnSegment(:A), fa, BranchPoint(Symbol("J1")), FreeEnd(Symbol("N_A2"))),
            ];
            check=false
        )
    elseif topo_code == 12  # (B2A2)2: two B2A2 units connected through one A-side J2 junction
        fa = fA / 4.0
        fb = (1.0 - fA) / 4.0
        chain = BlockCopolymer(
            Symbol("B2A2_2"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:B), fb, BranchPoint(Symbol("J1a")), FreeEnd(Symbol("N_B1"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("J1a")), FreeEnd(Symbol("N_B2"))),
                PolymerBlock(Symbol("a1"), KuhnSegment(:A), fa, BranchPoint(Symbol("J1a")), FreeEnd(Symbol("N_A1"))),
                PolymerBlock(Symbol("a2"), KuhnSegment(:A), fa, BranchPoint(Symbol("J1a")), BranchPoint(Symbol("J2"))),
                PolymerBlock(Symbol("a3"), KuhnSegment(:A), fa, BranchPoint(Symbol("J2")), BranchPoint(Symbol("J1b"))),
                PolymerBlock(Symbol("a4"), KuhnSegment(:A), fa, BranchPoint(Symbol("J1b")), FreeEnd(Symbol("N_A4"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:B), fb, BranchPoint(Symbol("J1b")), FreeEnd(Symbol("N_B3"))),
                PolymerBlock(Symbol("b4"), KuhnSegment(:B), fb, BranchPoint(Symbol("J1b")), FreeEnd(Symbol("N_B4"))),
            ];
            check=false
        )
    elseif topo_code == 13  # ABABAB alternating linear hexablock
        fa = fA / 3.0
        fb = (1.0 - fA) / 3.0
        chain = BlockCopolymer(
            Symbol("ABABAB"),
            [
                PolymerBlock(Symbol("b1"), KuhnSegment(:A), fa, FreeEnd(Symbol("N1")), BranchPoint(Symbol("N2"))),
                PolymerBlock(Symbol("b2"), KuhnSegment(:B), fb, BranchPoint(Symbol("N2")), BranchPoint(Symbol("N3"))),
                PolymerBlock(Symbol("b3"), KuhnSegment(:A), fa, BranchPoint(Symbol("N3")), BranchPoint(Symbol("N4"))),
                PolymerBlock(Symbol("b4"), KuhnSegment(:B), fb, BranchPoint(Symbol("N4")), BranchPoint(Symbol("N5"))),
                PolymerBlock(Symbol("b5"), KuhnSegment(:A), fa, BranchPoint(Symbol("N5")), BranchPoint(Symbol("N6"))),
                PolymerBlock(Symbol("b6"), KuhnSegment(:B), fb, BranchPoint(Symbol("N6")), FreeEnd(Symbol("N7")))
            ];
            check=false
        )
    end

    sys = TwoSpeciesSystem()
    try
        res = RPA.compute_stability_limit(chain, sys, chiN0, k2_max)
        results[i] = res[1]
    catch e
        results[i] = NaN
    end
end

println("RESULT:", join(results, ","))
'''

    result = subprocess.run(
        ['julia', '-e', julia_script],
        capture_output=True,
        text=True,
        timeout=600
    )

    output = result.stdout.strip()

    match = re.search(r'RESULT:([\d.,nan-]+)', output)
    if match:
        values = []
        for v in match.group(1).split(','):
            if v.lower() == 'nan':
                values.append(np.nan)
            else:
                values.append(float(v))
        return np.array(values)
    else:
        print(f"Julia output: {output}")
        print(f"Julia stderr: {result.stderr}")
        raise RuntimeError(f"Failed to parse Julia results")


def compute_rpa_phase_boundary(topology: str,
                               fA_range: Tuple[float, float] = (0.01, 0.99),
                               num_points: int = 100,
                               chiN0: float = 15.0,
                               julia_threads: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full RPA phase boundary using Julia batched calculation.

    Args:
        topology: Topology ('AB', 'ABA', 'ABAB', 'ABABA')
        fA_range: Range of fA values
        num_points: Number of fA points
        chiN0: Initial guess for chiN
        julia_threads: Number of threads for Julia BLAS

    Returns:
        Tuple of (fA array, chiN_star array)
    """
    fA_vals = np.linspace(fA_range[0], fA_range[1], num_points)
    print(f"Computing RPA phase boundary for {topology} ({num_points} points)...")

    chiN_vals = compute_rpa_batch_julia(topology, fA_vals.tolist(), chiN0, julia_threads)

    return fA_vals, chiN_vals


def get_all_phase_boundaries(fA_range: Tuple[float, float] = (0.01, 0.99),
                              num_points: int = 100,
                              chiN0: float = 15.0,
                              julia_threads: int = 112,
                              topologies: List[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute RPA phase boundaries for all topologies using Julia.

    Args:
        fA_range: Range of fA to compute
        num_points: Number of fA points
        chiN0: Initial guess for chiN
        julia_threads: Number of threads for Julia BLAS
        topologies: List of topologies to compute (default: AB, ABA, ABAB, ABABA)

    Returns:
        Dictionary mapping topology to (fA, chiN_star) tuples
    """
    if topologies is None:
        topologies = ['AB', 'ABA', 'ABAB', 'ABABA', 'AB2', 'AB3', 'M1', 'M2']
    results = {}

    for topo in topologies:
        fA_vals, chiN_vals = compute_rpa_phase_boundary(topo, fA_range, num_points, chiN0, julia_threads)
        results[topo] = (fA_vals, chiN_vals)

    return results


def compute_rpa_stability_limit_julia(fA: float, topology: str, chiN0: float = 15.0) -> Tuple[float, float]:
    """Single point calculation (convenience function)."""
    result = compute_rpa_batch_julia(topology, [fA], chiN0)
    return result[0], 0.0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python rpa_calculator_julia.py <fA> <topology>")
        print("Example: python rpa_calculator_julia.py 0.5 AB")
        sys.exit(1)

    fA = float(sys.argv[1])
    topo = sys.argv[2]

    chiN, k = compute_rpa_stability_limit_julia(fA, topo)
    print(f"chiN* = {chiN:.6f}")
