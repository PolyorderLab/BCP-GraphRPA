"""Recompute every bundled RPA value on exactly the published f_A grids.

The bundled JSON files are immutable provenance snapshots.  This script writes
fresh calculations to ``results/rpa_recomputed.json`` and a pointwise audit to
``results/rpa_recomputed_comparison.csv`` without overwriting the snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_DIR / "scripts"))

from rpa_calculator_julia import compute_rpa_batch_julia  # noqa: E402

INPUTS = {
    "global": ("global_phase_diagrams.json", ["AB", "ABA", "ABAB", "ABABA", "AB2", "AB3", "M1", "M2"]),
    "m1_m2_local": ("m1_m2_local.json", ["M1", "M2"]),
    "btabat": ("btabat.json", ["BTABAT"]),
    "shi": ("shi_b2a2.json", ["AB", "B2A2", "(B2A2)2"]),
    "ababab": ("ababab.json", ["ABABAB"]),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8, help="Julia BLAS threads")
    parser.add_argument("--chiN0", type=float, default=15.0, help="RPA initial guess")
    args = parser.parse_args()

    input_dir = BUNDLE_DIR / "data" / "rpa_inputs"
    result_dir = BUNDLE_DIR / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    recomputed = {}
    audit_rows = []

    for case, (filename, topologies) in INPUTS.items():
        source = load_json(input_dir / filename)
        recomputed[case] = {}
        for topology in topologies:
            f_a = np.asarray(source[topology]["rpa"]["fA"], dtype=float)
            archived = np.asarray(source[topology]["rpa"]["chiN"], dtype=float)
            calculated = compute_rpa_batch_julia(
                topology, f_a.tolist(), chiN0=args.chiN0, julia_threads=args.threads
            )
            recomputed[case][topology] = {
                "rpa": {"fA": f_a.tolist(), "chiN": calculated.tolist()}
            }
            for index, (x, old, new) in enumerate(zip(f_a, archived, calculated), start=1):
                audit_rows.append({
                    "case": case,
                    "topology": topology,
                    "point_index": index,
                    "fA": float(x),
                    "archived_chiN": float(old),
                    "recomputed_chiN": float(new),
                    "absolute_difference": float(abs(new - old)),
                    "relative_difference_percent": float(abs(new - old) / max(abs(old), 1e-12) * 100),
                })

    with (result_dir / "rpa_recomputed.json").open("w", encoding="utf-8") as handle:
        json.dump({"cases": recomputed}, handle, indent=2)
    fields = list(audit_rows[0])
    with (result_dir / "rpa_recomputed_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Wrote {result_dir / 'rpa_recomputed.json'}")
    print(f"Wrote {result_dir / 'rpa_recomputed_comparison.csv'}")


if __name__ == "__main__":
    main()
