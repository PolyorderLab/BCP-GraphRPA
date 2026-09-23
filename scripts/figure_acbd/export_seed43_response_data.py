#!/usr/bin/env python3
"""Export the b-panel validation response table from the original Seed-43 YAMLs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/datasets/seed43_2k"
OUTPUT = ROOT / "data/figure_acbd/responses/validation_seed43_2k_response_data.csv"


def main() -> None:
    rows = []
    for path in sorted(SOURCE.glob("*.yml")):
        with path.open(encoding="utf-8") as handle:
            system = yaml.safe_load(handle)["system"]
        chi_n = float(system["χN_star"])
        rows.append({
            "sample_id": path.stem,
            "dataset_topology": system["topology"],
            "n_blocks": len(system["components"][0]["blocks"]),
            "chiN_star": chi_n,
            "k_star": float(system["k_star"]),
            "log10_chiN_star": math.log10(chi_n),
        })
    if len(rows) != 2000:
        raise RuntimeError(f"Expected 2,000 Seed-43 YAMLs, found {len(rows)}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
