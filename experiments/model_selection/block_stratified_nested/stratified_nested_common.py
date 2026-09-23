#!/usr/bin/env python3
"""Shared audited block-count-stratified nested subset construction."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


BLOCK_RE = re.compile(r"^(\d+)blocks_sample_")


def load_sample_ids(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"sample_id column missing from {csv_path}")
        return [str(row["sample_id"]) for row in reader]


def block_labels(sample_ids: list[str]) -> np.ndarray:
    labels = []
    for sample_id in sample_ids:
        match = BLOCK_RE.match(sample_id)
        if match is None:
            raise RuntimeError(f"Cannot parse block count from {sample_id!r}")
        labels.append(int(match.group(1)))
    result = np.asarray(labels, dtype=np.int64)
    if set(result.tolist()) != {2, 3, 4, 5}:
        raise RuntimeError(f"Expected block strata 2/3/4/5, got {sorted(set(result.tolist()))}")
    return result


def _largest_remainder_quotas(counts: dict[int, int], size: int) -> dict[int, int]:
    total = sum(counts.values())
    exact = {key: size * value / total for key, value in counts.items()}
    quotas = {key: min(counts[key], int(np.floor(value))) for key, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"Unable to allocate {size} samples across strata")
    return quotas


def build_stratified_nested_indices(
    pool_indices: np.ndarray,
    labels: np.ndarray,
    sizes: list[int] | tuple[int, ...],
    random_state: int = 42,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    pool = np.asarray(pool_indices, dtype=np.int64)
    if len(pool) != len(np.unique(pool)):
        raise RuntimeError("Training pool contains duplicate indices")
    strata = sorted(set(int(labels[index]) for index in pool))
    rng = np.random.default_rng(random_state)
    ordered: dict[int, np.ndarray] = {}
    for stratum in strata:
        members = pool[labels[pool] == stratum].copy()
        ordered[stratum] = rng.permutation(members)
    counts = {key: len(value) for key, value in ordered.items()}

    subsets: dict[int, np.ndarray] = {}
    audit: list[dict] = []
    previous: set[int] = set()
    previous_quotas = {key: 0 for key in strata}
    for size in sorted(sizes):
        quotas = _largest_remainder_quotas(counts, int(size))
        if any(quotas[key] < previous_quotas[key] for key in strata):
            raise RuntimeError(f"Non-monotone stratum quota at size={size}: {quotas}")
        subset = np.sort(np.concatenate([ordered[key][: quotas[key]] for key in strata]))
        current = set(int(value) for value in subset)
        if len(subset) != size or len(current) != size:
            raise RuntimeError(f"Invalid subset size/uniqueness at size={size}")
        if not previous.issubset(current):
            raise RuntimeError(f"Nested-set invariant failed at size={size}")
        observed = Counter(int(labels[index]) for index in subset)
        if dict(observed) != quotas:
            raise RuntimeError(f"Stratum audit failed at size={size}")
        subsets[int(size)] = subset
        audit.append({
            "train_size": int(size),
            "counts_by_block": {str(key): int(quotas[key]) for key in strata},
            "fractions_by_block": {str(key): quotas[key] / size for key in strata},
        })
        previous, previous_quotas = current, quotas

    if max(sizes) == len(pool) and set(subsets[max(sizes)].tolist()) != set(pool.tolist()):
        raise RuntimeError("Largest subset does not recover the complete frozen training pool")
    return subsets, audit


def write_indices(
    output_dir: Path,
    subsets: dict[int, np.ndarray],
    audit: list[dict],
    *,
    validation_indices: np.ndarray | None = None,
    train_key: str = "train_index",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for size, subset in subsets.items():
        label = f"{size // 1000}k"
        payload = {train_key: subset, "subset_random_state": np.asarray(42)}
        if validation_indices is not None:
            payload["validation_indices"] = np.asarray(validation_indices, dtype=np.int64)
        np.savez_compressed(output_dir / f"nested_train_{label}.npz", **payload)
    (output_dir / "stratification_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
