#!/usr/bin/env python3
"""Build a 20k test pool disjoint from n_train_100k by MAT content."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path




def digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'train-dir', 'fixed-test-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--per-block', type=int, default=5000)
    args = parser.parse_args()
    SOURCE, TRAIN, FIXED_TEST, DEST = args.source_dir, args.train_dir, args.fixed_test_dir, args.output_dir
    PER_BLOCK = args.per_block
    if PER_BLOCK < 1 or any(not p.is_dir() for p in (SOURCE, TRAIN, FIXED_TEST)):
        parser.error('Input directories must exist and count must be positive')
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {DEST}")

    train_hashes = {digest(path) for path in TRAIN.glob("*.mat")}
    fixed_hashes = {digest(path) for path in FIXED_TEST.glob("*.mat")}
    selected = []
    source_counts = {}
    overlap_counts = {}

    for block_count in (2, 3, 4, 5):
        source_paths = sorted((SOURCE / f"{block_count}blocks").glob("*.mat"))
        source_counts[str(block_count)] = len(source_paths)
        nonoverlap = []
        overlap = 0
        for mat_path in source_paths:
            mat_hash = digest(mat_path)
            if mat_hash in train_hashes:
                overlap += 1
            else:
                nonoverlap.append((mat_path, mat_hash))
        overlap_counts[str(block_count)] = overlap
        if len(nonoverlap) != PER_BLOCK:
            raise RuntimeError(
                f"Expected exactly {PER_BLOCK} non-overlapping {block_count}-block samples; "
                f"found {len(nonoverlap)}"
            )
        selected.extend((block_count, path, mat_hash) for path, mat_hash in nonoverlap)

    selected_hashes = {mat_hash for _, _, mat_hash in selected}
    missing_fixed = fixed_hashes - selected_hashes
    if missing_fixed:
        raise RuntimeError(f"The fixed leakage-free 2k set has {len(missing_fixed)} samples absent from new pool")

    DEST.mkdir(parents=True)
    rows = []
    for block_count, mat_path, mat_hash in selected:
        yml_path = mat_path.with_suffix(".yml")
        dest_stem = f"{block_count}blocks_{mat_path.stem}"
        dest_mat = DEST / f"{dest_stem}.mat"
        dest_yml = DEST / f"{dest_stem}.yml"
        shutil.copy2(mat_path, dest_mat)
        shutil.copy2(yml_path, dest_yml)
        rows.append({
            "sample_name": dest_stem,
            "block_count": block_count,
            "source_mat": str(mat_path),
            "overlaps_train_100k": False,
            "belongs_to_fixed_test_2k": mat_hash in fixed_hashes,
        })

    with (DEST / "pool_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final_counts = Counter(row["block_count"] for row in rows)
    fixed_counts = Counter(row["block_count"] for row in rows if row["belongs_to_fixed_test_2k"])
    audit = {
        "source_directory": str(SOURCE),
        "training_directory": str(TRAIN),
        "fixed_test_2k_directory": str(FIXED_TEST),
        "destination": str(DEST),
        "identity_rule": "identical MAT content",
        "source_count_by_block": source_counts,
        "removed_train_overlap_by_block": overlap_counts,
        "final_count_by_block": {str(n): final_counts[n] for n in (2, 3, 4, 5)},
        "fixed_test_2k_preserved_by_block": {str(n): fixed_counts[n] for n in (2, 3, 4, 5)},
        "total_final_samples": len(rows),
        "total_fixed_test_samples_preserved": sum(fixed_counts.values()),
        "regenerated_samples": 0,
        "note": "The original 120k source contained exactly 5,000 non-training samples per block count, so regeneration was unnecessary.",
    }
    (DEST / "pool_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
