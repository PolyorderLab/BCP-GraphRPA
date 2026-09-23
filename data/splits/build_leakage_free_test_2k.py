#!/usr/bin/env python3
"""Build a block-balanced 2k test set with no MAT-content overlap with training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path




def digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-dir', type=Path, action='append', default=[],
                        help='Additional dataset to exclude by MAT or YML content; repeatable')
    parser.add_argument('--per-block', type=int, default=500)
    args = parser.parse_args()
    TRAIN, POOL, DEST = args.train_dir, args.pool_dir, args.output_dir
    SEED, PER_BLOCK = args.seed, args.per_block
    if not TRAIN.is_dir() or not any(TRAIN.glob('*.mat')):
        parser.error('--train-dir must contain MAT records')
    if not POOL.is_dir() or PER_BLOCK < 1:
        parser.error('--pool-dir must exist and --per-block must be positive')
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {DEST}")
    train_hashes = {digest(path) for path in TRAIN.glob("*.mat")}
    excluded = {'.mat': set(), '.yml': set()}
    for directory in args.exclude_dir:
        if not directory.is_dir() or not any(directory.glob('*.mat')):
            parser.error(f'Exclusion directory must contain MAT records: {directory}')
        for suffix in excluded:
            excluded[suffix].update(digest(p) for p in directory.glob('*' + suffix))
    eligible = defaultdict(list)
    pool_rows = []
    for mat_path in sorted(POOL.glob("*.mat")):
        block_count = int(mat_path.name.split("blocks", 1)[0])
        mat_hash = digest(mat_path)
        leaked = mat_hash in train_hashes
        yml_path = mat_path.with_suffix('.yml')
        if not yml_path.is_file():
            raise FileNotFoundError(yml_path)
        leaked = leaked or mat_hash in excluded['.mat'] or (
            bool(excluded['.yml']) and digest(yml_path) in excluded['.yml'])
        pool_rows.append((mat_path, block_count, mat_hash, leaked))
        if not leaked:
            eligible[block_count].append((mat_path, mat_hash))

    rng = random.Random(SEED)
    selected = []
    for block_count in (2, 3, 4, 5):
        if len(eligible[block_count]) < PER_BLOCK:
            raise RuntimeError(f"Only {len(eligible[block_count])} eligible {block_count}-block samples")
        chosen = rng.sample(eligible[block_count], PER_BLOCK)
        selected.extend((block_count, path, mat_hash) for path, mat_hash in chosen)

    for _, mat_path, _ in selected:
        if not mat_path.with_suffix(".yml").is_file():
            raise FileNotFoundError(mat_path.with_suffix(".yml"))
    DEST.mkdir(parents=True)
    manifest_rows = []
    for block_count, mat_path, mat_hash in sorted(selected, key=lambda item: item[1].name):
        yml_path = mat_path.with_suffix(".yml")
        shutil.copy2(mat_path, DEST / mat_path.name)
        shutil.copy2(yml_path, DEST / yml_path.name)
        manifest_rows.append({
            "sample_name": mat_path.stem,
            "block_count": block_count,
            "source_mat": str(mat_path),
            "overlaps_train_100k": False,
        })

    with (DEST / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    audit = {
        "training_directory": str(TRAIN), "source_pool": str(POOL),
        "destination": str(DEST), "seed": SEED, "per_block": PER_BLOCK,
        "additional_exclusion_directories": [str(p) for p in args.exclude_dir],
        "pool_count_by_block": {str(n): sum(row[1] == n for row in pool_rows) for n in (2, 3, 4, 5)},
        "eligible_nonoverlap_by_block": {str(n): len(eligible[n]) for n in (2, 3, 4, 5)},
        "selected_by_block": {str(n): sum(row[0] == n for row in selected) for n in (2, 3, 4, 5)},
        "selection_rule": "Exclude records whose MAT or YML content duplicates the training or additional exclusion directories, then sample within each block count using the fixed seed",
    }
    (DEST / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
