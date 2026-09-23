#!/usr/bin/env python3
"""Parse the frozen 100k YML corpus once for reuse by all training runs."""

import hashlib
import json
import pickle
import sys
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
DATA_DIR = ROOT / "TrainData/data/chiNlimited/n_train_100k"
OUTPUT = ROOT / "baselinemodels/pub0818/transfomer/data_cache/n_train_100k_records.pkl"
MODEL_DIR = ROOT / "baselinemodels/models"
sys.path.insert(0, str(MODEL_DIR))

from baseline_common import load_yml_records  # noqa: E402


def main():
    records = load_yml_records(DATA_DIR, "chiN")
    if len(records) != 100000:
        raise RuntimeError(f"Expected 100000 records, found {len(records)}")
    payload = {
        "data_directory": str(DATA_DIR.resolve()),
        "record_count": len(records),
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(OUTPUT)
    metadata = {key: value for key, value in payload.items() if key != "records"}
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
