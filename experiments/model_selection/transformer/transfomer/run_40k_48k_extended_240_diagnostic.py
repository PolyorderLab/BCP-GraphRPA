#!/usr/bin/env python3
"""Rerun the 40k/48k nested Transformer points for exactly 240 epochs.

The initial OFAT learning rate and every other frozen hyperparameter remain
unchanged.  Results are isolated from the formal 120-epoch study so this
post-hoc convergence diagnostic cannot silently replace preregistered runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
BASE_RUNNER = CODE_ROOT / "run_nested_data_scaling_80k_pool.py"
FORMAL_STUDY = CODE_ROOT / "nested_data_scaling_main_ofat_80k_pool_seed42"
STUDY_ROOT = FORMAL_STUDY / "extended_240epoch_diagnostic_40k_48k"
PYTHON = ROOT / ".venv/bin/python"

SIZES_K = [40, 48]
SEEDS = [202601, 202602, 202603, 202604, 202605]
EPOCHS = 240
FROZEN_CONFIG = {
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "d_model": 192,
    "num_heads": 6,
    "num_layers": 3,
    "ff_mult": 6,
    "dropout": 0.0,
    "readout": "attention",
    "position_encoding": "sinusoidal",
    "representation": "full_topology",
    "architecture": "transformer",
}


def load_base_runner():
    spec = importlib.util.spec_from_file_location("nested_scaling_base_240", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STUDY_ROOT = STUDY_ROOT
    module.SIZES_K = SIZES_K
    module.SEEDS = SEEDS
    module.MIN_EPOCHS = EPOCHS
    module.MAX_EPOCHS = EPOCHS
    module.FROZEN_CONFIG = FROZEN_CONFIG
    return module


def verify_frozen_config() -> None:
    manifest = json.loads(
        (CODE_ROOT / "confirmation/experiment_manifest.json").read_text(encoding="utf-8")
    )
    combined = next(config for config in manifest["configs"] if config["config_id"] == "combined_candidate")
    for key, value in FROZEN_CONFIG.items():
        if combined.get(key) != value:
            raise RuntimeError(f"Main OFAT config mismatch for {key}: {combined.get(key)!r} != {value!r}")


def run_final_test(base) -> None:
    environment = os.environ.copy()
    environment["TRANSFORMER_EXPERIMENT_ROOT"] = str(STUDY_ROOT)
    environment["CUDA_VISIBLE_DEVICES"] = os.environ.get("TRANSFORMER_DIAGNOSTIC_TEST_GPU", "0")
    subprocess.run(
        [
            str(PYTHON), str(base.EVALUATOR),
            "--phase", "training",
            "--test-dir", str(base.TEST_DIR),
            "--output-dir", str(STUDY_ROOT / "final_test"),
            "--device", "cuda:0",
            "--save-predictions",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    verify_frozen_config()
    base = load_base_runner()
    base.prepare_protocol()
    manifest_path = STUDY_ROOT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "study": "post_hoc_40k_48k_extended_training_diagnostic",
        "formal_study": str(FORMAL_STUDY),
        "diagnostic_only": True,
        "epochs_per_seed": EPOCHS,
        "initial_learning_rate_unchanged": True,
        "scheduler_unchanged": "ReduceLROnPlateau(factor=0.5, patience=10)",
        "final_test_rule": "Evaluate all frozen size-by-seed checkpoints only after 10/10 diagnostic training runs complete",
        "frozen_hyperparameters_source": str(CODE_ROOT / "confirmation/experiment_manifest.json"),
        "frozen_hyperparameters": FROZEN_CONFIG,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    max_per_gpu = int(os.environ.get("TRANSFORMER_DIAGNOSTIC_MAX_PER_GPU", "1"))
    base.run_training(max_per_gpu=max_per_gpu)
    base.aggregate_validation()
    if sum(1 for _ in (STUDY_ROOT / "training/runs").glob("*/seed*/metrics.json")) != 10:
        raise RuntimeError("Refusing test access before all 10 diagnostic runs complete")
    run_final_test(base)
    base.aggregate_test()
    completion = {
        "status": "complete",
        "training_runs": 10,
        "test_runs": 10,
        "epochs_per_seed": EPOCHS,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (STUDY_ROOT / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
