#!/usr/bin/env python3
"""Finite, resume-safe L0--L4 local coordinate search around Transformer OFAT.

The search is validation-only.  It uses the verified 100k record cache, one
frozen random_state=42 80k/20k split, and five declared seeds.  L0 reuses the
already completed OFAT anchor (after an identity/hash audit); L1--L3 are
sequential one-coordinate searches, and L4 is a frozen diagnostic audit of
the remaining coordinates.  No final-test directory is opened by this file.

Examples
--------
python run_iterative_local_grid_ofat100k.py --phase prepare
python run_iterative_local_grid_ofat100k.py --phase smoke
python run_iterative_local_grid_ofat100k.py --phase all --wait-for-gnn
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path("/home/zxy/project/TAPD")
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
OUTPUT = CODE_ROOT / "iterative_local_grid_ofat100k_baseline_center"
DATA_DIR = ROOT / "TrainData/data/chiNlimited/n_train_100k"
CACHE = CODE_ROOT / "data_cache/n_train_100k_records.pkl"
TRAINER = CODE_ROOT / "train_validation_only.py"
SOURCE_OFAT = CODE_ROOT / "ofat"
PYTHON = ROOT / ".venv/bin/python"

SPLIT_SEED = 42
TRAIN_COUNT = 80_000
VALIDATION_COUNT = 20_000
EPOCHS = 120
PATIENCE = 25
SEEDS = (202601, 202602, 202603, 202604, 202605)
PROTOCOL_VERSION = "transformer_ofat100k_iterative_local_grid_L0-L4_20260827"

ANCHOR = {
    "batch_size": 40,
    "lr": 2e-4,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 2,
    "ff_mult": 4,
    "dropout": 0.1,
    "readout": "cls",
    "position_encoding": "learnable",
    "weight_decay": 1e-5,
    "representation": "full_topology",
    "architecture": "transformer",
}

TARGET = {
    **ANCHOR,
    "num_layers": 3,
    "dropout": 0.0,
    "weight_decay": 0.0,
}

L1_GRID = {"num_layers": (1, 2, 3)}
L2_GRID = {"dropout": (0.0, 0.05, 0.1)}
L3_GRID = {"weight_decay": (0.0, 1e-6, 1e-5)}
L4_GRID = {
    "batch_size": (32, 40, 64),
    "lr": (1e-4, 2e-4, 5e-4),
    "model_size": ((192, 6), (256, 8)),
    "ff_mult": (2, 4, 6),
    "readout": ("cls", "attention"),
    "position_encoding": ("learnable", "sinusoidal"),
}


def json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_immutable(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise RuntimeError(f"Refusing to overwrite symlink artifact: {path}")
        existing = path.read_bytes()
        desired = payload.encode("utf-8") if isinstance(payload, str) else payload
        if existing != desired:
            raise RuntimeError(f"Immutable artifact differs: {path}")
        return
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)


def write_json(path: Path, payload: object) -> None:
    write_immutable(path, json_text(payload))


def load_cache() -> dict:
    if not CACHE.is_file():
        raise FileNotFoundError(f"Missing verified cache: {CACHE}")
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("data_directory") != str(DATA_DIR.resolve()):
        raise RuntimeError("Cache data-directory mismatch")
    records = payload.get("records", [])
    if payload.get("record_count") != 100_000 or len(records) != 100_000:
        raise RuntimeError("Expected exactly 100,000 cached development records")
    names = [str(record["base_name"]) for record in records]
    if len(set(names)) != len(names):
        raise RuntimeError("Cache contains duplicate base names")
    return payload


def prepare_split(payload: dict) -> dict:
    records = payload["records"]
    all_indices = np.arange(len(records), dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        all_indices, test_size=0.2, random_state=SPLIT_SEED
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if len(train_indices) != TRAIN_COUNT or len(validation_indices) != VALIDATION_COUNT:
        raise RuntimeError("Expected 80k/20k split")
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Train/validation split overlaps")
    split_path = OUTPUT / "fixed_80k_20k_split_indices_seed42.npz"
    if split_path.exists():
        with np.load(split_path) as old:
            if not np.array_equal(old["train_indices"], train_indices) or not np.array_equal(
                old["validation_indices"], validation_indices
            ):
                raise RuntimeError("Existing fixed split differs from deterministic split")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            split_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
    return {
        "path": str(split_path.resolve()),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
    }


def float_slug(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("+", "").replace("-0", "-")


def slug(value: object) -> str:
    if isinstance(value, tuple):
        return f"d{value[0]}h{value[1]}"
    if isinstance(value, float):
        return float_slug(value)
    return str(value).replace("_", "-")


def config_id(config: dict) -> str:
    return (
        f"cfg_b{config['batch_size']}_lr{slug(config['lr'])}_"
        f"d{config['d_model']}h{config['num_heads']}_L{config['num_layers']}_"
        f"ff{config['ff_mult']}_do{slug(config['dropout'])}_"
        f"ro{slug(config['readout'])}_pe{slug(config['position_encoding'])}_"
        f"wd{slug(config['weight_decay'])}"
    )


def canonical_config(config: dict) -> dict:
    result = {key: config[key] for key in ANCHOR}
    result["config_id"] = config_id(result)
    result["protocol_version"] = PROTOCOL_VERSION
    result["selection_metric"] = "five-seed mean validation_MAPE"
    result["checkpoint_metric"] = "validation log-MSE"
    result["test_loaded"] = False
    return result


def apply_coordinate(center: dict, family: str, value: object) -> dict:
    result = deepcopy(center)
    if family == "model_size":
        result["d_model"], result["num_heads"] = value
    else:
        result[family] = value
    return result


def assert_coordinate_grid(center: dict, configs: list[dict], family: str) -> None:
    """Fail before training if a stage changes any frozen coordinate."""
    varied = {"d_model", "num_heads"} if family == "model_size" else {family}
    for config in configs:
        for key in ANCHOR:
            if key not in varied and config[key] != center[key]:
                raise RuntimeError(
                    f"{family} grid changed frozen coordinate {key}: "
                    f"center={center[key]!r} candidate={config[key]!r} "
                    f"config={config['config_id']}"
                )


def report_stage_plan(stage: str, incumbent: dict, configs: list[dict]) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "incumbent": incumbent["config_id"],
                "candidate_config_ids": [config["config_id"] for config in configs],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def unique_configs(configs: list[dict]) -> list[dict]:
    output = []
    seen = set()
    for config in configs:
        item = canonical_config(config)
        if item["config_id"] in seen:
            continue
        seen.add(item["config_id"])
        output.append(item)
    return output


def config_path(config: dict) -> Path:
    return OUTPUT / "configs" / f"{config['config_id']}.json"


def write_config(config: dict) -> None:
    write_json(config_path(config), config)


def ensure_layout() -> None:
    for name in ("configs", "logs", "runs", "summary", "failures", "smoke"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    write_immutable(
        OUTPUT / "logs" / "README.txt",
        "Formal training logs are created only by --phase l1/l2/l3/l4.\n"
        "L0 logs are immutable symlinks to the audited OFAT anchor logs.\n",
    )


def audit_anchor(split: dict) -> dict:
    source_manifest_path = SOURCE_OFAT / "experiment_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    anchors = [item for item in manifest.get("configs", []) if item.get("config_id") == "anchor"]
    if len(anchors) != 1:
        raise RuntimeError("OFAT manifest does not contain exactly one anchor")
    if any(anchors[0].get(key) != ANCHOR[key] for key in ANCHOR):
        raise RuntimeError("OFAT anchor differs from declared baseline")
    source_runs = {}
    for seed in SEEDS:
        source = SOURCE_OFAT / "runs" / "anchor" / f"seed{seed}"
        required = ["metrics.json", "run_config.json", "data_manifest.json", "best_model.pth"]
        if any(not (source / name).is_file() for name in required):
            raise RuntimeError(f"Incomplete source anchor run: {source}")
        metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
        data = json.loads((source / "data_manifest.json").read_text(encoding="utf-8"))
        for key in ANCHOR:
            if config.get(key) != ANCHOR[key]:
                raise RuntimeError(f"Anchor config mismatch {key}: {source}")
        if config.get("epochs") != EPOCHS or config.get("seed") != seed:
            raise RuntimeError(f"Anchor run control mismatch: {source}")
        if metrics.get("seed") != seed or metrics.get("train_count") != TRAIN_COUNT:
            raise RuntimeError(f"Anchor metric identity mismatch: {source}")
        if metrics.get("validation_count") != VALIDATION_COUNT or metrics.get("test_loaded") is not False:
            raise RuntimeError(f"Anchor validation/test audit failed: {source}")
        if data.get("split_seed") != SPLIT_SEED or data.get("train_count") != TRAIN_COUNT:
            raise RuntimeError(f"Anchor data split mismatch: {source}")
        if data.get("validation_count") != VALIDATION_COUNT or data.get("test_loaded") is not False:
            raise RuntimeError(f"Anchor data test audit failed: {source}")
        source_runs[str(seed)] = {
            "source_run": str(source.resolve()),
            "source_log": str((SOURCE_OFAT / "logs" / f"anchor__seed{seed}.log").resolve()),
            "validation_MAPE": metrics["validation_MAPE"],
            "epochs_completed": metrics["epochs_completed"],
        }
    return {
        "source_manifest": str(source_manifest_path.resolve()),
        "runs": source_runs,
    }


def prepare() -> dict:
    ensure_layout()
    payload = load_cache()
    split = prepare_split(payload)
    anchor_audit = audit_anchor(split)
    anchor = canonical_config(ANCHOR)
    l1 = unique_configs([apply_coordinate(ANCHOR, "num_layers", value) for value in L1_GRID["num_layers"]])
    for config in [anchor, *l1]:
        write_config(config)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "baseline-centered Transformer local coordinate search",
        "center": ANCHOR,
        "target_third_column_candidate": TARGET,
        "stage_rule": "Each stage uses the preceding stage's lowest five-seed mean validation MAPE winner as its center.",
        "selection_rule": "Select lowest five-seed mean validation_MAPE; paired 5%/4-of-5 statistics are reported but are not a hard gate.",
        "stages": {
            "L0": {"coordinates": "anchor", "grid": [ANCHOR], "seeds": list(SEEDS), "reuse": "read-only OFAT anchor"},
            "L1": {"coordinate": "num_layers", "grid": list(L1_GRID["num_layers"]), "seeds": list(SEEDS)},
            "L2": {"coordinate": "dropout", "grid": list(L2_GRID["dropout"]), "seeds": list(SEEDS)},
            "L3": {"coordinate": "weight_decay", "grid": list(L3_GRID["weight_decay"]), "seeds": list(SEEDS)},
            "L4": {"coordinates": "single-coordinate diagnostic audit", "grid": L4_GRID, "seeds": list(SEEDS), "selection_reopened": False},
        },
        "training": {"epochs": EPOCHS, "early_stopping_patience": PATIENCE, "trainer": str(TRAINER.resolve()), "cache": str(CACHE.resolve())},
        "data": {"directory": str(DATA_DIR.resolve()), "record_count": 100_000, "fixed_split": split},
        "test_firewall": {"final_test_access_during_search": False, "final_test_directory_not_opened_by_runner": True},
        "resource_policy": {"default_slots_per_gpu": 1, "wait_for_gnn_flag": "--wait-for-gnn", "blocker_names": ["gnn_external_validation_worker.py", "gnn_validation_worker.py"]},
        "planned_budget": {
            "stage_config_slots": {"L0": 1, "L1": 3, "L2": 3, "L3": 3, "L4": 10},
            "all_stage_run_slots": 100,
            "source_or_prior_reused_run_slots": 25,
            "new_training_runs": 75,
            "unique_canonical_configs_on_one_path": 16,
        },
    }
    write_json(OUTPUT / "protocol.json", protocol)
    manifest = {
        "experiment": "iterative_local_grid_ofat100k_baseline_center",
        "protocol_version": PROTOCOL_VERSION,
        "runner": str(Path(__file__).resolve()),
        "development_directory": str(DATA_DIR.resolve()),
        "records_cache": {"path": str(CACHE.resolve())},
        "fixed_split": split,
        "seeds": list(SEEDS),
        "anchor": anchor,
        "target": canonical_config(TARGET),
        "anchor_provenance": anchor_audit,
        "final_test_access": "forbidden",
        "stage_protocol": protocol["stages"],
        "planned_budget": protocol["planned_budget"],
    }
    write_json(OUTPUT / "experiment_manifest.json", manifest)
    write_json(OUTPUT / "anchor_provenance.json", anchor_audit)
    return {"split": split, "anchor": anchor_audit, "protocol": protocol}


def ensure_anchor_reuse(anchor_audit: dict, anchor: dict) -> None:
    for seed in SEEDS:
        source = Path(anchor_audit["runs"][str(seed)]["source_run"])
        target = OUTPUT / "runs" / anchor["config_id"] / f"seed{seed}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise RuntimeError(f"Immutable L0 reuse target differs: {target}")
        else:
            target.symlink_to(source, target_is_directory=True)
        source_log = Path(anchor_audit["runs"][str(seed)]["source_log"])
        log_target = OUTPUT / "logs" / f"{anchor['config_id']}__seed{seed}.log"
        if source_log.is_file():
            if log_target.exists() or log_target.is_symlink():
                if not log_target.is_symlink() or log_target.resolve() != source_log.resolve():
                    raise RuntimeError(f"Immutable L0 log target differs: {log_target}")
            else:
                log_target.symlink_to(source_log)


def metric_path(config: dict, seed: int) -> Path:
    return OUTPUT / "runs" / config["config_id"] / f"seed{seed}" / "metrics.json"


def run_valid(config: dict, seed: int) -> bool:
    run_dir = metric_path(config, seed).parent
    required = ["metrics.json", "run_config.json", "data_manifest.json", "best_model.pth"]
    if any(not (run_dir / name).is_file() for name in required):
        return False
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        control = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        data = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
        for key in ANCHOR:
            if control.get(key) != config[key]:
                return False
        return (
            control.get("seed") == seed
            and control.get("epochs") == EPOCHS
            and metrics.get("seed") == seed
            and metrics.get("train_count") == TRAIN_COUNT
            and metrics.get("validation_count") == VALIDATION_COUNT
            and metrics.get("test_loaded") is False
            and data.get("train_count") == TRAIN_COUNT
            and data.get("validation_count") == VALIDATION_COUNT
            and data.get("test_loaded") is False
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def load_manifest() -> dict:
    return json.loads((OUTPUT / "experiment_manifest.json").read_text(encoding="utf-8"))


def command_for(config: dict, seed: int, run_dir: Path) -> list[str]:
    split_path = load_manifest()["fixed_split"]["path"]
    command = [
        str(PYTHON), str(TRAINER),
        "--data-dir", str(DATA_DIR), "--records-cache", str(CACHE),
        "--split-indices", split_path, "--output-dir", str(run_dir),
        "--run-name", f"{config['config_id']}__seed{seed}", "--seed", str(seed),
        "--device", "cuda:0", "--epochs", str(EPOCHS),
        "--early-stopping-patience", str(PATIENCE),
    ]
    flags = {
        "batch_size": "--batch-size", "lr": "--lr", "weight_decay": "--weight-decay",
        "d_model": "--d-model", "num_heads": "--num-heads", "num_layers": "--num-layers",
        "ff_mult": "--ff-mult", "dropout": "--dropout", "readout": "--readout",
        "position_encoding": "--position-encoding", "representation": "--representation",
        "architecture": "--architecture",
    }
    for key, flag in flags.items():
        command.extend([flag, str(config[key])])
    return command


def blocker_processes() -> list[tuple[int, str]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(name in command for name in ("gnn_external_validation_worker.py", "gnn_validation_worker.py")):
            found.append((int(entry.name), command))
    return found


def wait_for_gnn(clear: bool, poll_seconds: int = 60) -> None:
    blockers = blocker_processes()
    if blockers and not clear:
        pids = ",".join(str(pid) for pid, _ in blockers)
        raise RuntimeError(f"GNN workers are active (pids={pids}); rerun with --wait-for-gnn")
    while blockers:
        pids = ",".join(str(pid) for pid, _ in blockers)
        print(f"waiting for GNN workers to clear: pids={pids}", flush=True)
        time.sleep(min(max(poll_seconds, 1), 60))
        blockers = blocker_processes()


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def parse_gpus(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError(f"Invalid --gpus value: {raw}")
    return values


def run_tasks(configs: list[dict], args: argparse.Namespace) -> None:
    if not cuda_available():
        raise RuntimeError("CUDA unavailable; formal validation runs refuse CPU fallback")
    wait_for_gnn(args.wait_for_gnn, args.wait_poll_seconds)
    for config in configs:
        write_config(config)
    gpus = parse_gpus(args.gpus)
    pending = [(config, seed, 0) for config in configs for seed in SEEDS if not run_valid(config, seed)]
    print(json.dumps({"pending_runs": len(pending), "gpus": gpus, "slots_per_gpu": args.slots_per_gpu}, ensure_ascii=False), flush=True)
    active: list[dict] = []
    counts = {gpu: 0 for gpu in gpus}
    while pending or active:
        if pending:
            wait_for_gnn(args.wait_for_gnn, args.wait_poll_seconds)
        for gpu in gpus:
            while pending and counts[gpu] < args.slots_per_gpu:
                config, seed, attempt = pending.pop(0)
                if run_valid(config, seed):
                    continue
                run_dir = OUTPUT / "runs" / config["config_id"] / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = OUTPUT / "logs" / f"{config['config_id']}__seed{seed}.log"
                handle = log_path.open("a", encoding="utf-8")
                env = os.environ.copy()
                env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
                process = subprocess.Popen(command_for(config, seed, run_dir), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active.append({"process": process, "gpu": gpu, "config": config, "seed": seed, "attempt": attempt, "run_dir": run_dir, "log": log_path, "handle": handle, "started": time.time()})
                counts[gpu] += 1
                print(f"launch {config['config_id']} seed={seed} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(10)
        remaining = []
        for item in active:
            process = item["process"]
            if process.poll() is None:
                remaining.append(item)
                continue
            item["handle"].close()
            counts[item["gpu"]] -= 1
            if process.returncode == 0 and run_valid(item["config"], item["seed"]):
                print(f"complete {item['config']['config_id']} seed={item['seed']} elapsed={time.time()-item['started']:.1f}s", flush=True)
                continue
            failure = {"config_id": item["config"]["config_id"], "seed": item["seed"], "return_code": process.returncode, "log": str(item["log"]), "attempt": item["attempt"] + 1}
            if item["attempt"] + 1 < args.max_retries + 1:
                pending.append((item["config"], item["seed"], item["attempt"] + 1))
                print(f"retry {failure}", flush=True)
            else:
                write_json(OUTPUT / "failures" / f"{item['config']['config_id']}__seed{item['seed']}.json", failure)
                raise RuntimeError(f"Training failed; inspect {item['log']}")
        active = remaining


def read_metrics(config: dict, seed: int) -> dict:
    if not run_valid(config, seed):
        raise RuntimeError(f"Invalid/incomplete run: {config['config_id']} seed={seed}")
    run_dir = metric_path(config, seed).parent
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"config_id": config["config_id"], "seed": seed, "validation_MAPE": float(metrics["validation_MAPE"]), "validation_RMSE": float(metrics["validation_RMSE"]), "validation_MAE": float(metrics["validation_MAE"]), "validation_R2": float(metrics["validation_R2"]), "best_epoch": metrics["best_epoch"], "epochs_completed": metrics["epochs_completed"], "trainable_parameters": metrics["trainable_parameters"], "run_dir": str(metric_path(config, seed).parent.resolve())}


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prior_selection(stage: str) -> dict:
    """Load the completed selection immediately preceding ``stage``."""
    previous = {"L1": "L0", "L2": "L1", "L3": "L2", "L4": "L3"}[stage]
    return json.loads((OUTPUT / "summary" / f"{previous}_selection.json").read_text(encoding="utf-8"))


def selection_for(stage: str, configs: list[dict], incumbent: dict | None) -> dict:
    per_run = []
    summaries = []
    for config in configs:
        values = []
        for seed in SEEDS:
            row = read_metrics(config, seed)
            per_run.append({"stage": stage, **row})
            values.append(row["validation_MAPE"])
        summaries.append({"stage": stage, "config_id": config["config_id"], **{key: config[key] for key in ANCHOR}, "n": len(values), "validation_MAPE_mean": mean(values), "validation_MAPE_sd": stdev(values), "validation_MAPE_sem": stdev(values) / len(values) ** 0.5, "validation_MAPE_per_seed": json.dumps(values), "config_path": str(config_path(config).resolve())})
    summaries.sort(key=lambda row: (row["validation_MAPE_mean"], row["validation_MAPE_sd"], row["config_id"]))
    write_rows(OUTPUT / "summary" / f"{stage}_per_run.csv", per_run)
    write_rows(OUTPUT / "summary" / f"{stage}_config_summary.csv", summaries)
    winner = next(config for config in configs if config["config_id"] == summaries[0]["config_id"])
    payload = {"stage": stage, "candidate_config_ids": [config["config_id"] for config in configs], "winner": winner, "winner_summary": summaries[0], "selection_rule": "lowest five-seed mean validation_MAPE", "paired_rule_is_diagnostic_only": True}
    if incumbent is not None:
        incumbent_values = [read_metrics(incumbent, seed)["validation_MAPE"] for seed in SEEDS]
        winner_values = [read_metrics(winner, seed)["validation_MAPE"] for seed in SEEDS]
        old_mean, new_mean = mean(incumbent_values), mean(winner_values)
        payload["incumbent"] = incumbent
        payload["paired_comparison"] = {"incumbent_mean": old_mean, "winner_mean": new_mean, "relative_improvement_percent": 100 * (old_mean - new_mean) / old_mean, "paired_seed_wins": sum(new < old for old, new in zip(incumbent_values, winner_values)), "incumbent_per_seed": incumbent_values, "winner_per_seed": winner_values, "five_percent_four_of_five_gate": bool(100 * (old_mean - new_mean) / old_mean >= 5 and sum(new < old for old, new in zip(incumbent_values, winner_values)) >= 4)}
    write_json(OUTPUT / "summary" / f"{stage}_selection.json", payload)
    return payload


def run_l0() -> dict:
    manifest = load_manifest()
    anchor = manifest["anchor"]
    ensure_anchor_reuse(manifest["anchor_provenance"], anchor)
    return selection_for("L0", [anchor], None)


def run_l1(args: argparse.Namespace) -> dict:
    incumbent = run_l0()["winner"]
    configs = unique_configs([apply_coordinate(incumbent, "num_layers", value) for value in L1_GRID["num_layers"]])
    assert_coordinate_grid(incumbent, configs, "num_layers")
    report_stage_plan("L1", incumbent, configs)
    run_tasks(configs, args)
    return selection_for("L1", configs, incumbent)


def run_l2(args: argparse.Namespace) -> dict:
    incumbent = prior_selection("L2")["winner"]
    configs = unique_configs([apply_coordinate(incumbent, "dropout", value) for value in L2_GRID["dropout"]])
    assert_coordinate_grid(incumbent, configs, "dropout")
    report_stage_plan("L2", incumbent, configs)
    run_tasks(configs, args)
    return selection_for("L2", configs, incumbent)


def run_l3(args: argparse.Namespace) -> dict:
    incumbent = prior_selection("L3")["winner"]
    configs = unique_configs([apply_coordinate(incumbent, "weight_decay", value) for value in L3_GRID["weight_decay"]])
    assert_coordinate_grid(incumbent, configs, "weight_decay")
    report_stage_plan("L3", incumbent, configs)
    run_tasks(configs, args)
    return selection_for("L3", configs, incumbent)


def run_l4(args: argparse.Namespace) -> dict:
    incumbent = prior_selection("L4")["winner"]
    # Build each coordinate audit independently around the actual L3
    # incumbent.  The incumbent is deliberately present in every family so
    # that every family has a like-for-like reference.  The global task list
    # is only a de-duplicated execution pool; using one family label per
    # config here would silently overwrite the incumbent label as the loop
    # advances through families and drop it from later audits.
    family_configs: dict[str, list[dict]] = {}
    configs: list[dict] = []
    seen_config_ids: set[str] = set()
    family_for_config: dict[str, str] = {incumbent["config_id"]: "incumbent"}
    for family, values in L4_GRID.items():
        family_candidates = unique_configs(
            [incumbent]
            + [apply_coordinate(incumbent, family, value) for value in values]
        )
        assert_coordinate_grid(incumbent, family_candidates, family)
        family_configs[family] = family_candidates
        for candidate in family_candidates:
            if candidate["config_id"] not in seen_config_ids:
                configs.append(candidate)
                seen_config_ids.add(candidate["config_id"])
            # Keep the explicit incumbent label for the shared center.  This
            # label is only for the pooled per-run table; family audits use
            # ``family_configs`` above and therefore retain the center in
            # every coordinate-specific group.
            family_for_config.setdefault(candidate["config_id"], family)
    report_stage_plan("L4", incumbent, configs)
    run_tasks(configs, args)
    per_run = []
    for config in configs:
        for seed in SEEDS:
            per_run.append({"stage": "L4", "coordinate": family_for_config[config["config_id"]], **read_metrics(config, seed)})
    write_rows(OUTPUT / "summary" / "L4_per_run.csv", per_run)
    audit = []
    incumbent_values = [read_metrics(incumbent, seed)["validation_MAPE"] for seed in SEEDS]
    for family, values in L4_GRID.items():
        candidates = family_configs[family]
        rows = []
        for config in candidates:
            vals = [read_metrics(config, seed)["validation_MAPE"] for seed in SEEDS]
            rows.append((config, vals, mean(vals)))
        rows.sort(key=lambda item: (item[2], item[0]["config_id"]))
        best, best_values, best_mean = rows[0]
        target_value = (
            (TARGET["d_model"], TARGET["num_heads"])
            if family == "model_size"
            else TARGET[family]
        )
        incumbent_value = (
            (incumbent["d_model"], incumbent["num_heads"])
            if family == "model_size"
            else incumbent[family]
        )
        best_value = (best["d_model"], best["num_heads"]) if family == "model_size" else best[family]
        audit.append({"coordinate": family, "incumbent_value": incumbent_value, "target_value": target_value, "best_challenger_config_id": best["config_id"], "best_challenger_value": best_value, "best_challenger_mean": best_mean, "incumbent_mean": mean(incumbent_values), "relative_improvement_percent": 100 * (mean(incumbent_values) - best_mean) / mean(incumbent_values), "paired_seed_wins": sum(new < old for old, new in zip(incumbent_values, best_values)), "target_center_retained": best_value == target_value})
    write_rows(OUTPUT / "summary" / "L4_coordinate_audit.csv", audit)
    result = {"stage": "L4", "diagnostic_only": True, "selection_reopened": False, "frozen_incumbent": incumbent, "coordinate_audit": audit, "target_config": canonical_config(TARGET), "target_reached_after_L3": config_id(incumbent) == config_id(TARGET), "test_accessed": False}
    write_json(OUTPUT / "summary" / "L4_completion.json", result)
    write_json(OUTPUT / "summary" / "search_result.json", {"target": canonical_config(TARGET), "L3_winner": incumbent, "target_reached": result["target_reached_after_L3"], "L4_coordinate_audit": audit, "test_accessed": False})
    return result


def smoke(args: argparse.Namespace) -> None:
    output = OUTPUT / "smoke" / "anchor_seed202601"
    log = OUTPUT / "logs" / "smoke.log"
    config = load_manifest()["anchor"]
    command = [str(PYTHON), str(TRAINER), "--data-dir", str(DATA_DIR), "--output-dir", str(output), "--run-name", "smoke_anchor__seed202601", "--seed", "202601", "--device", "cpu", "--max-samples", "200", "--epochs", "2", "--early-stopping-patience", "25"]
    for key, flag in (("batch_size", "--batch-size"), ("lr", "--lr"), ("weight_decay", "--weight-decay"), ("d_model", "--d-model"), ("num_heads", "--num-heads"), ("num_layers", "--num-layers"), ("ff_mult", "--ff-mult"), ("dropout", "--dropout"), ("readout", "--readout"), ("position_encoding", "--position-encoding"), ("representation", "--representation"), ("architecture", "--architecture")):
        command.extend([flag, str(config[key])])
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("test_loaded") is not False or metrics.get("epochs_completed") != 2:
        raise RuntimeError("Smoke test audit failed")
    write_json(OUTPUT / "summary" / "smoke_result.json", {"status": "passed", "command": command, "metrics": metrics, "test_accessed": False})
    print(json.dumps({"smoke": "passed", "metrics": metrics}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "smoke", "l0", "l1", "l2", "l3", "l4", "all"), required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--wait-for-gnn", action="store_true")
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.slots_per_gpu < 1 or args.max_retries < 0:
        raise ValueError("slots-per-gpu must be positive and max-retries non-negative")
    prepare()
    if args.phase == "prepare":
        print(json.dumps({"output": str(OUTPUT.resolve()), "planned_new_training_runs": 75, "stages": ["L0", "L1", "L2", "L3", "L4"]}, ensure_ascii=False), flush=True)
        return
    if args.phase == "smoke":
        smoke(args)
        return
    if args.phase == "l0":
        print(json.dumps(run_l0(), ensure_ascii=False), flush=True)
        return
    if args.phase in ("l1", "all"):
        print(json.dumps(run_l1(args), ensure_ascii=False), flush=True)
    if args.phase in ("l2", "all"):
        print(json.dumps(run_l2(args), ensure_ascii=False), flush=True)
    if args.phase in ("l3", "all"):
        print(json.dumps(run_l3(args), ensure_ascii=False), flush=True)
    if args.phase in ("l4", "all"):
        print(json.dumps(run_l4(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
