#!/usr/bin/env python3
"""Resume-safe compact L0--L4 search on a fixed 64k/16k split.

The protocol is deliberately narrower than ``run_iterative_local_grid_v2.py``:
it searches optimizer/width values, encoder/decoder depth, and the three
declared node inputs only.  The original bidirectional virtual-node graph and
full ``[f, A/B]`` edge input are frozen throughout.  Validation phases never
load Test-A or Test-B.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
GNN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselinemodels.pub0818.gnn import run_iterative_local_grid as legacy
from mainmodel.scripts import train_single_chin_supplementary_confirmation as source


OUTPUT = GNN_ROOT / "iterative_local_grid_c2_64k_train_16k_val"
CONDITION = "C2_virtualnode_learnable_node_full_edge"
NODE_CONDITIONS = (
    ("S01", "S01_full_virtualnode_free_branch_degree_full_edge"),
    ("C2", "C2_virtualnode_learnable_node_full_edge"),
    ("C3", "C3_virtualnode_degree_avg_neighbor_full_edge"),
)
SEEDS = (202601, 202602, 202603, 202604, 202605)
SCREEN_SEEDS = SEEDS[:3]

L0_HYPER = {
    "batch_size": 128,
    "learning_rate": 0.002,
    "hidden_dim": 512,
    "num_layers": 5,
    "dec_layers": 1,
}
L1_BATCH = (64, 128, 256)
L1_LR = (0.001, 0.002, 0.004)
L1_HIDDEN = (256, 512, 1024)
DEPTH_ENCODER = (4, 5, 6)
DEPTH_DECODER = (1, 2)

TRAIN_DATA = ROOT / "TrainData/data/chiNlimited/n_train_100k"
SOURCE_SPLIT = (
    GNN_ROOT
    / "iterative_local_grid_c2_e5d1_80k_train_20k_val/fixed_80k_20k_split_indices.npz"
)
SPLIT_NAME = "fixed_64k_16k_split_indices.npz"
SHARED_CACHE = (
    GNN_ROOT / "training_pool_size_80k/runs_10k_to_100k_validation_only/dataset_cache"
)
WORKER = GNN_ROOT / "gnn_validation_worker.py"
PROTOCOL_VERSION = "c2_iterative_local_grid_64k_16k_20260825"
TRAIN_SIZE = 64_000
VALIDATION_SIZE = 16_000
MINIMUM_EPOCHS = 120
MAX_EPOCHS = 200
PATIENCE = 30


def hyper_key(hyper: dict) -> str:
    return legacy.hyper_key(hyper)


def hyper_tuple(hyper: dict) -> tuple:
    return (
        int(hyper["batch_size"]),
        float(hyper["learning_rate"]),
        int(hyper["hidden_dim"]),
        int(hyper["num_layers"]),
        int(hyper["dec_layers"]),
    )


def unique_hypers(hypers: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for hyper in hypers:
        key = hyper_tuple(hyper)
        if key not in seen:
            seen.add(key)
            result.append(dict(hyper))
    return result


def l1_hypers() -> list[dict]:
    """Nine batch/lr cross cells at h512 plus the three-width center line."""
    rows = []
    for batch in L1_BATCH:
        for learning_rate in L1_LR:
            rows.append(
                {
                    "batch_size": batch,
                    "learning_rate": learning_rate,
                    "hidden_dim": 512,
                    "num_layers": 5,
                    "dec_layers": 1,
                }
            )
    for hidden_dim in L1_HIDDEN:
        rows.append(
            {
                "batch_size": 128,
                "learning_rate": 0.002,
                "hidden_dim": hidden_dim,
                "num_layers": 5,
                "dec_layers": 1,
            }
        )
    return unique_hypers(rows)


def build_config(condition: str, hyper: dict, seed: int) -> dict:
    cfg = source.condition_config(condition)
    cfg.update(
        {
            **hyper,
            "run_id": f"{condition}__{hyper_key(hyper)}__seed{seed}",
            "seed": int(seed),
            "train_data_dir": str(TRAIN_DATA),
            "dataset_cache_dir": str(SHARED_CACHE),
            "train_size": TRAIN_SIZE,
            "validation_size": VALIDATION_SIZE,
            "split_random_state": 42,
            "split_rule": "source train_index[:64000] + val_index[:16000]",
            "test_data_dir": "SEALED: not available to validation worker",
            "epochs": MAX_EPOCHS,
            "minimum_epochs": MINIMUM_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "optimizer": "Adam",
            "loss": "MSE on log10(chiN*)",
            "protocol_version": PROTOCOL_VERSION,
            "test_a_accessed_during_training": False,
            "test_b_accessed": False,
            "model_identity": (
                f"{condition}|b{hyper['batch_size']}|lr{hyper['learning_rate']}|"
                f"h{hyper['hidden_dim']}|E{hyper['num_layers']}|D{hyper['dec_layers']}"
            ),
        }
    )
    return cfg


def local_metrics(output: Path, cfg: dict) -> Path:
    return output / "runs" / cfg["run_id"] / "metrics.json"


def validate_metrics(output: Path, cfg: dict) -> dict:
    path = local_metrics(output, cfg)
    if not path.is_file():
        raise RuntimeError(f"Missing metrics artifact: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_id": cfg["run_id"],
        "condition_key": cfg["condition_key"],
        "model_identity": cfg["model_identity"],
        "seed": cfg["seed"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "dec_layers": cfg["dec_layers"],
        "graph_builder": "virtual_node",
        "edge_mode": "full",
        "minimum_epochs": MINIMUM_EPOCHS,
        "epochs": MAX_EPOCHS,
    }
    mismatches = [
        key for key, value in expected.items() if metrics.get(key) != value
    ]
    if metrics.get("final_test_accessed") is not False:
        mismatches.append("final_test_accessed")
    if metrics.get("test_a_accessed_during_training") is not False:
        mismatches.append("test_a_accessed_during_training")
    if metrics.get("test_b_accessed") is not False:
        mismatches.append("test_b_accessed")
    if mismatches:
        raise RuntimeError(f"Metric identity/firewall mismatch {mismatches}: {path}")
    return metrics


def write_config(output: Path, cfg: dict) -> None:
    path = output / "configs" / f"{cfg['run_id']}.json"
    payload = json.dumps(cfg, indent=2)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != cfg:
            raise RuntimeError(f"Immutable config changed: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def deduplicate_configs(configs: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for cfg in configs:
        if cfg["run_id"] not in seen:
            seen.add(cfg["run_id"])
            result.append(cfg)
    return result


def run_tasks(
    output: Path,
    tasks: list[dict],
    gpus: list[str],
    slots_per_gpu: int,
    retries: int,
) -> None:
    """Use the repository scheduler pattern with the 64k/16k split filename."""
    tasks = deduplicate_configs(tasks)
    for cfg in tasks:
        write_config(output, cfg)
    pending = []
    for cfg in tasks:
        metrics_path = local_metrics(output, cfg)
        if metrics_path.is_file():
            validate_metrics(output, cfg)
        else:
            pending.append(cfg)
    if not pending:
        print(f"resume: all {len(tasks)} requested tasks already complete", flush=True)
        return
    queue = deque(pending)
    active = []
    retry_count: dict[str, int] = {}
    slots = [gpu for gpu in gpus for _ in range(slots_per_gpu)]
    split = output / SPLIT_NAME
    while queue or active:
        counts = {
            gpu: sum(item["gpu"] == gpu for item in active) for gpu in set(slots)
        }
        free = []
        for gpu in slots:
            if counts.get(gpu, 0):
                counts[gpu] -= 1
            else:
                free.append(gpu)
        while queue and free:
            cfg = queue.popleft()
            gpu = free.pop(0)
            run_dir = output / "runs" / cfg["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = output / "logs" / f"{cfg['run_id']}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                sys.executable,
                str(WORKER),
                "--config-json",
                str(output / "configs" / f"{cfg['run_id']}.json"),
                "--split-indices",
                str(split),
                "--output-dir",
                str(run_dir),
                "--device",
                "cuda:0",
            ]
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append(
                {
                    "cfg": cfg,
                    "gpu": gpu,
                    "proc": proc,
                    "log": log_handle,
                    "log_path": log_path,
                }
            )
            print(f"started gpu={gpu} {cfg['run_id']}", flush=True)
        if not active:
            continue
        time.sleep(5)
        remaining = []
        for item in active:
            code = item["proc"].poll()
            if code is None:
                remaining.append(item)
                continue
            item["log"].close()
            cfg = item["cfg"]
            if code == 0 and local_metrics(output, cfg).is_file():
                metrics = validate_metrics(output, cfg)
                print(
                    f"finished {cfg['run_id']} validation_MAPE="
                    f"{float(metrics['validation_mape_percent']):.6f}%",
                    flush=True,
                )
            elif retry_count.get(cfg["run_id"], 0) < retries:
                retry_count[cfg["run_id"]] = retry_count.get(cfg["run_id"], 0) + 1
                queue.append(cfg)
                print(
                    f"retry={retry_count[cfg['run_id']]} {cfg['run_id']}; "
                    f"log={item['log_path']}",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"Failed after retries: {cfg['run_id']}; see {item['log_path']}"
                )
        active = remaining


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rows_for(
    output: Path,
    condition: str,
    hypers: list[dict],
    seeds: tuple[int, ...],
) -> list[dict]:
    rows = []
    for hyper in unique_hypers(hypers):
        values = []
        for seed in seeds:
            cfg = build_config(condition, hyper, seed)
            metrics = validate_metrics(output, cfg)
            values.append(float(metrics["validation_mape_percent"]))
        rows.append(
            {
                **hyper,
                "condition_key": condition,
                "hyper_key": hyper_key(hyper),
                "n_seeds": len(values),
                "validation_mape_mean_percent": mean(values),
                "validation_mape_sd_percent": stdev(values)
                if len(values) > 1
                else 0.0,
                "validation_mape_per_seed_percent": json.dumps(values),
            }
        )
    rows.sort(
        key=lambda row: (
            row["validation_mape_mean_percent"],
            row["validation_mape_sd_percent"],
        )
    )
    return rows


def hyper_from_row(row: dict) -> dict:
    return {
        "batch_size": int(row["batch_size"]),
        "learning_rate": float(row["learning_rate"]),
        "hidden_dim": int(row["hidden_dim"]),
        "num_layers": int(row["num_layers"]),
        "dec_layers": int(row["dec_layers"]),
    }


def values_for(output: Path, condition: str, hyper: dict) -> list[float]:
    return [
        float(validate_metrics(output, build_config(condition, hyper, seed))["validation_mape_percent"])
        for seed in SEEDS
    ]


def paired_decision(
    incumbent: dict,
    challenger: dict,
    incumbent_values: list[float],
    challenger_values: list[float],
) -> dict:
    if len(incumbent_values) != 5 or len(challenger_values) != 5:
        raise RuntimeError("Promotion requires exactly five paired seeds")
    valid = bool(
        all(np.isfinite(incumbent_values))
        and all(np.isfinite(challenger_values))
    )
    old_mean = mean(incumbent_values)
    new_mean = mean(challenger_values)
    improvement = 100.0 * (old_mean - new_mean) / old_mean
    wins = sum(new < old for old, new in zip(incumbent_values, challenger_values))
    return {
        "incumbent": incumbent,
        "challenger": challenger,
        "incumbent_per_seed_validation_mape_percent": incumbent_values,
        "challenger_per_seed_validation_mape_percent": challenger_values,
        "paired_seed_difference_percent": [
            new - old for old, new in zip(incumbent_values, challenger_values)
        ],
        "incumbent_mean_validation_mape_percent": old_mean,
        "challenger_mean_validation_mape_percent": new_mean,
        "relative_improvement_percent": improvement,
        "paired_seed_wins": wins,
        "valid_metrics": valid,
        "promote": bool(valid and improvement >= 5.0 and wins >= 4),
        "rule": "mean improvement >=5% and >=4/5 paired-seed wins",
    }


def run_and_summarize(
    output: Path,
    condition: str,
    hypers: list[dict],
    seeds: tuple[int, ...],
    summary_name: str,
    gpus: list[str],
    slots_per_gpu: int,
    retries: int,
) -> list[dict]:
    tasks = [
        build_config(condition, hyper, seed)
        for hyper in unique_hypers(hypers)
        for seed in seeds
    ]
    run_tasks(output, tasks, gpus, slots_per_gpu, retries)
    rows = rows_for(output, condition, hypers, seeds)
    write_csv(output / "summary" / summary_name, rows)
    return rows


def l1_boundary_candidates(best: dict) -> tuple[list[dict], dict]:
    candidates = []
    notes = {"hidden_upper_bound": False}
    replacements = []
    if best["batch_size"] == 64:
        replacements.append(("batch_size", 32))
    elif best["batch_size"] == 256:
        replacements.append(("batch_size", 512))
    if best["learning_rate"] == 0.001:
        replacements.append(("learning_rate", 0.0005))
    elif best["learning_rate"] == 0.004:
        replacements.append(("learning_rate", 0.008))
    if best["hidden_dim"] == 256:
        replacements.append(("hidden_dim", 128))
    elif best["hidden_dim"] == 1024:
        notes["hidden_upper_bound"] = True
    for key, value in replacements:
        candidate = dict(best)
        candidate[key] = value
        candidates.append(candidate)
    return unique_hypers(candidates), notes


def prepare(output: Path) -> None:
    for name in ("configs", "runs", "logs", "summary"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if not SOURCE_SPLIT.is_file():
        raise RuntimeError(f"Missing source split: {SOURCE_SPLIT}")
    with np.load(SOURCE_SPLIT) as source_split:
        if len(source_split["train_index"]) != 80_000:
            raise RuntimeError("Expected 80,000 source train indices")
        if len(source_split["val_index"]) != 20_000:
            raise RuntimeError("Expected 20,000 source validation indices")
        train_index = np.asarray(source_split["train_index"][:TRAIN_SIZE], dtype=np.int64)
        val_index = np.asarray(source_split["val_index"][:VALIDATION_SIZE], dtype=np.int64)
    if np.intersect1d(train_index, val_index).size:
        raise RuntimeError("Derived train and validation indices overlap")
    target = output / SPLIT_NAME
    if target.exists():
        with np.load(target) as existing:
            if not np.array_equal(existing["train_index"], train_index) or not np.array_equal(
                existing["val_index"], val_index
            ):
                raise RuntimeError(f"Existing derived split changed: {target}")
    else:
        np.savez(target, train_index=train_index, val_index=val_index)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "compact L0-L4 search at 64k train / 16k validation",
        "ledger": str(GNN_ROOT / "ITERATIVE_LOCAL_GRID_RECORD.md"),
        "development_pool": str(TRAIN_DATA),
        "source_split": str(SOURCE_SPLIT),
        "derived_split": str(target),
        "split_rule": "train_index[:64000] + val_index[:16000] from original n_train_100k L0-L4 split",
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "condition_scope": {
            "graph_builder": "virtual_node",
            "graph_label": "original bidirectional + virtual node",
            "edge_mode": "full",
            "edge_features": "[f, A/B]",
            "node_conditions": [key for key, _ in NODE_CONDITIONS],
        },
        "frozen_training": {
            "optimizer": "Adam",
            "loss": "MSE on log10(chiN*)",
            "epochs": MAX_EPOCHS,
            "minimum_epochs": MINIMUM_EPOCHS,
            "early_stopping_patience": PATIENCE,
        },
        "seeds": list(SEEDS),
        "screen_seeds": list(SCREEN_SEEDS),
        "l0": {"condition": CONDITION, "hyperparameters": L0_HYPER},
        "l1": {
            "configuration_count": len(l1_hypers()),
            "batch": list(L1_BATCH),
            "learning_rate": list(L1_LR),
            "hidden": list(L1_HIDDEN),
            "construction": "batch x learning-rate cross at hidden=512 plus hidden line at batch=128, lr=0.002",
            "screen_then_confirm": "all 11 on 3 seeds; top 2 to 5 seeds",
            "boundary_expansion": "at most one adjacent expansion per winning batch/lr boundary; hidden=1024 is upper-bound limited",
        },
        "l2": {
            "encoder": list(DEPTH_ENCODER),
            "decoder": list(DEPTH_DECODER),
            "configuration_count": len(DEPTH_ENCODER) * len(DEPTH_DECODER),
            "screen_then_confirm": "all 6 on 3 seeds; top 2 to 5 seeds",
            "e7_forbidden": True,
        },
        "l3": {
            "condition": "only if L2 promotes",
            "construction": "center plus immediate one-coordinate neighbors in declared L1 values",
            "maximum_configuration_count": 7,
            "screen_then_confirm": "screen on 3 seeds; top 2 to 5 seeds",
            "one_cycle_only": True,
        },
        "l4": {
            "conditions": [condition for _, condition in NODE_CONDITIONS],
            "configuration_count": 3,
            "seeds": list(SEEDS),
            "diagnostic_only": True,
            "edge_and_graph_ablations_forbidden": True,
        },
        "promotion_rule": "five-seed mean improvement >=5% and >=4/5 paired-seed wins; all metrics finite",
        "test_a_policy": "not accessed by any selection or ablation phase",
        "test_b_policy": "sealed and not constructed",
        "estimated_seed_evaluations": {
            "l0": 5,
            "l1_core": 37,
            "l1_boundary_max": 11,
            "l2": 22,
            "l3_if_triggered": 25,
            "l4": 15,
            "total_without_boundary_or_l3": 79,
            "maximum_with_boundary_and_l3": 115,
        },
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(f"Immutable protocol changed: {protocol_path}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    manifest = []
    manifest_hypers = [L0_HYPER] + l1_hypers() + [
        dict(L0_HYPER, num_layers=encoder, dec_layers=decoder)
        for encoder in DEPTH_ENCODER
        for decoder in DEPTH_DECODER
    ]
    for hyper in unique_hypers(manifest_hypers):
        for seed in SCREEN_SEEDS:
            manifest.append(build_config(CONDITION, hyper, seed))
    for _, condition in NODE_CONDITIONS:
        for seed in SEEDS:
            manifest.append(build_config(condition, L0_HYPER, seed))
    manifest = deduplicate_configs(manifest)
    write_json(output / "manifest.json", manifest)


def bootstrap(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    hypers = l1_hypers()
    tasks = [
        build_config(CONDITION, hyper, seed)
        for hyper in hypers
        for seed in SCREEN_SEEDS
    ] + [build_config(CONDITION, L0_HYPER, seed) for seed in SEEDS]
    run_tasks(output, tasks, gpus, slots, retries)
    l0_rows = rows_for(output, CONDITION, [L0_HYPER], SEEDS)
    l1_rows = rows_for(output, CONDITION, hypers, SCREEN_SEEDS)
    write_csv(output / "summary/l0_c2_e5d1.csv", l0_rows)
    write_csv(output / "summary/l1_screen.csv", l1_rows)
    write_json(output / "summary/l1_screen_top2.json", l1_rows[:2])
    print(json.dumps({"l0": l0_rows, "l1_top2": l1_rows[:2]}, indent=2), flush=True)


def l1_confirm(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    path = output / "summary/l1_screen_top2.json"
    if not path.is_file():
        raise RuntimeError("L1 screen summary missing; run --phase bootstrap first")
    top2 = [hyper_from_row(row) for row in json.loads(path.read_text(encoding="utf-8"))]
    confirmed = run_and_summarize(
        output, CONDITION, top2, SEEDS, "l1_top2_five_seed.csv", gpus, slots, retries
    )
    incumbent = dict(L0_HYPER)
    best = hyper_from_row(confirmed[0])
    core_decision = paired_decision(
        incumbent, best, values_for(output, CONDITION, incumbent), values_for(output, CONDITION, best)
    )
    selected = dict(best) if core_decision["promote"] else dict(incumbent)
    expansion = {
        "triggered": False,
        "candidates": [],
        "screen": [],
        "confirmation": [],
        "decision": None,
        "notes": {},
    }
    if core_decision["promote"]:
        boundary, notes = l1_boundary_candidates(best)
        expansion["notes"] = notes
        if boundary:
            expansion["triggered"] = True
            expansion["candidates"] = boundary
            boundary_screen = run_and_summarize(
                output,
                CONDITION,
                boundary,
                SCREEN_SEEDS,
                "l1_boundary_screen.csv",
                gpus,
                slots,
                retries,
            )
            expansion["screen"] = boundary_screen
            boundary_best = hyper_from_row(boundary_screen[0])
            boundary_confirmed = run_and_summarize(
                output,
                CONDITION,
                [boundary_best],
                SEEDS,
                "l1_boundary_five_seed.csv",
                gpus,
                slots,
                retries,
            )
            expansion["confirmation"] = boundary_confirmed
            boundary_best = hyper_from_row(boundary_confirmed[0])
            boundary_decision = paired_decision(
                selected,
                boundary_best,
                values_for(output, CONDITION, selected),
                values_for(output, CONDITION, boundary_best),
            )
            expansion["decision"] = boundary_decision
            if boundary_decision["promote"]:
                selected = dict(boundary_best)
    result = {
        "phase": "L1",
        "screen_configuration_count": len(l1_hypers()),
        "top2": confirmed,
        "core_decision": core_decision,
        "boundary_expansion": expansion,
        "selected_hyperparameters": selected,
        "test_accessed": False,
    }
    write_json(output / "summary/l1_selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def l2_hypers(center: dict) -> list[dict]:
    rows = []
    for encoder in DEPTH_ENCODER:
        for decoder in DEPTH_DECODER:
            hyper = dict(center)
            hyper.update(num_layers=encoder, dec_layers=decoder)
            rows.append(hyper)
    return unique_hypers(rows)


def l2(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    path = output / "summary/l1_selection.json"
    if not path.is_file():
        raise RuntimeError("L1 selection missing; run --phase l1-confirm first")
    incumbent = json.loads(path.read_text(encoding="utf-8"))["selected_hyperparameters"]
    hypers = l2_hypers(incumbent)
    screen = run_and_summarize(
        output, CONDITION, hypers, SCREEN_SEEDS, "l2_depth_screen.csv", gpus, slots, retries
    )
    top2 = [hyper_from_row(row) for row in screen[:2]]
    confirmed = run_and_summarize(
        output, CONDITION, top2, SEEDS, "l2_depth_top2_five_seed.csv", gpus, slots, retries
    )
    # Always audit the best non-incumbent depth as the challenger.  If the
    # incumbent ranks first after five-seed confirmation, comparing it with
    # itself would preserve the selection but erase the evidence for rejecting
    # the actual depth challenger from l2_selection.json.
    challenger_rows = [
        row
        for row in confirmed
        if hyper_tuple(hyper_from_row(row)) != hyper_tuple(incumbent)
    ]
    best = hyper_from_row(challenger_rows[0]) if challenger_rows else dict(incumbent)
    decision = paired_decision(
        incumbent, best, values_for(output, CONDITION, incumbent), values_for(output, CONDITION, best)
    )
    selected = dict(best) if decision["promote"] else dict(incumbent)
    result = {
        "phase": "L2",
        "screen_configuration_count": len(hypers),
        "screen": screen,
        "top2_confirmation": confirmed,
        "decision": decision,
        "selected_hyperparameters": selected,
        "encoder_boundary_limited": bool(decision["promote"] and best["num_layers"] == 6),
        "e7_forbidden": True,
        "test_accessed": False,
    }
    write_json(output / "summary/l2_selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def immediate_neighbors(value, declared: list) -> list:
    ordered = sorted(set(declared + [value]))
    index = ordered.index(value)
    neighbors = []
    if index > 0:
        neighbors.append(ordered[index - 1])
    if index + 1 < len(ordered):
        neighbors.append(ordered[index + 1])
    return neighbors


def l3_hypers(center: dict, l1_record: dict) -> list[dict]:
    declared_batches = list(L1_BATCH)
    declared_lrs = list(L1_LR)
    declared_hidden = list(L1_HIDDEN)
    expansion = l1_record.get("boundary_expansion", {})
    for candidate in expansion.get("candidates", []):
        declared_batches.append(int(candidate["batch_size"]))
        declared_lrs.append(float(candidate["learning_rate"]))
        declared_hidden.append(int(candidate["hidden_dim"]))
    rows = [dict(center)]
    for key, declared in (
        ("batch_size", declared_batches),
        ("learning_rate", declared_lrs),
        ("hidden_dim", declared_hidden),
    ):
        for value in immediate_neighbors(center[key], declared):
            candidate = dict(center)
            candidate[key] = value
            rows.append(candidate)
    return unique_hypers(rows)


def l3(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    path = output / "summary/l2_selection.json"
    if not path.is_file():
        raise RuntimeError("L2 selection missing; run --phase l2 first")
    l2_record = json.loads(path.read_text(encoding="utf-8"))
    incumbent = l2_record["selected_hyperparameters"]
    if not l2_record["decision"]["promote"]:
        result = {
            "phase": "L3",
            "executed": False,
            "reason": "L2 did not promote a new depth",
            "selected_hyperparameters": incumbent,
            "test_accessed": False,
        }
        write_json(output / "summary/l3_selection.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    l1_record = json.loads((output / "summary/l1_selection.json").read_text(encoding="utf-8"))
    hypers = l3_hypers(incumbent, l1_record)
    screen = run_and_summarize(
        output, CONDITION, hypers, SCREEN_SEEDS, "l3_one_cycle_screen.csv", gpus, slots, retries
    )
    top2 = [hyper_from_row(row) for row in screen[:2]]
    confirmed = run_and_summarize(
        output, CONDITION, top2, SEEDS, "l3_one_cycle_top2_five_seed.csv", gpus, slots, retries
    )
    best = hyper_from_row(confirmed[0])
    decision = paired_decision(
        incumbent, best, values_for(output, CONDITION, incumbent), values_for(output, CONDITION, best)
    )
    selected = dict(best) if decision["promote"] else dict(incumbent)
    result = {
        "phase": "L3",
        "executed": True,
        "one_cycle_only": True,
        "screen_configuration_count": len(hypers),
        "screen": screen,
        "top2_confirmation": confirmed,
        "decision": decision,
        "selected_hyperparameters": selected,
        "test_accessed": False,
    }
    write_json(output / "summary/l3_selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def l4(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    path = output / "summary/l3_selection.json"
    if not path.is_file():
        raise RuntimeError("L3 selection missing; run --phase l3 first")
    hyper = json.loads(path.read_text(encoding="utf-8"))["selected_hyperparameters"]
    tasks = [
        build_config(condition, hyper, seed)
        for _, condition in NODE_CONDITIONS
        for seed in SEEDS
    ]
    run_tasks(output, tasks, gpus, slots, retries)
    rows = []
    for label, condition in NODE_CONDITIONS:
        values = [
            float(
                validate_metrics(output, build_config(condition, hyper, seed))["validation_mape_percent"]
            )
            for seed in SEEDS
        ]
        first = build_config(condition, hyper, SEEDS[0])
        rows.append(
            {
                "node_label": label,
                "condition_key": condition,
                "graph_builder": first["graph_builder"],
                "node_builder": first["node_builder"],
                "edge_mode": first["edge_mode"],
                "n_seeds": 5,
                "validation_mape_mean_percent": mean(values),
                "validation_mape_sd_percent": stdev(values),
                "validation_mape_per_seed_percent": json.dumps(values),
            }
        )
    rows.sort(key=lambda row: row["validation_mape_mean_percent"])
    write_csv(output / "summary/l4_node_input.csv", rows)
    result = {
        "phase": "L4",
        "diagnostic_only": True,
        "does_not_reopen_model_selection": True,
        "frozen_graph_builder": "virtual_node",
        "frozen_edge_mode": "full",
        "frozen_hyperparameters": hyper,
        "n_conditions": 3,
        "conditions": [condition for _, condition in NODE_CONDITIONS],
        "test_accessed": False,
    }
    write_json(output / "summary/l4_completion.json", result)
    print(json.dumps(result, indent=2), flush=True)


def active_workers(output: Path) -> list[int]:
    return legacy.active_workers_for_output(output)


def check_runtime(output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    live = active_workers(output)
    if live:
        raise RuntimeError(f"Live workers already target output: {live}")


def run_all(args, output: Path, gpus: list[str]) -> None:
    bootstrap(output, gpus, args.slots_per_gpu, args.max_retries)
    l1_confirm(output, gpus, args.slots_per_gpu, args.max_retries)
    l2(output, gpus, args.slots_per_gpu, args.max_retries)
    l3(output, gpus, args.slots_per_gpu, args.max_retries)
    l4(output, gpus, args.slots_per_gpu, args.max_retries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "bootstrap", "l1-confirm", "l2", "l3", "l4", "all"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    prepare(output)
    if args.phase == "prepare":
        print(
            json.dumps(
                {
                    "prepared_output": str(output),
                    "split": str(output / SPLIT_NAME),
                    "l1_configuration_count": len(l1_hypers()),
                    "l2_configuration_count": len(DEPTH_ENCODER) * len(DEPTH_DECODER),
                    "l4_configuration_count": len(NODE_CONDITIONS),
                    "minimum_epochs": MINIMUM_EPOCHS,
                    "max_epochs": MAX_EPOCHS,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    check_runtime(output)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("No GPUs supplied")
    if args.phase == "bootstrap":
        bootstrap(output, gpus, args.slots_per_gpu, args.max_retries)
    elif args.phase == "l1-confirm":
        l1_confirm(output, gpus, args.slots_per_gpu, args.max_retries)
    elif args.phase == "l2":
        l2(output, gpus, args.slots_per_gpu, args.max_retries)
    elif args.phase == "l3":
        l3(output, gpus, args.slots_per_gpu, args.max_retries)
    elif args.phase == "l4":
        l4(output, gpus, args.slots_per_gpu, args.max_retries)
    else:
        run_all(args, output, gpus)


if __name__ == "__main__":
    main()
