#!/usr/bin/env python3
"""Run the pre-registered five-seed PolymerType Transformer OFAT study.

The leakage-free 2k corpus is deliberately treated as a selection/reporting set,
not as an untouched final test set, per the revised experiment definition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from copy import deepcopy
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
PYTHON = ROOT / ".venv/bin/python"
TRAIN_SCRIPT = ROOT / "baselinemodels/models/train_transformer_polymer_type.py"
DATA_ROOT = ROOT / "TrainData/data/chiNlimited"
SELECTION_DIR = DATA_ROOT / "n_test_2k_leakage_free_seed42"
SEEDS = [202601, 202602, 202603, 202604, 202605]

ANCHOR = {
    "batch_size": 40,
    "lr": 2e-4,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 2,
    "ff_mult": 4,
    "dropout": 0.10,
    "readout": "cls",
    "position_encoding": "learnable",
    "weight_decay": 1e-5,
    "development_pool": "10k",
}

FAMILIES = {
    "batch_size": [16, 32, 40, 64, 128],
    "lr": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "model_size": [(64, 2), (128, 4), (192, 6), (256, 8)],
    "num_layers": [1, 2, 3, 4],
    "ff_mult": [2, 4, 6],
    "dropout": [0.0, 0.05, 0.10, 0.20],
    "readout": ["cls", "mean", "cls_mean", "attention"],
    "position_encoding": ["learnable", "sinusoidal", "none"],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "development_pool": ["2k", "5k", "10k", "20k", "50k", "100k"],
}


def slug(value) -> str:
    if isinstance(value, tuple):
        return f"d{value[0]}h{value[1]}"
    if isinstance(value, float):
        return f"{value:.0e}".replace("+", "").replace("-0", "-")
    return str(value).replace("_", "-")


def anchor_value(family):
    if family == "model_size":
        return (ANCHOR["d_model"], ANCHOR["num_heads"])
    return ANCHOR[family]


def apply_value(config, family, value):
    config = deepcopy(config)
    if family == "model_size":
        config["d_model"], config["num_heads"] = value
    else:
        config[family] = value
    return config


def build_ofat_configs():
    configs = [
        {
            "config_id": "anchor",
            "phase": "ofat",
            "families": list(FAMILIES),
            "varied_family": "anchor",
            "varied_value": "anchor",
            **deepcopy(ANCHOR),
        }
    ]
    for family, values in FAMILIES.items():
        for value in values:
            if value == anchor_value(family):
                continue
            config = apply_value(ANCHOR, family, value)
            configs.append(
                {
                    "config_id": f"{family}__{slug(value)}",
                    "phase": "ofat",
                    "families": [family],
                    "varied_family": family,
                    "varied_value": list(value) if isinstance(value, tuple) else value,
                    **config,
                }
            )
    if len(configs) != 33:
        raise AssertionError(f"Expected 33 unique OFAT configs, generated {len(configs)}")
    return configs


def load_configs(path):
    return json.loads(path.read_text(encoding="utf-8"))["configs"]


def write_manifest(batch_root, phase, configs):
    phase_root = batch_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "selection_set": str(SELECTION_DIR.resolve()),
        "selection_set_role": "ablation-selection and reporting; not an untouched final test",
        "seeds": SEEDS,
        "epochs": 120,
        "early_stopping_patience": 25,
        "lr_scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 10},
        "optimizer": "Adam",
        "primary_selection_metric": "five-seed mean original-scale MAPE on selection set",
        "configs": configs,
    }
    path = phase_root / "experiment_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_dir_for(outcome_root, run_name):
    matches = list(outcome_root.glob(f"**/{run_name}"))
    return matches[0] if matches else None


def command_for(config, seed, outcome_root, run_name):
    train_dir = DATA_ROOT / f"n_train_{config['development_pool']}"
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--train-dir", str(train_dir),
        "--test-dir", str(SELECTION_DIR),
        "--outcome-root", str(outcome_root),
        "--run-name", run_name,
        "--seed", str(seed),
        "--epochs", "120",
        "--early-stopping-patience", "25",
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--d-model", str(config["d_model"]),
        "--num-heads", str(config["num_heads"]),
        "--num-layers", str(config["num_layers"]),
        "--ff-mult", str(config["ff_mult"]),
        "--dropout", str(config["dropout"]),
        "--readout", str(config["readout"]),
        "--position-encoding", str(config["position_encoding"]),
        "--max-len", "128",
        "--canonicalization", "strict",
        "--representation", str(config.get("representation", "full_topology")),
        "--numeric-mode", "side_channel",
        "--order-variant", str(config.get("order_variant", "canonical")),
        "--device", "cuda:0",
    ]


def run_jobs(batch_root, phase, configs, max_per_gpu):
    outcome_root = batch_root / phase / "runs"
    log_root = batch_root / phase / "logs"
    failure_root = batch_root / phase / "failures"
    for path in (outcome_root, log_root, failure_root):
        path.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"config": config, "seed": seed, "run_name": f"{config['config_id']}__seed{seed}"}
        for config in configs
        for seed in SEEDS
    ]
    waiting, active, failures = [], [], []
    for job in jobs:
        run_dir = run_dir_for(outcome_root, job["run_name"])
        failure_path = failure_root / f"{job['run_name']}.json"
        if run_dir and (run_dir / "metrics.json").exists():
            print(f"skip completed {job['run_name']}", flush=True)
        elif failure_path.exists():
            print(f"skip recorded failure {job['run_name']}", flush=True)
        else:
            waiting.append(job)

    gpu_counts = [0, 0, 0, 0]
    next_gpu = 0
    while waiting or active:
        while waiting:
            candidates = [gpu for gpu, count in enumerate(gpu_counts) if count < max_per_gpu]
            if not candidates:
                break
            gpu = min(candidates, key=lambda item: (gpu_counts[item], (item - next_gpu) % 4))
            next_gpu = (gpu + 1) % 4
            job = waiting.pop(0)
            log_path = log_root / f"{job['run_name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                }
            )
            process = subprocess.Popen(
                command_for(job["config"], job["seed"], outcome_root, job["run_name"]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active.append((process, job, gpu, log_path, handle, time.time()))
            gpu_counts[gpu] += 1
            print(f"launch {job['run_name']} gpu={gpu} pid={process.pid}", flush=True)
        if active:
            time.sleep(10)
        remaining = []
        for process, job, gpu, log_path, handle, started in active:
            code = process.poll()
            if code is None:
                remaining.append((process, job, gpu, log_path, handle, started))
                continue
            handle.close()
            gpu_counts[gpu] -= 1
            print(f"finish {job['run_name']} gpu={gpu} returncode={code}", flush=True)
            if code != 0:
                failure = {
                    "phase": phase,
                    "run_name": job["run_name"],
                    "config_id": job["config"]["config_id"],
                    "seed": job["seed"],
                    "returncode": code,
                    "gpu": gpu,
                    "runtime_seconds": time.time() - started,
                    "log": str(log_path.resolve()),
                    "config": job["config"],
                }
                failures.append(failure)
                (failure_root / f"{job['run_name']}.json").write_text(
                    json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        active = remaining
    (batch_root / phase / "failure_summary.json").write_text(
        json.dumps({"failure_count": len(failures), "failures": failures}, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; failures were preserved")


def aggregate(batch_root, phase, configs):
    outcome_root = batch_root / phase / "runs"
    summary_root = batch_root / phase / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    per_run = []
    incomplete = []
    for config in configs:
        for seed in SEEDS:
            run_name = f"{config['config_id']}__seed{seed}"
            run_dir = run_dir_for(outcome_root, run_name)
            if not run_dir or not (run_dir / "metrics.json").exists():
                incomplete.append(run_name)
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            per_run.append(
                {
                    "phase": phase,
                    "config_id": config["config_id"],
                    "seed": seed,
                    "MAPE": metrics["MAPE"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "R2": metrics["R2"],
                    "MaxAE": metrics["MaxAE"],
                    "best_val_loss": metrics["best_val_loss"],
                    "best_epoch": metrics["best_epoch"],
                    "trainable_parameters": run_config["trainable_parameters"],
                    "run_dir": str(run_dir.resolve()),
                }
            )
    if incomplete:
        raise RuntimeError(f"Cannot aggregate {phase}; incomplete runs: {incomplete[:10]}")

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(summary_root / "per_run.csv", per_run)
    config_map = {item["config_id"]: item for item in configs}
    aggregate_rows = []
    for config_id, config in config_map.items():
        rows = [row for row in per_run if row["config_id"] == config_id]
        result = {"phase": phase, "config_id": config_id, "n": len(rows)}
        for metric in ("MAPE", "RMSE", "MAE", "R2", "MaxAE", "best_val_loss"):
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sd"] = statistics.stdev(values)
            result[f"{metric}_sem"] = statistics.stdev(values) / math.sqrt(len(values))
        result["trainable_parameters"] = int(rows[0]["trainable_parameters"])
        result["config_json"] = json.dumps(config, sort_keys=True)
        aggregate_rows.append(result)
    aggregate_rows.sort(key=lambda row: (row["MAPE_mean"], row["MAPE_sd"], row["trainable_parameters"]))
    for rank, row in enumerate(aggregate_rows, start=1):
        row["raw_mape_rank"] = rank
    write_csv(summary_root / "config_summary.csv", aggregate_rows)
    return aggregate_rows


def family_rows(ofat_summary, configs):
    config_map = {config["config_id"]: config for config in configs}
    output = []
    for family, values in FAMILIES.items():
        for value in values:
            config_id = "anchor" if value == anchor_value(family) else f"{family}__{slug(value)}"
            row = next(item for item in ofat_summary if item["config_id"] == config_id)
            output.append(
                {
                    "family": family,
                    "value": json.dumps(value),
                    "config_id": config_id,
                    **{key: value for key, value in row.items() if key not in {"phase", "config_json"}},
                    "config_json": json.dumps(config_map[config_id], sort_keys=True),
                }
            )
    return output


def select_family_candidates(rows):
    selected = {}
    for family in FAMILIES:
        candidates = sorted(
            [row for row in rows if row["family"] == family],
            key=lambda row: (row["MAPE_mean"], row["MAPE_sd"], row["trainable_parameters"]),
        )
        leader = candidates[0]
        statistically_close = []
        for candidate in candidates:
            se_difference = math.sqrt(leader["MAPE_sem"] ** 2 + candidate["MAPE_sem"] ** 2)
            if candidate["MAPE_mean"] - leader["MAPE_mean"] <= se_difference:
                statistically_close.append(candidate)
        winner = min(
            statistically_close,
            key=lambda row: (row["trainable_parameters"], row["MAPE_mean"], row["MAPE_sd"]),
        )
        selected[family] = {
            "winner": winner,
            "top_two_by_mean": candidates[:2],
            "raw_mean_leader": leader,
            "one_se_tie_candidates": [row["config_id"] for row in statistically_close],
        }
    return selected


def build_interactions(selection):
    specs = [
        ("lr_x_batch", "lr", "batch_size"),
        ("size_x_layers", "model_size", "num_layers"),
        ("dropout_x_weight_decay", "dropout", "weight_decay"),
    ]
    configs = []
    for grid_name, family_a, family_b in specs:
        values_a = [json.loads(row["value"]) for row in selection[family_a]["top_two_by_mean"]]
        values_b = [json.loads(row["value"]) for row in selection[family_b]["top_two_by_mean"]]
        for value_a in values_a:
            if family_a == "model_size":
                value_a = tuple(value_a)
            for value_b in values_b:
                config = apply_value(ANCHOR, family_a, value_a)
                config = apply_value(config, family_b, value_b)
                configs.append(
                    {
                        "config_id": f"{grid_name}__{slug(value_a)}__{slug(value_b)}",
                        "phase": "interactions",
                        "grid": grid_name,
                        "families": [family_a, family_b],
                        "values": {
                            family_a: list(value_a) if isinstance(value_a, tuple) else value_a,
                            family_b: value_b,
                        },
                        **config,
                    }
                )
    unique = {config["config_id"]: config for config in configs}
    if len(unique) > 12:
        raise AssertionError("Interaction study exceeds 12 configurations")
    return list(unique.values())


def build_confirmation(batch_root):
    """Combine measured grid winners with non-interacted OFAT winners."""
    family_selection = json.loads(
        (batch_root / "ofat/summary/family_selection.json").read_text(encoding="utf-8")
    )
    interaction_configs = {
        item["config_id"]: item
        for item in load_configs(batch_root / "interactions/experiment_manifest.json")
    }
    with (batch_root / "interactions/summary/config_summary.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        interaction_rows = list(csv.DictReader(f))

    config = deepcopy(ANCHOR)
    grid_winners = {}
    for grid in ("lr_x_batch", "size_x_layers", "dropout_x_weight_decay"):
        rows = [row for row in interaction_rows if row["config_id"].startswith(grid + "__")]
        winner = min(rows, key=lambda row: (float(row["MAPE_mean"]), float(row["MAPE_sd"])))
        measured = interaction_configs[winner["config_id"]]
        grid_winners[grid] = {
            "config_id": winner["config_id"],
            "MAPE_mean": float(winner["MAPE_mean"]),
            "MAPE_sd": float(winner["MAPE_sd"]),
        }
        for key in (
            "batch_size", "lr", "d_model", "num_heads", "num_layers",
            "dropout", "weight_decay",
        ):
            if key in measured and key in measured.get("families", []) or (
                grid == "lr_x_batch" and key in {"batch_size", "lr"}
            ) or (
                grid == "size_x_layers" and key in {"d_model", "num_heads", "num_layers"}
            ) or (
                grid == "dropout_x_weight_decay" and key in {"dropout", "weight_decay"}
            ):
                config[key] = measured[key]

    for family in ("ff_mult", "readout", "position_encoding"):
        winner = family_selection[family]["winner"]
        value = json.loads(winner["value"])
        config = apply_value(config, family, value)

    # Hyperparameters are selected on 10k; 100k is used only after freezing.
    config["development_pool"] = "10k"
    return [{
        "config_id": "combined_candidate",
        "phase": "confirmation",
        "families": [
            "batch_size", "lr", "model_size", "num_layers", "ff_mult",
            "dropout", "readout", "position_encoding", "weight_decay",
        ],
        "grid_winners": grid_winners,
        **config,
    }]


def frozen_config(batch_root):
    manifest = json.loads(
        (batch_root / "confirmation/experiment_manifest.json").read_text(encoding="utf-8")
    )
    config = deepcopy(manifest["configs"][0])
    config.pop("grid_winners", None)
    config["development_pool"] = "100k"
    config["phase"] = "frozen_100k"
    return config


def build_frozen_100k(batch_root):
    base = frozen_config(batch_root)
    configs = []
    for representation in (
        "full_topology", "no_shared_junction", "species_fraction_only"
    ):
        config = deepcopy(base)
        config.update({
            "config_id": representation,
            "representation": representation,
            "order_variant": "canonical",
            "architecture": "transformer",
            "families": ["representation"],
        })
        configs.append(config)
    return configs


def write_selection_outputs(batch_root, configs, summary):
    rows = family_rows(summary, configs)
    summary_root = batch_root / "ofat/summary"
    with (summary_root / "family_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = select_family_candidates(rows)
    (summary_root / "family_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    interactions = build_interactions(selection)
    write_manifest(batch_root, "interactions", interactions)
    return selection, interactions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True)
    parser.add_argument(
        "--action",
        choices=[
            "prepare-ofat", "run-ofat", "aggregate-ofat",
            "run-interactions", "aggregate-interactions",
            "prepare-confirmation", "run-confirmation", "aggregate-confirmation",
            "prepare-frozen", "run-frozen", "aggregate-frozen",
        ],
        required=True,
    )
    parser.add_argument("--max-per-gpu", type=int, default=2)
    args = parser.parse_args()
    batch_root = Path(args.batch_root)

    if args.action == "prepare-ofat":
        path = write_manifest(batch_root, "ofat", build_ofat_configs())
        print(path)
        return
    if args.action in {"run-ofat", "aggregate-ofat"}:
        manifest = batch_root / "ofat/experiment_manifest.json"
        if not manifest.exists():
            write_manifest(batch_root, "ofat", build_ofat_configs())
        configs = load_configs(manifest)
        if args.action == "run-ofat":
            run_jobs(batch_root, "ofat", configs, args.max_per_gpu)
        else:
            summary = aggregate(batch_root, "ofat", configs)
            selection, interactions = write_selection_outputs(batch_root, configs, summary)
            print(json.dumps({"selection": selection, "interaction_config_count": len(interactions)}, indent=2))
        return
    if args.action == "prepare-confirmation":
        path = write_manifest(batch_root, "confirmation", build_confirmation(batch_root))
        print(path)
        return
    if args.action in {"run-confirmation", "aggregate-confirmation"}:
        manifest = batch_root / "confirmation/experiment_manifest.json"
        if not manifest.exists():
            write_manifest(batch_root, "confirmation", build_confirmation(batch_root))
        configs = load_configs(manifest)
        if args.action == "run-confirmation":
            run_jobs(batch_root, "confirmation", configs, args.max_per_gpu)
        else:
            print(json.dumps(aggregate(batch_root, "confirmation", configs), indent=2))
        return
    phase_actions = {
        "prepare-frozen": ("frozen_100k", build_frozen_100k),
        "run-frozen": ("frozen_100k", build_frozen_100k),
        "aggregate-frozen": ("frozen_100k", build_frozen_100k),
    }
    if args.action in phase_actions:
        phase, builder = phase_actions[args.action]
        manifest = batch_root / phase / "experiment_manifest.json"
        if not manifest.exists():
            write_manifest(batch_root, phase, builder(batch_root))
        configs = load_configs(manifest)
        if args.action.startswith("prepare-"):
            print(manifest)
        elif args.action.startswith("run-"):
            run_jobs(batch_root, phase, configs, args.max_per_gpu)
        else:
            print(json.dumps(aggregate(batch_root, phase, configs), indent=2))
        return
    interaction_manifest = batch_root / "interactions/experiment_manifest.json"
    if not interaction_manifest.exists():
        raise FileNotFoundError("Run aggregate-ofat first to generate interaction configs")
    configs = load_configs(interaction_manifest)
    if args.action == "run-interactions":
        run_jobs(batch_root, "interactions", configs, args.max_per_gpu)
    else:
        print(json.dumps(aggregate(batch_root, "interactions", configs), indent=2))


if __name__ == "__main__":
    main()
