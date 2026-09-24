#!/usr/bin/env python3
"""Resume-safe staged local-grid search starting from the five-seed C2 baseline.

The final 2k development test is never read during model selection.  It is
evaluated exactly once for the final selected configuration after all
validation-only stages have finished.  A future, disjoint Test-B remains
required for the final paper number.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapd_iterative_grid_mplconfig")

import numpy as np
import torch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[4]
GNN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.gnn_validation_worker import evaluate
from mainmodel.scripts import train_single_chin_supplementary_confirmation as source
from src.data_utils import load_and_process_dataset


SEEDS = (202601, 202602, 202603, 202604, 202605)
SCREEN_SEEDS = SEEDS[:3]
BASELINE_HYPER = {
    "batch_size": 128,
    "learning_rate": 1e-3,
    "hidden_dim": 128,
    "num_layers": 4,
    "dec_layers": 1,
}
SCREEN_GRID = {
    "batch_size": (64, 128, 256),
    "learning_rate": (5e-4, 1e-3, 2e-3),
    "hidden_dim": (64, 128, 256),
}
STRUCTURE_GRID = {
    "num_layers": (3, 4, 5),
    "dec_layers": (1, 2),
}
BASELINE_CONDITION = "C2_virtualnode_learnable_node_full_edge"
SOURCE_ABLATION_ROOT = GNN_ROOT / "ablation_80k/runs_100k_fixed80_20_validation_only"
SOURCE_SPLIT = SOURCE_ABLATION_ROOT / "fixed_80k_20k_split_indices.npz"
SOURCE_BASELINE_RUNS = SOURCE_ABLATION_ROOT / "runs"
TRAIN_DATA = ROOT / "data/datasets/development_100k"
DEVELOPMENT_TEST = ROOT / "data/datasets/validation_seed42"
DEFAULT_OUTPUT = GNN_ROOT / "iterative_local_grid_80k_train_20k_val"


def hyper_key(hyper: dict) -> str:
    lr = f"{float(hyper['learning_rate']):.4g}".replace(".", "p").replace("-", "m")
    return (
        f"b{int(hyper['batch_size'])}_lr{lr}_h{int(hyper['hidden_dim'])}"
        f"_g{int(hyper['num_layers'])}_d{int(hyper['dec_layers'])}"
    )


def is_baseline_hyper(hyper: dict) -> bool:
    return all(float(hyper[key]) == float(value) for key, value in BASELINE_HYPER.items())


def build_config(condition_key: str, hyper: dict, seed: int, stage: str) -> dict:
    cfg = source.condition_config(condition_key)
    cfg.update(hyper)
    cfg.update({
        "run_id": f"{condition_key}__{hyper_key(hyper)}__seed{seed}",
        "seed": int(seed),
        # The stage is deliberately not part of training identity.  A run from
        # the 3-seed screen must be reusable when the same configuration is
        # promoted to 5-seed confirmation.
        "train_data_dir": str(TRAIN_DATA),
        "dataset_cache_dir": str(
            GNN_ROOT
            / "hyperparameter_selection_80k/runs_100k_fixed80_20_validation_only/dataset_cache"
        ),
        "test_data_dir": "SEALED: not available to validation worker",
        "epochs": 200,
        "early_stopping_patience": 30,
        "val_fraction": 0.2,
        "split_random_state": 42,
        "optimizer": "Adam",
        "loss": "MSE on log10(chiN*)",
    })
    return cfg


def source_ablation_metrics(condition: str, seed: int) -> Path:
    return SOURCE_BASELINE_RUNS / f"{condition}__seed{seed}" / "metrics.json"


def local_metrics(output: Path, cfg: dict) -> Path:
    return output / "runs" / cfg["run_id"] / "metrics.json"


def metrics_path(output: Path, cfg: dict) -> Path:
    if is_baseline_hyper(cfg):
        source_path = source_ablation_metrics(cfg["condition_key"], int(cfg["seed"]))
        if source_path.is_file():
            return source_path
    return local_metrics(output, cfg)


def write_config(output: Path, cfg: dict) -> None:
    path = output / "configs" / f"{cfg['run_id']}.json"
    payload = json.dumps(cfg, indent=2)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # Compatibility with configs prepared before search_stage was removed
        # from the immutable training identity.
        existing.pop("search_stage", None)
        if existing != cfg:
            raise RuntimeError(f"Existing config changed: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def active_workers_for_output(output: Path) -> list[int]:
    """Return live worker PIDs for this output root to prevent duplicate runs."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError):
            continue
        if "gnn_validation_worker.py" in argv and str(output) in argv:
            found.append(int(proc.name))
    return found


def run_tasks(output: Path, tasks: list[dict], gpus: list[str], slots_per_gpu: int, retries: int) -> None:
    for cfg in tasks:
        write_config(output, cfg)
    pending = deque(cfg for cfg in tasks if not metrics_path(output, cfg).is_file())
    active: list[dict] = []
    retry_count: dict[str, int] = {}
    slots = [gpu for gpu in gpus for _ in range(slots_per_gpu)]
    worker = GNN_ROOT / "gnn_validation_worker.py"
    split = output / "fixed_80k_20k_split_indices.npz"
    while pending or active:
        counts = {gpu: sum(item["gpu"] == gpu for item in active) for gpu in set(slots)}
        free: list[str] = []
        for gpu in slots:
            if counts.get(gpu, 0):
                counts[gpu] -= 1
            else:
                free.append(gpu)
        while pending and free:
            cfg = pending.popleft()
            gpu = free.pop(0)
            run_dir = output / "runs" / cfg["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = output / "logs" / f"{cfg['run_id']}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                sys.executable,
                str(worker),
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
            active.append({
                "cfg": cfg,
                "gpu": gpu,
                "proc": proc,
                "log": log_handle,
                "log_path": log_path,
            })
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
            run_id = cfg["run_id"]
            if code == 0 and local_metrics(output, cfg).is_file():
                score = json.loads(local_metrics(output, cfg).read_text(encoding="utf-8"))[
                    "validation_mape_percent"
                ]
                print(f"finished {run_id} validation_MAPE={score:.6f}%", flush=True)
            elif retry_count.get(run_id, 0) < retries:
                retry_count[run_id] = retry_count.get(run_id, 0) + 1
                pending.append(cfg)
                print(f"retry={retry_count[run_id]} {run_id}; log={item['log_path']}", flush=True)
            else:
                raise RuntimeError(f"Failed after retries: {run_id}; see {item['log_path']}")
        active = remaining


def score(output: Path, cfg: dict) -> float:
    return float(json.loads(metrics_path(output, cfg).read_text(encoding="utf-8"))["validation_mape_percent"])


def summarize_hyper(output: Path, configurations: list[dict], seeds: tuple[int, ...], name: str) -> list[dict]:
    rows = []
    for hyper in configurations:
        configs = [build_config(BASELINE_CONDITION, hyper, seed, name) for seed in seeds]
        values = [score(output, cfg) for cfg in configs]
        rows.append({
            **hyper,
            "hyper_key": hyper_key(hyper),
            "n_seeds": len(values),
            "validation_mape_mean_percent": mean(values),
            "validation_mape_sd_percent": stdev(values) if len(values) > 1 else 0.0,
            "validation_mape_per_seed_percent": json.dumps(values),
        })
    rows.sort(key=lambda item: (item["validation_mape_mean_percent"], item["validation_mape_sd_percent"]))
    write_csv(output / "summary" / f"{name}.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_decision(old_values: list[float], new_values: list[float]) -> dict:
    old_mean = mean(old_values)
    new_mean = mean(new_values)
    improvement = (old_mean - new_mean) / old_mean * 100.0
    wins = sum(new < old for old, new in zip(old_values, new_values))
    return {
        "old_mean_validation_mape_percent": old_mean,
        "new_mean_validation_mape_percent": new_mean,
        "relative_improvement_percent": improvement,
        "seed_wins": wins,
        "n_seeds": len(old_values),
        "promote": bool(improvement >= 5.0 and wins >= 4),
        "rule": "mean relative improvement >=5% and improvement on >=4/5 paired seeds",
    }


def values_for(output: Path, condition: str, hyper: dict) -> list[float]:
    return [score(output, build_config(condition, hyper, seed, "lookup")) for seed in SEEDS]


def screen_configurations() -> list[dict]:
    configs = []
    for batch, lr, hidden in itertools.product(
        SCREEN_GRID["batch_size"], SCREEN_GRID["learning_rate"], SCREEN_GRID["hidden_dim"]
    ):
        configs.append({
            "batch_size": batch,
            "learning_rate": lr,
            "hidden_dim": hidden,
            "num_layers": BASELINE_HYPER["num_layers"],
            "dec_layers": BASELINE_HYPER["dec_layers"],
        })
    return configs


def structure_configurations(center: dict) -> list[dict]:
    configs = []
    for layers, decoder in itertools.product(STRUCTURE_GRID["num_layers"], STRUCTURE_GRID["dec_layers"]):
        cfg = dict(center)
        cfg.update(num_layers=layers, dec_layers=decoder)
        configs.append(cfg)
    return configs


def summarize_ablation(output: Path, hyper: dict) -> list[dict]:
    rows = []
    for condition in source.CONDITIONS:
        configs = [build_config(condition, hyper, seed, "final_representation_confirmation") for seed in SEEDS]
        values = [score(output, cfg) for cfg in configs]
        first = configs[0]
        rows.append({
            "condition_key": condition,
            "aliases": ";".join(first["aliases"]),
            "condition_label": first["condition_label"],
            "n_seeds": len(values),
            "validation_mape_mean_percent": mean(values),
            "validation_mape_sd_percent": stdev(values),
            "validation_mape_per_seed_percent": json.dumps(values),
        })
    rows.sort(key=lambda item: item["validation_mape_mean_percent"])
    write_csv(output / "summary/final_representation_confirmation.csv", rows)
    return rows


def checkpoint_path(output: Path, condition: str, hyper: dict, seed: int) -> Path:
    cfg = build_config(condition, hyper, seed, "lookup")
    if is_baseline_hyper(hyper):
        source_path = SOURCE_BASELINE_RUNS / f"{condition}__seed{seed}" / "best_model.pth"
        if source_path.is_file():
            return source_path
    return output / "runs" / cfg["run_id"] / "best_model.pth"


def evaluate_final_development_test(output: Path, condition: str, hyper: dict, device_name: str) -> dict:
    result_path = output / "development_test/final_selected_summary.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    cfg = build_config(condition, hyper, SEEDS[0], "final_development_test")
    test_source = load_and_process_dataset(str(DEVELOPMENT_TEST), task_type="chiN")
    test_data = source.build_wrapped_dataset(test_source, cfg)
    loader = DataLoader(test_data, batch_size=512, shuffle=False)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in SEEDS:
        seed_cfg = build_config(condition, hyper, seed, "final_development_test")
        model = source.build_model(seed_cfg, test_data, device)
        ckpt = checkpoint_path(output, condition, hyper, seed)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        _, _, metrics = evaluate(model, loader, device)
        rows.append({"seed": seed, "mape_percent": float(metrics["MAPE"]), "checkpoint": str(ckpt)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    values = [row["mape_percent"] for row in rows]
    result = {
        "status": "development-test only; not final untouched Test-B",
        "condition_key": condition,
        "hyperparameters": hyper,
        "development_test_dir": str(DEVELOPMENT_TEST),
        "n_seeds": len(values),
        "development_test_mape_mean_percent": mean(values),
        "development_test_mape_sd_percent": stdev(values),
        "per_seed": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def prepare_output(output: Path) -> None:
    for name in ("configs", "runs", "logs", "summary", "development_test"):
        (output / name).mkdir(parents=True, exist_ok=True)
    source_split = np.load(SOURCE_SPLIT)
    target = output / "fixed_80k_20k_split_indices.npz"
    if target.exists():
        current = np.load(target)
        if not all(np.array_equal(current[key], source_split[key]) for key in ("train_index", "val_index")):
            raise RuntimeError("Local split differs from the frozen OFAT/ablation split")
    else:
        np.savez(target, train_index=source_split["train_index"], val_index=source_split["val_index"])
    protocol = {
        "objective": "iterative local joint-grid search from C2 + OFAT baseline",
        "development_pool": str(TRAIN_DATA),
        "actual_train_size": int(len(source_split["train_index"])),
        "validation_size": int(len(source_split["val_index"])),
        "split_source": str(SOURCE_SPLIT),
        "baseline_condition": BASELINE_CONDITION,
        "baseline_hyperparameters": BASELINE_HYPER,
        "screen_grid": SCREEN_GRID,
        "screen_seeds": SCREEN_SEEDS,
        "confirmation_seeds": SEEDS,
        "structure_grid": STRUCTURE_GRID,
        "promotion_rule": "mean relative improvement >=5% and improvement on >=4/5 paired seeds",
        "development_test": str(DEVELOPMENT_TEST),
        "development_test_policy": "evaluate only final selected model after validation search",
        "final_test_b_status": "not available; must be disjoint and untouched",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    prepare_output(output)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; refusing to run the GPU experiment on CPU")
    live_workers = active_workers_for_output(output)
    if live_workers:
        raise RuntimeError(
            f"Found live workers for this output root: {live_workers}; "
            "refusing to duplicate them. Wait for or explicitly terminate them before resuming."
        )

    screen_hypers = screen_configurations()
    screen_tasks = [
        build_config(BASELINE_CONDITION, hyper, seed, "round1_screen")
        for hyper in screen_hypers
        for seed in SCREEN_SEEDS
    ]
    for cfg in screen_tasks:
        write_config(output, cfg)
    if args.prepare_only:
        print(f"Prepared {len(screen_tasks)} screen tasks; baseline center reuses 3 existing runs.")
        return

    print("phase=round1_screen", flush=True)
    run_tasks(output, screen_tasks, gpus, args.slots_per_gpu, args.max_retries)
    screen_rows = summarize_hyper(output, screen_hypers, SCREEN_SEEDS, "round1_screen_summary")
    top_hypers = [{key: row[key] for key in BASELINE_HYPER} for row in screen_rows[:3]]
    (output / "summary/round1_top3.json").write_text(json.dumps(top_hypers, indent=2), encoding="utf-8")

    print("phase=round1_top3_five_seed_confirmation", flush=True)
    confirmation_tasks = [
        build_config(BASELINE_CONDITION, hyper, seed, "round1_top3_five_seed_confirmation")
        for hyper in top_hypers
        for seed in SEEDS
    ]
    run_tasks(output, confirmation_tasks, gpus, args.slots_per_gpu, args.max_retries)
    confirmation = summarize_hyper(
        output, top_hypers, SEEDS, "round1_top3_five_seed_confirmation_summary"
    )
    round1_best = {key: confirmation[0][key] for key in BASELINE_HYPER}
    baseline_values = values_for(output, BASELINE_CONDITION, BASELINE_HYPER)
    round1_values = values_for(output, BASELINE_CONDITION, round1_best)
    round1_decision = paired_decision(baseline_values, round1_values)
    round1_decision.update({"incumbent": BASELINE_HYPER, "challenger": round1_best})
    (output / "summary/round1_decision.json").write_text(json.dumps(round1_decision, indent=2), encoding="utf-8")

    incumbent = dict(BASELINE_HYPER)
    decisions = {"round1": round1_decision}
    if round1_decision["promote"]:
        incumbent = dict(round1_best)
        print("phase=round2_structure_grid", flush=True)
        structure_hypers = structure_configurations(incumbent)
        structure_tasks = [
            build_config(BASELINE_CONDITION, hyper, seed, "round2_structure_grid")
            for hyper in structure_hypers
            for seed in SEEDS
        ]
        run_tasks(output, structure_tasks, gpus, args.slots_per_gpu, args.max_retries)
        structure_rows = summarize_hyper(output, structure_hypers, SEEDS, "round2_structure_grid_summary")
        structure_best = {key: structure_rows[0][key] for key in BASELINE_HYPER}
        structure_decision = paired_decision(
            values_for(output, BASELINE_CONDITION, incumbent),
            values_for(output, BASELINE_CONDITION, structure_best),
        )
        structure_decision.update({"incumbent": incumbent, "challenger": structure_best})
        (output / "summary/round2_decision.json").write_text(
            json.dumps(structure_decision, indent=2), encoding="utf-8"
        )
        decisions["round2"] = structure_decision
        if structure_decision["promote"]:
            incumbent = dict(structure_best)
    else:
        print("round1 improvement below promotion rule; structure grid skipped", flush=True)

    if is_baseline_hyper(incumbent):
        print("final hyperparameters equal B0; reusing existing A/B/C confirmation", flush=True)
        representation_rows = list(csv.DictReader(
            (SOURCE_ABLATION_ROOT / "summary/ablation_validation_summary.csv").open(encoding="utf-8")
        ))
    else:
        print("phase=final_representation_confirmation", flush=True)
        ablation_tasks = [
            build_config(condition, incumbent, seed, "final_representation_confirmation")
            for condition in source.CONDITIONS
            for seed in SEEDS
        ]
        run_tasks(output, ablation_tasks, gpus, args.slots_per_gpu, args.max_retries)
        representation_rows = summarize_ablation(output, incumbent)
    representation_best = min(
        representation_rows, key=lambda row: float(row["validation_mape_mean_percent"])
    )["condition_key"]
    representation_decision = paired_decision(
        values_for(output, BASELINE_CONDITION, incumbent),
        values_for(output, representation_best, incumbent),
    )
    representation_decision.update({
        "incumbent_condition": BASELINE_CONDITION,
        "challenger_condition": representation_best,
        "interpretation": "joint graph/node/edge representation search, not a pure one-factor ablation",
    })
    (output / "summary/representation_decision.json").write_text(
        json.dumps(representation_decision, indent=2), encoding="utf-8"
    )
    selected_condition = representation_best if representation_decision["promote"] else BASELINE_CONDITION

    final_result = {
        "validation_selection_complete": True,
        "baseline_condition": BASELINE_CONDITION,
        "baseline_hyperparameters": BASELINE_HYPER,
        "selected_condition": selected_condition,
        "selected_hyperparameters": incumbent,
        "decisions": decisions,
        "representation_decision": representation_decision,
        "final_test_b_status": "sealed/not yet constructed",
    }
    (output / "summary/final_selection.json").write_text(json.dumps(final_result, indent=2), encoding="utf-8")
    print("phase=final_selected_development_test", flush=True)
    test_result = evaluate_final_development_test(output, selected_condition, incumbent, "cuda:0")
    final_result["development_test"] = test_result
    (output / "summary/final_selection.json").write_text(json.dumps(final_result, indent=2), encoding="utf-8")
    print(json.dumps(final_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
