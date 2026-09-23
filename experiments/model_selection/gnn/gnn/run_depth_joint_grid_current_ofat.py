#!/usr/bin/env python3
"""Run the complete E2--E5 x D1--D3 depth grid under the current OFAT setup."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
GNN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselinemodels.pub0818.gnn.run_iterative_local_grid import (
    BASELINE_HYPER,
    SEEDS,
    SOURCE_SPLIT,
    TRAIN_DATA,
    active_workers_for_output,
    build_config,
    local_metrics,
    metrics_path,
    run_tasks,
    score,
    write_config,
)


CONDITION = "S01_full_virtualnode_free_branch_degree_full_edge"
ENCODER_DEPTHS = (2, 3, 4, 5)
DECODER_DEPTHS = (1, 2, 3)
DEFAULT_OUTPUT = GNN_ROOT / "depth_joint_grid_100k_current_ofat"


def prepare_output(output: Path) -> dict:
    for name in ("configs", "runs", "logs", "summary"):
        (output / name).mkdir(parents=True, exist_ok=True)
    source_split = np.load(SOURCE_SPLIT)
    train = source_split["train_index"]
    val = source_split["val_index"]
    target = output / "fixed_80k_20k_split_indices.npz"
    if target.exists():
        old = np.load(target)
        if not (np.array_equal(old["train_index"], train) and np.array_equal(old["val_index"], val)):
            raise RuntimeError("Existing depth-grid split differs from the frozen OFAT split")
    else:
        np.savez(target, train_index=train, val_index=val)
    protocol = {
        "purpose": "complete encoder x decoder depth interaction grid requested before local-grid redesign",
        "model_condition": CONDITION,
        "graph": "original bidirectional + virtual node",
        "node_features": "[is_free_end, is_branch_point, degree]",
        "edge_features": "[f, A/B]",
        "fixed_hyperparameters": {
            "batch_size": 128,
            "learning_rate": 0.001,
            "hidden_dim": 128,
        },
        "encoder_depths": ENCODER_DEPTHS,
        "decoder_depths": DECODER_DEPTHS,
        "development_pool": str(TRAIN_DATA),
        "actual_train_size": int(len(train)),
        "validation_size": int(len(val)),
        "split_source": str(SOURCE_SPLIT),
        "seeds": SEEDS,
        "selection_metric": "five-seed mean original-scale validation MAPE",
        "test_policy": "no Test-A or Test-B access",
        "reuse": "E4D1 exactly matches the existing S01 + assembled-OFAT five-seed ablation runs",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def depth_hypers() -> list[dict]:
    rows = []
    for encoder, decoder in itertools.product(ENCODER_DEPTHS, DECODER_DEPTHS):
        hyper = dict(BASELINE_HYPER)
        hyper.update(num_layers=encoder, dec_layers=decoder)
        rows.append(hyper)
    return rows


def summarize(output: Path, hypers: list[dict]) -> list[dict]:
    rows = []
    for hyper in hypers:
        configs = [build_config(CONDITION, hyper, seed, "depth_joint_grid") for seed in SEEDS]
        values = [score(output, cfg) for cfg in configs]
        rows.append({
            "depth_config": f"E{hyper['num_layers']}D{hyper['dec_layers']}",
            "num_layers": hyper["num_layers"],
            "dec_layers": hyper["dec_layers"],
            "batch_size": hyper["batch_size"],
            "learning_rate": hyper["learning_rate"],
            "hidden_dim": hyper["hidden_dim"],
            "n_seeds": len(values),
            "validation_mape_mean_percent": mean(values),
            "validation_mape_sd_percent": stdev(values),
            "validation_mape_per_seed_percent": json.dumps(values),
        })
    rows.sort(key=lambda row: (row["validation_mape_mean_percent"], row["validation_mape_sd_percent"]))
    summary_path = output / "summary/depth_joint_grid_validation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = {
        "selection_metric": "five-seed mean original-scale validation MAPE",
        "selected": rows[0],
        "all_results": rows,
        "test_accessed": False,
    }
    (output / "summary/selected_depth.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return rows


def live_worker_records(output: Path) -> list[dict]:
    """Discover all live workers for this output, including adopted workers."""
    records = []
    output_text = str(output)
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            argv = (proc_dir / "cmdline").read_bytes().split(b"\0")
            args = [item.decode(errors="replace") for item in argv if item]
            if "gnn_validation_worker.py" not in " ".join(args) or output_text not in " ".join(args):
                continue
            config_path = Path(args[args.index("--config-json") + 1])
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            environ = (proc_dir / "environ").read_bytes().split(b"\0")
            gpu = next(
                item.split(b"=", 1)[1].decode()
                for item in environ
                if item.startswith(b"CUDA_VISIBLE_DEVICES=")
            )
        except (FileNotFoundError, PermissionError, ValueError, StopIteration, json.JSONDecodeError):
            continue
        records.append({"pid": int(proc_dir.name), "run_id": cfg["run_id"], "gpu": gpu})
    return records


def run_tasks_adopting(
    output: Path,
    tasks: list[dict],
    gpus: list[str],
    slots_per_gpu: int,
    retries: int,
) -> None:
    """Fill around already-running workers without duplicating their run IDs."""
    for cfg in tasks:
        write_config(output, cfg)
    by_id = {cfg["run_id"]: cfg for cfg in tasks}
    worker = GNN_ROOT / "gnn_validation_worker.py"
    split = output / "fixed_80k_20k_split_indices.npz"
    owned: dict[int, dict] = {}
    retry_count: dict[str, int] = {}

    while True:
        # Reap workers created by this scheduler and enforce the retry budget.
        for pid, item in list(owned.items()):
            code = item["proc"].poll()
            if code is None:
                continue
            item["log"].close()
            cfg = item["cfg"]
            run_id = cfg["run_id"]
            del owned[pid]
            if code == 0 and local_metrics(output, cfg).is_file():
                value = json.loads(local_metrics(output, cfg).read_text(encoding="utf-8"))[
                    "validation_mape_percent"
                ]
                print(f"finished {run_id} validation_MAPE={value:.6f}%", flush=True)
            else:
                retry_count[run_id] = retry_count.get(run_id, 0) + 1
                if retry_count[run_id] > retries:
                    raise RuntimeError(f"Failed after retries: {run_id}; see {item['log_path']}")
                print(f"retry={retry_count[run_id]} {run_id}; log={item['log_path']}", flush=True)

        live = live_worker_records(output)
        live_ids = {item["run_id"] for item in live}
        counts = {gpu: sum(item["gpu"] == gpu for item in live) for gpu in gpus}
        pending = [
            cfg for cfg in tasks
            if not metrics_path(output, cfg).is_file() and cfg["run_id"] not in live_ids
        ]

        while pending:
            free_gpus = [gpu for gpu in gpus if counts[gpu] < slots_per_gpu]
            if not free_gpus:
                break
            gpu = min(free_gpus, key=lambda item: (counts[item], gpus.index(item)))
            cfg = pending.pop(0)
            run_dir = output / "runs" / cfg["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = output / "logs" / f"{cfg['run_id']}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                sys.executable,
                str(worker),
                "--config-json", str(output / "configs" / f"{cfg['run_id']}.json"),
                "--split-indices", str(split),
                "--output-dir", str(run_dir),
                "--device", "cuda:0",
            ]
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            owned[proc.pid] = {
                "cfg": cfg,
                "proc": proc,
                "log": log_handle,
                "log_path": log_path,
            }
            live_ids.add(cfg["run_id"])
            counts[gpu] += 1
            print(f"started gpu={gpu} {cfg['run_id']}", flush=True)

        missing = [cfg for cfg in tasks if not metrics_path(output, cfg).is_file()]
        if not missing and not live_worker_records(output):
            return
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--adopt-live-workers",
        action="store_true",
        help="Safely fill free slots around workers started by an earlier scheduler.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    prepare_output(output)
    hypers = depth_hypers()
    tasks = [
        build_config(CONDITION, hyper, seed, "depth_joint_grid")
        for hyper in hypers
        for seed in SEEDS
    ]
    for cfg in tasks:
        write_config(output, cfg)
    reused = sum(metrics_path(output, cfg).is_file() for cfg in tasks)
    if args.prepare_only:
        print(f"Prepared {len(tasks)} tasks across 12 depth combinations; reusable={reused}.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    live = active_workers_for_output(output)
    if live and not args.adopt_live_workers:
        raise RuntimeError(f"Live workers already target this output: {live}")
    print(f"depth-grid tasks={len(tasks)} reusable={reused} new={len(tasks)-reused}", flush=True)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.adopt_live_workers:
        run_tasks_adopting(output, tasks, gpu_ids, args.slots_per_gpu, args.max_retries)
    else:
        run_tasks(output, tasks, gpu_ids, args.slots_per_gpu, args.max_retries)
    rows = summarize(output, hypers)
    print(json.dumps(rows[0], indent=2), flush=True)


if __name__ == "__main__":
    main()
