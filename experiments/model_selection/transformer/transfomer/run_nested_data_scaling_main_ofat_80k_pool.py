#!/usr/bin/env python3
"""Run the nested training-size study with the confirmed 100k OFAT config.

This runner deliberately writes to a new study directory.  It reuses the
protocol and aggregation helpers from ``run_nested_data_scaling_80k_pool.py``
but replaces its frozen hyperparameters and GPU scheduler.  The development
set is split once (80k pool / 20k validation, random_state=42), every smaller
training set is a prefix of that same pool, and the final 2k test is evaluated
only after all planned training runs pass their audits.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
BASE_RUNNER = CODE_ROOT / "run_nested_data_scaling_80k_pool.py"
STUDY_ROOT = CODE_ROOT / "nested_data_scaling_main_ofat_80k_pool_seed42"
PYTHON = ROOT / ".venv/bin/python"

SIZES_K = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
SEEDS = [202601, 202602, 202603, 202604, 202605]
MIN_EPOCHS = 120
MAX_EPOCHS = 120
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
    "config_id": "combined_candidate",
}


def load_base_runner():
    spec = importlib.util.spec_from_file_location("nested_scaling_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import base runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STUDY_ROOT = STUDY_ROOT
    module.SIZES_K = SIZES_K
    module.SEEDS = SEEDS
    module.MIN_EPOCHS = MIN_EPOCHS
    module.MAX_EPOCHS = MAX_EPOCHS
    module.FROZEN_CONFIG = {k: v for k, v in FROZEN_CONFIG.items() if k != "config_id"}
    return module


def parse_gpu_ids() -> list[int]:
    raw = os.environ.get("TRANSFORMER_MAIN_OFAT_GPU_IDS", "1,2,3")
    ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(gpu < 0 for gpu in ids):
        raise ValueError(f"Invalid GPU list: {raw}")
    return ids


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def command_for(base, size_k: int, seed: int, output_dir: Path, device: str) -> list[str]:
    command = [
        str(PYTHON), str(base.TRAINER),
        "--data-dir", str(base.DATA_DIR),
        "--records-cache", str(base.CACHE),
        "--split-indices", str(
            STUDY_ROOT / "nested_indices" / f"nested_train_{size_k}k_common_val20k_seed42.npz"
        ),
        "--output-dir", str(output_dir),
        "--run-name", f"{size_k}k__seed{seed}",
        "--seed", str(seed),
        "--device", device,
        "--epochs", str(MAX_EPOCHS),
        "--min-epochs", str(MIN_EPOCHS),
        "--early-stopping-patience", "25",
    ]
    for key, flag in (
        ("batch_size", "--batch-size"), ("lr", "--lr"),
        ("weight_decay", "--weight-decay"), ("d_model", "--d-model"),
        ("num_heads", "--num-heads"), ("num_layers", "--num-layers"),
        ("ff_mult", "--ff-mult"), ("dropout", "--dropout"),
        ("readout", "--readout"), ("position_encoding", "--position-encoding"),
        ("representation", "--representation"), ("architecture", "--architecture"),
    ):
        command.extend([flag, str(FROZEN_CONFIG[key])])
    return command


def reusable_run(metrics_path: Path, checkpoint_path: Path, size_k: int, base) -> bool:
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(metrics_path.with_name("run_config.json").read_text(encoding="utf-8"))
        data_manifest = json.loads(metrics_path.with_name("data_manifest.json").read_text(encoding="utf-8"))
        study_manifest = json.loads((STUDY_ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))
        expected_keys = set(FROZEN_CONFIG) - {"config_id"}
        config_matches = all(config.get(key) == FROZEN_CONFIG[key] for key in expected_keys)
        membership_matches = data_manifest.get("test_loaded") is False
        return (
            metrics.get("train_count") == size_k * 1000
            and metrics.get("validation_count") == 20_000
            and metrics.get("test_loaded") is False
            and int(metrics.get("epochs_completed", -1)) >= MIN_EPOCHS
            and config_matches
            and membership_matches
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def run_training(base, gpu_ids: list[int], max_per_gpu: int, cpu_slots: int) -> None:
    runs_root = STUDY_ROOT / "training/runs"
    logs_root = STUDY_ROOT / "training/logs"
    failures_root = STUDY_ROOT / "training/failures"
    for path in (runs_root, logs_root, failures_root):
        path.mkdir(parents=True, exist_ok=True)

    waiting: list[tuple[int, int, int]] = []
    for size_k in SIZES_K:
        for seed in SEEDS:
            output_dir = runs_root / f"{size_k}k" / f"seed{seed}"
            if not reusable_run(output_dir / "metrics.json", output_dir / "best_model.pth", size_k, base):
                waiting.append((size_k, seed, 0))

    use_cuda = cuda_available() and bool(gpu_ids)
    devices = [(gpu, "cuda:0") for gpu in gpu_ids] if use_cuda else [(-1, "cpu")]
    slots = max_per_gpu if use_cuda else max(1, cpu_slots)
    if not use_cuda:
        print("CUDA unavailable; using CPU fallback with one conservative slot.", flush=True)
    print(
        json.dumps({
            "pending_runs": len(waiting),
            "cuda": use_cuda,
            "gpu_ids": gpu_ids if use_cuda else [],
            "slots_per_gpu": slots,
            "study_root": str(STUDY_ROOT),
        }, ensure_ascii=False), flush=True,
    )

    active: list[dict] = []
    counts = {gpu: 0 for gpu, _ in devices}
    max_attempts = int(os.environ.get("TRANSFORMER_MAIN_OFAT_MAX_ATTEMPTS", "3"))
    while waiting or active:
        for gpu, device in devices:
            while waiting and counts[gpu] < slots:
                size_k, seed, attempt = waiting.pop(0)
                output_dir = runs_root / f"{size_k}k" / f"seed{seed}"
                # A previous worker or a verified external reuse may have
                # completed this run since the queue was built.
                if reusable_run(output_dir / "metrics.json", output_dir / "best_model.pth", size_k, base):
                    print(f"skip reusable size={size_k}k seed={seed}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_root / f"{size_k}k__seed{seed}.log"
                handle = log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment.update({
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": os.environ.get("TRANSFORMER_MAIN_OFAT_OMP_THREADS", "4"),
                    "MKL_NUM_THREADS": os.environ.get("TRANSFORMER_MAIN_OFAT_MKL_THREADS", "4"),
                    "MPLCONFIGDIR": "/tmp/matplotlib-tapd-nested-transformer-main-ofat",
                })
                if use_cuda:
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                else:
                    environment.pop("CUDA_VISIBLE_DEVICES", None)
                process = subprocess.Popen(
                    command_for(base, size_k, seed, output_dir, device),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    env=environment,
                )
                active.append({
                    "process": process, "gpu": gpu, "size_k": size_k, "seed": seed,
                    "attempt": attempt, "output_dir": output_dir,
                    "log_path": log_path, "handle": handle, "started": time.time(),
                })
                counts[gpu] += 1
                print(
                    f"launch size={size_k}k seed={seed} gpu={gpu} device={device} "
                    f"attempt={attempt + 1} pid={process.pid}", flush=True,
                )
        time.sleep(10)
        remaining = []
        for item in active:
            process = item["process"]
            if process.poll() is None:
                remaining.append(item)
                continue
            item["handle"].close()
            counts[item["gpu"]] -= 1
            metrics_path = item["output_dir"] / "metrics.json"
            checkpoint_path = item["output_dir"] / "best_model.pth"
            ok = process.returncode == 0 and reusable_run(
                metrics_path, checkpoint_path, item["size_k"], base
            )
            if ok:
                print(
                    f"complete size={item['size_k']}k seed={item['seed']} "
                    f"gpu={item['gpu']} elapsed={time.time() - item['started']:.1f}s",
                    flush=True,
                )
                continue
            log_text = ""
            try:
                log_text = item["log_path"].read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
            oom = any(token in log_text.lower() for token in ("out of memory", "cuda error", "cublas_status_alloc_failed"))
            failure = {
                "size_k": item["size_k"], "seed": item["seed"],
                "attempt": item["attempt"] + 1, "return_code": process.returncode,
                "gpu": item["gpu"], "oom_detected": oom, "log": str(item["log_path"]),
            }
            if item["attempt"] + 1 < max_attempts:
                waiting.append((item["size_k"], item["seed"], item["attempt"] + 1))
                print(f"retry size={item['size_k']}k seed={item['seed']} failure={failure}", flush=True)
            else:
                (failures_root / f"{item['size_k']}k__seed{item['seed']}.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                raise RuntimeError(f"Failed {item['size_k']}k seed={item['seed']}; see {item['log_path']}")
        active = remaining


def run_final_test(base, device: str, gpu_id: int | None) -> None:
    environment = os.environ.copy()
    environment["TRANSFORMER_EXPERIMENT_ROOT"] = str(STUDY_ROOT)
    if gpu_id is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        # The evaluator receives cuda:0, so bind that logical device to the
        # selected physical GPU rather than accidentally using GPU0.
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(
        [
            str(PYTHON), str(base.EVALUATOR), "--phase", "training",
            "--test-dir", str(base.TEST_DIR), "--output-dir", str(STUDY_ROOT / "final_test"),
            "--device", device, "--save-predictions",
        ], cwd=ROOT, env=environment, check=True,
    )


def main() -> None:
    base = load_base_runner()
    base.prepare_protocol()
    manifest_path = STUDY_ROOT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = CODE_ROOT / "confirmation/experiment_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    candidates = [
        config for config in source_manifest.get("configs", [])
        if config.get("config_id") == "combined_candidate"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one combined_candidate in {source_manifest_path}, found {len(candidates)}"
        )
    source_config = candidates[0]
    compare_keys = set(FROZEN_CONFIG) - {"config_id"}
    mismatches = {
        key: (source_config.get(key), FROZEN_CONFIG[key])
        for key in sorted(compare_keys)
        if source_config.get(key) != FROZEN_CONFIG[key]
    }
    if source_config.get("config_id") != FROZEN_CONFIG["config_id"] or mismatches:
        raise RuntimeError(
            "Hard-coded frozen config does not match confirmation combined_candidate: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    manifest.update({
        "runner": str(Path(__file__).resolve()),
        "config_id": "combined_candidate",
        "frozen_hyperparameters_source": str(CODE_ROOT / "confirmation/experiment_manifest.json"),
        "frozen_hyperparameters": FROZEN_CONFIG,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    gpu_ids = parse_gpu_ids()
    max_per_gpu = int(os.environ.get("TRANSFORMER_MAIN_OFAT_MAX_PER_GPU", "1"))
    cpu_slots = int(os.environ.get("TRANSFORMER_MAIN_OFAT_CPU_SLOTS", "1"))
    run_training(base, gpu_ids, max_per_gpu, cpu_slots)
    base.aggregate_validation()
    if not all((STUDY_ROOT / "training/runs" / f"{size}k" / f"seed{seed}" / "metrics.json").is_file() for size in SIZES_K for seed in SEEDS):
        raise RuntimeError("Refusing test access before all planned training runs complete")
    test_gpu_id = gpu_ids[0] if cuda_available() and gpu_ids else None
    device = "cuda:0" if test_gpu_id is not None else "cpu"
    run_final_test(base, device, test_gpu_id)
    base.aggregate_test()
    completion = {
        "status": "complete", "training_runs": len(SIZES_K) * len(SEEDS), "test_runs": len(SIZES_K) * len(SEEDS),
        "config_id": "combined_candidate", "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (STUDY_ROOT / "COMPLETED.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
