#!/usr/bin/env python3
"""Resume-safe C2 E5D1 iterative local grid, matching ledger Section 8.

Phases are explicit.  Validation phases never load Test-A or Test-B.  The
bootstrap phase combines L0 and the independent L1 three-seed screen only to
keep the four GPUs occupied; L0 is still summarized before any L1 decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[3]
GNN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselinemodels.pub0818.gnn import run_iterative_local_grid as legacy
from baselinemodels.pub0818.gnn.gnn_validation_worker import evaluate
from mainmodel.scripts import train_single_chin_supplementary_confirmation as source
from src.data_utils import load_and_process_dataset


OUTPUT = GNN_ROOT / "iterative_local_grid_c2_e5d1_80k_train_20k_val"
CONDITION = "C2_virtualnode_learnable_node_full_edge"
SEEDS = (202601, 202602, 202603, 202604, 202605)
SCREEN_SEEDS = SEEDS[:3]
L0_HYPER = {
    "batch_size": 128,
    "learning_rate": 1e-3,
    "hidden_dim": 128,
    "num_layers": 5,
    "dec_layers": 1,
}
L1_GRID = {
    "batch_size": (64, 128, 256),
    "learning_rate": (5e-4, 1e-3, 2e-3),
    "hidden_dim": (64, 128, 256),
}
def build_config(condition: str, hyper: dict, seed: int) -> dict:
    cfg = legacy.build_config(condition, hyper, seed, "identity_not_stage")
    cfg.update({
        "protocol_version": "c2_iterative_local_grid_v2_20260822",
        "test_a_accessed_during_training": False,
        "test_b_accessed": False,
        "model_identity": (
            f"{condition}|b{hyper['batch_size']}|lr{hyper['learning_rate']}|"
            f"h{hyper['hidden_dim']}|E{hyper['num_layers']}|D{hyper['dec_layers']}"
        ),
    })
    return cfg


def l1_hypers() -> list[dict]:
    return [
        {
            "batch_size": batch,
            "learning_rate": lr,
            "hidden_dim": hidden,
            "num_layers": 5,
            "dec_layers": 1,
        }
        for batch, lr, hidden in itertools.product(
            L1_GRID["batch_size"], L1_GRID["learning_rate"], L1_GRID["hidden_dim"]
        )
    ]


def local_metrics(output: Path, cfg: dict) -> Path:
    return output / "runs" / cfg["run_id"] / "metrics.json"


def validated_metrics(output: Path, cfg: dict) -> dict:
    path = local_metrics(output, cfg)
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
        "graph_builder": cfg["graph_builder"],
        "node_builder": cfg["node_builder"],
        "edge_mode": cfg["edge_mode"],
    }
    mismatches = [key for key, value in expected.items() if metrics.get(key) != value]
    if metrics.get("final_test_accessed") is not False:
        mismatches.append("final_test_accessed")
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


def run_tasks(output: Path, tasks: list[dict], gpus: list[str], slots: int, retries: int) -> None:
    # The legacy scheduler is used only for process supervision.  E5D1 cannot
    # trigger its explicit E4D1 historical-reuse branch.
    for cfg in tasks:
        write_config(output, cfg)
    original_write = legacy.write_config
    original_metrics = legacy.metrics_path
    try:
        legacy.write_config = write_config
        legacy.metrics_path = local_metrics
        legacy.run_tasks(output, tasks, gpus, slots, retries)
    finally:
        legacy.write_config = original_write
        legacy.metrics_path = original_metrics


def rows_for(output: Path, hypers: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for hyper in hypers:
        values = []
        for seed in seeds:
            cfg = build_config(CONDITION, hyper, seed)
            metrics = validated_metrics(output, cfg)
            values.append(float(metrics["validation_mape_percent"]))
        rows.append({
            **hyper,
            "hyper_key": legacy.hyper_key(hyper),
            "n_seeds": len(values),
            "validation_mape_mean_percent": mean(values),
            "validation_mape_sd_percent": stdev(values),
            "validation_mape_per_seed_percent": json.dumps(values),
        })
    rows.sort(key=lambda row: (row["validation_mape_mean_percent"], row["validation_mape_sd_percent"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(output: Path) -> None:
    for name in ("configs", "runs", "logs", "summary"):
        (output / name).mkdir(parents=True, exist_ok=True)
    source = legacy.SOURCE_SPLIT
    target = output / "fixed_80k_20k_split_indices.npz"
    source_npz = np.load(source)
    if target.exists():
        target_npz = np.load(target)
        if not all(np.array_equal(source_npz[key], target_npz[key]) for key in ("train_index", "val_index")):
            raise RuntimeError("Output split differs from frozen source split")
    else:
        np.savez(target, train_index=source_npz["train_index"], val_index=source_npz["val_index"])
    protocol = {
        "protocol_version": "c2_iterative_local_grid_v2_20260822",
        "ledger": str(GNN_ROOT / "ITERATIVE_LOCAL_GRID_RECORD.md"),
        "development_pool": str(legacy.TRAIN_DATA),
        "train_size": 80000,
        "validation_size": 20000,
        "split_source": str(source),
        "l0": {"condition": CONDITION, "hyperparameters": L0_HYPER, "seeds": SEEDS},
        "l1_grid": L1_GRID,
        "l1_screen_seeds": SCREEN_SEEDS,
        "promotion_rule": "mean improvement >=5% and >=4/5 paired-seed wins",
        "test_a_policy": "not accessed by bootstrap or any validation phase",
        "test_b_policy": "sealed and not constructed",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def bootstrap(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    hypers = l1_hypers()
    tasks_by_id = {}
    for hyper in hypers:
        for seed in SCREEN_SEEDS:
            cfg = build_config(CONDITION, hyper, seed)
            tasks_by_id[cfg["run_id"]] = cfg
    for seed in SEEDS:
        cfg = build_config(CONDITION, L0_HYPER, seed)
        tasks_by_id[cfg["run_id"]] = cfg
    tasks = list(tasks_by_id.values())
    for cfg in tasks:
        write_config(output, cfg)
    print(f"phase=bootstrap unique_tasks={len(tasks)} slots_per_gpu={slots}", flush=True)
    run_tasks(output, tasks, gpus, slots, retries)
    l0_rows = rows_for(output, [L0_HYPER], SEEDS)
    l1_rows = rows_for(output, hypers, SCREEN_SEEDS)
    write_csv(output / "summary/l0_c2_e5d1.csv", l0_rows)
    write_csv(output / "summary/l1_screen.csv", l1_rows)
    (output / "summary/l1_screen_top3.json").write_text(
        json.dumps(l1_rows[:3], indent=2), encoding="utf-8"
    )
    print(json.dumps({"l0": l0_rows[0], "l1_top3": l1_rows[:3]}, indent=2), flush=True)


def hyper_from_row(row: dict) -> dict:
    return {
        "batch_size": int(row["batch_size"]),
        "learning_rate": float(row["learning_rate"]),
        "hidden_dim": int(row["hidden_dim"]),
        "num_layers": int(row["num_layers"]),
        "dec_layers": int(row["dec_layers"]),
    }


def values_for(output: Path, condition: str, hyper: dict) -> list[float]:
    values = []
    for seed in SEEDS:
        cfg = build_config(condition, hyper, seed)
        metrics = validated_metrics(output, cfg)
        values.append(float(metrics["validation_mape_percent"]))
    return values


def paired_decision(old_hyper: dict, new_hyper: dict, old_values: list[float], new_values: list[float]) -> dict:
    improvement = 100.0 * (mean(old_values) - mean(new_values)) / mean(old_values)
    wins = sum(new < old for old, new in zip(old_values, new_values))
    return {
        "incumbent": old_hyper,
        "challenger": new_hyper,
        "incumbent_per_seed_validation_mape_percent": old_values,
        "challenger_per_seed_validation_mape_percent": new_values,
        "incumbent_mean_validation_mape_percent": mean(old_values),
        "challenger_mean_validation_mape_percent": mean(new_values),
        "relative_improvement_percent": improvement,
        "paired_seed_wins": wins,
        "promote": bool(improvement >= 5.0 and wins >= 4),
        "rule": "mean improvement >=5% and >=4/5 paired-seed wins",
    }


def run_and_summarize(
    output: Path,
    hypers: list[dict],
    seeds: tuple[int, ...],
    summary_name: str,
    gpus: list[str],
    slots: int,
    retries: int,
) -> list[dict]:
    tasks = [build_config(CONDITION, hyper, seed) for hyper in hypers for seed in seeds]
    run_tasks(output, tasks, gpus, slots, retries)
    rows = rows_for(output, hypers, seeds)
    write_csv(output / "summary" / summary_name, rows)
    return rows


def boundary_candidates(best: dict) -> list[dict]:
    candidates = []
    replacements = []
    if best["batch_size"] == 256:
        replacements.append(("batch_size", 512))
    if best["learning_rate"] == 5e-4:
        replacements.append(("learning_rate", 2.5e-4))
    elif best["learning_rate"] == 2e-3:
        replacements.append(("learning_rate", 4e-3))
    if best["hidden_dim"] == 256:
        replacements.append(("hidden_dim", 512))
    for key, value in replacements:
        candidate = dict(best)
        candidate[key] = value
        candidates.append(candidate)
    return candidates


def l1_confirm(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    top3_path = output / "summary/l1_screen_top3.json"
    if not top3_path.is_file():
        raise RuntimeError("Bootstrap summary missing; run --phase bootstrap first")
    top3 = [hyper_from_row(row) for row in json.loads(top3_path.read_text(encoding="utf-8"))]
    rows = run_and_summarize(
        output, top3, SEEDS, "l1_top3_five_seed.csv", gpus, slots, retries
    )
    best = hyper_from_row(rows[0])
    incumbent = dict(L0_HYPER)
    decision = paired_decision(
        incumbent, best, values_for(output, CONDITION, incumbent), values_for(output, CONDITION, best)
    )
    selected = dict(best) if decision["promote"] else dict(incumbent)
    expansion = {"triggered": False, "candidates": [], "decision": None}
    candidates = boundary_candidates(best) if decision["promote"] else []
    if candidates:
        expansion["triggered"] = True
        expansion["candidates"] = candidates
        screen_rows = run_and_summarize(
            output, candidates, SCREEN_SEEDS, "l1_boundary_screen.csv", gpus, slots, retries
        )
        boundary_best = hyper_from_row(screen_rows[0])
        confirmed = run_and_summarize(
            output, [boundary_best], SEEDS, "l1_boundary_five_seed.csv", gpus, slots, retries
        )
        boundary_best = hyper_from_row(confirmed[0])
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
        "top3": rows,
        "core_decision": decision,
        "boundary_expansion": expansion,
        "selected_hyperparameters": selected,
        "test_accessed": False,
    }
    (output / "summary/l1_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def l2_hypers(center: dict) -> list[dict]:
    rows = []
    for encoder, decoder in itertools.product((4, 5, 6), (1, 2)):
        hyper = dict(center)
        hyper.update(num_layers=encoder, dec_layers=decoder)
        rows.append(hyper)
    return rows


def l2(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    selection_path = output / "summary/l1_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("L1 selection missing; run --phase l1-confirm first")
    incumbent = json.loads(selection_path.read_text(encoding="utf-8"))["selected_hyperparameters"]
    hypers = l2_hypers(incumbent)
    rows = run_and_summarize(output, hypers, SEEDS, "l2_depth_grid.csv", gpus, slots, retries)
    best = hyper_from_row(rows[0])
    decision = paired_decision(
        incumbent, best, values_for(output, CONDITION, incumbent), values_for(output, CONDITION, best)
    )
    selected = dict(best) if decision["promote"] else dict(incumbent)
    result = {
        "phase": "L2",
        "decision": decision,
        "selected_hyperparameters": selected,
        "encoder_boundary_limited": bool(decision["promote"] and best["num_layers"] == 6),
        "e7_forbidden": True,
        "test_accessed": False,
    }
    (output / "summary/l2_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def neighbours(value: float, declared: list[float]) -> list[float]:
    ordered = sorted(set(declared + [value]))
    index = ordered.index(value)
    return ordered[max(0, index - 1): min(len(ordered), index + 2)]


def l3_hypers(center: dict, l1_record: dict) -> list[dict]:
    batches = list(L1_GRID["batch_size"])
    lrs = list(L1_GRID["learning_rate"])
    hidden = list(L1_GRID["hidden_dim"])
    boundary = l1_record.get("boundary_expansion", {}).get("decision")
    if boundary and boundary.get("promote"):
        promoted = boundary["challenger"]
        batches.append(int(promoted["batch_size"]))
        lrs.append(float(promoted["learning_rate"]))
        hidden.append(int(promoted["hidden_dim"]))
    rows = []
    for batch, lr, width in itertools.product(
        neighbours(int(center["batch_size"]), batches),
        neighbours(float(center["learning_rate"]), lrs),
        neighbours(int(center["hidden_dim"]), hidden),
    ):
        hyper = dict(center)
        hyper.update(batch_size=int(batch), learning_rate=float(lr), hidden_dim=int(width))
        rows.append(hyper)
    return rows


def l3(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    l2_record = json.loads((output / "summary/l2_selection.json").read_text(encoding="utf-8"))
    incumbent = l2_record["selected_hyperparameters"]
    if not l2_record["decision"]["promote"]:
        result = {
            "phase": "L3",
            "executed": False,
            "reason": "L2 did not promote a new depth",
            "selected_hyperparameters": incumbent,
            "test_accessed": False,
        }
    else:
        l1_record = json.loads((output / "summary/l1_selection.json").read_text(encoding="utf-8"))
        hypers = l3_hypers(incumbent, l1_record)
        screen = run_and_summarize(
            output, hypers, SCREEN_SEEDS, "l3_one_cycle_screen.csv", gpus, slots, retries
        )
        top3 = [hyper_from_row(row) for row in screen[:3]]
        confirmed = run_and_summarize(
            output, top3, SEEDS, "l3_one_cycle_top3_five_seed.csv", gpus, slots, retries
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
            "decision": decision,
            "selected_hyperparameters": selected,
            "test_accessed": False,
        }
    (output / "summary/l3_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def custom_config(base_condition: str, key: str, overrides: dict, hyper: dict, seed: int) -> dict:
    cfg = build_config(base_condition, hyper, seed)
    cfg.update(overrides)
    cfg["condition_key"] = key
    cfg["run_id"] = f"{key}__{legacy.hyper_key(hyper)}__seed{seed}"
    cfg["model_identity"] = (
        f"{key}|b{hyper['batch_size']}|lr{hyper['learning_rate']}|"
        f"h{hyper['hidden_dim']}|E{hyper['num_layers']}|D{hyper['dec_layers']}"
    )
    return cfg


def l4_conditions() -> list[tuple[str, str, dict]]:
    node_bases = {
        "S01": "S01_full_virtualnode_free_branch_degree_full_edge",
        "C2": "C2_virtualnode_learnable_node_full_edge",
        "C3": "C3_virtualnode_degree_avg_neighbor_full_edge",
    }
    rows = []
    for node, base in node_bases.items():
        for edge, label in (("full", "[f, A/B]"), ("f", "[f]"), ("type", "[A/B]")):
            if edge == "full":
                rows.append((base, base, {}))
            else:
                key = f"L4_{node}_virtualnode_edge_{edge}"
                rows.append((base, key, {"edge_mode": edge, "edge_features_label": label}))
    # The alternative constructions change the semantic roles of tensors
    # (e.g. original blocks become line-graph nodes).  Reuse the declared
    # S01-compatible constructions instead of pretending C2 junction-node
    # embeddings transfer unchanged across those representations.
    graph_rows = (
        "B2_bidirectional_free_branch_degree_full_edge",
        "B3_line_graph_free_branch_degree_full_edge",
        "B4_bipartite_free_branch_degree_full_edge",
        "B5_self_loop_free_branch_degree_full_edge",
    )
    for condition in graph_rows:
        rows.append((condition, condition, {}))
    return rows


def l4(output: Path, gpus: list[str], slots: int, retries: int) -> None:
    l3_path = output / "summary/l3_selection.json"
    if not l3_path.is_file():
        raise RuntimeError("L3 gate missing; run --phase l3 first (it may record a skip)")
    hyper = json.loads(l3_path.read_text(encoding="utf-8"))["selected_hyperparameters"]
    definitions = l4_conditions()
    tasks = []
    for base, key, overrides in definitions:
        for seed in SEEDS:
            if key == base and not overrides:
                tasks.append(build_config(base, hyper, seed))
            else:
                tasks.append(custom_config(base, key, overrides, hyper, seed))
    run_tasks(output, tasks, gpus, slots, retries)
    rows = []
    for base, key, overrides in definitions:
        values = []
        for seed in SEEDS:
            cfg = build_config(base, hyper, seed) if key == base and not overrides else custom_config(
                base, key, overrides, hyper, seed
            )
            metrics = validated_metrics(output, cfg)
            values.append(float(metrics["validation_mape_percent"]))
        first = build_config(base, hyper, SEEDS[0]) if key == base and not overrides else custom_config(
            base, key, overrides, hyper, SEEDS[0]
        )
        rows.append({
            "condition_key": key,
            "graph_builder": first["graph_builder"],
            "node_builder": first["node_builder"],
            "edge_mode": first["edge_mode"],
            "n_seeds": 5,
            "validation_mape_mean_percent": mean(values),
            "validation_mape_sd_percent": stdev(values),
            "validation_mape_per_seed_percent": json.dumps(values),
        })
    rows.sort(key=lambda row: row["validation_mape_mean_percent"])
    write_csv(output / "summary/l4_frozen_protocol_diagnostic_ablation.csv", rows)
    result = {
        "phase": "L4",
        "diagnostic_only": True,
        "does_not_reopen_model_selection": True,
        "frozen_selected_condition": CONDITION,
        "frozen_hyperparameters": hyper,
        "n_conditions": len(rows),
        "test_accessed": False,
    }
    (output / "summary/l4_completion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def test_a(output: Path, device_name: str, test_dir: Path, overwrite: bool = False) -> None:
    completion_path = output / "summary/l4_completion.json"
    ablation_path = output / "summary/l4_frozen_protocol_diagnostic_ablation.csv"
    if not completion_path.is_file() or not ablation_path.is_file():
        raise RuntimeError("L4 is incomplete; Test-A firewall remains closed")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    ablation_rows = list(csv.DictReader(ablation_path.open(encoding="utf-8")))
    if completion.get("n_conditions") != 13 or len(ablation_rows) != 13:
        raise RuntimeError("L4 completion artifact is stale or incomplete")
    result_path = output / "development_test/final_frozen_c2_test_a.json"
    if result_path.exists() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return
    hyper = json.loads((output / "summary/l3_selection.json").read_text(encoding="utf-8"))[
        "selected_hyperparameters"
    ]
    test_dir = test_dir.resolve()
    test_source = load_and_process_dataset(str(test_dir), task_type="chiN")
    if len(test_source) != 2000:
        raise RuntimeError(f"Expected 2,000 Test-A graphs, found {len(test_source)}")
    test_a_names = sorted(path.stem for path in test_dir.glob("*.yml"))
    cfg = build_config(CONDITION, hyper, SEEDS[0])
    test_data = source.build_wrapped_dataset(test_source, cfg)
    loader = DataLoader(test_data, batch_size=512, shuffle=False)
    device = torch.device(device_name)
    per_seed = []
    for seed in SEEDS:
        seed_cfg = build_config(CONDITION, hyper, seed)
        model = source.build_model(seed_cfg, test_data, device)
        checkpoint = output / "runs" / seed_cfg["run_id"] / "best_model.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        _, _, metrics = evaluate(model, loader, device)
        per_seed.append({"seed": seed, "mape_percent": float(metrics["MAPE"]), "checkpoint": str(checkpoint)})
        del model
        torch.cuda.empty_cache()
    values = [row["mape_percent"] for row in per_seed]
    result = {
        "status": "Test-A development benchmark; not sealed final Test-B",
        "condition": CONDITION,
        "hyperparameters": hyper,
        "test_a_path": str(test_dir),
        "test_a_graph_count": len(test_source),
        "mape_mean_percent": mean(values),
        "mape_sd_percent": stdev(values),
        "per_seed": per_seed,
        "test_b_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "bootstrap", "l1-confirm", "l2", "l3", "l4", "test-a"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--test-a-dir",
        type=Path,
        default=legacy.DEVELOPMENT_TEST,
        help="Dataset used only by the explicit test-a phase.",
    )
    parser.add_argument(
        "--overwrite-test-a",
        action="store_true",
        help="Replace an existing Test-A result artifact after rerunning inference.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    prepare(output)
    if args.phase == "prepare":
        tasks = len({
            build_config(CONDITION, hyper, seed)["run_id"]
            for hyper in l1_hypers() for seed in SCREEN_SEEDS
        } | {
            build_config(CONDITION, L0_HYPER, seed)["run_id"] for seed in SEEDS
        })
        print(f"prepared output={output} bootstrap_unique_tasks={tasks}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    live = legacy.active_workers_for_output(output)
    if live:
        raise RuntimeError(f"Live workers already target output: {live}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
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
        test_a(output, "cuda:0", args.test_a_dir, args.overwrite_test_a)


if __name__ == "__main__":
    main()
