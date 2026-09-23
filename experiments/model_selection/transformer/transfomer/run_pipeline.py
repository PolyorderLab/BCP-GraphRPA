#!/usr/bin/env python3
"""Run all validation-only Transformer phases, then the frozen 2k test evaluation."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import run_experiment as exp


def top_two_by_family(summary, configs):
    by_id = {config["config_id"]: config for config in configs}
    selected = {}
    for family in exp.FAMILIES:
        candidates = []
        for row in summary:
            config = by_id[row["config_id"]]
            if row["config_id"] == "anchor" or config.get("varied_family") == family:
                candidates.append(row)
        candidates.sort(key=lambda row: (row["validation_MAPE_mean"], row["validation_MAPE_sd"]))
        selected[family] = candidates[:2]
    return selected


def build_interactions(selection):
    specs = [
        ("lr_x_batch", "lr", "batch_size"),
        ("size_x_layers", "model_size", "num_layers"),
        ("dropout_x_weight_decay", "dropout", "weight_decay"),
    ]
    configs = []
    for grid, family_a, family_b in specs:
        for row_a in selection[family_a]:
            for row_b in selection[family_b]:
                def value_for(row, family):
                    raw = json.loads(row["config_json"])
                    return (raw["d_model"], raw["num_heads"]) if family == "model_size" else raw[family]
                value_a, value_b = value_for(row_a, family_a), value_for(row_b, family_b)
                config = exp.apply_value(exp.ANCHOR, family_a, value_a)
                config = exp.apply_value(config, family_b, value_b)
                configs.append({
                    "config_id": f"{grid}__{exp.slug(value_a)}__{exp.slug(value_b)}",
                    "phase": "interactions", "families": [family_a, family_b], "grid": grid, **config,
                })
    return list({config["config_id"]: config for config in configs}.values())


def build_confirmation(ofat_summary, ofat_configs, interaction_summary, interaction_configs):
    config = deepcopy(exp.ANCHOR)
    by_id = {item["config_id"]: item for item in interaction_configs}
    for grid in ("lr_x_batch", "size_x_layers", "dropout_x_weight_decay"):
        rows = [row for row in interaction_summary if row["config_id"].startswith(grid + "__")]
        winner = min(rows, key=lambda row: (row["validation_MAPE_mean"], row["validation_MAPE_sd"]))
        measured = by_id[winner["config_id"]]
        for family in measured["families"]:
            if family == "model_size":
                config["d_model"], config["num_heads"] = measured["d_model"], measured["num_heads"]
            else:
                config[family] = measured[family]
    by_id = {item["config_id"]: item for item in ofat_configs}
    for family in ("ff_mult", "readout", "position_encoding"):
        candidates = [row for row in ofat_summary if row["config_id"] == "anchor" or by_id[row["config_id"]].get("varied_family") == family]
        winner = min(candidates, key=lambda row: (row["validation_MAPE_mean"], row["validation_MAPE_sd"]))
        config = exp.apply_value(config, family, json.loads(winner["config_json"])[family])
    config.update({"config_id": "combined_candidate", "phase": "confirmation", "families": list(exp.FAMILIES)})
    return [config]


def main():
    max_per_gpu = 2
    # The pool-size GNN workers are CPU-heavy and use little VRAM, so retain
    # two Transformer workers per shared card.  Once a
    # card is free of GNN workers, use the spare compute with up to three
    # Transformer workers on that card.
    idle_gpu_slots = 3
    if not exp.manifest_path("ofat").exists():
        exp.write_manifest("ofat", exp.build_ofat_configs())
    exp.run_phase("ofat", max_per_gpu, idle_gpu_slots=idle_gpu_slots)
    ofat_summary = exp.aggregate("ofat")
    ofat_configs = exp.load_configs("ofat")
    selection = top_two_by_family(ofat_summary, ofat_configs)
    interaction_configs = build_interactions(selection)
    exp.write_manifest("interactions", interaction_configs)
    exp.run_phase("interactions", max_per_gpu, idle_gpu_slots=idle_gpu_slots)
    interaction_summary = exp.aggregate("interactions")
    confirmation_configs = build_confirmation(ofat_summary, ofat_configs, interaction_summary, interaction_configs)
    exp.write_manifest("confirmation", confirmation_configs)
    exp.run_phase("confirmation", max_per_gpu, idle_gpu_slots=idle_gpu_slots)
    confirmation_summary = exp.aggregate("confirmation")
    final_configs = []
    base = deepcopy(confirmation_configs[0])
    for representation in ("full_topology", "no_shared_junction", "species_fraction_only"):
        config = deepcopy(base)
        config.update({"config_id": representation, "phase": "final", "representation": representation, "architecture": "transformer", "families": ["representation"]})
        final_configs.append(config)
    exp.write_manifest("final", final_configs)
    exp.run_phase("final", max_per_gpu, idle_gpu_slots=idle_gpu_slots)
    exp.aggregate("final")
    subprocess.run([
        str(exp.PYTHON), str(exp.CODE_ROOT / "evaluate_final_test.py"),
        "--phase", "final", "--test-dir", str(exp.FINAL_TEST_DIR),
        "--output-dir", str(exp.EXPERIMENT_ROOT / "final_test"), "--device", "cuda:0",
    ], cwd=exp.ROOT, check=True)
    print(json.dumps({"status": "complete", "phases": ["ofat", "interactions", "confirmation", "final", "final_test"]}, indent=2))


if __name__ == "__main__":
    main()
