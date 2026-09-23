#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "baselinemodels" / "models"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT))

from baseline_common import load_yml_records, predict, set_seed
from train_transformer_polymer_type import (
    PolymerTypeDataset,
    PolymerTypeTransformerRegressor,
    forward_batch,
    git_provenance,
    polymer_type_collate,
)
from src.plotting import compute_metrics


def infer_model_config(state, run_config):
    d_model = state["token_embed.weight"].shape[1]
    layer_ids = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("encoder.layers.") and key.split(".")[2].isdigit()
    }
    head_input = state["head.0.weight"].numel()
    if "attn_score.weight" in state:
        readout = "attention"
    elif head_input == d_model * 2:
        readout = "cls_mean"
    else:
        readout = run_config.get("readout", "cls")
    if "pos" in state:
        position_encoding = run_config.get("position_encoding", "learnable")
        max_len = state["pos"].shape[1]
    else:
        position_encoding = run_config.get("position_encoding", "none")
        max_len = int(run_config.get("max_len", 128))
    return {
        "vocab_size": state["token_embed.weight"].shape[0],
        "d_model": d_model,
        "num_heads": int(run_config.get("num_heads", 8)),
        "num_layers": len(layer_ids),
        "dim_feedforward": state["encoder.layers.0.linear1.weight"].shape[0],
        "dropout": float(run_config.get("dropout", 0.2)),
        "max_len": max_len,
        "readout": readout,
        "position_encoding": position_encoding,
        "numeric_mode": run_config.get(
            "numeric_mode", "side_channel" if "number_embed.weight" in state else "none"
        ),
    }


def write_predictions(path, names, y_true, y_pred):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["base_name", "true", "pred"])
        writer.writerows(zip(names, y_true.tolist(), y_pred.tolist()))


def evaluate_one(model, records, vocab, model_config, args, variant, eval_seed, output_dir, device):
    dataset = PolymerTypeDataset(
        records,
        vocab=vocab,
        order_variant=variant,
        seed=eval_seed,
        max_len=model_config["max_len"],
        numeric_mode=args.numeric_mode or model_config["numeric_mode"],
        representation=args.representation,
        canonicalization=args.canonicalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=polymer_type_collate,
    )
    y_pred, y_true, names = predict(model, loader, device, forward_batch)
    metrics = compute_metrics(y_true, y_pred)
    label = variant if variant != "randomized" else f"randomized_seed{eval_seed}"
    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(run_dir / "predictions.csv", names, y_true, y_pred)
    result = {
        "evaluation_label": label,
        "order_variant": variant,
        "evaluation_seed": eval_seed,
        "test_count": len(records),
        "MAPE": float(metrics["MAPE"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "R2": float(metrics["R²"]),
        "MaxAE": float(metrics["MaxAE"]),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PolymerType checkpoint without retraining.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--canonicalization", choices=["legacy", "strict"], default="strict")
    parser.add_argument(
        "--representation",
        choices=["species_fraction_only", "no_shared_junction", "full_topology"],
        default="full_topology",
    )
    parser.add_argument("--numeric-mode", choices=["side_channel", "string", "none"])
    parser.add_argument(
        "--order-variants", default="canonical,reversed,randomized",
        help="Comma-separated canonical/reversed/randomized variants.",
    )
    parser.add_argument(
        "--randomized-seeds", default="11,23,37,53,71",
        help="Comma-separated evaluation seeds used when randomized is requested.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = checkpoint_dir / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.exists()
        else {}
    )
    state = torch.load(checkpoint_dir / "best_model.pth", map_location="cpu", weights_only=True)
    model_config = infer_model_config(state, run_config)
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
    if len(vocab) != model_config["vocab_size"]:
        raise ValueError("vocab.json size does not match checkpoint embedding")
    model = PolymerTypeTransformerRegressor(**model_config)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device)
    set_seed(args.seed)
    records = load_yml_records(args.test_dir)

    variants = [item.strip() for item in args.order_variants.split(",") if item.strip()]
    randomized_seeds = [int(item) for item in args.randomized_seeds.split(",") if item.strip()]
    rows = []
    for variant in variants:
        seeds = randomized_seeds if variant == "randomized" else [args.seed]
        for eval_seed in seeds:
            result = evaluate_one(
                model, records, vocab, model_config, args, variant, eval_seed, output_dir, device
            )
            rows.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    with (output_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "test_dir": str(Path(args.test_dir).resolve()),
        "canonicalization": args.canonicalization,
        "representation": args.representation,
        "model_config": model_config,
        "checkpoint_run_config": run_config,
        "test_base_names": [record["base_name"] for record in records],
        "source_provenance": git_provenance(),
        "results": rows,
    }
    audit_path = Path(args.test_dir) / "selection_audit.json"
    if audit_path.exists():
        metadata["test_selection_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
