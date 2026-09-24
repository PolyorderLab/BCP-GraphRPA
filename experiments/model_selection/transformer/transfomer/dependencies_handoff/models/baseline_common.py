import argparse
import csv
import glob
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.plotting import compute_metrics, plot_parity_log


TRAIN_DATA_DIR = str(ROOT / "data/datasets/development_100k")
TEST_DATA_DIR = str(ROOT / "data/datasets/seed43_2k")
DEFAULT_IMAGE_DIR = str(ROOT / "results/chain_visualizations")
OUTCOME_ROOT = str(ROOT / "results/transformer")
SEEDS = [202601, 202602, 202603, 202604, 202605]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def normalize_sample_id(stem):
    parts = stem.split("_")
    if "Random" in parts:
        stem = "_".join(parts[: parts.index("Random")])
        parts = stem.split("_")
    if len(parts) >= 4 and parts[0].endswith("blocks") and parts[1] == parts[0] and parts[2] == "sample":
        stem = "_".join([parts[0], "sample", parts[3]])
    return stem


def image_aliases(path):
    stem = sample_id_from_path(path)
    aliases = {stem, normalize_sample_id(stem)}
    parts = stem.split("_")
    if "Random" in parts:
        aliases.add("_".join(parts[: parts.index("Random")]))
    return aliases


def build_image_index(image_dir):
    index = {}
    for path in glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True):
        for alias in image_aliases(path):
            index.setdefault(alias, path)
    return index


def load_yml_records(
    data_dir, task_type="chiN", image_index=None, require_image=False, max_samples=None
):
    records = []
    yml_paths = sorted(glob.glob(os.path.join(data_dir, "*.yml")))
    if max_samples is not None:
        yml_paths = yml_paths[:max_samples]
    for yml_path in yml_paths:
        with open(yml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        system = config["system"]
        target_raw = float(system["χN_star"] if task_type == "chiN" else system["k_star"])
        base_name = sample_id_from_path(yml_path)
        record = {
            "base_name": base_name,
            "sample_key": normalize_sample_id(base_name),
            "yml_path": yml_path,
            "system": system,
            "target_log": float(np.log10(max(target_raw, 1e-6))),
            "target_raw": target_raw,
        }
        if image_index is not None:
            img_path = image_index.get(record["sample_key"]) or image_index.get(base_name)
            record["img_path"] = img_path
            if require_image and img_path is None:
                continue
        records.append(record)
    return records


def blocks_from_record(record):
    return record["system"]["components"][0]["blocks"]


def node_degrees(blocks):
    deg = Counter()
    for block in blocks:
        e1, e2 = block["ends"]
        deg[e1] += 1
        deg[e2] += 1
    return deg


def descriptor_from_record(record, max_blocks=5):
    blocks = blocks_from_record(record)
    n_blocks = len(blocks)
    lengths = np.array([float(b["length"]) for b in blocks], dtype=np.float32)
    is_a = np.array([1.0 if str(b["segment"]) == "A" else 0.0 for b in blocks], dtype=np.float32)
    is_b = 1.0 - is_a
    deg_values = np.array(list(node_degrees(blocks).values()), dtype=np.float32)

    def stats(arr):
        if arr.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())]

    topo_num = float(n_blocks)
    features = [
        n_blocks / max_blocks,
        topo_num / max_blocks,
        float(is_a.sum()) / max_blocks,
        float(is_b.sum()) / max_blocks,
        float((lengths * is_a).sum()),
        float((lengths * is_b).sum()),
    ]
    features.extend(stats(lengths))
    features.extend(stats(lengths[is_a > 0.5]))
    features.extend(stats(lengths[is_b > 0.5]))
    features.extend(stats(deg_values))
    features.extend(
        [
            float(len(deg_values)) / (max_blocks + 1),
            float(np.sum(deg_values == 1)) / (max_blocks + 1),
            float(np.sum(deg_values > 1)) / (max_blocks + 1),
            float(deg_values.max() if deg_values.size else 0.0) / max_blocks,
        ]
    )

    length_pad = np.zeros(max_blocks, dtype=np.float32)
    seg_pad = np.zeros(max_blocks, dtype=np.float32)
    length_pad[: min(max_blocks, n_blocks)] = lengths[:max_blocks]
    seg_pad[: min(max_blocks, n_blocks)] = is_a[:max_blocks]
    features.extend(length_pad.tolist())
    features.extend(seg_pad.tolist())
    return np.array(features, dtype=np.float32)


def tokens_from_record(record, max_blocks=5):
    blocks = blocks_from_record(record)
    deg = node_degrees(blocks)
    n_blocks = len(blocks)
    tokens = []
    for pos, block in enumerate(blocks[:max_blocks]):
        e1, e2 = block["ends"]
        is_a = 1.0 if str(block["segment"]) == "A" else 0.0
        length = float(block["length"])
        tokens.append(
            [
                is_a,
                1.0 - is_a,
                length,
                deg[e1] / max_blocks,
                deg[e2] / max_blocks,
                pos / max(1, max_blocks - 1),
                n_blocks / max_blocks,
            ]
        )
    return np.array(tokens, dtype=np.float32)


class DescriptorDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.features = [descriptor_from_record(r) for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.features[idx], dtype=torch.float32),
            "y": torch.tensor([self.records[idx]["target_log"]], dtype=torch.float32),
            "base_name": self.records[idx]["base_name"],
        }


class TokenDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.tokens = [tokens_from_record(r) for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {
            "tokens": torch.tensor(self.tokens[idx], dtype=torch.float32),
            "y": torch.tensor([self.records[idx]["target_log"]], dtype=torch.float32),
            "base_name": self.records[idx]["base_name"],
        }


def token_collate(batch):
    max_len = max(item["tokens"].size(0) for item in batch)
    feat_dim = batch[0]["tokens"].size(1)
    tokens = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["tokens"].size(0)
        tokens[i, :n] = item["tokens"]
        mask[i, :n] = False
    y = torch.stack([item["y"] for item in batch], dim=0)
    return {"tokens": tokens, "mask": mask, "y": y, "base_name": [item["base_name"] for item in batch]}


class ImageDataset(Dataset):
    def __init__(self, records, img_size=128, cache=True, cache_path=None):
        self.records = records
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.cached_images = None
        if cache:
            if cache_path and os.path.exists(cache_path):
                payload = torch.load(cache_path, weights_only=False, map_location="cpu")
                if payload["img_size"] == img_size and payload["base_names"] == [r["base_name"] for r in records]:
                    self.cached_images = payload["images"]
                else:
                    raise ValueError(f"Image cache does not match records: {cache_path}")
            else:
                cached = []
                for record in records:
                    img = Image.open(record["img_path"]).convert("RGB")
                    img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                    img = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1).float() / 255.0
                    img = (img - self.mean) / self.std
                    cached.append(img.contiguous())
                self.cached_images = torch.stack(cached, dim=0)
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
                    torch.save(
                        {
                            "img_size": img_size,
                            "base_names": [r["base_name"] for r in records],
                            "images": self.cached_images,
                        },
                        tmp_path,
                    )
                    os.replace(tmp_path, cache_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if self.cached_images is None:
            img = Image.open(record["img_path"]).convert("RGB")
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
            img = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1)
        else:
            img = self.cached_images[idx]
        return {
            "image": img,
            "y": torch.tensor([record["target_log"]], dtype=torch.float32),
            "base_name": record["base_name"],
        }


def split_train_val(dataset, seed=42, val_fraction=0.2):
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=val_fraction, random_state=42)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_loaders(train_dataset, test_dataset, batch_size, seed, collate_fn=None):
    train_subset, val_subset = split_train_val(train_dataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    drop_last = len(train_subset) > batch_size and len(train_subset) % batch_size == 1
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, forward_batch):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        y = batch["y"].to(device).view(-1, 1)
        optimizer.zero_grad()
        out = forward_batch(model, batch, device).view(-1, 1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
        count += y.size(0)
    return total / count


def evaluate_loss(model, loader, criterion, device, forward_batch):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device).view(-1, 1)
            out = forward_batch(model, batch, device).view(-1, 1)
            loss = criterion(out, y)
            total += loss.item() * y.size(0)
            count += y.size(0)
    return total / count


def predict(model, loader, device, forward_batch):
    model.eval()
    preds, trues, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = forward_batch(model, batch, device).view(-1).cpu().numpy()
            y = batch["y"].view(-1).numpy()
            preds.append(out)
            trues.append(y)
            names.extend(batch["base_name"])
    return 10 ** np.concatenate(preds), 10 ** np.concatenate(trues), names


def save_loss_curve(train_hist, val_hist, path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_hist, label="Train Loss")
    plt.plot(val_hist, label="Val Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_training(
    *,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    forward_batch,
    result_dir,
    epochs,
    lr,
    weight_decay,
    title,
    seed,
    early_stopping_patience=None,
    evaluate_test=True,
):
    os.makedirs(result_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()
    model_path = os.path.join(result_dir, "best_model.pth")
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, forward_batch)
        val_loss = evaluate_loss(model, val_loader, criterion, device, forward_batch)
        scheduler.step(val_loss)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}")
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d}; "
                f"best epoch={best_epoch:03d}, best val={best_val:.6f}"
            )
            break

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    save_loss_curve(train_hist, val_hist, os.path.join(result_dir, "loss_curve.png"))
    with open(os.path.join(result_dir, "training_history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_log_mse", "val_log_mse"])
        writer.writerows((i, tr, va) for i, (tr, va) in enumerate(zip(train_hist, val_hist), start=1))
    result = {
        "title": title,
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(train_hist),
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "best_val_loss": float(best_val),
    }
    if evaluate_test:
        y_pred, y_true, names = predict(model, test_loader, device, forward_batch)
        metrics = compute_metrics(y_true, y_pred)
        plot_parity_log(
            y_true,
            y_pred,
            os.path.join(result_dir, "parity_plot_nature"),
            title=title,
            xlabel=r"True $\chi N^*$",
            ylabel=r"Predicted $\chi N^*$",
            stats_metrics=metrics,
            figsize=(5, 5),
        )
        result.update(
            {
                "MAPE": float(metrics["MAPE"]),
                "RMSE": float(metrics["RMSE"]),
                "MAE": float(metrics["MAE"]),
                "R2": float(metrics["R²"]),
                "MaxAE": float(metrics["MaxAE"]),
            }
        )
    with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if evaluate_test:
        with open(os.path.join(result_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["base_name", "true", "pred"])
            writer.writerows(zip(names, y_true.tolist(), y_pred.tolist()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def add_common_args(parser):
    parser.add_argument("--train-dir", default=TRAIN_DATA_DIR)
    parser.add_argument("--test-dir", default=TEST_DATA_DIR)
    parser.add_argument("--outcome-root", default=OUTCOME_ROOT)
    parser.add_argument("--task-type", choices=["chiN", "k"], default="chiN")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test limit for train and test records.")
    return parser


def result_dir(args, model_name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{model_name}_seed{args.seed}_{stamp}"
    return os.path.join(args.outcome_root, model_name, run_name)


def maybe_limit(records, max_samples):
    if max_samples is None:
        return records
    return records[:max_samples]
