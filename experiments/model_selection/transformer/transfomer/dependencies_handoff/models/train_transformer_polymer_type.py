import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from baseline_common import (
    add_common_args,
    get_device,
    load_yml_records,
    make_loaders,
    maybe_limit,
    result_dir,
    run_training,
    set_seed,
)


PAD_TOKEN = "<PAD>"
CLS_TOKEN = "<CLS>"
UNK_TOKEN = "<UNK>"
NUM_TOKEN = "<NUM>"


def blocks_from_record(record):
    return record["system"]["components"][0]["blocks"]


def node_degrees(blocks):
    deg = Counter()
    for block in blocks:
        e1, e2 = block["ends"]
        deg[e1] += 1
        deg[e2] += 1
    return deg


def first_seen_endpoint_maps(blocks):
    degrees = node_degrees(blocks)
    free_map = {}
    branch_map = {}
    for block in blocks:
        for endpoint in block["ends"]:
            if degrees[endpoint] == 1:
                free_map.setdefault(endpoint, f"E{len(free_map) + 1}")
            else:
                branch_map.setdefault(endpoint, f"J{len(branch_map) + 1}")
    return free_map, branch_map


def canonical_block_ids(blocks):
    return {id(block): f"B{i + 1}" for i, block in enumerate(blocks)}


def legacy_topology_order(blocks):
    by_node = defaultdict(list)
    for idx, block in enumerate(blocks):
        for endpoint in block["ends"]:
            by_node[endpoint].append(idx)

    neighbors = defaultdict(set)
    for indices in by_node.values():
        for idx in indices:
            neighbors[idx].update(j for j in indices if j != idx)

    degrees = node_degrees(blocks)
    free_node_rank = {
        idx: min((endpoint for endpoint in block["ends"] if degrees[endpoint] == 1), default="~")
        for idx, block in enumerate(blocks)
    }

    def rank(idx):
        block = blocks[idx]
        return (free_node_rank[idx], str(block["segment"]), str(block["label"]), tuple(block["ends"]))

    order = []
    seen = set()
    for start in sorted(range(len(blocks)), key=rank):
        if start in seen:
            continue
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
            for neighbor in sorted(neighbors[idx] - seen, key=rank, reverse=True):
                stack.append(neighbor)
    return [blocks[idx] for idx in order]


def legacy_ordered_blocks(record, variant, seed):
    blocks = legacy_topology_order(list(blocks_from_record(record)))
    if variant in {"canonical", "topology"}:
        return blocks
    if variant == "reversed":
        return list(reversed(blocks))
    if variant == "randomized":
        rng = random.Random(f'{record["base_name"]}:{seed}')
        shuffled = list(blocks)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown order variant: {variant}")


def endpoint_tokens(endpoint, degrees, free_map, branch_map):
    if degrees[endpoint] == 1:
        return ["FREE", free_map[endpoint]]
    return ["BRANCH", branch_map[endpoint]]


def chain_label(blocks):
    return "".join(str(block["segment"]) for block in blocks)


def float_token(value):
    return f"NUM_{float(value):.8g}"


def legacy_polymer_type_sequence_from_record(
    record, variant="canonical", seed=0, numeric_mode="side_channel"
):
    canonical_blocks = list(blocks_from_record(record))
    blocks = legacy_ordered_blocks(record, variant, seed)
    degrees = node_degrees(canonical_blocks)
    canonical_order = legacy_ordered_blocks(record, "canonical", seed)
    free_map, branch_map = first_seen_endpoint_maps(canonical_order)
    block_ids = canonical_block_ids(canonical_order)

    tokens = [CLS_TOKEN, "BCP", "LABEL", chain_label(canonical_order)]
    numbers = [[0.0, 0.0] for _ in tokens]

    for block in blocks:
        e1, e2 = block["ends"]
        length = float(block["length"])
        value_token = NUM_TOKEN if numeric_mode in {"side_channel", "none"} else float_token(length)
        block_tokens = [
            "BLOCK",
            block_ids[id(block)],
            "SPECIES",
            str(block["segment"]),
            "F",
            value_token,
            "END1",
            *endpoint_tokens(e1, degrees, free_map, branch_map),
            "END2",
            *endpoint_tokens(e2, degrees, free_map, branch_map),
        ]
        tokens.extend(block_tokens)
        block_numbers = [[0.0, 0.0] for _ in block_tokens]
        if numeric_mode == "side_channel":
            block_numbers[block_tokens.index(value_token)] = [length, 1.0]
        numbers.extend(block_numbers)

    return tokens, numbers


def _compressed_colors(signatures):
    unique = {signature: i for i, signature in enumerate(sorted(set(signatures.values())))}
    return {node: unique[signature] for node, signature in signatures.items()}


def _canonical_edge_code(record):
    """Return an exact small-graph canonical code independent of all YML names/order."""
    raw_blocks = list(blocks_from_record(record))
    endpoints = sorted({endpoint for block in raw_blocks for endpoint in block["ends"]})
    incident = {endpoint: [] for endpoint in endpoints}
    edges = []
    for block in raw_blocks:
        u, v = block["ends"]
        attr = (str(block["segment"]), float(block["length"]))
        edges.append((u, v, attr))
        incident[u].append((v, attr))
        incident[v].append((u, attr))

    colors = _compressed_colors(
        {
            node: (len(incident[node]), tuple(sorted(attr for _, attr in incident[node])))
            for node in endpoints
        }
    )
    for _ in range(len(endpoints)):
        signatures = {
            node: (
                colors[node],
                tuple(sorted((attr, colors[neighbor]) for neighbor, attr in incident[node])),
            )
            for node in endpoints
        }
        refined = _compressed_colors(signatures)
        if refined == colors:
            break
        colors = refined

    cells = []
    for color in sorted(set(colors.values())):
        cells.append([node for node in endpoints if colors[node] == color])
    permutation_count = 1
    for cell in cells:
        permutation_count *= math.factorial(len(cell))
    if permutation_count > 100000:
        raise ValueError(
            f"Canonicalization would require {permutation_count} permutations; "
            "the exact routine is intended for the current <=5-block dataset."
        )

    best_code = None
    for cell_orders in itertools.product(*(itertools.permutations(cell) for cell in cells)):
        ordered_nodes = [node for cell_order in cell_orders for node in cell_order]
        node_id = {node: i for i, node in enumerate(ordered_nodes)}
        code = tuple(
            sorted(
                (
                    min(node_id[u], node_id[v]),
                    max(node_id[u], node_id[v]),
                    attr[0],
                    attr[1],
                )
                for u, v, attr in edges
            )
        )
        if best_code is None or code < best_code:
            best_code = code
    return best_code or tuple()


def strict_canonical_blocks(record, representation="full_topology"):
    code = _canonical_edge_code(record)
    blocks = [
        {
            "canonical_u": u,
            "canonical_v": v,
            "segment": segment,
            "length": length,
        }
        for u, v, segment, length in code
    ]
    if representation == "species_fraction_only":
        blocks.sort(key=lambda block: (block["segment"], block["length"]))
    for i, block in enumerate(blocks):
        block["canonical_index"] = i
    return blocks


def _strict_ordered_blocks(record, variant, seed, representation):
    blocks = strict_canonical_blocks(record, representation)
    if variant in {"canonical", "topology"}:
        return blocks
    if variant == "reversed":
        return list(reversed(blocks))
    if variant == "randomized":
        canonical_text = json.dumps(blocks, sort_keys=True, separators=(",", ":"))
        rng = random.Random(f"{canonical_text}:{seed}")
        shuffled = list(blocks)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown order variant: {variant}")


def strict_polymer_type_sequence_from_record(
    record,
    variant="canonical",
    seed=0,
    numeric_mode="side_channel",
    representation="full_topology",
):
    if representation not in {"species_fraction_only", "no_shared_junction", "full_topology"}:
        raise ValueError(f"Unknown representation: {representation}")
    canonical_blocks = strict_canonical_blocks(record, representation)
    blocks = _strict_ordered_blocks(record, variant, seed, representation)
    endpoint_degree = Counter()
    for block in canonical_blocks:
        endpoint_degree[block["canonical_u"]] += 1
        endpoint_degree[block["canonical_v"]] += 1
    free_nodes = sorted(node for node, degree in endpoint_degree.items() if degree == 1)
    junction_nodes = sorted(node for node, degree in endpoint_degree.items() if degree != 1)
    free_map = {node: f"E{i + 1}" for i, node in enumerate(free_nodes)}
    junction_map = {node: f"J{i + 1}" for i, node in enumerate(junction_nodes)}

    if representation == "species_fraction_only":
        tokens = [CLS_TOKEN, "BCP"]
    else:
        tokens = [CLS_TOKEN, "BCP", "LABEL", chain_label(canonical_blocks)]
    numbers = [[0.0, 0.0] for _ in tokens]

    for block in blocks:
        length = float(block["length"])
        value_token = NUM_TOKEN if numeric_mode in {"side_channel", "none"} else float_token(length)
        block_tokens = ["BLOCK"]
        if representation != "species_fraction_only":
            block_tokens.append(f"B{block['canonical_index'] + 1}")
        block_tokens.extend(["SPECIES", block["segment"], "F", value_token])
        if representation != "species_fraction_only":
            for end_name, node in (("END1", block["canonical_u"]), ("END2", block["canonical_v"])):
                endpoint_type = "FREE" if endpoint_degree[node] == 1 else "BRANCH"
                block_tokens.extend([end_name, endpoint_type])
                if representation == "full_topology":
                    block_tokens.append(
                        free_map[node] if endpoint_degree[node] == 1 else junction_map[node]
                    )
        tokens.extend(block_tokens)
        block_numbers = [[0.0, 0.0] for _ in block_tokens]
        if numeric_mode == "side_channel":
            block_numbers[block_tokens.index(value_token)] = [length, 1.0]
        numbers.extend(block_numbers)
    return tokens, numbers


def polymer_type_sequence_from_record(
    record,
    variant="canonical",
    seed=0,
    numeric_mode="side_channel",
    representation="full_topology",
    canonicalization="strict",
):
    if canonicalization == "legacy":
        if representation != "full_topology":
            raise ValueError("Legacy canonicalization only supports full_topology representation")
        return legacy_polymer_type_sequence_from_record(record, variant, seed, numeric_mode)
    if canonicalization != "strict":
        raise ValueError(f"Unknown canonicalization: {canonicalization}")
    return strict_polymer_type_sequence_from_record(
        record, variant, seed, numeric_mode, representation
    )


def sequence_to_text(tokens, numbers):
    pieces = []
    for token, number in zip(tokens, numbers):
        if token == NUM_TOKEN:
            pieces.append(f"{token}({number[0]:.8g})")
        else:
            pieces.append(token)
    return " ".join(pieces)


class PolymerTypeDataset(Dataset):
    def __init__(
        self,
        records,
        vocab=None,
        order_variant="canonical",
        seed=0,
        max_len=None,
        numeric_mode="side_channel",
        dynamic_randomized=False,
        representation="full_topology",
        canonicalization="strict",
    ):
        self.records = records
        self.order_variant = order_variant
        self.seed = seed
        self.numeric_mode = numeric_mode
        self.dynamic_randomized = dynamic_randomized
        self.representation = representation
        self.canonicalization = canonicalization
        self.sequences = [
            polymer_type_sequence_from_record(
                r, order_variant, seed, numeric_mode, representation, canonicalization
            )
            for r in records
        ]
        if vocab is None:
            vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
            for tokens, _ in self.sequences:
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)
        self.vocab = vocab
        self.max_len = max_len or max(len(tokens) for tokens, _ in self.sequences)
        self.cached_ids = None
        self.cached_numbers = None
        self.cached_lengths = None
        self.cached_targets = torch.tensor(
            [record["target_log"] for record in records], dtype=torch.float32
        ).view(-1, 1)
        if not self.dynamic_randomized:
            cache_width = min(
                self.max_len, max(len(tokens) for tokens, _ in self.sequences)
            )
            self.cached_ids = torch.zeros(
                (len(records), cache_width), dtype=torch.long
            )
            self.cached_numbers = torch.zeros(
                (len(records), cache_width, 2), dtype=torch.float32
            )
            self.cached_lengths = []
            for idx, (tokens, numbers) in enumerate(self.sequences):
                ids_tensor, numbers_tensor = self.encode(tokens, numbers)
                length = min(ids_tensor.numel(), cache_width)
                self.cached_ids[idx, :length] = ids_tensor[:length]
                self.cached_numbers[idx, :length] = numbers_tensor[:length]
                self.cached_lengths.append(length)

    def __len__(self):
        return len(self.records)

    def sequence_for_idx(self, idx):
        if not self.dynamic_randomized:
            return self.sequences[idx]
        seed = random.randint(0, 2**31 - 1)
        return polymer_type_sequence_from_record(
            self.records[idx],
            "randomized",
            seed,
            self.numeric_mode,
            self.representation,
            self.canonicalization,
        )

    def encode(self, tokens, numbers):
        tokens = tokens[: self.max_len]
        numbers = numbers[: self.max_len]
        ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokens]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(numbers, dtype=torch.float32)

    def __getitem__(self, idx):
        if self.cached_ids is None:
            ids, numbers = self.encode(*self.sequence_for_idx(idx))
        else:
            length = self.cached_lengths[idx]
            ids = self.cached_ids[idx, :length]
            numbers = self.cached_numbers[idx, :length]
        return {
            "ids": ids,
            "numbers": numbers,
            "y": self.cached_targets[idx],
            "base_name": self.records[idx]["base_name"],
        }


def sinusoidal_position_encoding(max_len, d_model):
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def polymer_type_collate(batch):
    max_len = max(item["ids"].numel() for item in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    numbers = torch.zeros(len(batch), max_len, 2, dtype=torch.float32)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["ids"].numel()
        ids[i, :n] = item["ids"]
        numbers[i, :n] = item["numbers"]
        mask[i, :n] = False
    y = torch.stack([item["y"] for item in batch], dim=0)
    return {
        "ids": ids,
        "numbers": numbers,
        "mask": mask,
        "y": y,
        "base_name": [item["base_name"] for item in batch],
    }


class PolymerTypeTransformerRegressor(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_len=128,
        readout="cls",
        position_encoding="learnable",
        numeric_mode="side_channel",
    ):
        super().__init__()
        self.readout = readout
        self.position_encoding = position_encoding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.number_embed = nn.Linear(2, d_model, bias=False) if numeric_mode == "side_channel" else None
        if position_encoding == "learnable":
            self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        elif position_encoding == "sinusoidal":
            self.register_buffer("pos", sinusoidal_position_encoding(max_len, d_model), persistent=False)
        elif position_encoding == "none":
            self.pos = None
        else:
            raise ValueError(f"Unknown position encoding: {position_encoding}")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if readout == "cls":
            head_in = d_model
        elif readout == "mean":
            head_in = d_model
        elif readout == "cls_mean":
            head_in = d_model * 2
        elif readout == "attention":
            head_in = d_model
            self.attn_score = nn.Linear(d_model, 1)
        else:
            raise ValueError(f"Unknown readout: {readout}")
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        if isinstance(self.pos, nn.Parameter):
            nn.init.trunc_normal_(self.pos, std=0.02)

    def readout_hidden(self, x, mask):
        if self.readout == "cls":
            return x[:, 0]
        valid = (~mask).unsqueeze(-1).to(x.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        mean = (x * valid).sum(dim=1) / denom
        if self.readout == "mean":
            return mean
        if self.readout == "cls_mean":
            return torch.cat([x[:, 0], mean], dim=-1)
        scores = self.attn_score(x).squeeze(-1).masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

    def forward(self, ids, numbers, mask):
        max_len = self.pos.size(1) if self.pos is not None else ids.size(1)
        if ids.size(1) > max_len:
            raise ValueError(f"Input length {ids.size(1)} exceeds max_len {max_len}")
        x = self.token_embed(ids)
        if self.number_embed is not None:
            x = x + self.number_embed(numbers)
        if self.pos is not None:
            x = x + self.pos[:, : ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(self.readout_hidden(x, mask))




def forward_batch(model, batch, device):
    return model(batch["ids"].to(device), batch["numbers"].to(device), batch["mask"].to(device))


def git_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    tracked_status = git(
        "status",
        "--short",
        "--",
        "baselinemodels/models/train_transformer_polymer_type.py",
        "baselinemodels/models/baseline_common.py",
    )
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(tracked_status),
        "git_status_short": tracked_status,
    }


def save_reproducibility_artifacts(out_dir, args, train_records, test_records, train_loader, val_loader):
    out_path = Path(out_dir)
    train_indices = list(train_loader.dataset.indices)
    val_indices = list(val_loader.dataset.indices)
    manifest = {
        "train_directory": str(Path(args.train_dir).resolve()),
        "test_directory": str(Path(args.test_dir).resolve()),
        "split_seed": 42,
        "train_count": len(train_indices),
        "validation_count": len(val_indices),
        "test_count": len(test_records),
        "train_base_names": [train_records[i]["base_name"] for i in train_indices],
        "validation_base_names": [train_records[i]["base_name"] for i in val_indices],
        "test_base_names": [record["base_name"] for record in test_records],
    }
    audit_path = Path(args.test_dir) / "selection_audit.json"
    if audit_path.exists():
        manifest["test_selection_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
    (out_path / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    provenance = git_provenance()
    (out_path / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    snapshot_dir = out_path / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot_dir / Path(__file__).name)
    shutil.copy2(Path(__file__).resolve().parent / "baseline_common.py", snapshot_dir / "baseline_common.py")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PolymerType Transformer baseline.")
    parser = add_common_args(parser)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument(
        "--canonicalization", choices=["strict", "legacy"], default="strict"
    )
    parser.add_argument(
        "--representation",
        choices=["species_fraction_only", "no_shared_junction", "full_topology"],
        default="full_topology",
    )
    parser.add_argument(
        "--numeric-mode",
        choices=["side_channel", "string", "none"],
        default="side_channel",
        help="How block fractions are represented.",
    )
    parser.add_argument(
        "--readout",
        choices=["cls", "mean", "cls_mean", "attention"],
        default="cls",
        help="How token states are pooled before regression.",
    )
    parser.add_argument(
        "--position-encoding",
        choices=["learnable", "sinusoidal", "none"],
        default="learnable",
    )
    parser.add_argument(
        "--train-randomized-augmentation",
        action="store_true",
        help="Use a new randomized equivalent block order each time a training sample is read.",
    )
    parser.add_argument(
        "--test-order-variant",
        choices=["canonical", "reversed", "topology", "randomized"],
        default=None,
    )
    parser.add_argument(
        "--order-variant",
        choices=["canonical", "reversed", "topology", "randomized"],
        default="canonical",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Select hyperparameters by validation loss without evaluating the held-out test set.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    train_records = load_yml_records(
        args.train_dir, args.task_type, max_samples=args.max_samples
    )
    test_records = load_yml_records(
        args.test_dir, args.task_type, max_samples=args.max_samples
    )
    train_dataset = PolymerTypeDataset(
        train_records,
        order_variant=args.order_variant,
        seed=args.seed,
        max_len=args.max_len,
        numeric_mode=args.numeric_mode,
        dynamic_randomized=args.train_randomized_augmentation,
        representation=args.representation,
        canonicalization=args.canonicalization,
    )
    test_dataset = PolymerTypeDataset(
        test_records,
        vocab=train_dataset.vocab,
        order_variant=args.test_order_variant or args.order_variant,
        seed=args.seed,
        max_len=train_dataset.max_len,
        numeric_mode=args.numeric_mode,
        representation=args.representation,
        canonicalization=args.canonicalization,
    )
    train_loader, val_loader, test_loader = make_loaders(
        train_dataset, test_dataset, args.batch_size, args.seed, collate_fn=polymer_type_collate
    )

    model = PolymerTypeTransformerRegressor(
                vocab_size=len(train_dataset.vocab),
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dim_feedforward=args.d_model * args.ff_mult,
                dropout=args.dropout,
                max_len=train_dataset.max_len,
                readout=args.readout,
                position_encoding=args.position_encoding,
                numeric_mode=args.numeric_mode,
            )
    variant_name = (
        f"Transformer_PolymerType_order-{args.order_variant}"
        f"_canon-{args.canonicalization}_repr-{args.representation}"
        f"_num-{args.numeric_mode}_readout-{args.readout}_pos-{args.position_encoding}"
    )
    if args.train_randomized_augmentation:
        variant_name += "_trainaug-randomized"
    if args.test_order_variant and args.test_order_variant != args.order_variant:
        variant_name += f"_testorder-{args.test_order_variant}"
    out_dir = result_dir(args, variant_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "serialization_examples.txt"), "w", encoding="utf-8") as f:
        for tokens, numbers in train_dataset.sequences[:10]:
            f.write(sequence_to_text(tokens, numbers) + "\n")
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset.vocab, f, indent=2, ensure_ascii=False)

    print(f"Device: {device}")
    print(f"Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Order variant: {args.order_variant}")
    print(f"Canonicalization: {args.canonicalization}")
    print(f"Representation: {args.representation}")
    print(f"Test order variant: {args.test_order_variant or args.order_variant}")
    print(f"Numeric mode: {args.numeric_mode}")
    print(f"Readout: {args.readout}")
    print(f"Position encoding: {args.position_encoding}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Train randomized augmentation: {args.train_randomized_augmentation}")
    print(f"Vocab size: {len(train_dataset.vocab)}")
    print(f"Max sequence length: {train_dataset.max_len}")
    print(f"Example: {sequence_to_text(*train_dataset.sequences[0])}")

    run_config = {
        **vars(args),
        "model_class": model.__class__.__name__,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    save_reproducibility_artifacts(
        out_dir, args, train_records, test_records, train_loader, val_loader
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        forward_batch=forward_batch,
        result_dir=out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        title=variant_name,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        evaluate_test=not args.skip_test_evaluation,
    )


if __name__ == "__main__":
    main()
