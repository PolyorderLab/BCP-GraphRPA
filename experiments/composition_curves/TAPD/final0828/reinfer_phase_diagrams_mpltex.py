"""Re-infer every mainmodel phase-diagram case with the five final0828 C2 seeds.

The existing RPA coordinates are treated as the analytical reference.  Every ML
value is recomputed from each checkpoint.  Figures show the five-seed mean and
one-standard-deviation envelope, while JSON/CSV retain every seed prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-final0828")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch import nn
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "final0828"
SOURCE_DIR = ROOT / "mainmodel" / "phase_diagrams"
FIGURE_DIR = FINAL_DIR / "figure"
STYLE_DIR = Path.home() / ".codex/skills/mpltex-skill/scripts"

sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(STYLE_DIR)]

from mainmodel.infer_phase_diagrams_from_rpa import add_mainmodel_features_and_virtual_node  # noqa: E402
from paper_style import COLORS, configure_paper_style, save_dual, style_axis  # noqa: E402
from src.graph_builder import build_copolymer_graph  # noqa: E402
from src.ira_graph_common import GINEEncoder, SingleTaskRegressor  # noqa: E402


SEEDS = [202601, 202602, 202603, 202604, 202605]
CHECKPOINTS = {
    seed: FINAL_DIR / f"C2_nested_train__80k__seed{seed}" / "best_model.pth"
    for seed in SEEDS
}
GLOBAL_SOURCE = SOURCE_DIR / "mainmodel_inferred_plot_data.json"
LOCAL_SOURCE = SOURCE_DIR / "mainmodel_inferred_m1_m2_comparison_data.json"
BTABAT_SOURCE = SOURCE_DIR / "btabat_spinodal_data.json"
SHI_SOURCE = SOURCE_DIR / "shi_b2a2_series_data.json"
ABABAB_SOURCE = SOURCE_DIR / "ababab_spinodal_data.json"
OUTPUT_JSON = FIGURE_DIR / "five_seed_phase_diagram_predictions.json"
OUTPUT_CSV = FIGURE_DIR / "five_seed_phase_diagram_predictions.csv"
LEGEND_SIZE = 26
AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 24
ML_MARKER_SIZE = 11

TOPOLOGY_COLORS = {
    "AB": COLORS["blue"],
    "ABA": COLORS["orange"],
    "ABAB": COLORS["green"],
    "ABABA": COLORS["purple"],
    "AB2": COLORS["teal"],
    "AB3": COLORS["red"],
    "M1": COLORS["amber"],
    "M2": COLORS["teal"],
    "BTABAT": COLORS["density_green"],
    "B2A2": COLORS["mean_orange"],
    "(B2A2)2": COLORS["purple"],
}
LABELS = {
    "AB": r"$AB$", "ABA": r"$ABA$", "ABAB": r"$ABAB$", "ABABA": r"$ABABA$",
    "AB2": r"$AB_2$", "AB3": r"$AB_3$", "ABABAB": r"$ABABAB$",
    "M1": r"$M1$", "M2": r"$M2$",
    "BTABAT": r"$(B_T)AB(A_T)$", "B2A2": r"$B_2A_2$", "(B2A2)2": r"$(B_2A_2)_2$",
}


class LearnableNodeEncoder(nn.Module):
    """C2 encoder: local node-position embeddings followed by a GINE encoder."""

    def __init__(self, max_nodes: int = 80, hidden_dim: int = 512, num_layers: int = 5):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.empty(max_nodes, hidden_dim))
        self.gine = GINEEncoder(hidden_dim, edge_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, data):
        counts = data.ptr[1:] - data.ptr[:-1]
        local_ids = torch.cat([torch.arange(int(n), device=data.x.device) for n in counts])
        if local_ids.numel() and int(local_ids.max()) >= self.node_embedding.shape[0]:
            raise ValueError("Graph exceeds the checkpoint's 80-node embedding table")
        embedded = data.clone()
        embedded.x = self.node_embedding[local_ids]
        return self.gine(embedded)


class C2Predictor:
    def __init__(self, checkpoint: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = LearnableNodeEncoder()
        self.model = SingleTaskRegressor(encoder, hidden_dim=512, num_dec_layers=1)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def predict(self, topology: str, f_a: np.ndarray) -> np.ndarray:
        graphs = [
            add_mainmodel_features_and_virtual_node(
                build_copolymer_graph(topology, float(value), N_max=32)
            )
            for value in f_a
        ]
        batch = Batch.from_data_list(graphs).to(self.device)
        with torch.inference_mode():
            pred_log10 = self.model(batch).cpu().numpy().reshape(-1)
        return np.power(10.0, pred_log10)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_rpa(source: dict, topologies: list[str]) -> dict:
    return {topology: source[topology]["rpa"] for topology in topologies}


def assemble_cases() -> dict[str, dict]:
    global_data = load_json(GLOBAL_SOURCE)
    local_data = load_json(LOCAL_SOURCE)
    btabat_data = load_json(BTABAT_SOURCE)
    shi_data = load_json(SHI_SOURCE)
    ababab_data = load_json(ABABAB_SOURCE)
    global_rpa = extract_rpa(global_data, ["AB", "ABA", "ABAB", "ABABA", "AB2", "AB3", "M1", "M2"])
    global_rpa.update(extract_rpa(ababab_data, ["ABABAB"]))
    return {
        "global": global_rpa,
        "m1_m2_local": extract_rpa(local_data, ["M1", "M2"]),
        "btabat": extract_rpa(btabat_data, ["BTABAT"]),
        "shi": extract_rpa(shi_data, ["AB", "B2A2", "(B2A2)2"]),
    }


def infer_all(cases: dict[str, dict]) -> dict:
    output = {}
    for case, topology_data in cases.items():
        output[case] = {}
        for topology, rpa in topology_data.items():
            f_a = np.asarray(rpa["fA"], dtype=float)
            output[case][topology] = {
                "rpa": {"fA": f_a.tolist(), "chiN": list(map(float, rpa["chiN"]))},
                "seeds": {},
            }

    for seed, checkpoint in CHECKPOINTS.items():
        print(f"Loading seed {seed}: {checkpoint}", flush=True)
        predictor = C2Predictor(checkpoint)
        for case, topology_data in output.items():
            for topology, values in topology_data.items():
                f_a = np.asarray(values["rpa"]["fA"], dtype=float)
                pred = predictor.predict(topology, f_a)
                values["seeds"][str(seed)] = pred.tolist()
                print(f"  {case}/{topology}: {len(pred)} points", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for topology_data in output.values():
        for values in topology_data.values():
            stack = np.asarray([values["seeds"][str(seed)] for seed in SEEDS])
            values["ml_mean"] = stack.mean(axis=0).tolist()
            values["ml_std"] = stack.std(axis=0, ddof=1).tolist()
    return output


def update_one_seed(data: dict, seed: int) -> dict:
    """Replace one seed in saved predictions and refresh mean/sample SD."""
    if seed not in CHECKPOINTS:
        raise ValueError(f"Unknown seed {seed}; choose one of {SEEDS}")
    predictor = C2Predictor(CHECKPOINTS[seed])
    for case, topology_data in data.items():
        for topology, values in topology_data.items():
            f_a = np.asarray(values["rpa"]["fA"], dtype=float)
            values["seeds"][str(seed)] = predictor.predict(topology, f_a).tolist()
            stack = np.asarray([values["seeds"][str(item)] for item in SEEDS])
            values["ml_mean"] = stack.mean(axis=0).tolist()
            values["ml_std"] = stack.std(axis=0, ddof=1).tolist()
            print(f"Updated {case}/{topology} for seed {seed}: {len(f_a)} points", flush=True)
    return data


def save_predictions(data: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump({"seeds": SEEDS, "cases": data}, handle, indent=2)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = ["case", "topology", "point_index", "fA", "rpa_chiN", *[f"seed_{s}" for s in SEEDS], "ml_mean", "ml_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, topology_data in data.items():
            for topology, values in topology_data.items():
                for idx, (f_a, rpa, mean, std) in enumerate(zip(values["rpa"]["fA"], values["rpa"]["chiN"], values["ml_mean"], values["ml_std"]), start=1):
                    row = {"case": case, "topology": topology, "point_index": idx, "fA": f_a, "rpa_chiN": rpa, "ml_mean": mean, "ml_std": std}
                    row.update({f"seed_{seed}": values["seeds"][str(seed)][idx - 1] for seed in SEEDS})
                    writer.writerow(row)


def draw_series(
    ax, values: dict, topology: str, *, labeled: bool = True,
    marker_step: int = 4, color_override: str | None = None,
    band_alpha: float = 0.28,
) -> None:
    x = np.asarray(values["rpa"]["fA"])
    ml_x = np.asarray(values.get("ml_fA", values["rpa"]["fA"]))
    rpa = np.asarray(values["rpa"]["chiN"])
    mean = np.asarray(values["ml_mean"])
    std = np.asarray(values["ml_std"])
    band_x = np.asarray(values.get("band_fA", ml_x))
    band_mean = np.asarray(values.get("band_mean", mean))
    band_std = np.asarray(values.get("band_std", std))
    color = color_override or TOPOLOGY_COLORS[topology]
    prefix = f"{LABELS[topology]} " if labeled else ""
    ax.plot(x, rpa, color=color, lw=3.4, label=f"{prefix}RPA", zorder=2)
    ax.fill_between(band_x, np.maximum(band_mean - band_std, 0), band_mean + band_std,
                    color=color, alpha=band_alpha, linewidth=0, zorder=1)
    ax.plot(ml_x[::marker_step], mean[::marker_step], linestyle="none", marker="o", ms=ML_MARKER_SIZE,
            color=color, markeredgecolor="none", markeredgewidth=0,
            label=f"{prefix}GNN", zorder=3)


def structure_label_size(ax) -> float:
    """Scale topology labels to match M1's 32 pt on a 5.8-inch panel."""
    spec = ax.get_subplotspec()
    rows = spec.get_gridspec().nrows if spec is not None else 1
    panel_height = ax.figure.get_figheight() / rows
    return 32.0 * panel_height / 5.8


def add_standard_legend(ax) -> None:
    """Use one legend design and method ordering throughout the figure set."""
    handles, labels = ax.get_legend_handles_labels()
    # Matplotlib fills multi-column legends column-first.  Reorder the natural
    # topology-wise pairs so every visible row is ``RPA | GNN`` for the
    # same architecture.
    if len(handles) % 2 == 0:
        order = list(range(0, len(handles), 2)) + list(range(1, len(handles), 2))
        handles = [handles[index] for index in order]
        labels = [labels[index] for index in order]
    ax.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.5,
        labelspacing=0.45,
        borderaxespad=0.55,
    )


def finish_axis(
    ax,
    *,
    xlim=(0, 1),
    ylim=(0, 210),
    annotation: str | None = None,
    annotation_x: float = 0.08,
    axis_label_size: float = AXIS_LABEL_SIZE,
    tick_label_size: float = TICK_LABEL_SIZE,
) -> None:
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"$f_A$", ylabel=r"$\chi N$")
    if xlim == (0, 1):
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    if ylim == (0, 210):
        ax.set_yticks([0, 50, 100, 150, 200])
    elif ylim == (0, 550):
        ax.set_yticks([0, 100, 200, 300, 400, 500])
    if annotation:
        # The lower-left branch of every U-shaped spinodal contains data.  Keep
        # the panel identifier in the open upper-left region instead.
        ax.text(annotation_x, 0.78, annotation, transform=ax.transAxes,
                fontsize=structure_label_size(ax), fontstyle="italic")
    style_axis(ax)
    # Keep both coordinate directions visually identical in every panel.
    ax.xaxis.label.set_fontsize(axis_label_size)
    ax.yaxis.label.set_fontsize(axis_label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)


def save(fig, name: str) -> None:
    paths = save_dual(fig, FIGURE_DIR / name)
    plt.close(fig)
    print(f"Saved {paths[0]} and {paths[1]}", flush=True)


def plot_single(data: dict, topologies: list[str], name: str, *, figsize=(14, 10), ylim=(0, 210)) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    single_topology = len(topologies) == 1
    for topology in topologies:
        draw_series(ax, data[topology], topology, labeled=not single_topology)
    annotation = LABELS[topologies[0]] if single_topology else None
    finish_axis(ax, ylim=ylim, annotation=annotation,
                annotation_x=0.14 if single_topology else 0.08)
    add_standard_legend(ax)
    save(fig, name)


def plot_grid(data: dict, topologies: list[str], name: str, shape: tuple[int, int], *, ylim=(0, 550), figsize=(18, 11)) -> None:
    fig, axes = plt.subplots(*shape, figsize=figsize, constrained_layout=True, squeeze=False)
    for ax, topology in zip(axes.ravel(), topologies):
        draw_series(ax, data[topology], topology, labeled=False)
        finish_axis(ax, ylim=ylim, annotation=LABELS[topology])
        add_standard_legend(ax)
    for ax in axes.ravel()[len(topologies):]:
        ax.axis("off")
    save(fig, name)


def plot_ab_series(data: dict) -> None:
    """Combined AB/AB2/AB3 panel using the finalized global-panel style."""
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    colors = {
        "AB": plt.rcParams["axes.labelcolor"],
        "AB2": COLORS["blue"],
        "AB3": COLORS["orange"],
    }
    legend_handles = []
    legend_labels = []
    for topology in ["AB", "AB2", "AB3"]:
        draw_series(
            ax, data[topology], topology, labeled=False,
            color_override=colors[topology], band_alpha=0.10,
        )
        handles, _ = ax.get_legend_handles_labels()
        legend_handles.extend([
            handles[-2],
            handles[-1],
            Patch(facecolor=colors[topology], alpha=0.10, edgecolor="none"),
        ])
        label = LABELS[topology]
        legend_labels.extend([
            f"{label} RPA calculation",
            f"{label} GNN prediction",
            f"{label} mean $\\pm$ 1 SD",
        ])

    finish_axis(
        ax, ylim=(0, 75),
        axis_label_size=32, tick_label_size=32,
    )
    ax.set_yticks([0, 25, 50, 75])
    ax.set_box_aspect(2 / 3)
    for topology, y_pos in zip(["AB", "AB2", "AB3"], [0.05, 0.12, 0.19]):
        ax.text(
            0.06, y_pos, LABELS[topology],
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=48, fontstyle="italic", color=colors[topology],
        )
    ax.legend(
        legend_handles[:3],
        ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."],
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=28,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0,
        alignment="center",
    )
    save(fig, "rpa_ab_series_publication")


def plot_shi_series(data: dict) -> None:
    """Combined Shi architecture series using the finalized AB-series style."""
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    topologies = ["AB", "B2A2", "(B2A2)2"]
    colors = {
        "AB": plt.rcParams["axes.labelcolor"],
        "B2A2": COLORS["blue"],
        "(B2A2)2": COLORS["orange"],
    }
    method_handles = []
    for topology in topologies:
        draw_series(
            ax, data[topology], topology, labeled=False,
            color_override=colors[topology], band_alpha=0.10,
        )
        if not method_handles:
            handles, _ = ax.get_legend_handles_labels()
            method_handles = [
                handles[-2],
                handles[-1],
                Patch(facecolor=colors[topology], alpha=0.10, edgecolor="none"),
            ]

    finish_axis(
        ax, ylim=(0, 150),
        axis_label_size=32, tick_label_size=32,
    )
    ax.set_yticks([0, 50, 100, 150])
    ax.set_box_aspect(2 / 3)
    for topology, y_pos in zip(topologies, [0.19, 0.12, 0.05]):
        ax.text(
            0.06, y_pos, LABELS[topology],
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=48, fontstyle="italic", color=colors[topology],
        )
    ax.legend(
        method_handles,
        ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."],
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=28,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0,
        alignment="center",
    )
    save(fig, "shi_b2a2_series_publication")


def plot_architecture_series(data: dict) -> None:
    """Combined AB/ABAB/ABABAB panel using the finalized series style."""
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    topologies = ["AB", "ABAB", "ABABAB"]
    colors = {
        "AB": plt.rcParams["axes.labelcolor"],
        "ABAB": COLORS["blue"],
        "ABABAB": COLORS["orange"],
    }
    method_handles = []
    for topology in topologies:
        draw_series(
            ax, data[topology], topology, labeled=False,
            color_override=colors[topology], band_alpha=0.10,
        )
        if not method_handles:
            handles, _ = ax.get_legend_handles_labels()
            method_handles = [
                handles[-2],
                handles[-1],
                Patch(facecolor=colors[topology], alpha=0.10, edgecolor="none"),
            ]

    finish_axis(
        ax, ylim=(0, 125),
        axis_label_size=32, tick_label_size=32,
    )
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_box_aspect(2 / 3)
    for topology, y_pos in zip(topologies, [0.19, 0.12, 0.05]):
        ax.text(
            0.025, y_pos, LABELS[topology],
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=48, fontstyle="italic", color=colors[topology],
            clip_on=False, in_layout=False,
        )
    ax.legend(
        method_handles,
        ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."],
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=28,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0,
        alignment="center",
    )
    save(fig, "rpa_ml_comparison_all_publication")


def plot_local_m1_m2(global_data: dict, local_data: dict) -> None:
    for topology in ["M1", "M2"]:
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        values = global_data[topology]
        x = np.asarray(values["rpa"]["fA"])
        selected = (x >= 0.10) & (x <= 0.20)
        local_rpa = local_data[topology]["rpa"]
        local_rpa_x = np.asarray(local_rpa["fA"], dtype=float)
        local_rpa_y = np.asarray(local_rpa["chiN"], dtype=float)
        global_rpa_y = np.asarray(values["rpa"]["chiN"], dtype=float)
        global_rpa_mask = (x >= 0.08) & (x <= 0.22)
        rpa_x = np.concatenate([x[global_rpa_mask], local_rpa_x])
        rpa_y = np.concatenate([global_rpa_y[global_rpa_mask], local_rpa_y])
        order = np.argsort(rpa_x)
        subset = {
            "rpa": {
                "fA": rpa_x[order].tolist(),
                "chiN": rpa_y[order].tolist(),
            },
            "ml_fA": x[selected].tolist(),
            "ml_mean": np.asarray(values["ml_mean"])[selected].tolist(),
            "ml_std": np.asarray(values["ml_std"])[selected].tolist(),
            # Include adjacent already-inferred global compositions so the
            # uncertainty band reaches both visible x boundaries continuously.
            "band_fA": x[global_rpa_mask].tolist(),
            "band_mean": np.asarray(values["ml_mean"])[global_rpa_mask].tolist(),
            "band_std": np.asarray(values["ml_std"])[global_rpa_mask].tolist(),
        }
        # Keep the dense analytical local RPA line, while GNN points come only
        # from the already inferred global grid.
        draw_series(ax, subset, topology, labeled=False, marker_step=1,
                    color_override=plt.rcParams["axes.labelcolor"],
                    band_alpha=0.10)
        finish_axis(
            ax, xlim=(0.10, 0.20), ylim=(45, 175),
            axis_label_size=32, tick_label_size=32,
        )
        ax.set_box_aspect(2 / 3)
        ax.set_xticks([0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
        ax.set_xticklabels(["0.10", "0.12", "0.14", "0.16", "0.18", "0.20"])
        ax.set_yticks([50, 75, 100, 125, 150, 175])
        ax.text(
            0.08, 0.10, LABELS[topology],
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=48, fontstyle="italic", color="#3568F4",
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=plt.rcParams["axes.labelcolor"], alpha=0.10,
                             edgecolor="none"))
        labels = ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."]
        ax.legend(
            handles,
            labels,
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.96),
            fontsize=28,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.7,
            labelspacing=0.45,
            borderaxespad=0,
            alignment="center",
        )
        save(fig, f"{topology.lower()}_comparison_publication")


def plot_btabat_local(data: dict) -> None:
    values = data["BTABAT"]
    x = np.asarray(values["rpa"]["fA"])
    marker_mask = (x > 0.10) & (x < 0.30)
    left = max(int(np.searchsorted(x, 0.10, side="left")) - 1, 0)
    right = min(int(np.searchsorted(x, 0.30, side="right")) + 1, len(x))
    continuous = slice(left, right)
    subset = {
        "rpa": {
            "fA": x[continuous].tolist(),
            "chiN": np.asarray(values["rpa"]["chiN"])[continuous].tolist(),
        },
        "ml_fA": x[marker_mask].tolist(),
        "ml_mean": np.asarray(values["ml_mean"])[marker_mask].tolist(),
        "ml_std": np.asarray(values["ml_std"])[marker_mask].tolist(),
        "band_fA": x[continuous].tolist(),
        "band_mean": np.asarray(values["ml_mean"])[continuous].tolist(),
        "band_std": np.asarray(values["ml_std"])[continuous].tolist(),
    }
    fig, ax = plt.subplots(figsize=(12, 8.5), constrained_layout=True)
    # The local window contains exactly 20 inferred compositions; show every
    # GNN prediction rather than applying the global marker thinning.
    draw_series(ax, subset, "BTABAT", labeled=False, marker_step=1,
                color_override=plt.rcParams["axes.labelcolor"], band_alpha=0.10)
    finish_axis(ax, xlim=(0.10, 0.30), ylim=(22, 175),
                axis_label_size=32, tick_label_size=32)
    ax.set_xticks([0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xticklabels(["0.10", "0.15", "0.20", "0.25", "0.30"])
    ax.margins(x=0)
    ax.set_xlim(0.10, 0.30)
    # Use the requested plotting-window width:height ratio of 3:2.
    ax.set_box_aspect(2 / 3)
    ax.set_yticks([25, 50, 75, 100, 125, 150, 175])
    ax.text(
        0.08, 0.10, LABELS["BTABAT"],
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=48, fontstyle="italic", color="#3568F4",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=plt.rcParams["axes.labelcolor"], alpha=0.10,
                         edgecolor="none"))
    labels = ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."]
    ax.legend(
        handles,
        labels,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.96),
        fontsize=28,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.7,
        labelspacing=0.45,
        borderaxespad=0,
        alignment="center",
    )
    save(fig, "btabat_spinodal_local_publication")


def plot_btabat_global(data: dict) -> None:
    """BTABAT full-range panel matched proportionally to its local panel."""
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    draw_series(ax, data["BTABAT"], "BTABAT", labeled=False,
                color_override=plt.rcParams["axes.labelcolor"], band_alpha=0.10)

    finish_axis(
        ax,
        ylim=(0, 200),
        axis_label_size=32,
        tick_label_size=32,
    )
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_box_aspect(2 / 3)

    # Architecture label follows the local-panel placement.
    ax.text(
        0.08, 0.08, LABELS["BTABAT"], transform=ax.transAxes,
        ha="left", va="bottom", fontsize=48,
        fontstyle="italic", color="#3568F4",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=plt.rcParams["axes.labelcolor"], alpha=0.10,
                         edgecolor="none"))
    labels = ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."]
    ax.legend(
        handles,
        labels,
        ncol=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        fontsize=28,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.7,
        labelspacing=0.45,
        borderaxespad=0,
        alignment="center",
    )
    save(fig, "btabat_spinodal_publication")


def plot_m1_m2_global(data: dict) -> None:
    """Save M1 and M2 as separate full-range BTABAT-style panels."""
    for topology in ["M1", "M2"]:
        fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
        draw_series(ax, data[topology], topology, labeled=False,
                    color_override=plt.rcParams["axes.labelcolor"],
                    band_alpha=0.10)
        finish_axis(
            ax, ylim=(0, 200),
            axis_label_size=32, tick_label_size=32,
        )
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_box_aspect(2 / 3)
        ax.text(
            0.08, 0.08, LABELS[topology], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=48,
            fontstyle="italic", color="#3568F4",
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=plt.rcParams["axes.labelcolor"], alpha=0.10,
                             edgecolor="none"))
        labels = ["RPA calculation", "GNN prediction", r"GNN mean $\pm$ 1 s.d."]
        ax.legend(
            handles,
            labels,
            ncol=1,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            fontsize=28,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.7,
            labelspacing=0.45,
            borderaxespad=0,
            alignment="center",
        )
        save(fig, f"{topology.lower()}_global_publication")


def make_figures(data: dict) -> None:
    configure_paper_style()
    plot_architecture_series(data["global"])
    plot_grid(data["global"], ["AB", "ABA", "ABAB", "ABABA"], "rpa_ml_comparison_grid_publication", (2, 2))
    plot_ab_series(data["global"])
    plot_grid(data["global"], ["AB", "AB2", "AB3"], "rpa_ab_series_grid_publication", (1, 3), figsize=(20, 6.8))
    plot_local_m1_m2(data["global"], data["m1_m2_local"])
    plot_m1_m2_global(data["global"])
    plot_btabat_global(data["btabat"])
    plot_btabat_local(data["btabat"])
    plot_shi_series(data["shi"])
    plot_grid(data["shi"], ["AB", "B2A2", "(B2A2)2"], "shi_b2a2_series_grid_publication", (1, 3), figsize=(20, 6.8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS,
                        help="Update only this seed in the existing JSON/CSV")
    args = parser.parse_args()
    missing = [str(path) for path in CHECKPOINTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    if args.seed is None:
        data = infer_all(assemble_cases())
    else:
        payload = load_json(OUTPUT_JSON)
        data = update_one_seed(payload["cases"], args.seed)
    save_predictions(data)
    make_figures(data)


if __name__ == "__main__":
    main()
