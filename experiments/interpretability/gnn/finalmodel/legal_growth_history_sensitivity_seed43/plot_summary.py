from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")

SKILL_DIR = Path(r"C:\Users\kachrel\.codex\skills\mpltex-skill")
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
with (root / "per_checkpoint_summary.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 5 and all(float(row["p95_abs_relative_change_percent"]) < 1 for row in rows)

x = range(1, len(rows) + 1)
series = [
    ("Median", "median_abs_relative_change_percent", COLORS["blue"], "o"),
    ("Mean", "mean_abs_relative_change_percent", COLORS["orange"], "s"),
    ("95th percentile", "p95_abs_relative_change_percent", COLORS["green"], "^"),
]

fig, ax = plt.subplots(figsize=(10.5, 8.5))
for label, key, color, marker in series:
    ax.scatter(x, [float(row[key]) for row in rows], marker=marker, s=120,
               color=color, edgecolors="#FFFFFF", linewidths=0.7, label=label)

ax.axhline(1.0, color="0.35", linestyle="--", linewidth=2.4, label="1% reference")
ax.set(
    xlim=(0.7, 5.3),
    ylim=(0.015, 1.6),
    yscale="log",
    xlabel="Model checkpoint",
    ylabel="Absolute relative prediction change (%)",
)
ax.set_xticks(list(x), [row["seed"] for row in rows])
ax.legend(loc="upper center", ncol=2)
style_axis(ax)
fig.subplots_adjust(left=0.18, right=0.96, bottom=0.17, top=0.96)
save_dual(fig, root / "legal_growth_history_sensitivity")
plt.close(fig)
