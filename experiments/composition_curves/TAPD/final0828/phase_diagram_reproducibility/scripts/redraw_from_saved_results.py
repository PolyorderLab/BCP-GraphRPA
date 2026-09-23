"""Redraw every figure from frozen predictions without loading checkpoints."""

import json
from pathlib import Path

import reproduce_phase_diagrams as pipeline

BUNDLE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS = BUNDLE_DIR / "results" / "five_seed_phase_diagram_predictions.json"


def main() -> None:
    with PREDICTIONS.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    pipeline.make_figures(payload["cases"])


if __name__ == "__main__":
    main()
