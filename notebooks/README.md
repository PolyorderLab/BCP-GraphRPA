# Figure notebooks

The notebooks in this directory are the executable plotting entries for the
manuscript and supplementary figures. They resolve the repository root from
the notebook location and read only data bundled under `sourcecode/code`.
Figures are written to `sourcecode/code/results/figures/` as PNG, SVG and PDF
files where applicable.

## Manuscript figures

| Notebook | Figures | Local inputs |
| --- | --- | --- |
| `schematics.ipynb` | Fig. 1 and Fig. 9 | `data/schematics/` |
| `figure_acbd.ipynb` | Fig. 2 | `data/figure_acbd/` |
| `seed43_final_evaluation.ipynb` | Fig. 3 | `results/source_data/final_evaluation/` |
| `block_extrapolation.ipynb` | Fig. 4 | `results/source_data/block_extrapolation/` |
| `mechanistic_interpretability.ipynb` | Fig. 5 | `data/mechanistic_interpretability/` |
| `kstar_transfer.ipynb` | Fig. 6 | `results/source_data/kstar_transfer/` |
| `composition_curves.ipynb` | Fig. 7 and Fig. 8 | `results/source_data/composition_curves/` |

## Supplementary figures

| Notebook | Figure families | Local inputs |
| --- | --- | --- |
| `gnn_ablations.ipynb` | GNN architecture, input, depth and training-size panels | `results/source_data/gnn/` and `experiments/model_selection/gnn/` |
| `mlp_ablations.ipynb` | Descriptor screening, validation and training-size panels | `results/source_data/mlp/` and `experiments/model_selection/mlp/` |
| `transformer_ablations.ipynb` | Transformer OFAT, interaction, training-size and representation panels | `results/source_data/transformer/` and `experiments/model_selection/transformer/` |
| `mechanistic_interpretability.ipynb` | Mechanistic robustness and growth-history panels | `data/mechanistic_interpretability/` |
| `kstar_transfer.ipynb` | Full-data transfer and signed-error panels | `results/source_data/kstar_transfer/` |
| `block_extrapolation.ipynb` | Per-seed OOD parity panels | `results/source_data/block_extrapolation/` |

## Running

Run a notebook with the project kernel from this directory. The plotting
cells create their output directories when needed and leave the input data
unchanged. The notebooks use the shared paper plotting style and export
publication-resolution figures.
