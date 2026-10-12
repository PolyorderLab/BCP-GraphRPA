# Figure generation

Run the notebooks from the repository root. The data figures read the bundled
numeric CSV, JSON, NPZ, and YAML files; their plotting cells draw Matplotlib
artists directly. They do not open a previously rendered panel to assemble a
figure. The names below identify the corresponding manuscript figures.

| Paper figure | Output stem | Plotting notebook | Bundled input |
| --- | --- | --- | --- |
| Main 1 | `figure1_architecture_representation` | `main_figures.ipynb` (PPTX source) | `results/GINE_concise_editable.pptx` |
| Main 2 | `figure_acbd_four_panel` | `main_figures.ipynb` | `data/figure_acbd/architecture/`, `data/figure_acbd/responses/` |
| Main 3 | `in_domain_model_comparison` | `main_figures.ipynb` | `results/source_data/final_evaluation/` |
| Main 4 | `Fig4_block_extrapolation` | `main_figures.ipynb` | `results/source_data/block_extrapolation/` |
| Main 5 | `fig5_mechanistic_interpretability` | `main_figures.ipynb` | `data/mechanistic_interpretability/rpa/`, `data/mechanistic_interpretability/legal_histories/` |
| Main 6 | `kstar_transfer_main_evidence` | `main_figures.ipynb` | `results/source_data/kstar_transfer/` |
| Main 7 | `figure6_stability_limit_dashboard` | `main_figures.ipynb` | `results/source_data/composition_curves/five_seed_phase_diagram_predictions.json` |
| Main 8 | `figure8_literature_architectures_stability_limits` | `main_figures.ipynb` | Same numeric JSON; architecture paths are drawn in the notebook |
| Main 9 | `figure9_gine_encoder` | `main_figures.ipynb` (PPTX source) | `results/GINE_concise_editable.pptx` |
| SI 1 | `in_domain_all_seed_parity` | `si_figures.ipynb` | `results/source_data/final_evaluation/` |
| SI 2 | `in_domain_signed_relative_error` | `si_figures.ipynb` | Same |
| SI 3 | `gnn_training_histories` | `si_figures.ipynb` | Same |
| SI 4 | `mlp_training_histories` | `si_figures.ipynb` | `results/source_data/mlp/` |
| SI 5 | `final_transformer_training_histories` | `si_figures.ipynb` | `results/source_data/transformer/` |
| SI 6 | `ood_all_seed_parity` | `si_figures.ipynb` | `results/source_data/block_extrapolation/` |
| SI 7 | `abtab2_grid_robustness` | `si_figures.ipynb` | `data/mechanistic_interpretability/rpa/` |
| SI 8 | `kstar_transfer_n100_all_seed_parity` | `si_figures.ipynb` | `results/source_data/kstar_transfer/limited_n100/` |
| SI 9 | `kstar_transfer_n100_signed_relative_error` | `si_figures.ipynb` | Same |
| SI 10 | `kstar_transfer_all_seed_parity` | `si_figures.ipynb` | `results/source_data/kstar_transfer/` |
| SI 11 | `kstar_transfer_signed_relative_error` | `si_figures.ipynb` | Same |
| SI 12 | `gnn_hyperparameter_screening` | `si_figures.ipynb` | `results/source_data/gnn/` |
| SI 13 | `gnn_representation_ablation` | `si_figures.ipynb` | Same |
| SI 14 | `gnn_training_size` | `si_figures.ipynb` | Same |
| SI 15 | `descriptor_screening_a` | `si_figures.ipynb` | `results/source_data/mlp/` |
| SI 16 | `descriptor_screening_b` | `si_figures.ipynb` | Same |
| SI 17 | `descriptor_screening_c` | `si_figures.ipynb` | Same |
| SI 18 | `descriptor_selection_signals` | `si_figures.ipynb` | Same |
| SI 19 | `mlp_hyperparameter_screening` | `si_figures.ipynb` | Same |
| SI 20 | `mlp_training_size` | `si_figures.ipynb` | Same |
| SI 21 | `descriptor_ablation` | `si_figures.ipynb` | Same |
| SI 22 | `transformer_hyperparameter_screening` | `si_figures.ipynb` | `results/source_data/transformer/` |
| SI 23 | `transformer_training_size` | `si_figures.ipynb` | Same |

Figs. 1 and 9 are editable schematics in `results/GINE_concise_editable.pptx`;
the main notebook displays their exported images. Other figures are plotted
from local numeric data. Both notebooks display an inline preview for every
figure. Generated plots represent the reported data, but their typography and
layout may differ from the final manuscript artwork.
