# TAPD source code

This directory is the self-contained code and figure-reproduction package for
the TAPD manuscript. It contains the shared Python source, fixed data splits,
model-selection and evaluation scripts, frozen checkpoints, local numerical
results, and notebooks used to regenerate the manuscript and supplementary
figures.

Run commands from this directory unless stated otherwise. Plotting the released
results is CPU-only and uses bundled source data. Training and inference usually
require the full datasets and, for practical runtimes, a CUDA-capable GPU.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Shared data loaders, constants, model workers, training helpers, and plotting utilities. |
| `data/` | Example records, figure inputs, fixed splits, selection tools, and Julia data-generation code. |
| `experiments/` | Current model-selection, final-evaluation, transfer, and extrapolation workflows. |
| `mainmodel/` | Main GNN training and supplementary-confirmation entry points. |
| `checkpoints/` | Frozen models used by the released inference workflows. |
| `notebooks/` | Executable entries for manuscript and supplementary figures. |
| `scripts/` | Shared validation, inference, and mechanistic-analysis scripts. |
| `results/source_data/` | Numerical inputs read by plotting notebooks. |
| `results/figures/` | Generated PNG, SVG, and PDF figures. |
| `results/table_sources/` | Machine-readable inputs for manuscript and supplementary tables. |

## Installation

The release environment uses Python 3.12.10. The simplest setup is Conda:

```bash
conda env create -f environment.yml
conda activate tapd
python -m ipykernel install --user --name tapd --display-name TAPD
```

Alternatively, create a Python 3.12 virtual environment and install the pinned
packages directly:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the complete Python dependency set. GPU workflows
may need a PyTorch build matched to the local CUDA driver; the released
environment records PyTorch 2.12.0. Figure notebooks can run on CPU.

## Quick verification

Check the bundled data pipeline and release package before larger workflows:

```bash
python data/check_data_pipeline.py
python src/check_label_conversion.py
python scripts/verify_bundle.py
```

The data-pipeline check uses bundled examples and temporary output. It tests
MAT/YAML pairing, split selection, exclusion rules, and target conversion.

## Data and target conventions

### Bundled example data

`data/example/n_train_1k/` contains 1,000 paired MAT/YML examples for loader
checks and small demonstrations. It is not the full development, validation,
test, or out-of-distribution dataset. Pass `data/example/n_train_1k` as the
dataset path when running from the repository root.

### Fixed splits

The canonical split package is under `data/splits/`:

- `development_fixed_80k_20k_seed42.npz` stores the fixed 80,000-record fitting
  pool and complementary 20,000-record internal-validation pool.
- `block_stratified_nested/` stores the exact nested subsets used by the GNN,
  MLP, and Transformer training-size studies.
- `release_manifests/` stores machine-readable released memberships for the
  seed-42 validation set, independent seed-43 test set, and 6–10-block OOD set.

Indices follow the record order in
`data/figure_acbd/responses/training_100k_response_data.csv`. Nested subsets are
formed within the 2-, 3-, 4-, and 5-block strata using one seeded permutation
per stratum. Smaller subsets are contained in larger subsets, and the 80k
endpoint is the full fitting pool.

To construct new leakage-free pools from raw data, use:

```bash
python data/splits/unified_sampling.py \
  --source-dir data/datasets/raw_2to5 \
  --output-dir data/datasets/pools \
  --seed 42

python data/splits/build_leakage_free_test_2k.py \
  --train-dir data/datasets/pools/n_train_100k \
  --pool-dir data/datasets/pools/n_test_pool \
  --output-dir data/datasets/validation_seed42 \
  --seed 42

python data/splits/build_leakage_free_test_2k.py \
  --train-dir data/datasets/pools/n_train_100k \
  --pool-dir data/datasets/pools/n_test_pool \
  --exclude-dir data/datasets/validation_seed42 \
  --output-dir data/datasets/test_seed43 \
  --seed 43
```

These commands create new selections; they do not redefine the released
experimental memberships.

### Wavelength and wavevector labels

Raw generated records use the historical field name `k_star` for the
dimensionless wavelength `L*/Rg`. The shared loader preserves that value for
`task_type="k"`. For `task_type="k_wavevector"`, it converts the label before
the logarithm:

```text
k*Rg = 2*pi/(L*/Rg)
target = log10(k*Rg)
```

Do not apply this reciprocal conversion again to a cache that already contains
wavevector targets. `src/data_utils.py` implements common loading and conversion.
`src/constants.py` provides topology mappings for random 2- through 6-block
architectures; unknown topology labels receive `-1`.

## Raw architecture and RPA generation

The generation utilities use Julia 1.12.7. Their environment is defined by
`data/generation/Project.toml` and `data/generation/Manifest.toml`:

```bash
julia --project=data/generation -e "using Pkg; Pkg.instantiate()"
julia --project=data/generation data/generation/rpa/smoke_test.jl

julia --threads=1 data/generation/number_5.jl \
  --count-per-block 2 \
  --seed 42 \
  --output-dir data/datasets/smoke_2to5

julia data/generation/generate_6block_transfer_test.jl \
  --block-counts 6,7,8,9,10 \
  --count-per-block 1 \
  --seed 20260707 \
  --output-dir data/datasets/smoke_ood
```

Use a new output directory: the generators refuse to overwrite an existing
one. For deterministic small checks, use one Julia thread because the raw
generator combines the supplied seed with the thread ID. Full generation uses
30,000 samples per block for the 2–5-block pool and 500 samples per block for
the 6–10-block OOD set.

The local RPA implementation is
`data/generation/rpa/ExtractedPolyorderRPA.jl`. It requires Polymer,
PolymerArchitecture, Graphs, LinearAlgebra, Optim, and Roots.

## Main experimental workflows

### Model selection and training-size studies

Model-selection code is grouped by model family:

- GNN: `experiments/model_selection/gnn/`
- descriptor MLP: `experiments/model_selection/mlp/`
- canonical Transformer: `experiments/model_selection/transformer/`
- shared nested study: `experiments/model_selection/block_stratified_nested/`

The shared nested-study entry points are:

```bash
python experiments/model_selection/block_stratified_nested/run_gnn_stratified_nested.py --help
python experiments/model_selection/block_stratified_nested/run_mlp_stratified_nested.py --help
python experiments/model_selection/block_stratified_nested/run_transformer_stratified_nested.py --help
```

The corresponding training-size curves are rendered by the GNN, MLP, and
Transformer ablation notebooks listed below.

The supplementary experiment matrices have dedicated numerical entry points:

```bash
# GNN encoder-depth x decoder-depth grid used by Fig. S12f
python experiments/model_selection/gnn/gnn/run_depth_joint_grid_current_ofat.py --help

# Selected 11-descriptor group/feature ablations and raw-35 controls in Fig. S19
python experiments/model_selection/mlp/mlp/ablation_validation_only/run_ablation_search.py --help
python experiments/model_selection/mlp/mlp/ablation_validation_only/run_raw_descriptor_ablation.py --help

# Transformer OFAT and local interaction studies
python experiments/model_selection/transformer/transfomer/run_experiment.py --help
python experiments/model_selection/transformer/transfomer/run_iterative_local_grid_ofat100k.py --help
```

Each model family uses the fixed 80k fitting pool, the same 20k internal
validation membership, and five training seeds (`202601`–`202605`). Validation
loss controls checkpointing and early stopping; original-scale validation MAPE
is used for model selection. The independent seed-43 test set is reserved until
model and hyperparameter choices are frozen.

### Final in-domain evaluation

The model-family inference entries for the independent seed-43 test set are:

```text
experiments/final_evaluation/gnn/infer_c2_nested_80k_seed43.py
experiments/final_evaluation/mlp/evaluate_seed43.py
experiments/final_evaluation/transformer/evaluate_80k_seed43_independent.py
```

Generate the comparison and parity plots with
`notebooks/seed43_final_evaluation.ipynb`.

### Block-count extrapolation

The GNN, MLP, and Transformer are evaluated on the same 2,500 architectures:
500 each with 6, 7, 8, 9, and 10 blocks. Inference scripts are under
`experiments/block_extrapolation/`, frozen models are under `checkpoints/`, and
predictions are under `results/source_data/block_extrapolation/`.

Run `notebooks/block_extrapolation.ipynb` to regenerate the main extrapolation
figure and supplementary per-seed parity figure. Plotting uses stored local
predictions and does not require a GPU.

### Transfer learning

The transfer workflow is under
`experiments/kstar_transfer/gnn/finalmodel/transfer/`. Its training entry point,
`scripts/train_c2_kstar_transfer.py`, includes both pretrained transfer and
randomly initialized scratch controls. The task is `k_wavevector`, so the target
convention above applies.

The frozen-embedding MLP is generated by
`scripts/train_c2_512_embedding_mlp_probe.py`. After those five full-data runs
have produced their embedding caches, limited-label MLP runs use
`scripts/train_kstar_label_curve_embedding_mlp.py`. The complete five-method,
five-seed label-budget grid is orchestrated by
`scripts/run_kstar_label_curve_5seeds.py`; its `--dry-run` option prints the
planned commands without starting GPU jobs.

Run `notebooks/kstar_transfer.ipynb` for the limited-label, full-data, parity,
and signed-error figures. Main evidence tables are stored under
`results/source_data/kstar_transfer/main_evidence_20260913/`.

### Mechanistic interpretation

Executable analyses are under `scripts/mechanistic_interpretability/`. Run
`notebooks/mechanistic_interpretability.ipynb` to regenerate the main mechanism
figure and robustness and legal-growth-history companion figures. Local inputs
are under `data/mechanistic_interpretability/`; outputs are written to
`results/figures/mechanistic_interpretability/`.

`run_abtab2_rpa_gnn.py` is the retained numerical entry point for the frozen-GNN
ABTAB2 analysis. It computes the RPA edge decomposition, constrained message
deletion analysis, grouped-by-arm-ratio PCA--ridge layer probes, and matched
block-placement counterfactuals consumed by the notebook.

### Composition curves

Run `notebooks/composition_curves.ipynb` for the composition-curve and
phase-diagram manuscript figures. The notebook is the sole plotting entry and
uses bundled numerical inputs in `results/source_data/composition_curves/`.

The numerical regeneration chain is under
`experiments/composition_curves/numerical/`. It includes the published RPA
composition grids, a Julia RPA recomputation entry point, the C2 graph/model
definitions, and `generate_five_seed_phase_diagram_predictions.py`, which
loads the five frozen GNN checkpoints and writes the CSV/JSON consumed by the
notebook. This script intentionally produces numerical source data only; it
does not contain plotting code.

## Figure reproduction

Start Jupyter from the repository root so relative paths resolve consistently:

```bash
conda activate tapd
jupyter notebook notebooks/
```

| Notebook | Main output | Primary local inputs |
| --- | --- | --- |
| `schematics.ipynb` | Figs. 1 and 9 | `data/schematics/` |
| `figure_acbd.ipynb` | Fig. 2 | `data/figure_acbd/` |
| `seed43_final_evaluation.ipynb` | Fig. 3 | `results/source_data/final_evaluation/` |
| `block_extrapolation.ipynb` | Fig. 4 and OOD parity panels | `results/source_data/block_extrapolation/` |
| `mechanistic_interpretability.ipynb` | Fig. 5 and robustness panels | `data/mechanistic_interpretability/` |
| `kstar_transfer.ipynb` | Fig. 6 and transfer supplementary panels | `results/source_data/kstar_transfer/` |
| `composition_curves.ipynb` | Figs. 7 and 8 | `results/source_data/composition_curves/` |
| `gnn_ablations.ipynb` | GNN ablation and training-size panels | `results/source_data/gnn/`, `experiments/model_selection/gnn/` |
| `mlp_ablations.ipynb` | MLP descriptor, validation, and training-size panels | `results/source_data/mlp/`, `experiments/model_selection/mlp/` |
| `transformer_ablations.ipynb` | Transformer selection and ablation panels | `results/source_data/transformer/`, `experiments/model_selection/transformer/` |

The notebooks read bundled inputs and write figures under `results/figures/`.
They create missing output directories without modifying numerical inputs.

## Practical notes

- Keep the working directory at the repository root when running scripts.
- Use the fixed files in `data/splits/` when exact experimental membership is
  required; newly generated subsets are not substitutes for released splits.
- Plotting notebooks use stored predictions and can run without retraining.
- Large training commands expose available paths and options through `--help`;
  check those options before launching multi-GPU jobs.
- Generated figures belong in `results/figures/`; newly generated datasets
  belong in `data/datasets/`.
