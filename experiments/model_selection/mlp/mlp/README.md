# MLP publication package — 2026-08-18

## Descriptor selection

`descriptor_selection_80k/` is the corrected, validation-isolated descriptor-screening package.
Every correlation, VIF, PCA, mutual-information, Lasso, and ExtraTrees decision uses only
the fixed 80,000-member fitting fold from `n_train_100k`; the 20,000-member validation fold
and 2,000-member final test set are excluded.  It contains the data tables, code, and Figures 01–06.

The descriptor set is therefore frozen from `descriptor_selection_80k/`.  By contrast,
`hyperparameter_selection_80k/data/n_train_100k_descriptor_matrix.csv` is the deterministic
raw descriptor matrix for all 100k development samples.  It is used only after the descriptor
set has been frozen, so the 20k validation rows are never used to select descriptors.

## Hyperparameter search

`hyperparameter_selection_80k/` runs 27 one-factor candidate configurations × five training seeds
(135 runs) on one saved `random_state=42` split: 80,000 fitting samples and 20,000 validation samples.
The worker intentionally has no test-set argument.  Validation MSE selects the checkpoint, and
five-seed mean original-scale validation MAPE selects each hyperparameter value.  The declared 2,000
sample final test set must remain unread until all selected values are frozen.

## Strictly nested training-size ablation

`training_size_nested_80k_common_val_seed42/` defines one `random_state=42` permutation of the
fixed 80,000-sample training fold and uses its 8k, 16k, ..., 72k prefixes as strictly nested
training subsets.  The 80k point reuses the five frozen joint-selected checkpoints.  Every
size shares the same fixed 20,000-sample validation fold for checkpoint selection and early
stopping, and all frozen checkpoints are evaluated on the same seed42 2,000-sample development
evaluation set.  The older per-directory 10k--100k experiment is superseded for S4(f) because
its independently formed train folds were not strictly nested.  The independent seed43 2k set
remains reserved for final reporting of the full 80k model.

## Independent final test

The replacement sealed final test is
`TrainData/data/chiNlimited/n_test_2k_leakage_free_seed43_independent`.  It has no sample-ID
overlap with either the 100k development pool or the seed42 ablation-evaluation set.  The five
frozen joint-selected checkpoints are evaluated by
`joint_selected_configuration/evaluate_joint_selected_seed43_independent.py`; outputs are kept
under `joint_selected_configuration/final_test_seed43_independent/`.  The superseded seed42
joint-selected inference outputs were removed after seed43 became the independent final test;
the separate seed42 training-pool-ablation evaluation remains unchanged.

Run with the project environment, for example:

```bash
./.venv/bin/python baselinemodels/pub0818/mlp/hyperparameter_selection_80k/run_hyperparameter_search.py --gpus 0,1,2,3 --slots-per-gpu 3
```
