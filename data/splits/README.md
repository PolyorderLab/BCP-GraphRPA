# Released data splits

This directory is the canonical split package for the experiments reported in
the manuscript and Supplementary Information. Indices refer to the row order of
`data/figure_acbd/responses/training_100k_response_data.csv`, whose `sample_id`
column follows the development-set record order.

## Fixed development split

`development_fixed_80k_20k_seed42.npz` contains:

- `train_index`: the fixed 80,000-record training pool.
- `validation_index`: the complementary 20,000-record internal-validation pool.
- `split_random_state`: 42.

The original dataset archive stores the same partition as
`release_manifests/development_fixed_split_original.pt`, with the 20,000-record
holdout divided into `val` and `test` arrays of 10,000 records each. Its
80,000-record `train` array is identical to `train_index`, and the union of its
`val` and `test` arrays is identical to `validation_index`.

## Nested training-size subsets

`block_stratified_nested/` contains the exact subsets used for Supplementary
Figs. S14, S18 and S21. The files are shared by the GNN, MLP and Transformer
studies. Each file contains `train_index`, `validation_index` and
`subset_random_state`.

The frozen 80,000-record training pool is separated into 2-, 3-, 4- and
5-block strata. One NumPy `default_rng(42)` permutation is generated within
each stratum. Every requested size uses monotone largest-remainder quotas and
takes a prefix from each permuted stratum. Consequently, every smaller subset
is contained in every larger subset, and the 80,000 endpoint recovers the full
training pool.

`manifest.json` records counts and split construction settings. The construction is implemented in
`experiments/model_selection/block_stratified_nested/stratified_nested_common.py`.

## Released validation and test sets

`release_manifests/` contains the selection manifests and provenance metadata
stored in the released dataset archives for the seed-42 validation set, the
independent seed-43 test set and the 6-10-block OOD set.

The scripts in this directory construct new leakage-free selections when given
raw source pools. They do not replace the released manifests when exact sample
membership is required.
