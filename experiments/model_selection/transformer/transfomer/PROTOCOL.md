# Transformer validation-only tuning protocol (2026-08-18)

## Frozen data boundary

- Development corpus: `/home/zxy/project/TAPD/TrainData/data/chiNlimited/n_train_100k`
- Fixed split: 80% fitting / 20% validation, `random_state=42`
- Final held-out test: `/home/zxy/project/TAPD/TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42`
- Training seeds: `202601, 202602, 202603, 202604, 202605`

The final-test directory is not accepted by the tuning trainer and must not be
loaded during OFAT, interaction, confirmation, or frozen-model training.

## Selection rules

1. Validation log-MSE controls ReduceLROnPlateau, early stopping, and the saved checkpoint.
2. Original-scale validation MAPE of the saved checkpoint selects hyperparameters.
3. Each candidate is ranked by five-seed mean validation MAPE, then sample SD.
4. Within one standard error of the raw family leader, prefer fewer trainable parameters.
5. All candidates use the same fitting and validation sample membership.
6. The final-test set is evaluated only after the hyperparameters, model list, and checkpoints are frozen.

## Fixed training settings

- Optimizer: Adam
- Maximum epochs: 120
- Early-stopping patience: 25
- ReduceLROnPlateau: factor 0.5, patience 10
- Maximum sequence length: 128
- Canonicalization: strict
- Representation during tuning: full topology
- Order: canonical
- Numeric representation: side channel
- Architecture during tuning: Transformer

## Search design

The 100k anchor is varied one factor at a time over nine hyperparameter
families, producing 28 unique configurations. The top two values from the
learning-rate/batch-size, model-size/layer-count, and dropout/weight-decay
families form at most twelve targeted 2x2 interaction configurations.

The data-pool-size family from the earlier experiment is excluded from model
selection. Any later data-scaling study is descriptive and uses already-frozen
hyperparameters.
