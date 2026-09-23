# Frozen-protocol figure status

The figures below were regenerated only from the frozen 80k/20k split and its
post-freeze 2k test evaluation.  Hyperparameter and checkpoint selection use
only validation MAPE; the 2k test figures are reporting-only.

## Regenerated

- `validation_20k/transformer_ofat_optimization`
- `validation_20k/transformer_ofat_architecture`
- `validation_20k/transformer_interaction_checks`
- `final_test_2k/transformer_ensemble_parity`
- `final_test_2k/transformer_best_single_seed_202604_parity`

## Not regenerated from old values

The frozen protocol did not include the independent experiments needed for the
following old figures, so they must not be redrawn from historical results:

- `transformer_data_scaling`: requires fixed-protocol runs at each training-set size.
- `transformer_same_checkpoint_order_robustness`: requires canonical, reversed,
  and randomized checkpoint-order runs using the same frozen checkpoints.
