# Descriptor screening on the fixed 80k training fold

## Protocol

- Source pool: `n_train_100k` (100,000 samples).
- Training-only subset: the 80,000 indices in
  `../model_11_vif_descriptors/split_indices.npz::train_index`.
- Excluded from every screening decision: the corresponding 20,000 validation
  indices.
- External check only: `n_test_2k_leakage_free_seed42` (2,000 samples).
- Split and analysis random state: 42.
- Thresholds: variance `1e-12`, absolute Spearman correlation `0.95`, VIF `10`.

Correlation ranking, variance filtering, iterative VIF, PCA, mutual
information, LassoCV, and ExtraTrees were all fitted using only the fixed 80k
training fold. The external test set was not used to select descriptors.

## Result

The result is stable relative to the earlier 100k-development-pool analysis:

| Stage | 100k analysis | 80k-only analysis |
|---|---:|---:|
| Candidate descriptors | 35 | 35 |
| After variance filtering | 35 | 35 |
| After correlation pruning | 24 | 24 |
| After VIF pruning | 11 | 11 |
| PCA components for at least 95% variance | 7 | 7 |
| Mutual-information selected count | 11 | 11 |

The correlation-pruned feature set, VIF removal set and removal order, final
VIF feature set, final descriptor order, and mutual-information ranking are
identical between the two analyses.

The final 11 descriptors and 80k-only VIF values are:

| Descriptor | VIF |
|---|---:|
| `SLE` | 7.268 |
| `r_deg` | 6.594 |
| `CV_arm` | 6.181 |
| `n_leaves` | 5.907 |
| `M_AB` | 3.716 |
| `IR` | 3.464 |
| `D_f` | 3.267 |
| `ISC` | 3.208 |
| `CAF` | 2.644 |
| `tau_mean` | 1.946 |
| `f_A` | 1.148 |

The 80k-only LassoCV retains all eleven descriptors, including `n_leaves`.
Consequently, the intersection of mutual-information and Lasso selections is
now exactly the same eleven-feature set already used by the final MLP.

## Consequence for the five-seed MLP

No five-seed retraining is required. The existing final MLP already:

1. uses exactly these eleven descriptors in the same order;
2. fits its scaler only on the same fixed 80k training indices;
3. trains model parameters only on those 80k samples; and
4. uses the fixed 20k validation fold only for validation and model selection.

The new 80k-only screening therefore removes the validation-isolation concern
without changing the model input or any fitted-model data. Retraining would
repeat the same experiment rather than correct a changed pipeline.

