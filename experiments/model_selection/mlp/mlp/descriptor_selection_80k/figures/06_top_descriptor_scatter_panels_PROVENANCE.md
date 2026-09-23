# Corrected top-descriptor scatter panels

- Data matrix: `baselinemodels/mlp_pub/analysis_80k_trainfold/train_descriptor_matrix.csv`.
- Population: the fixed 80,000-sample fitting fold selected with split seed 42 from `n_train_100k`.
- Display subsample: every fourth matrix row, giving approximately 20,000 plotted samples.
- Descriptor ranking: the first six rows of the fitting-fold-only `descriptor_log10_chiN_star_correlation_train.csv`.
- Category: true polymer block count parsed from the `sample_id` prefix (`2blocks` through `5blocks`).
- Excluded: the 20,000-sample validation fold and the leakage-free 2,000-sample selection/reporting set.

The superseded figure grouped on the graph descriptor `Nnodes` and incorrectly labelled that value as the block count. Its apparent `6 blocks` category consisted entirely of true five-block samples. The corrected figure contains no six-block category or six-block data.
