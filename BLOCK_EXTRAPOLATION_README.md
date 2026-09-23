# Block-count extrapolation

The GNN, descriptor MLP and canonical transformer are evaluated on the same
2,500 six-to-ten-block architectures, with 500 samples per block count. Five
frozen models per family are used, and uncertainty is the sample standard
deviation across seeds.

Inference entry points are under `experiments/block_extrapolation/`, frozen
models are under `checkpoints/`, and local prediction records are under
`results/source_data/block_extrapolation/`. Open
`notebooks/block_extrapolation.ipynb` and run all cells to regenerate Fig. 4
and the corresponding supplementary parity figure in
`results/figures/block_extrapolation/`.

All metrics are evaluated on the original target scale. Figure generation uses
the stored local predictions and does not require a GPU or data outside
`sourcecode/code`.
