# Corrected block-stratified nested training-size study

These scripts are corrected copies/wrappers of the publication GNN, MLP, and
Transformer training-size studies.  They preserve each model, seed set,
hyperparameters, and validation definition, while changing only the nested
training-subset construction.

For every model, the frozen 80k training pool is divided by the block count
encoded in `sample_id` (`2blocks` through `5blocks`).  Each stratum receives a
single deterministic `random_state=42` permutation.  At every requested size,
the subset takes a prefix from every stratum using largest-remainder quotas
proportional to the frozen 80k pool.  Quotas and memberships are monotone, so
every smaller subset is a strict subset of every larger one.  The 80k endpoint
recovers the complete frozen training pool.

The generated `stratification_audit.json` files record per-size stratum counts,
fractions, and index hashes. This publication snapshot contains the executable
scripts, audits, per-seed tabulations, and summary tables needed to reproduce
Supplementary Figs. S14, S18, and S21. The GNN C2 scripts, protocol, manifest,
audit, ten-point summary, and 50-row per-seed tabulation are stored locally
under `results/gnn`.

`plot_stratified_curves.py` writes the three title-free manuscript figures as
PNG, SVG, and PDF under `sourcecode/code/results/figures/{gnn,mlp,transformer}`.
