# Architecture and response distributions

Run `notebooks/figure_acbd.ipynb` with the project `.venv` kernel to regenerate the four-panel architecture and response-distribution figure. The notebook reads only the copied inputs under `data/figure_acbd` and writes the PNG, SVG, and PDF outputs under `results/figures/figure_acbd`.

The input directories contain the representative architecture YAML files and the local training, seed43 validation, and six-to-ten-block OOD response tables. The plotting scripts in `scripts/figure_acbd` are called by the notebook and can also be run directly from the repository root.
