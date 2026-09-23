# Mechanistic interpretability figures

Run `notebooks/mechanistic_interpretability.ipynb` with the project `.venv` kernel to regenerate the main mechanism figure and its grid-wise and legal-growth-history companion figures. The notebook reads the local tables under `data/mechanistic_interpretability` and writes all figure outputs under `results/figures/mechanistic_interpretability`.

The executable plotting entry points are kept in `scripts/mechanistic_interpretability`. They resolve paths relative to `sourcecode/code` and do not require a source directory outside the repository.
