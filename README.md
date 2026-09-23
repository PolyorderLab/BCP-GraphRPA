# TAPD source code

This directory contains the scripts, experiment records, local figure data and
notebooks used for the manuscript and Supplementary Information. All plotting
notebooks read inputs from this directory and write figures to
`results/figures/`; no figure notebook requires data outside `sourcecode/code`.

## Layout

- `notebooks/`: executable entries for every figure cited in `draft.md` and
  `SI.md`; see `notebooks/README.md` for the figure map.
- `experiments/`: model selection, training-size, transfer and extrapolation
  scripts together with their run records.
- `results/source_data/`: local numerical inputs consumed by the notebooks.
- `results/figures/`: manuscript and supplementary figure exports.
- `checkpoints/`: frozen trained models used by the inference scripts.
- `data/`: data-generation, exact released splits and figure-input resources.
- `src/`: shared model, loader and training utilities.

## Environment

The release environment uses Python 3.12.10. Create it from this directory:

```bash
conda env create -f environment.yml
conda activate tapd
python -m ipykernel install --user --name tapd --display-name TAPD
```

`requirements.txt` pins the complete Python dependency set. The reported GPU
runs used the CUDA 13.0 build of PyTorch 2.12.0; install the matching PyTorch
wheel for the local CUDA driver when GPU training is required. Figure notebooks
also run on CPU.

The RPA and raw-data generation utilities use Julia 1.12.7. Their complete
Julia environment is locked by `data/generation/Project.toml` and
`data/generation/Manifest.toml` and can be instantiated with:

```bash
julia --project=data/generation -e "using Pkg; Pkg.instantiate()"
```

## Reproduction entry points

Run notebooks from this directory. They resolve paths relative to the release
root and create output directories when needed:

```bash
jupyter notebook notebooks/
```

The exact fixed and nested dataset memberships are in `data/splits/`; see its
README and machine-readable manifest. `notebooks/README.md` maps every cited
figure to its executable notebook and local inputs.
