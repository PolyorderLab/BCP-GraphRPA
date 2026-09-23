# Data generation, selection and examples

Run all commands from the code repository root.

## Install and check

```sh
python -m pip install -r data/requirements.txt
python data/check_data_pipeline.py
```

The Python check uses a temporary directory and bundled real samples. It checks
pool selection, seed42/43 exclusion, MAT/YAML loading and wavelength conversion.
See generation/README.md for Julia installation and small-generation commands.

## Shared selection commands

```sh
python data/splits/unified_sampling.py --source-dir data/datasets/raw_2to5 --output-dir data/datasets/pools --seed 42
python data/splits/build_leakage_free_test_2k.py --train-dir data/datasets/pools/n_train_100k --pool-dir data/datasets/pools/n_test_pool --output-dir data/datasets/validation_seed42 --seed 42
python data/splits/build_leakage_free_test_2k.py --train-dir data/datasets/pools/n_train_100k --pool-dir data/datasets/pools/n_test_pool --exclude-dir data/datasets/validation_seed42 --output-dir data/datasets/test_seed43 --seed 43
```

The pool builder defaults to 25000 development and 5000 candidate records per
block count. Use --train-per-block and --pool-per-block for smaller runs. It
sorts source filenames before seeded shuffling and scans the requested source
freshly, avoiding stale caches. Legacy training-size sampling runs only with
--legacy-subsets; it is not the manuscript's final nested-index protocol.

The 2k builder is shared by both sets: change the pool, seed, exclusions and
output path. Defaults select 500 per block; --per-block changes this count.
Additional exclusions compare both MAT and YAML hashes. Existing output sets
are not overwritten. Historical release membership remains defined by the
published manifests, not these newly parameterized generation commands.

build_leakage_free_test_2w.py is an optional historical pool reconstruction tool:
use --source-dir (raw block subdirectories), --train-dir, --fixed-test-dir and
--output-dir, with --per-block 5000 by default. It requires exactly that many
nonoverlapping records per block and requires the fixed set to be contained in
the reconstructed pool. It is not needed for the new shared workflow above.

## Data locations

- generation/: random architectures, RPA module and Julia environment.
- splits/: pool selection and exclusion scripts.
- example/n_train_1k/: bundled example records, not the final experimental split.
- datasets/: default location for newly generated or downloaded full datasets.

Training/validation and nested-subset membership must use the verified saved
indices for each experiment. No new index file is presented as a historical one.

