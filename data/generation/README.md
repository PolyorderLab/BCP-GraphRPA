# Architecture generation and RPA

Run from the code repository root. Both entry points activate this directory's
Project.toml and load rpa/ExtractedPolyorderRPA.jl; the full Polyorder package
is no longer imported. Polymer and PolymerArchitecture remain dependencies.

```sh
julia --project=data/generation -e "using Pkg; Pkg.instantiate()"
julia --project=data/generation data/generation/rpa/smoke_test.jl
julia --threads=1 data/generation/number_5.jl --count-per-block 2 --seed 42 --output-dir data/datasets/smoke_2to5
julia data/generation/generate_6block_transfer_test.jl --block-counts 6,7,8,9,10 --count-per-block 1 --seed 20260707 --output-dir data/datasets/smoke_ood
```

For full generation use 30000 per block in number_5.jl and 500 per block
in the OOD generator. Use a new output directory; existing paths are refused.
The first script writes block-count subdirectories, which unified_sampling.py
scans recursively. OOD output is flat. Original numerical algorithms and legacy
wavelength labels are preserved. The k_wavevector loader converts the latter.

The raw generator uses seed + thread ID. Use one thread for repeatable small
checks; changing thread count or scheduling can change the generated pool.
The extracted source license and provenance are in rpa/upstream and
rpa/PROVENANCE.json. The installed environment is not claimed to be the original
historical data-generation environment.
