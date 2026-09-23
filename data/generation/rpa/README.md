# Extracted RPA module

Source: `D:/2026/paper/OPF/Polyorder.jl`. Original files, Apache-2.0 LICENSE
and NOTICE are preserved in `upstream/`; source hashes are in `PROVENANCE.json`.
This local copy has not been verified as the version used to generate TAPD labels.

`ExtractedPolyorderRPA.jl` loads the original RPA core and an adapted numerical
extension. Only extension import paths were changed; numerical expressions,
optimization bounds and return values are preserved. SCFT code is not included.

Dependencies: Polymer, PolymerArchitecture, Graphs, LinearAlgebra, Optim and Roots.
The dependency compatibility ranges come from the source package. The parent
generation/Manifest.toml locks the current verification environment, not the historical environment. Polymer and PolymerArchitecture must be available
from your configured registry or local package sources.

```julia
include("ExtractedPolyorderRPA.jl")
using .ExtractedPolyorderRPA: RPA
chiN_star, wavelength = RPA.compute_stability_limit(chain; χN0=15.0, k2_max=100.0)
```

The second result is the dimensionless wavelength `L*/Rg = 2π/sqrt(xstar)`.
The original dataset generators store this value under the historical name
`k_star`. Preserve this convention when reproducing the raw datasets.

The current C2 transfer task uses `task_type="k_wavevector"`. Its training script
`sourcecode/pub/gnn/finalmodel/transfer/scripts/train_c2_kstar_transfer.py`
sets this task explicitly and, when deriving labels from the chiN graph cache,
computes `wavevector = 2.0 * np.pi / legacy_period`. The regression target is
`log10(k*Rg)`, where `k*Rg = 2π/(L*/Rg) = sqrt(xstar)`.
The direct-loading branch delegates to `load_and_process_dataset` with the same
task type. The author-supplied loader is now included at `src/data_utils.py` and
confirms this conversion before log10 transformation. The author-supplied
`src/constants.py` now provides its `TOPO_MAP` dependency.

Historical tasks using `task_type="k"` retain wavelength labels, as clarified by
the author. Do not apply the reciprocal conversion again to already converted
wavevector caches. The name `kstar` alone does not determine the stored quantity.

This explains the raw-label naming difference; it does not establish the exact
historical Polyorder version. Both generators now load this extracted copy explicitly.

From the repository root, run `julia --project=data/generation data/generation/rpa/smoke_test.jl` after installing the dependencies.
The numerical smoke test passed in the environment locked by the parent Manifest.toml. Both generation entry points were run and their nine small-run outputs loaded successfully; see data/VALIDATION.json.
