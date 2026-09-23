# Example data

`n_train_1k/` is copied from the author-provided local `TAPD/n_train_1k/`
directory. It contains 1,000 paired MAT/YML samples and the accompanying PT file.
The example set preserves the source filenames and paired MAT/YML records.

Use this directory for loader checks and small demonstration runs. It does not
replace the full development, validation, final-test or OOD datasets used in
the manuscript. Its membership and accompanying PT file have not been verified
against the final experimental split indices.

Pass `data/example/n_train_1k` as the dataset path from the repository root.
The raw `k_star` field retains the legacy wavelength label; use the loader's
`k_wavevector` task for the dimensionless wavevector target.
