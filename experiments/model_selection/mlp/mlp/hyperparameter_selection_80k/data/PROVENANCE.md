# Development descriptor matrix provenance

`n_train_100k_descriptor_matrix.csv` is a copy of
`baselinemodels/mlp_pub/analysis_100k/train_descriptor_matrix.csv`. It holds
the deterministic descriptor values and target for the 100,000 samples in
`TrainData/data/chiNlimited/n_train_100k`.

It is **not** a descriptor-selection result. The 11 input columns used by the
hyperparameter search were frozen beforehand from the 80k-only analyses in
`../../descriptor_selection_80k/`:

`SLE, r_deg, CV_arm, n_leaves, M_AB, IR, D_f, ISC, CAF, tau_mean, f_A`.

The fixed split is generated with `train_test_split(..., test_size=0.2,
random_state=42)`, yielding 80,000 training and 20,000 validation records.

The external 2k final-test directory is deliberately not an input to this
search and must only be read after hyperparameters are frozen.
