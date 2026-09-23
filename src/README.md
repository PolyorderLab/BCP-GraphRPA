# Shared source status

`data_utils.py` is the shared loader used by the bundled training and inference
entry points.

The loader preserves raw wavelength labels for `task_type='k'`, and converts
`L*/Rg` to `k*Rg = 2*pi/(L*/Rg)` before taking log10 for `k_wavevector`.

Required packages: PyYAML, torch, NumPy, SciPy, h5py and torch_geometric.
`constants.py` contains the topology mapping and colors used by the experiments.
It covers Random_2blocks through
Random_6blocks. The loader assigns -1 to unmapped labels, including seven- to
ten-block labels; the mapping has not been extended. The current data-module check installs h5py and exercises paired sample loading
and both period and wavevector targets; see data/check_data_pipeline.py.

The transfer trainer at
`experiments/kstar_transfer/gnn/finalmodel/transfer/scripts/train_c2_kstar_transfer.py`
includes the randomly initialized scratch control without loading a chiN checkpoint.
