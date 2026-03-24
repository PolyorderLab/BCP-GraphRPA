import pickle
import torch
import numpy as np
import copy
from torch_geometric.data import Data

def get_block_number_label(sample):
    """Extract block number label (n_b) from PyG Data object."""
    # Ensure n_b attribute exists and is not None to prevent TypeError
    return int(sample.n_b) if getattr(sample, 'n_b', None) is not None else 0

def process_single_file(file_path):
    """Parse the generated .pkl file containing PyG Data objects."""
    print(f"Reading and processing: {file_path}")
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    samples, labels, ys = [], [], []
    
    for s in raw_data:
        try:
            n_b_label = get_block_number_label(s)
            # Move tensor to CPU before converting to numpy to prevent CUDA errors
            y_val = s.y.cpu().numpy().flatten()
            
            samples.append(s)
            labels.append(n_b_label)
            ys.append(y_val)
        except Exception:
            continue
            
    return samples, np.array(labels), np.array(ys)

def create_pyg_list(samples, labels, y_values, scaler=None):
    """
    Update PyG Data objects with scaled targets and n_b labels.
    Targets are scaled using log10.
    """
    y_values_arr = np.array(y_values)
    if y_values_arr.ndim == 1:
        y_values_arr = y_values_arr.reshape(-1, 1)
        
    y_log = np.log10(np.clip(y_values_arr, 1e-6, None))
    y_target = y_log if scaler is None else scaler.transform(y_log)
    
    data_list = []
    for i, s in enumerate(samples):
        data = copy.deepcopy(s)
        data.y = torch.tensor(y_target[i], dtype=torch.float).view(1, -1)
        data.topo_type = torch.tensor([labels[i]], dtype=torch.long)
        
        data_list.append(data)
        
    return data_list