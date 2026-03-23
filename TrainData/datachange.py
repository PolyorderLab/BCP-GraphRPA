import h5py
import torch
import pickle
import os
from torch_geometric.data import Data
from tqdm import tqdm

# Path configuration
bridge_path = "/home/zxy/project/TAPD/TrainData/Generate/output/data_bridge.h5"
final_pkl_path = "/home/zxy/project/TAPD/TrainData/Generate/output/dataset_4n_2w.pkl"

def convert_bridge_to_pyg():
    if not os.path.exists(bridge_path):
        print(f"❌ Error: {bridge_path} not found!")
        return

    pyg_data_list = []
    
    print(f"📖 Reading bridge file: {bridge_path}")
    with h5py.File(bridge_path, 'r') as f:
        # Sort keys to maintain order s0, s1, s2...
        keys = sorted(f.keys(), key=lambda x: int(x[1:]))
        
        for k in tqdm(keys, desc="Converting to PyG Objects"):
            group = f[k]
            
            # 1. Node Features: Transpose due to Julia (Col-major) vs Python (Row-major)
            # x shape from h5: (D, N) -> torch shape: (N, D)
            x = torch.from_numpy(group['x'][:]).T.float()
            
            # 2. Edge Index: Correct 1-based to 0-based index and transpose
            # edge_index shape from h5: (E, 2) -> torch shape: (2, E)
            edge_index = torch.from_numpy(group['edge_index'][:]).T.long() - 1
            
            # 3. Edge Attributes: Transpose
            # edge_attr shape from h5: (D_edge, E) -> torch shape: (E, D_edge)
            edge_attr = torch.from_numpy(group['edge_attr'][:]).T.float()
            
            # 4. Targets: Combine chiN and k_star
            y = torch.tensor([[group['chiN'][()], group['k_star'][()]]]).float()
            
            # 5. Metadata
            n_b = int(group['n_b'][()])
            
            # Construct PyG Data object (aligning with TopoGNN/GC-GNN structures)
            # [cite: 1114, 2239]
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                n_b=n_b
            )
            pyg_data_list.append(data)

    print(f"💾 Saving to Pickle: {final_pkl_path}")
    with open(final_pkl_path, 'wb') as f_out:
        pickle.dump(pyg_data_list, f_out, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Success! Total samples: {len(pyg_data_list)}")

if __name__ == "__main__":
    convert_bridge_to_pyg()