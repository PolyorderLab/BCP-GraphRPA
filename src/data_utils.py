import os
import glob
import yaml
import pickle
import torch
import numpy as np
import scipy.io as sio
import h5py
from torch_geometric.data import Data
from constants import TOPO_MAP

def load_and_process_dataset(
    data_dir, task_type='chiN', scaler=None, use_desc=False, desc_scaler=None,
    desc_names=None
):
    """
    扫描目录下的所有 .mat 和 .yml 文件，解析并直接构建 PyG Data 列表。
    
    Args:
        data_dir: 数据目录
        task_type: 任务类型，支持 'chiN', 'k', 'reconstruction', 'dual_regression', 'mtl', 'chiN_recon', 'k_recon', 'chiN_cls', 'k_cls', 'dual_recon', 'dual_cls'
        scaler: 目标值缩放器
        use_desc: 是否计算描述符
        desc_scaler: 描述符缩放器（use_desc=True 时，None 则自动 fit）
        desc_names: 描述符名称列表，仅 use_desc=True 时有效

    Returns:
        use_desc=False: data_list
        use_desc=True:  (data_list, fitted_desc_scaler)
    """
    label = f"{task_type} (with descriptors)" if use_desc else task_type
    print(f"🔍 正在扫描数据目录: {data_dir} for task: {label}")
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        print("❌ 未在该目录下找到任何 .mat 文件！")
        return ([], None) if use_desc else []

    if use_desc:
        from desc_utils import get_descriptors, DescriptorScaler

    data_list = []
    chiN_raw_list = []
    kstar_raw_list = []
    
    for mat_path in mat_files:
        base_path = mat_path.replace('.mat', '')
        yml_path = base_path + '.yml'
        
        if not os.path.exists(yml_path):
            continue
            
        try:
            # ================= 1. 读取 MAT 文件 =================
            try:
                mat_data = sio.loadmat(mat_path)
                Fp = np.array(mat_data['F_pad'], dtype=np.float32)
                edge_idx = np.array(mat_data['edge_index'], dtype=np.int64)
                edge_attr = np.array(mat_data['edge_attr'], dtype=np.float32)
            except (NotImplementedError, ValueError):
                with h5py.File(mat_path, 'r') as f:
                    Fp = np.array(f['F_pad'], dtype=np.float32).T
                    edge_idx = np.array(f['edge_index'], dtype=np.int64).T
                    edge_attr = np.array(f['edge_attr'], dtype=np.float32).T

            # ================= 2. 读取 YML 文件 =================
            with open(yml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            system_cfg = config['system']
            chiN_star = float(system_cfg['χN_star'])
            k_star = float(system_cfg['k_star'])
            topo_str = system_cfg['topology']
            n_b = len(system_cfg['components'][0]['blocks'])
            
            # ================= 3. 构建 PyTorch Tensors =================
            x_tensor = torch.tensor(Fp[:, :3], dtype=torch.float)
            edge_index = torch.tensor(edge_idx - 1, dtype=torch.long)
            if edge_index.dim() == 2 and edge_index.shape[0] != 2:
                edge_index = edge_index.t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
            topo_label = TOPO_MAP.get(topo_str, -1)
            
            # ================= 4. 构建 Data 对象 =================
            data = Data(
                x=x_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr_tensor,
                num_blocks=torch.tensor([n_b], dtype=torch.long),
                topo_type=torch.tensor([topo_label], dtype=torch.long)
            )

            if use_desc:
                desc_raw = get_descriptors(edge_index, edge_attr_tensor,
                                                     desc_names=desc_names)
                data.desc_raw = torch.tensor(
                    desc_raw, dtype=torch.float
                ).unsqueeze(0)

            if task_type in ['reconstruction', 'mtl', 'chiN_recon', 'k_recon', 'dual_recon']:
                max_nodes = Fp.shape[0]
                dense_adj = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
                dense_edge_attr = torch.zeros((max_nodes, max_nodes, 2), dtype=torch.float)
                for idx in range(edge_index.shape[1]):
                    u, v = edge_index[0, idx], edge_index[1, idx]
                    dense_adj[u, v] = 1.0
                    dense_edge_attr[u, v] = edge_attr_tensor[idx]
                
                data.dense_adj = dense_adj
                data.dense_F = x_tensor
                data.dense_edge_attr = dense_edge_attr

            data_list.append(data)
            chiN_raw_list.append(chiN_star)
            kstar_raw_list.append(k_star)
            
        except Exception as e:
            print(f"⚠️ 解析样本 {os.path.basename(base_path)} 时出错: {e}")
            continue
            
    print(f"✅ 成功加载了 {len(data_list)} 个样本！")
    
    # ================= 5. 目标值预处理 =================
    chiN_raw_arr = np.array(chiN_raw_list).reshape(-1, 1)
    kstar_raw_arr = np.array(kstar_raw_list).reshape(-1, 1)
    
    chiN_log = np.log10(np.clip(chiN_raw_arr, 1e-6, None))
    kstar_log = np.log10(np.clip(kstar_raw_arr, 1e-6, None))
    
    chiN_target = chiN_log if scaler is None else scaler.transform(chiN_log)
    kstar_target = kstar_log if scaler is None else scaler.transform(kstar_log)

    # ================= 6. 描述符标准化 =================
    if use_desc:
        desc_matrix = np.vstack(
            [d.desc_raw.numpy()[0] for d in data_list]
        )
        desc_matrix = np.where(np.isnan(desc_matrix), -2, desc_matrix)
        desc_matrix = np.where(np.isinf(desc_matrix), -2, desc_matrix)

        if desc_scaler is None:
            desc_scaler = DescriptorScaler()
            desc_scaled = desc_scaler.fit_transform(desc_matrix)
            print("✅ 描述符标准化器已fit")
        else:
            desc_scaled = desc_scaler.transform(desc_matrix)
            print("✅ 使用已存在的描述符标准化器")

    # ================= 7. 设置目标值和描述符 =================
    for i, data in enumerate(data_list):
        if task_type == 'chiN':
            data.y = torch.tensor(chiN_target[i], dtype=torch.float).view(1, 1)
        elif task_type == 'k':
            data.y = torch.tensor(kstar_target[i], dtype=torch.float).view(1, 1)
        elif task_type == 'reconstruction':
            # y is not used strictly as a single target here, but we can set it to k_star for consistency with old code
            data.y = torch.tensor(kstar_target[i], dtype=torch.float).view(1, 1)
            data.chiN_star = torch.tensor(chiN_target[i], dtype=torch.float).view(1, 1)
            data.k_star = torch.tensor(kstar_target[i], dtype=torch.float).view(1, 1)
        elif task_type in ['mtl', 'dual_regression', 'chiN_recon', 'k_recon', 'chiN_cls', 'k_cls', 'dual_recon', 'dual_cls']:
            data.chiN_star = torch.tensor(chiN_target[i], dtype=torch.float).view(1, 1)
            data.k_star = torch.tensor(kstar_target[i], dtype=torch.float).view(1, 1)

        if use_desc:
            data.desc = torch.tensor(
                desc_scaled[i], dtype=torch.float
            ).unsqueeze(0)
            delattr(data, "desc_raw")

    if use_desc:
        return data_list, desc_scaler
    return data_list
