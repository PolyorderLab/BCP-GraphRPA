import os
import glob
import yaml
import torch
import numpy as np
import scipy.io as sio
import h5py
from torch_geometric.data import Data

# 定义拓扑类型到数字分类标签的映射字典
TOPO_MAP = {
    "Linear": 0,
    "Comb": 1,
    "Branch": 2,
    "Star": 3,
    "Dendrimer": 4
}

def load_and_process_dataset(data_dir, scaler=None):
    """
    扫描目录下的所有 .mat 和 .yml 文件，解析并直接构建 PyG Data 列表。
    """
    print(f"🔍 正在扫描数据目录: {data_dir}")
    
    # 获取所有的 .mat 文件路径
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        print("❌ 未在该目录下找到任何 .mat 文件！")
        return []

    data_list = []
    ys_raw = []
    
    for mat_path in mat_files:
        # 推断对应的 yml 文件路径
        base_path = mat_path.replace('.mat', '')
        yml_path = base_path + '.yml'
        
        if not os.path.exists(yml_path):
            continue
            
        try:
            # ================= 1. 读取 MAT 文件 (张量特征) =================
            try:
                mat_data = sio.loadmat(mat_path)
                Fp = np.array(mat_data['F_pad'], dtype=np.float32)
                edge_idx = np.array(mat_data['edge_index'], dtype=np.int64)
                edge_attr = np.array(mat_data['edge_attr'], dtype=np.float32)
            except (NotImplementedError, ValueError) as e:
                # Handle Matlab v7.3 files which are HDF5 based
                if 'HDF' in str(e) or 'v7.3' in str(e):
                    with h5py.File(mat_path, 'r') as f:
                        # h5py loads datasets in transposed order relative to Matlab/Scipy
                        Fp = np.array(f['F_pad'], dtype=np.float32).T
                        edge_idx = np.array(f['edge_index'], dtype=np.int64).T
                        edge_attr = np.array(f['edge_attr'], dtype=np.float32).T
                else:
                    raise e

            
            # ================= 2. 读取 YML 文件 (物理量与配置) =================
            with open(yml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            system_cfg = config['system']
            chiN_star = float(system_cfg['χN_star'])
            k_star = float(system_cfg['k_star'])
            topo_str = system_cfg['topology']
            
            # 从嵌段列表中直接推断嵌段数量 (替代原来的 s[-1])
            n_b = len(system_cfg['components'][0]['blocks'])
            
            # ================= 3. 构建 PyTorch Tensors =================
            x_tensor = torch.tensor(Fp[:, :3], dtype=torch.float)
            
            # 【重要】Julia 的索引是从 1 开始的，转换为 Python 的 0-based 索引
            edge_index = torch.tensor(edge_idx - 1, dtype=torch.long)
            if edge_index.dim() == 2 and edge_index.shape[0] != 2:
                edge_index = edge_index.t().contiguous()
                
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
            
            # 获取拓扑分类标签 (0 到 4)
            topo_label = TOPO_MAP.get(topo_str, -1)
            
            # ================= 4. 组装 PyG Data =================
            data = Data(
                x=x_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr_tensor,
                y=torch.tensor([[k_star]], dtype=torch.float),  # <-- 这里换成 k_star
                chiN_star=torch.tensor([[chiN_star]], dtype=torch.float), # 备用存起来
                num_blocks=torch.tensor([n_b], dtype=torch.long),
                topo_type=torch.tensor([topo_label], dtype=torch.long)
            )
            
            data_list.append(data)
            ys_raw.append(k_star)
            
        except Exception as e:
            print(f"⚠️ 解析样本 {os.path.basename(base_path)} 时出错: {e}")
            continue
            
    print(f"✅ 成功加载了 {len(data_list)} 个样本！")
    
    # ================= 5. 目标值预处理 (Log 化与归一化) =================
    ys_raw = np.array(ys_raw)
    y_log = np.log10(np.clip(ys_raw.reshape(-1, 1), 1e-6, None))
    y_target = y_log if scaler is None else scaler.transform(y_log)
    
    # 更新回 Data 列表中
    for i, data in enumerate(data_list):
        data.y = torch.tensor(y_target[i], dtype=torch.float).view(1, 1)
        
    return data_list