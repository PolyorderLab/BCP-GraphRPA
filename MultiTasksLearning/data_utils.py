import pickle
import torch
import numpy as np
from torch_geometric.data import Data

def get_block_number_label(sample_tuple):
    """
    取代原有的拓扑分类逻辑。
    直接从 Julia 存入的元组中提取第 5 个元素 (index 4) 作为分类标签。
    该标签代表嵌段数量 (2, 3, 4, 5, 6, 7, 8)。
    """
    try:
        # 新的数据集格式: push!(dataset, (gnn_out..., Float32(χN), Float32(k_star), Int32(n_b)))
        # 使用 -1 保证无论是新格式还是旧格式都能准确获取到最后的 n_b
        return int(sample_tuple[-1])
    except IndexError:
        # 兼容旧版本数据，如果不存在则返回 0
        return 0

def process_single_file(file_path):
    """解析 Julia 生成的 .pkl 文件"""
    print(f"读取并处理: {file_path}")
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    samples, labels, ys = [], [], []
    
    for s in raw_data:
        try:
            # 基础物理数据
            Fp = np.array(s[0], dtype=np.float32)
            edge_idx = np.array(s[1], dtype=np.int64)
            edge_attr = np.array(s[2], dtype=np.float32)
            y = float(s[3])
            
            # --- 核心修改：直接获取嵌段数作为标签 ---
            n_b_label = get_block_number_label(s)
            
            samples.append((Fp, edge_idx, edge_attr, y))
            labels.append(n_b_label)
            ys.append(y)
        except Exception as e:
            continue
            
    return samples, np.array(labels), np.array(ys)

def create_pyg_list(samples, labels, y_values, scaler=None):
    """
    创建 PyG 数据列表。
    y 被转换为 log10 尺度。
    topo_type 现在存储的是嵌段数量 (n_b)。
    """
    # 转换为对数尺度 (稳定性预测通常在对数空间更易收敛)
    y_log = np.log10(np.clip(y_values.reshape(-1, 1), 1e-6, None))
    y_target = y_log if scaler is None else scaler.transform(y_log)
    
    data_list = []
    for i in range(len(samples)):
        Fp, edge_idx, edge_attr, _ = samples[i]
        
        # 特征列：[度数, 是否支化, A/B类型]
        x_tensor = torch.tensor(Fp[:, :3], dtype=torch.float)
        
        # 索引对齐 (Julia 1-based -> PyG 0-based)
        edge_index = torch.tensor(edge_idx - 1, dtype=torch.long)
        
        # 确保 edge_index 形状为 [2, E]
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        
        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(y_target[i], dtype=torch.float).view(1, 1),
            # 嵌段数标签：例如 2, 3, 4...
            topo_type=torch.tensor([labels[i]], dtype=torch.long)
        )
        data_list.append(data)
        
    return data_list