import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset  

# 【新增】引入 PyG 原生工具，将稀疏图动态转化为稠密矩阵
from torch_geometric.utils import to_dense_adj, to_dense_batch

# ⚠️ 注意：确保这两个模块的名字和你的实际文件名一致
from reSTLdata_graph import load_and_process_dataset
from reSTLmodel_graph import get_model 

# ================= 1. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topo_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph/RecSTL_1w"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# ================= 2. 辅助函数 =================
def save_loss_curve(train_hist, val_hist, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_hist, label='Train Loss (Total Rec)', color='royalblue')
    plt.plot(val_hist, label='Val Loss (Total Rec)', color='orange')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Reconstruction Training and Validation Loss")
    plt.legend()
    plt.grid(False) 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_reconstruction(model, loader):
    """专门用于评估重构任务的函数：计算 A 的准确率，以及 F 和 E 的 MSE"""
    model.eval()
    
    total_A_acc = 0.0
    total_F_mse = 0.0
    total_E_mse = 0.0
    total_graphs = 0
    
    criterion_reg = nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out_A, out_F, out_E = model(batch)
            
            # 【修复点 1】：动态生成目标稠密矩阵
            target_adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=32)
            target_F, _ = to_dense_batch(batch.x, batch=batch.batch, max_num_nodes=32)
            target_E = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr, max_num_nodes=32)
            
            # 1. 评估 A 矩阵 (连通性还原率)
            pred_adj_bin = (out_A > 0.5).float() # 大于 0.5 认为有边，否则无边
            A_acc = (pred_adj_bin == target_adj).float().mean().item()
            
            # 2. 评估 F 和 E 的 MSE
            F_mse = criterion_reg(out_F, target_F).item()
            E_mse = criterion_reg(out_E, target_E).item()
            
            total_A_acc += A_acc * batch.num_graphs
            total_F_mse += F_mse * batch.num_graphs
            total_E_mse += E_mse * batch.num_graphs
            total_graphs += batch.num_graphs

    final_A_acc = (total_A_acc / total_graphs) * 100
    final_F_mse = total_F_mse / total_graphs
    final_E_mse = total_E_mse / total_graphs
    
    return final_A_acc, final_F_mse, final_E_mse

# ================= 3. 主训练流程 =================
def train_reconstruction_model():
    print(f"🖥️ 当前实际运行设备: {DEVICE}")
    
    # --- A. 数据加载与固定划分 ---
    print(f"📖 正在扫描并加载总数据集 (这可能需要几分钟)...")
    full_dataset = load_and_process_dataset(DATA_DIR, scaler=None)
    
    if len(full_dataset) == 0:
        print("❌ 数据集为空，请检查路径是否正确！")
        return

    split_info_path = os.path.join(os.path.dirname(DATA_DIR), "fixed_dataset_splits_1w.pt")

    if os.path.exists(split_info_path):
        print(f"📥 检测到统一的数据集划分文件，正在加载以确保严格的对比实验...")
        splits = torch.load(split_info_path, weights_only=False)
        train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    else:
        print(f"🔨 未检测到统一划分文件，正在生成固定的索引切分并保存...")
        indices = np.arange(len(full_dataset))
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx}, split_info_path)
        print(f"💾 切分索引已保存至: {split_info_path}")

    tr_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)
    test_data = Subset(full_dataset, test_idx)

    print(f"🛠️ 正在装载至 DataLoader...")
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    print(f"✅ 数据划分完毕 | 训练集: {len(tr_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}")

    # --- B. 模型初始化 ---
    model = get_model(in_feats=3, edge_dim=2, hidden_dim=128, num_enc_layers=4, max_nodes=32, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 重构任务需要两把尺子：BCE 和 MSE
    criterion_rec_A = nn.BCELoss()
    criterion_rec_FE = nn.MSELoss()
    
    best_val_loss = float('inf')
    model_path = os.path.join(BASE_RESULT_ROOT, "best_reconstruction_model.pth")
    train_hist, val_hist = [], []

    print(f"\n🔥 开始训练自编码器图重构模型 (设定总计 200 Epoch)...")
    
    # --- C. 训练循环 ---
    for epoch in tqdm(range(1, 201), desc="Training Progress"):
        model.train()
        total_t_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # 1. 前向传播拿到三个矩阵
            out_A, out_F, out_E = model(batch)
            
            # 【修复点 2】：动态生成目标稠密矩阵 (训练阶段)
            target_adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=32)
            target_F, _ = to_dense_batch(batch.x, batch=batch.batch, max_num_nodes=32)
            target_E = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr, max_num_nodes=32)
            
            # 3. 分别计算 Loss
            loss_A = criterion_rec_A(out_A, target_adj)
            loss_F = criterion_rec_FE(out_F, target_F)
            loss_E = criterion_rec_FE(out_E, target_E)
            
            # 4. 组合 Total Loss
            loss = loss_A + loss_F + loss_E
            
            loss.backward()
            optimizer.step()
            total_t_loss += loss.item() * batch.num_graphs
        
        # --- 验证环节 ---
        model.eval()
        total_v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out_A, out_F, out_E = model(batch)
                
                # 【修复点 3】：动态生成目标稠密矩阵 (验证阶段)
                target_adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=32)
                target_F, _ = to_dense_batch(batch.x, batch=batch.batch, max_num_nodes=32)
                target_E = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr, max_num_nodes=32)
                
                v_loss_A = criterion_rec_A(out_A, target_adj)
                v_loss_F = criterion_rec_FE(out_F, target_F)
                v_loss_E = criterion_rec_FE(out_E, target_E)
                
                total_v_loss += (v_loss_A + v_loss_F + v_loss_E).item() * batch.num_graphs
        
        avg_t = total_t_loss / len(tr_data)
        avg_v = total_v_loss / len(val_data)
        train_hist.append(avg_t)
        val_hist.append(avg_v)
        
        scheduler.step(avg_v)
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), model_path)

    # --- D. 最终评估 ---
    print("\n💾 训练结束，正在生成分析报告...")
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
    # 载入最优权重在 Test 集上体检
    model.load_state_dict(torch.load(model_path))
    A_acc, F_mse, E_mse = evaluate_reconstruction(model, test_loader)
    
    report_text = f"""
======================================
单任务自编码图重构 (Graph Reconstruction) 最终评估报告 
======================================
- 结构连通性还原准确率 (A Matrix) : {A_acc:.2f}%
- 节点度数还原均方误差 (F Matrix) : {F_mse:.6f}
- 边属性还原均方误差   (E Matrix) : {E_mse:.6f}
======================================
"""
    print(report_text)
    
    report_path = os.path.join(BASE_RESULT_ROOT, "reconstruction_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"📁 请前往目录查看 Loss 曲线及结果报告: {BASE_RESULT_ROOT}")

if __name__ == "__main__":
    train_reconstruction_model()