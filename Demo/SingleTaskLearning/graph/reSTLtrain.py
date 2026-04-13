import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add project root to sys.path if needed
sys.path.append("/home/zxy/project/TAPD")
from data_utils import load_and_process_dataset, get_splits
from model_utils import get_model
from train_utils import Trainer, save_loss_curve

# ================= 1. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/RecSTL_1w_refactored"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# ================= 2. 辅助函数 =================
def evaluate_reconstruction(model, loader):
    """评估重构任务：计算 A 的准确率，以及 F 和 E 的 MSE"""
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
            
            # 使用 data_utils 预处理好的稠密矩阵
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            # 1. 评估 A 矩阵 (连通性还原率)
            pred_adj_bin = (out_A > 0.5).float()
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
    print(f"🖥️ 设备: {DEVICE}")
    
    # 1. Load Data
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='reconstruction')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # 2. Initialize Model
    model = get_model(task_type='reconstruction', in_feats=3, edge_dim=2, hidden_dim=128, num_enc_layers=4, max_nodes=32, device=DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    criterion_rec_A = nn.BCELoss()
    criterion_rec_FE = nn.MSELoss()
    
    best_val_loss = float('inf')
    model_path = os.path.join(BASE_RESULT_ROOT, "best_reconstruction_model.pth")
    train_hist, val_hist = [], []

    print(f"\n🔥 开始训练自编码器图重构模型 (设定总计 200 Epoch)...")
    
    # 3. Train
    for epoch in tqdm(range(1, 201), desc="Training Progress"):
        model.train()
        total_t_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out_A, out_F, out_E = model(batch)
            
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            loss_A = criterion_rec_A(out_A, target_adj)
            loss_F = criterion_rec_FE(out_F, target_F)
            loss_E = criterion_rec_FE(out_E, target_E)
            
            loss = loss_A + loss_F + loss_E
            loss.backward()
            optimizer.step()
            total_t_loss += loss.item() * batch.num_graphs
        
        # Validation
        model.eval()
        total_v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out_A, out_F, out_E = model(batch)
                
                target_adj = batch.dense_adj.view(-1, 32, 32)
                target_F = batch.dense_F.view(-1, 32, 3)
                target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
                
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

    # 4. Evaluation
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
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
    
    with open(os.path.join(BASE_RESULT_ROOT, "reconstruction_evaluation.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    train_reconstruction_model()
