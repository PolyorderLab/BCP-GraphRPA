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

# PyG 相关的导入
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from TLdata_utils import process_single_file, create_pyg_list
from TLmodel_utils import get_model

# --- 1. 配置路径与全局参数 ---
# 注意：请将这里的路径替换为你真实用于 TL (GraphVAE) 训练的数据路径
DATA_PKL = "/home/zxy/project/TAPD/TrainData/Generate/output/dataset_4n_2w.pkl"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/TL/model_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 必须与 Decoder 中保持一致，用于 to_dense_adj 的填充对齐
MAX_NODES = 24  

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# --- 2. 辅助函数：VAE 损失曲线绘制 ---
def save_vae_loss_curve(train_hist, val_hist, save_path):
    plt.figure(figsize=(10, 6))
    
    # 绘制总损失
    plt.plot([h['total'] for h in train_hist], label='Train Total Loss', color='royalblue')
    plt.plot([h['total'] for h in val_hist], label='Val Total Loss', color='orange')
    
    # 绘制重构与 KL 损失（仅展示训练集做参考，使用虚线）
    plt.plot([h['recon'] for h in train_hist], label='Train Recon Loss', color='royalblue', linestyle='--')
    plt.plot([h['kld'] for h in train_hist], label='Train KLD Loss', color='green', linestyle=':')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title("GraphVAE Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def vae_loss_function(pred_adj, pred_edge_attr, true_adj, true_edge_attr, mu, logvar, kld_weight=1e-3):
    """
    计算 GraphVAE 的损失 = 拓扑重构损失 + 边属性重构损失 + KLD损失
    """
    # 1. 拓扑结构重构损失 (BCELoss，因为 pred_adj 已经经过 Sigmoid)
    bce_loss = nn.BCELoss()(pred_adj, true_adj.float())
    
    # 2. 边特征重构损失 (MSELoss，或者用 BCELoss 如果边特征也在 [0,1] 之间)
    mse_loss = nn.MSELoss()(pred_edge_attr, true_edge_attr.float())
    
    # 3. KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    
    # 总重构损失
    recon_loss = bce_loss + mse_loss
    # 整体损失
    total_loss = recon_loss + kld_weight * kld_loss
    
    return total_loss, recon_loss, kld_loss

# --- 3. 主训练流程 ---
def train_vae_model():
    print(f"当前实际运行设备: {DEVICE}")
    
    if not os.path.exists(DATA_PKL):
        print(f"⚠️ 找不到数据文件 {DATA_PKL}。")
        return

    print(f"📖 正在加载总数据: {DATA_PKL}")
    all_samples, all_labels, all_ys = process_single_file(DATA_PKL)

    # 第一次划分：分离出 20% 作为完全独立的测试集
    train_val_s, te_samples, train_val_l, te_labels, train_val_y, te_ys = train_test_split(
        all_samples, all_labels, all_ys, test_size=0.2, random_state=42
    )

    # 第二次划分：从剩下的里面，进一步抽出大约 10% 作为随验证用的集合
    tr_s, val_s, tr_l, val_l, tr_y, val_y = train_test_split(
        train_val_s, train_val_l, train_val_y, test_size=0.1, random_state=42
    )

    print(f"🛠️ 正在预处理数据为 PyG 格式...")
    train_loader = DataLoader(create_pyg_list(tr_s, tr_l, tr_y), batch_size=64, shuffle=True)
    val_loader = DataLoader(create_pyg_list(val_s, val_l, val_y), batch_size=64)
    test_loader = DataLoader(create_pyg_list(te_samples, te_labels, te_ys), batch_size=64)

    print(f"✅ 数据就绪 | 训练: {len(tr_s)} | 验证: {len(val_s)} | 测试: {len(te_samples)}")

    # 初始化 GraphVAE 模型
    model = get_model(
        in_feats=2,              # 节点特征维度 (参考你的业务要求)
        edge_dim=2,              # 边特征维度
        hidden_dim=64,           # Encoder隐层维度
        decoder_hidden_dim=128,  # 我们刚刚优化的 Decoder隐层维度
        latent_dim=32,           # VAE 潜在空间维度 z
        max_nodes=MAX_NODES,     # 最大节点生成数
        num_enc_layers=3, 
        device=DEVICE
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 根据验证集的 total loss 来自动缩减学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    model_path = os.path.join(BASE_RESULT_ROOT, "best_graph_vae.pth")
    train_hist, val_hist = [], []

    print(f"🔥 开始训练 GraphVAE 模型 (预计 200 Epoch)...")
    
    for epoch in tqdm(range(1, 201), desc="Training"):
        model.train()
        total_t_loss = total_t_recon = total_t_kld = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # 1. 前向传播：返回预测对角阵、边特征阵、以及高斯隐特征 mu, logvar
            pred_adj, pred_edge_attr, mu, logvar = model(batch)
            
            # 2. VAE 对齐：使用 PyG 的 to_dense_adj 获取真实的密集邻接矩阵和边特征矩阵作为重构目标
            # 注意: 为了与 Decoder 保持一致（固定输出 MAX_NODES * MAX_NODES），我们强制填充至 max_num_nodes
            true_adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=MAX_NODES)
            true_edge_attr = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr, max_num_nodes=MAX_NODES)
            
            # 3. 计算 VAE 组合损失 (重构误差 + KL发散)
            loss, recon, kld = vae_loss_function(pred_adj, pred_edge_attr, true_adj, true_edge_attr, mu, logvar, kld_weight=1e-3)
            
            loss.backward()
            optimizer.step()
            
            # 累加损失 (乘以 batch 内包含的图数量进行加权平均还原)
            num_graphs = batch.num_graphs
            total_t_loss += loss.item() * num_graphs
            total_t_recon += recon.item() * num_graphs
            total_t_kld += kld.item() * num_graphs
        
        # 验证集评估
        model.eval()
        total_v_loss = total_v_recon = total_v_kld = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred_adj, pred_edge_attr, mu, logvar = model(batch)
                
                true_adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=MAX_NODES)
                true_edge_attr = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr, max_num_nodes=MAX_NODES)
                
                v_loss, v_recon, v_kld = vae_loss_function(pred_adj, pred_edge_attr, true_adj, true_edge_attr, mu, logvar, kld_weight=1e-3)
                
                num_graphs = batch.num_graphs
                total_v_loss += v_loss.item() * num_graphs
                total_v_recon += v_recon.item() * num_graphs
                total_v_kld += v_kld.item() * num_graphs
        
        # 记录每轮平均 Loss
        avg_t = {'total': total_t_loss / len(tr_s), 'recon': total_t_recon / len(tr_s), 'kld': total_t_kld / len(tr_s)}
        avg_v = {'total': total_v_loss / len(val_s), 'recon': total_v_recon / len(val_s), 'kld': total_v_kld / len(val_s)}
        
        train_hist.append(avg_t)
        val_hist.append(avg_v)
        
        # LR 调整
        scheduler.step(avg_v['total'])
        
        # 保存最佳模型
        if avg_v['total'] < best_val_loss:
            best_val_loss = avg_v['total']
            torch.save(model.state_dict(), model_path)

    # 训练结束，保存损失曲线图供可视化分析
    loss_curve_path = os.path.join(BASE_RESULT_ROOT, "vae_loss_curve.png")
    save_vae_loss_curve(train_hist, val_hist, loss_curve_path)
    
    print(f"\n✨ 训练完成！")
    print(f"最佳模型已保存至: {model_path}")
    print(f"最优验证集 Total Loss: {best_val_loss:.4f}")
    print(f"📊 损失曲线图已保存至: {loss_curve_path}")

if __name__ == "__main__":
    train_vae_model()
