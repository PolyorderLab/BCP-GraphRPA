import os
import pickle
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

from data_utils import process_single_file, create_pyg_list
from model_utils import get_model

# --- 1. 配置路径与全局参数 ---
# 对应你刚生成的 50万 样本文件
TRAIN_PKL = "/home/zxy/project/GNN/randomclear/train_data_random/train_6n_5106.pkl"
TEST_PKL = "/home/zxy/project/GNN/randomclear/train_data_random/test6n_5106.pkl"
BASE_RESULT_ROOT = "/home/zxy/project/GNN/randomclear/random_outcome/model_6n_number"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# --- 2. 嵌段数配色配置 (Key 为 n_b) ---
COLOR_MAP_NB = {2: 'green', 3: 'red', 4: 'magenta', 5: 'royalblue', 6: 'gold', 7: 'gray', 8: 'orange', 9: 'brown', 10: 'purple'}

# --- 3. 辅助函数 ---
def save_loss_curve(train_hist, val_hist, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(train_hist, label='Train Loss', color='royalblue')
    plt.plot(val_hist, label='Val Loss', color='orange')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(False) 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_parity_plot(model, loader, save_path, title="Parity Plot (Grouped by Block Number)"):
    model.eval()
    all_preds_raw, all_trues_raw, all_nbs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            preds = model(batch).cpu().numpy()
            trues = batch.y.cpu().numpy()
            all_preds_raw.append(preds.reshape(-1, 1))
            all_trues_raw.append(trues.reshape(-1, 1))
            # 这里的 topo_type 现在存储的是 n_b (2, 3, 4)
            if hasattr(batch, 'topo_type'):
                all_nbs.extend(batch.topo_type.cpu().numpy().flatten())

    y_pred = 10 ** np.vstack(all_preds_raw).flatten()
    y_true = 10 ** np.vstack(all_trues_raw).flatten()
    all_nbs = np.array(all_nbs)
    
    # --- 计算分嵌段 MAPE 并导出 CSV ---
    mape_data = []
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    
    unique_nbs = np.unique(all_nbs)
    for nb in unique_nbs:
        mask = (all_nbs == nb)
        if not np.any(mask): continue
        
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        t_mape = np.mean(np.abs((t_pred - t_true) / t_true)) * 100
        
        # 记录统计
        mape_data.append({
            "Block_Number": int(nb),
            "Samples": len(t_true),
            "MAPE": round(t_mape, 4)
        })
        
        # --- 绘图优化：大样本降采样显示 ---
        MAX_PTS = 15000 
        if len(t_true) > MAX_PTS:
            idx = np.random.choice(len(t_true), MAX_PTS, replace=False)
            p_true, p_pred = t_true[idx], t_pred[idx]
        else:
            p_true, p_pred = t_true, t_pred

        ax.scatter(p_true, p_pred, c=COLOR_MAP_NB.get(nb, 'black'), 
                   label=f"{nb}-block (MAPE: {t_mape:.2f}%)", 
                   alpha=0.5, edgecolors='none', s=25)

    # 导出 CSV
    df_mape = pd.DataFrame(mape_data)
    csv_out_path = os.path.join(BASE_RESULT_ROOT, "block_mape_stats.csv")
    df_mape.to_csv(csv_out_path, index=False)
    print(f"📊 分嵌段统计已保存至: {csv_out_path}")

    total_mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
    # 绘制对角线
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('True $\chi N^*$'); ax.set_ylabel('Predicted $\chi N^*$')
    ax.set_title(f"{title}\nTotal MAPE: {total_mape:.2f}%")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return total_mape

# --- 4. 主训练流程 ---
def train_mixed_model():
    print(f"当前实际运行设备: {DEVICE}")
    
    print(f"📖 正在加载训练数据: {TRAIN_PKL}")
    tr_samples, tr_labels, tr_ys = process_single_file(TRAIN_PKL)
    print(f"📖 正在加载测试数据: {TEST_PKL}")
    te_samples, te_labels, te_ys = process_single_file(TEST_PKL)

    # 验证集比例保持 10% 或 20%
    tr_s, val_s, tr_l, val_l, tr_y, val_y = train_test_split(
        tr_samples, tr_labels, tr_ys, test_size=0.1, random_state=42
    )

    print(f"🛠️ 正在预处理数据为 PyG 格式...")
    # 增加 batch_size 到 128 以适配大样本
    train_loader = DataLoader(create_pyg_list(tr_s, tr_l, tr_y), batch_size=64, shuffle=True)
    val_loader = DataLoader(create_pyg_list(val_s, val_l, val_y), batch_size=64)
    test_loader = DataLoader(create_pyg_list(te_samples, te_labels, te_ys), batch_size=64)

    print(f"✅ 数据就绪 | 训练: {len(tr_s)} | 验证: {len(val_s)} | 测试: {len(te_samples)}")

    model = get_model(in_feats=3, edge_dim=2, hidden_dim=64, num_enc_layers=2, num_dec_layers=2, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    model_path = os.path.join(BASE_RESULT_ROOT, "best_mixed_model.pth")
    train_hist, val_hist = [], []

    print(f"🔥 开始训练模型 (预计 200 Epoch)...")
    
    for epoch in tqdm(range(1, 201), desc="Training"):
        model.train()
        total_t_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_t_loss += loss.item() * batch.num_graphs
        
        model.eval()
        total_v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                total_v_loss += criterion(model(batch), batch.y).item() * batch.num_graphs
        
        avg_t = total_t_loss / len(tr_s)
        avg_v = total_v_loss / len(val_s)
        train_hist.append(avg_t)
        val_hist.append(avg_v)
        
        scheduler.step(avg_v)
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), model_path)

    # 保存并评估
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    model.load_state_dict(torch.load(model_path))
    final_mape = save_parity_plot(model, test_loader, os.path.join(BASE_RESULT_ROOT, "parity_plot.png"))
    
    print(f"\n✨ 训练完成！测试集总 MAPE: {final_mape:.2f}%")

if __name__ == "__main__":
    train_mixed_model()