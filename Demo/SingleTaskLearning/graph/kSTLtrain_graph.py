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
from torch.utils.data import Subset  # 新增
from kSTLdata_graph import load_and_process_dataset
from kSTLmodel_graph import get_model 

# ================= 1. 配置路径与全局参数 =================
# 你刚生成的 10万 样本文件夹 (包含 .mat 和 .yml)
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topo_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph/ktest1wSTL"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# ================= 2. 拓扑结构配色与映射配置 =================
# 0: Linear, 1: Comb, 2: Branch, 3: Star, 4: Dendrimer
TOPO_MAP_REV = {0: "Linear", 1: "Comb", 2: "Branch", 3: "Star", 4: "Dendrimer"}

# 给每种拓扑分配一个高区分度的颜色
COLOR_MAP_TOPO = {
    0: 'green',       # Linear
    1: 'royalblue',   # Comb
    2: 'magenta',     # Branch
    3: 'orange',      # Star
    4: 'purple'       # Dendrimer
}

# ================= 3. 辅助函数 =================
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

def save_parity_plot(model, loader, save_path, title="Parity Plot (Grouped by Topology)"):
    model.eval()
    all_preds_raw, all_trues_raw, all_topos = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            preds = model(batch).cpu().numpy()
            trues = batch.y.cpu().numpy()
            all_preds_raw.append(preds.reshape(-1, 1))
            all_trues_raw.append(trues.reshape(-1, 1))
            
            # 提取拓扑类型的分类标签 (0~4)
            if hasattr(batch, 'topo_type'):
                all_topos.extend(batch.topo_type.cpu().numpy().flatten())

    y_pred = 10 ** np.vstack(all_preds_raw).flatten()
    y_true = 10 ** np.vstack(all_trues_raw).flatten()
    all_topos = np.array(all_topos)
    
    # --- 计算分拓扑结构的 MAPE 并导出 CSV ---
    mape_data = []
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    
    unique_topos = np.unique(all_topos)
    for topo_idx in unique_topos:
        mask = (all_topos == topo_idx)
        if not np.any(mask): continue
        
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        t_mape = np.mean(np.abs((t_pred - t_true) / t_true)) * 100
        
        # 获取拓扑结构的真实名称
        topo_name = TOPO_MAP_REV.get(int(topo_idx), f"Unknown-{topo_idx}")
        
        # 记录统计
        mape_data.append({
            "Topology": topo_name,
            "Samples": len(t_true),
            "MAPE(%)": round(t_mape, 4)
        })
        
        # --- 绘图优化：大样本降采样显示，防止点太密集变成糊状 ---
        MAX_PTS = 15000 
        if len(t_true) > MAX_PTS:
            idx = np.random.choice(len(t_true), MAX_PTS, replace=False)
            p_true, p_pred = t_true[idx], t_pred[idx]
        else:
            p_true, p_pred = t_true, t_pred

        ax.scatter(p_true, p_pred, c=COLOR_MAP_TOPO.get(topo_idx, 'black'), 
                   label=f"{topo_name} (MAPE: {t_mape:.2f}%)", 
                   alpha=0.6, edgecolors='black', linewidths=0.5, s=25)

    # 导出 CSV 统计表格
    df_mape = pd.DataFrame(mape_data)
    csv_out_path = os.path.join(BASE_RESULT_ROOT, "topology_mape_stats.csv")
    df_mape.to_csv(csv_out_path, index=False)
    print(f"📊 分拓扑结构测试统计已保存至: {csv_out_path}")

    total_mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
    # 绘制对角线基准
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    # 【修改点1】：将图表轴标签改为 k^*
    ax.set_xlabel(r'True $k^*$')
    ax.set_ylabel(r'Predicted $k^*$')
    ax.set_title(f"{title}\nTotal Test MAPE: {total_mape:.2f}%")
    
    # 把图例放在图外侧，避免遮挡数据点
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    return total_mape

# ================= 4. 主训练流程 =================
def train_mixed_model():
    print(f"🖥️ 当前实际运行设备: {DEVICE}")
    
    # --- A. 数据加载与固定划分 ---
    print(f"📖 正在扫描并加载总数据集 (这可能需要几分钟)...")
    full_dataset = load_and_process_dataset(DATA_DIR, scaler=None)
    
    if len(full_dataset) == 0:
        print("❌ 数据集为空，请检查路径是否正确！")
        return

    # 定义统一的索引存放路径（存放在数据根目录的同级，即 TrainData/data 下）
    split_info_path = os.path.join(os.path.dirname(DATA_DIR), "fixed_dataset_splits_1w.pt")

    if os.path.exists(split_info_path):
        print(f"📥 检测到统一的数据集划分文件，正在加载以确保严格的对比实验...")
        splits = torch.load(split_info_path, weights_only=False)
        train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    else:
        print(f"🔨 未检测到统一划分文件，正在生成固定的索引切分并保存...")
        indices = np.arange(len(full_dataset))
        # 第一次划分：抽出 20% 作为暂存的 Val+Test，剩下 80% 作为 Train
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
        # 第二次划分：把刚才的 20% 对半切，得到 10% Val 和 10% Test
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx}, split_info_path)
        print(f"💾 切分索引已保存至: {split_info_path}")

    # 提取子集
    tr_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)
    test_data = Subset(full_dataset, test_idx)

    print(f"🛠️ 正在装载至 DataLoader...")
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    print(f"✅ 数据划分完毕 | 训练集: {len(tr_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}")

    # --- B. 模型初始化 ---
    model = get_model(in_feats=3, edge_dim=2, hidden_dim=64, num_enc_layers=3, num_dec_layers=2, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    model_path = os.path.join(BASE_RESULT_ROOT, "best_mixed_model.pth")
    train_hist, val_hist = [], []

    print(f"\n🔥 开始训练模型 (设定总计 200 Epoch)...")
    
    # --- C. 训练循环 ---
    for epoch in tqdm(range(1, 201), desc="Training Progress"):
        model.train()
        total_t_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            # 注意: 使用 batch.num_graphs 而不是直接用 batch 数量
            total_t_loss += loss.item() * batch.num_graphs
        
        model.eval()
        total_v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                total_v_loss += criterion(model(batch), batch.y).item() * batch.num_graphs
        
        avg_t = total_t_loss / len(tr_data)
        avg_v = total_v_loss / len(val_data)
        train_hist.append(avg_t)
        val_hist.append(avg_v)
        
        scheduler.step(avg_v)
        # 保存验证集表现最好的模型参数
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), model_path)

    # --- D. 最终评估与出图 ---
    print("\n💾 训练结束，正在生成分析图表...")
    # 保存 Loss 曲线
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
    # 加载表现最好的权重，在从未见过的 Test 集上做最终测试
    model.load_state_dict(torch.load(model_path))
    final_mape = save_parity_plot(model, test_loader, os.path.join(BASE_RESULT_ROOT, "parity_plot_topology.png"))
    
    print(f"\n✨ 全部任务完美收官！模型已保存，测试集最终总 MAPE: {final_mape:.2f}%")
    print(f"📁 请前往目录查看结果: {BASE_RESULT_ROOT}")

if __name__ == "__main__":
    train_mixed_model()