import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset  # 【新增】引入 Subset 用于固定索引提取

# 导入你的模块
from MLdata_utils import load_and_process_dataset
from MLmodel_utils import get_model 

# ================= 1. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_topo_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/test1wML"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# ================= 2. 拓扑结构配色与映射配置 =================
TOPO_MAP_REV = {0: "Linear", 1: "Comb", 2: "Branch", 3: "Star", 4: "Dendrimer"}
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
    plt.plot(train_hist, label='Train Loss (Total)', color='royalblue')
    plt.plot(val_hist, label='Val Loss (Total)', color='orange')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Multi-Task Training and Validation Loss")
    plt.legend()
    plt.grid(False) 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_parity_plot(y_true, y_pred, all_topos, save_path, title, xlabel, ylabel):
    """通用的奇偶校验图绘制函数"""
    mape_data = []
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    
    unique_topos = np.unique(all_topos)
    for topo_idx in unique_topos:
        mask = (all_topos == topo_idx)
        if not np.any(mask): continue
        
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        t_mape = np.mean(np.abs((t_pred - t_true) / t_true)) * 100
        topo_name = TOPO_MAP_REV.get(int(topo_idx), f"Unknown-{topo_idx}")
        
        mape_data.append({"Topology": topo_name, "Samples": len(t_true), "MAPE(%)": round(t_mape, 4)})
        
        MAX_PTS = 15000 
        if len(t_true) > MAX_PTS:
            idx = np.random.choice(len(t_true), MAX_PTS, replace=False)
            p_true, p_pred = t_true[idx], t_pred[idx]
        else:
            p_true, p_pred = t_true, t_pred

        ax.scatter(p_true, p_pred, c=COLOR_MAP_TOPO.get(topo_idx, 'black'), 
                   label=f"{topo_name} (MAPE: {t_mape:.2f}%)", 
                   alpha=0.6, edgecolors='black', linewidths=0.5, s=25)

    df_mape = pd.DataFrame(mape_data)
    csv_out_path = save_path.replace('.png', '_stats.csv')
    df_mape.to_csv(csv_out_path, index=False)

    total_mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nTotal Test MAPE: {total_mape:.2f}%")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    return total_mape


def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=150)
    
    labels = [TOPO_MAP_REV[i] for i in range(5)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Topology')
    plt.ylabel('True Topology')
    plt.title('Topology Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def evaluate_multitask(model, loader):
    """评估多任务模型并生成相关图表与指标"""
    model.eval()
    
    preds_chiN, trues_chiN = [], []
    preds_kstar, trues_kstar = [], []
    preds_cls, trues_cls = [], []
    all_topos = []
    
    total_rec_A_acc = 0.0
    total_rec_F_mse = 0.0
    total_rec_E_mse = 0.0
    total_graphs = 0
    
    criterion_reg = nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out_chiN, out_kstar, out_cls, (out_A, out_F, out_E) = model(batch)
            
            preds_chiN.append(out_chiN.cpu().numpy().reshape(-1, 1))
            trues_chiN.append(batch.chiN_star.cpu().numpy().reshape(-1, 1))
            preds_kstar.append(out_kstar.cpu().numpy().reshape(-1, 1))
            trues_kstar.append(batch.k_star.cpu().numpy().reshape(-1, 1))

            # --- 分类任务 ---
            # 通过 argmax 获取概率最大的类别索引
            pred_classes = torch.argmax(out_cls, dim=1).cpu().numpy()
            true_classes = batch.topo_type.flatten().cpu().numpy()
            preds_cls.extend(pred_classes)
            trues_cls.extend(true_classes)
            
            if hasattr(batch, 'topo_type'):
                all_topos.extend(true_classes)
                
            # --- 重构任务评估 ---
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            # 将 Sigmoid 输出的概率矩阵转化为 0 和 1，计算预测准确率
            pred_adj_bin = (out_A > 0.5).float()
            A_acc = (pred_adj_bin == target_adj).float().mean().item()
            
            total_rec_A_acc += A_acc * batch.num_graphs
            total_rec_F_mse += criterion_reg(out_F, target_F).item() * batch.num_graphs
            total_rec_E_mse += criterion_reg(out_E, target_E).item() * batch.num_graphs
            total_graphs += batch.num_graphs

    # --- 处理与画图：回归任务 ---
    y_pred_chiN = 10 ** np.vstack(preds_chiN).flatten()
    y_true_chiN = 10 ** np.vstack(trues_chiN).flatten()
    y_pred_kstar = 10 ** np.vstack(preds_kstar).flatten()
    y_true_kstar = 10 ** np.vstack(trues_kstar).flatten()
    all_topos = np.array(all_topos)
    
    mape_chiN = save_parity_plot(
        y_true_chiN, y_pred_chiN, all_topos, 
        os.path.join(BASE_RESULT_ROOT, "parity_plot_chiN.png"),
        title=r"Parity Plot for $\chi N^*$", xlabel=r'True $\chi N^*$', ylabel=r'Predicted $\chi N^*$'
    )
    
    mape_kstar = save_parity_plot(
        y_true_kstar, y_pred_kstar, all_topos, 
        os.path.join(BASE_RESULT_ROOT, "parity_plot_kstar.png"),
        title=r"Parity Plot for $k^*$", xlabel=r'True $k^*$', ylabel=r'Predicted $k^*$'
    )
    
    # --- 处理与画图：分类任务 ---
    cls_acc = accuracy_score(trues_cls, preds_cls) * 100
    save_confusion_matrix(trues_cls, preds_cls, os.path.join(BASE_RESULT_ROOT, "confusion_matrix_cls.png"))
    
    # --- 计算重构任务最终平均指标 ---
    final_rec_A_acc = (total_rec_A_acc / total_graphs) * 100
    final_rec_F_mse = total_rec_F_mse / total_graphs
    final_rec_E_mse = total_rec_E_mse / total_graphs
    
    return mape_chiN, mape_kstar, cls_acc, final_rec_A_acc, final_rec_F_mse, final_rec_E_mse

# ================= 4. 主训练流程 =================
def train_mixed_model():
    print(f"🖥️ 当前实际运行设备: {DEVICE}")
    
    print(f"📖 正在扫描并加载总数据集 (这可能需要几分钟)...")
    full_dataset = load_and_process_dataset(DATA_DIR, scaler=None)
    
    if len(full_dataset) == 0:
        print("❌ 数据集为空，请检查路径是否正确！")
        return

    # ================= 【核心修改】固化数据集切分逻辑 =================
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

    # 使用 Subset 提取固定的数据集
    tr_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)
    test_data = Subset(full_dataset, test_idx)
    # =================================================================

    print(f"🛠️ 正在装载至 DataLoader...")
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    print(f"✅ 数据划分完毕 | 训练集: {len(tr_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}")

    # --- B. 模型初始化 ---
    model = get_model(in_feats=3, edge_dim=2, hidden_dim=64, num_enc_layers=4, max_nodes=32, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    criterion_reg = nn.MSELoss()          
    criterion_cls = nn.CrossEntropyLoss() 
    criterion_rec = nn.BCELoss()          
    
    best_val_loss = float('inf')
    model_path = os.path.join(BASE_RESULT_ROOT, "best_mtl_model.pth")
    train_hist, val_hist = [], []

    print(f"\n🔥 开始训练 4-Task MTL 多任务模型 (设定总计 200 Epoch)...")
    
    # --- C. 训练循环 ---
    for epoch in tqdm(range(1, 201), desc="Training Progress"):
        model.train()
        total_t_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out_chiN, out_kstar, out_cls, (out_A, out_F, out_E) = model(batch)
            
            loss_chiN = criterion_reg(out_chiN, batch.chiN_star)
            loss_kstar = criterion_reg(out_kstar, batch.k_star)
            loss_cls = criterion_cls(out_cls, batch.topo_type.flatten())
            
            target_adj = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            
            loss_rec_A = criterion_rec(out_A, target_adj)
            loss_rec_F = criterion_reg(out_F, target_F)
            loss_rec_E = criterion_reg(out_E, target_E)
            
            loss = loss_chiN + loss_kstar + 0.5 * loss_cls + 0.1 * (loss_rec_A + loss_rec_F + loss_rec_E)
            
            loss.backward()
            optimizer.step()
            total_t_loss += loss.item() * batch.num_graphs
        
        # --- 验证环节 ---
        model.eval()
        total_v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out_chiN, out_kstar, out_cls, (out_A, out_F, out_E) = model(batch)
                
                l_chiN = criterion_reg(out_chiN, batch.chiN_star)
                l_kstar = criterion_reg(out_kstar, batch.k_star)
                l_cls = criterion_cls(out_cls, batch.topo_type.flatten())
                
                target_adj = batch.dense_adj.view(-1, 32, 32)
                target_F = batch.dense_F.view(-1, 32, 3)
                target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
                
                l_rA = criterion_rec(out_A, target_adj)
                l_rF = criterion_reg(out_F, target_F)
                l_rE = criterion_reg(out_E, target_E)
                
                v_loss = l_chiN + l_kstar + 0.5 * l_cls + 0.1 * (l_rA + l_rF + l_rE)
                total_v_loss += v_loss.item() * batch.num_graphs
        
        avg_t = total_t_loss / len(tr_data)
        avg_v = total_v_loss / len(val_data)
        train_hist.append(avg_t)
        val_hist.append(avg_v)
        
        scheduler.step(avg_v)
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), model_path)

    # --- D. 最终评估与出图 ---
    print("\n💾 训练结束，正在生成全套分析图表...")
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
    # 加载最优权重进行全方位体检
    model.load_state_dict(torch.load(model_path))
    mape_chiN, mape_kstar, cls_acc, rec_A_acc, rec_F_mse, rec_E_mse = evaluate_multitask(model, test_loader)
    
    # 构造超级华丽的终极报告字符串 (注意 LaTeX 转义)
    report_text = f"""
4-Task MTL最终评估报告
[Task 1 & 2] 宏观物理量预测:
   - \\chi N^* 总 MAPE : {mape_chiN:.2f}%
   - k^* 总 MAPE      : {mape_kstar:.2f}%

[Task 3] 拓扑分类识别:
   - 分类准确率 (Accuracy): {cls_acc:.2f}%

[Task 4] 底层图全息重构:
   - 结构连通性还原准确率 (A Matrix) : {rec_A_acc:.2f}%
   - 节点度数还原均方误差 (F Matrix) : {rec_F_mse:.4f}
   - 边属性还原均方误差   (E Matrix) : {rec_E_mse:.4f}
"""
    
    # 打印到终端
    print("\n" + report_text)
    print(f"📁 请前往目录查看结果及相关文件: {BASE_RESULT_ROOT}")
    
    # 保存到 TXT 文件
    report_path = os.path.join(BASE_RESULT_ROOT, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📝 评估报告已保存为文本文件: {report_path}")

if __name__ == "__main__":
    train_mixed_model()