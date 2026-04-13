import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.loader import DataLoader

# ================= 1. 将项目根目录加入模块搜索路径 =================
sys.path.append("/home/zxy/project/TAPD")
from src.data_utils import load_and_process_dataset
from src.model_utils import get_model
from src.train_utils import Trainer, get_splits
from src.viz_utils import evaluate_and_plot_mtl, save_loss_curve

# ================= 2. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/MTLGridSearch"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(BASE_RESULT_ROOT, exist_ok=True)
CSV_RECORD_PATH = os.path.join(BASE_RESULT_ROOT, "grid_search_records.csv")

class MTLTrainerGrid(Trainer):
    """
    修改过的Trainer，能够接受各任务损失的动态权重，方便网格搜索。
    """
    def __init__(self, model, optimizer, device, scheduler, save_path, loss_weights):
        # 原本的 criterion 我们不使用，因此传 None
        super().__init__(model, optimizer, None, device, scheduler=scheduler, save_path=save_path)
        self.loss_weights = loss_weights # dict: {'w_chiN': ..., 'w_kstar': ..., 'w_cls': ..., 'w_recon': ...}

    def calculate_loss(self, out, batch):
        w_chiN = self.loss_weights['w_chiN']
        w_kstar = self.loss_weights['w_kstar']
        w_cls = self.loss_weights['w_cls']
        w_recon = self.loss_weights['w_recon']
        
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()
        
        # 1. 回归及分类任务
        loss_chiN = criterion_mse(out[0], batch.chiN_star)
        loss_kstar = criterion_mse(out[1], batch.k_star)
        loss_cls = criterion_ce(out[2], batch.topo_type)
        
        # 2. 重建任务 loss 计算
        out_A, out_F, out_E = out[3]
        target_A = batch.dense_adj.view(-1, 32, 32)
        target_F = batch.dense_F.view(-1, 32, 3)
        target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)

        loss_A = nn.functional.binary_cross_entropy(out_A, target_A)
        loss_F = criterion_mse(out_F, target_F)
        loss_E = criterion_mse(out_E, target_E)
        loss_recon = loss_A + loss_F + loss_E
        
        # 加权求和
        return (w_chiN * loss_chiN) + (w_kstar * loss_kstar) + (w_cls * loss_cls) + (w_recon * loss_recon)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.calculate_loss(out, batch)
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)


def grid_search_mtl():
    print(f"🖥️ Device: {DEVICE}")
    print(f"📁 Results will be saved to: {BASE_RESULT_ROOT}")
    
    # 1. 载入统一的数据集及切分
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='mtl')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # 2. 定义搜索空间参数 (可以根据自己需求调整，这里举例 0.1, 1.0, 10.0，避免组合过多)
    weights_range = [0.1, 1.0, 10.0]
    
    # 3. 如果已有记录，读取继续训练，避免中断后重新算
    if os.path.exists(CSV_RECORD_PATH):
        records_df = pd.read_csv(CSV_RECORD_PATH)
        records = records_df.to_dict('records')
        print(f"🔄 发现以往记录！从 {CSV_RECORD_PATH} 读取了 {len(records)} 组结果...")
    else:
        records = []

    # 4. 执行网格搜索循环
    for w_chiN in weights_range:
        for w_kstar in weights_range:
            for w_cls in weights_range:
                for w_recon in weights_range:
                    
                    # 命名此组合的文件名
                    comb_name = f"wChiN_{w_chiN}_wK_{w_kstar}_wCls_{w_cls}_wRec_{w_recon}"
                    
                    # 检查是否已处理过此组合
                    already_done = False
                    for r in records:
                        if (np.isclose(r.get("w_chiN"), w_chiN) and 
                            np.isclose(r.get("w_kstar"), w_kstar) and 
                            np.isclose(r.get("w_cls"), w_cls) and 
                            np.isclose(r.get("w_recon"), w_recon)):
                            already_done = True
                            break
                            
                    if already_done:
                        print(f"⏭️ Skipping already evaluated combination: {comb_name}")
                        continue
                        
                    print("\n" + "="*50)
                    print(f"🚀 Training for grid parameters: {comb_name}")
                    
                    run_dir = os.path.join(BASE_RESULT_ROOT, comb_name)
                    os.makedirs(run_dir, exist_ok=True)
                    model_path = os.path.join(run_dir, "best_mtl_model.pth")
                    
                    # 每轮初始化新的模型和优化器
                    model = get_model(task_type='mtl', device=DEVICE, in_feats=3, edge_dim=2, hidden_dim=64, num_classes=5, max_nodes=32)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
                    
                    loss_weights = {
                        'w_chiN': w_chiN,
                        'w_kstar': w_kstar,
                        'w_cls': w_cls,
                        'w_recon': w_recon
                    }
                    
                    trainer = MTLTrainerGrid(model, optimizer, DEVICE, scheduler=scheduler, save_path=model_path, loss_weights=loss_weights)
                    
                    # 这里设定训练 100个Epoch，你可以根据需要调大（如原本的200）以得到更准的结果
                    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=200)

                    # 记录和绘图
                    save_loss_curve(train_hist, val_hist, os.path.join(run_dir, "loss_curve.png"))
                    
                    print("💾 Evaluating best model on test set...")
                    model.load_state_dict(torch.load(model_path))
                    res = evaluate_and_plot_mtl(model, test_loader, DEVICE, run_dir)
                    
                    # 汇总字典
                    record = {
                        "w_chiN": w_chiN,
                        "w_kstar": w_kstar,
                        "w_cls": w_cls,
                        "w_recon": w_recon,
                        "chiN_mape(%)": res['chiN_mape'],
                        "kstar_mape(%)": res['kstar_mape'],
                        "cls_acc(%)": res['cls_acc'],
                        "rec_A_acc(%)": res['rec_A_acc'],
                        "rec_F_mse": res['rec_F_mse'],
                        "rec_E_mse": res['rec_E_mse']
                    }
                    records.append(record)
                    
                    # 增量写入CSV文件
                    pd.DataFrame(records).to_csv(CSV_RECORD_PATH, index=False)
                    print(f"✅ Combination finished. Results appended to {CSV_RECORD_PATH}")

    print("\n🎉 全部的目标组合网格搜索计算完毕！")
    print(f"详细的结果记录在： {CSV_RECORD_PATH}")

if __name__ == "__main__":
    # 使用 numpy 提供的小数容差去重
    import numpy as np
    grid_search_mtl()
