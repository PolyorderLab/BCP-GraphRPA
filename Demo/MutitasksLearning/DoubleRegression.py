import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add project root to sys.path if needed
sys.path.append("/home/zxy/project/TAPD/src")
from data_utils import load_and_process_dataset, get_splits
from model_utils import get_model
from train_utils import Trainer, save_loss_curve
from viz_utils import evaluate_and_plot_dual

# ================= 1. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/test1wDual_refactored"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

class DualTrainer(Trainer):
    """
    Dual specific trainer to handle two heads and custom loss.
    """
    def calculate_loss(self, out, batch):
        criterion_mse = nn.MSELoss()
        
        loss_chiN = criterion_mse(out[0], batch.chiN_star)
        loss_kstar = criterion_mse(out[1], batch.k_star)
        
        return loss_chiN + loss_kstar

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

def train_dual():
    print(f"🖥️ Device: {DEVICE}")
    
    # 1. Load Data
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='dual_regression')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # 2. Initialize Model
    model = get_model(task_type='dual_regression', device=DEVICE, in_feats=3, edge_dim=2, hidden_dim=64)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 3. Train
    model_path = os.path.join(BASE_RESULT_ROOT, "best_dual_model.pth")
    # Custom trainer for Dual
    trainer = DualTrainer(model, optimizer, None, DEVICE, scheduler=scheduler, save_path=model_path)
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=200)

    # 4. Evaluation
    print("\n💾 Training finished, generating evaluation reports and plots...")
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
    model.load_state_dict(torch.load(model_path))
    res = evaluate_and_plot_dual(model, test_loader, DEVICE, BASE_RESULT_ROOT)
    
    report_text = f"""
======================================
Dual Regression (chiN* & k*) Final Evaluation Report
======================================
[Physical Regression]
   - chiN* Total MAPE   : {res['chiN_mape']:.2f}%
   - k* Total MAPE      : {res['kstar_mape']:.2f}%
======================================
"""
    print(report_text)
    with open(os.path.join(BASE_RESULT_ROOT, "evaluation_report.txt"), "w") as f:
        f.write(report_text)
    
    print(f"✨ Dual Task Evaluation completed. Results saved to {BASE_RESULT_ROOT}")

if __name__ == "__main__":
    train_dual()
