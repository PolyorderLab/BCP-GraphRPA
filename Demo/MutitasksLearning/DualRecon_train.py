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
from viz_utils import evaluate_and_plot_dual_recon

DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/test1w_DualRecon_refactored"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

class DualReconTrainer(Trainer):
    def calculate_loss(self, out, batch):
        criterion_mse = nn.MSELoss()
        
        out_chiN, out_kstar, (out_A, out_F, out_E) = out
        
        loss_chiN = criterion_mse(out_chiN, batch.chiN_star)
        loss_kstar = criterion_mse(out_kstar, batch.k_star)
        
        target_A = batch.dense_adj.view(-1, 32, 32)
        target_F = batch.dense_F.view(-1, 32, 3)
        target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)

        loss_A = nn.functional.binary_cross_entropy(out_A, target_A)
        loss_F = criterion_mse(out_F, target_F)
        loss_E = criterion_mse(out_E, target_E)
        
        return loss_chiN + loss_kstar + 0.1*(loss_A + loss_F + loss_E)

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

def train_dual_recon():
    print(f"🖥️ Device: {DEVICE}")
    
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='dual_recon')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    model = get_model(task_type='dual_recon', device=DEVICE, in_feats=3, edge_dim=2, hidden_dim=64, max_nodes=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    model_path = os.path.join(BASE_RESULT_ROOT, "best_DualRecon_model.pth")
    trainer = DualReconTrainer(model, optimizer, None, DEVICE, scheduler=scheduler, save_path=model_path)
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=200)

    print("\n💾 Training finished, generating evaluation reports and plots...")
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
    model.load_state_dict(torch.load(model_path))
    res = evaluate_and_plot_dual_recon(model, test_loader, DEVICE, BASE_RESULT_ROOT)
    
    report_text = f"""
======================================
Three-Task: chiN* & k* Regression and Graph Reconstruction Final Evaluation Report
======================================
[Physical Regression]
   - chiN* Total MAPE   : {res['chiN_mape']:.2f}%
   - k* Total MAPE      : {res['kstar_mape']:.2f}%

[Graph Reconstruction]
   - Connectivity (A)   : {res['rec_A_acc']:.2f}%
   - Node Feats (F) MSE : {res['rec_F_mse']:.4f}
   - Edge Attr (E) MSE  : {res['rec_E_mse']:.4f}
======================================
"""
    print(report_text)
    with open(os.path.join(BASE_RESULT_ROOT, "evaluation_report.txt"), "w") as f:
        f.write(report_text)
    
    print(f"✨ Three-Task Evaluation completed. Results saved to {BASE_RESULT_ROOT}")

if __name__ == "__main__":
    train_dual_recon()
