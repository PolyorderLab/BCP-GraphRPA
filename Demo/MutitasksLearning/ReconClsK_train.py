import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add project root to sys.path if needed
sys.path.append("/home/zxy/project/TAPD/src")
from data_utils import load_and_process_dataset
from train_utils import get_splits
from model_utils import get_model
from train_utils import Trainer
from viz_utils import evaluate_and_plot_mtl, save_loss_curve

# Paths and device configuration
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/ReconClsK"
os.makedirs(BASE_RESULT_ROOT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReconClsKTrainer(Trainer):
    """Trainer handling reconstruction, classification and k* regression.

    The model is expected to return a tuple:
        out_kstar, out_cls, (out_A, out_F, out_E)
    where:
        - out_kstar: regression output for k*
        - out_cls: raw class logits for topology classification
        - out_A: reconstructed adjacency matrix (values in [0, 1])
        - out_F: reconstructed node features
        - out_E: reconstructed edge attributes
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()

    def calculate_loss(self, out, batch):
        # unpack model outputs
        out_chiN, out_kstar, out_cls, (out_A, out_F, out_E) = out

        # k* regression loss (log‑scale values, use MSE)
        loss_kstar = self.criterion_mse(out_kstar, batch.k_star)

        # classification loss (topology type)
        loss_cls = self.criterion_ce(out_cls, batch.topo_type)

        # reconstruction losses
        target_adj = batch.dense_adj.view(-1, 32, 32)
        loss_adj = self.criterion_bce(out_A, target_adj)
        target_F = batch.dense_F.view(-1, 32, 3)
        loss_F = self.criterion_mse(out_F, target_F)
        target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
        loss_E = self.criterion_mse(out_E, target_E)

        # total loss – weighted sum (reconstruction weighted lightly)
        total_loss = loss_kstar + loss_cls + 0.1 * (loss_adj + loss_F + loss_E)
        return total_loss

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
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
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.calculate_loss(out, batch)
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

def train_ReconClsK():
    print(f"🖥️ Device: {DEVICE}")

    # Load dataset – use the multi‑task (mtl) type which provides chiN, k*, and reconstruction data
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='mtl')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)

    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # Model – reuse the multi‑task model (outputs chiN, k*, cls, rec). We'll ignore chiN.
    model = get_model(task_type='mtl', device=DEVICE,
                     in_feats=3, edge_dim=2, hidden_dim=64, num_classes=5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    model_path = os.path.join(BASE_RESULT_ROOT, "best_ReconClsK_model.pth")
    trainer = ReconClsKTrainer(model, optimizer, None, DEVICE,
                               scheduler=scheduler, save_path=model_path)
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=200)

    print("\n💾 Training finished, generating evaluation reports and plots...")
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))

    # Load best model for evaluation
    model.load_state_dict(torch.load(model_path))
    # Evaluate using the generic MTL helper – it will compute both chiN and k* metrics.
    res = evaluate_and_plot_mtl(model, test_loader, DEVICE, BASE_RESULT_ROOT)

    report_text = f"""
======================================
Three‑Task: k* Regression, Reconstruction & Topology Classification
======================================
[Physical Regression]
   - k* Total MAPE   : {res['kstar_mape']:.2f}%
[Reconstruction]
   - Adj Accuracy       : {res['rec_A_acc']:.2f}%
   - Feature MSE        : {res['rec_F_mse']:.4f}
   - Edge Attr MSE      : {res['rec_E_mse']:.4f}
[Classification]
   - Accuracy           : {res['cls_acc']:.2f}%
======================================
"""
    print(report_text)
    with open(os.path.join(BASE_RESULT_ROOT, "evaluation_report.txt"), "w") as f:
        f.write(report_text)

    print(f"✨ Evaluation completed. Results saved to {BASE_RESULT_ROOT}")

if __name__ == "__main__":
    train_ReconClsK()
