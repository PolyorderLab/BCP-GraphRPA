import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

sys.path.append("/home/zxy/project/TAPD")
from src.data_utils import load_and_process_dataset
from src.model_utils import get_model
from src.train_utils import Trainer, get_splits
from src.viz_utils import save_confusion_matrix, save_loss_curve

# ================= 1. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/ClsSTL_1w_refactored"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

class ClsTrainer(Trainer):
    """
    Classification specific trainer to handle CrossEntropyLoss
    """
    def calculate_loss(self, out, batch):
        criterion_ce = nn.CrossEntropyLoss()
        return criterion_ce(out, batch.topo_type.flatten())

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

def evaluate_classification(model, loader):
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out_cls = model(batch)
            
            pred_classes = torch.argmax(out_cls, dim=1).cpu().numpy()
            true_classes = batch.topo_type.flatten().cpu().numpy()
            
            all_preds.extend(pred_classes)
            all_trues.extend(true_classes)
            
    acc = accuracy_score(all_trues, all_preds) * 100
    save_confusion_matrix(all_trues, all_preds, os.path.join(BASE_RESULT_ROOT, "confusion_matrix.png"))
    
    return acc

def train_classification_model():
    print(f"🖥️ Device: {DEVICE}")
    
    # 1. Load Data
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='cls')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # 2. Initialize Model
    model = get_model(task_type='cls', in_feats=3, edge_dim=2, hidden_dim=64, num_classes=5, device=DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 3. Train
    model_path = os.path.join(BASE_RESULT_ROOT, "best_cls_model.pth")
    trainer = ClsTrainer(model, optimizer, None, DEVICE, scheduler=scheduler, save_path=model_path)
    
    print(f"\n🔥 开始训练单任务拓扑分类模型 (设定总计 200 Epoch)...")
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=200)

    # 4. Evaluation
    print("\n💾 Training finished, generating evaluation reports and plots...")
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    
    model.load_state_dict(torch.load(model_path))
    acc = evaluate_classification(model, test_loader)
    
    report_text = f"""
======================================
单任务拓扑分类 (Topology Classification) 最终评估报告 
======================================
- 拓扑分类准确度 (Accuracy) : {acc:.2f}%
======================================
"""
    print(report_text)
    
    with open(os.path.join(BASE_RESULT_ROOT, "classification_evaluation.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    train_classification_model()
