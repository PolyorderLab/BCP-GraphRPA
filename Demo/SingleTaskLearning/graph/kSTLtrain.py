import os
import sys
import pickle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add project root to sys.path if needed
sys.path.append("src")
from data_utils import load_and_process_dataset
from model_utils import get_model, get_desc_model
from desc_utils import DESCRIPTOR_NAMES as DEFAULT_DESCRIPTOR_NAMES
from train_utils import Trainer, get_splits
from viz_utils import save_parity_plot, save_loss_curve

# ================= 1. 配置路径与全局参数 =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/k1wSTL_refactored"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

def train_k(use_desc=False, desc_names=None):
    print(f"🖥️ Device: {DEVICE}")

    if use_desc and desc_names is None:
        desc_names = DEFAULT_DESCRIPTOR_NAMES

    # 1. Load Data
    if use_desc:
        full_dataset, desc_scaler = load_and_process_dataset(
            DATA_DIR, task_type='k', use_desc=True, desc_names=desc_names
        )
        desc_dim = full_dataset[0].desc.shape[1]
    else:
        full_dataset = load_and_process_dataset(DATA_DIR, task_type='k')
        desc_scaler = None

    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    
    train_loader = DataLoader(tr_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # 2. Initialize Model
    if use_desc:
        model = get_desc_model(
            task_type='k', device=DEVICE, in_feats=3, edge_dim=2, hidden_dim=64,
            use_desc=True, desc_dim=desc_dim,
        )
    else:
        model = get_model(task_type='k', device=DEVICE, in_feats=3, edge_dim=2, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    # 3. Train
    model_path = os.path.join(BASE_RESULT_ROOT, "best_model.pth")
    trainer = Trainer(model, optimizer, criterion, DEVICE, scheduler=scheduler, save_path=model_path)
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=200)

    # 4. Evaluation
    save_loss_curve(train_hist, val_hist, os.path.join(BASE_RESULT_ROOT, "loss_curve.png"))
    model.load_state_dict(torch.load(model_path))
    final_mape = save_parity_plot(model, test_loader, os.path.join(BASE_RESULT_ROOT, "parity_plot.png"), DEVICE, title=r"$k^*$ Prediction")
    
    print(f"\n✨ Final Test MAPE: {final_mape:.2f}%")

    if use_desc and desc_scaler is not None:
        scaler_path = os.path.join(BASE_RESULT_ROOT, "topo_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(desc_scaler, f)

if __name__ == "__main__":
    train_k()
