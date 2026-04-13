import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
import json
from torch_geometric.loader import DataLoader
from datetime import datetime

# ================= 1. Project Path Setup =================
sys.path.append("/home/zxy/project/TAPD")
sys.path.append("/home/zxy/project/TAPD/src")
from src.data_utils import load_and_process_dataset
from src.model_utils import get_model
from src.train_utils import Trainer, get_splits
from src.viz_utils import evaluate_and_plot_mtl, save_loss_curve

# ================= 2. Config =================
DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/train_nnumber_1w"
BASE_RESULT_ROOT = "/home/zxy/project/TAPD/Demo/outcome/graph_trainn/MTL_Optuna_Ablation"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOBAL_STATS_PATH = os.path.join(BASE_RESULT_ROOT, "global_stats.json")
os.makedirs(BASE_RESULT_ROOT, exist_ok=True)

# Default Epochs & Trials (Adjustable via CLI or config)
EPOCHS_PER_TRIAL = 50
TRIALS_PER_COMB = 20

# ================= 3. Global Stats Manager =================
class GlobalStats:
    """Tracks min/max of metrics across all trials to allow normalization."""
    def __init__(self, path):
        self.path = path
        self.stats = self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                return json.load(f)
        return {
            "mape": {"min": 1e9, "max": 0},
            "rmse": {"min": 1e9, "max": 0},
            "r2":   {"min": 1.0, "max": -1.0} # R2: max is best (1.0)
        }

    def update(self, mape, rmse, r2):
        mape, rmse, r2 = float(mape), float(rmse), float(r2)
        changed = False
        if mape < self.stats["mape"]["min"]: self.stats["mape"]["min"] = mape; changed = True
        if mape > self.stats["mape"]["max"]: self.stats["mape"]["max"] = mape; changed = True
        if rmse < self.stats["rmse"]["min"]: self.stats["rmse"]["min"] = rmse; changed = True
        if rmse > self.stats["rmse"]["max"]: self.stats["rmse"]["max"] = rmse; changed = True
        if r2   > self.stats["r2"]["max"]:   self.stats["r2"]["max"]   = r2;   changed = True
        if r2   < self.stats["r2"]["min"]:   self.stats["r2"]["min"]   = r2;   changed = True
        
        if changed:
            with open(self.path, 'w') as f:
                json.dump(self.stats, f, indent=4)

    def normalize(self, mape, rmse, r2):
        s = self.stats
        # Norm MAPE: (val - min) / (max - min) -> lower is better (0)
        mape_n = (mape - s["mape"]["min"]) / (s["mape"]["max"] - s["mape"]["min"] + 1e-9)
        # Norm RMSE: (val - min) / (max - min) -> lower is better (0)
        rmse_n = (rmse - s["rmse"]["min"]) / (s["rmse"]["max"] - s["rmse"]["min"] + 1e-9)
        # Norm R2: (max - val) / (max - min) -> higher val is better, so (max-val) makes lower better (0)
        r2_n = (s["r2"]["max"] - r2) / (s["r2"]["max"] - s["r2"]["min"] + 1e-9)
        
        # Clip to [0, 1]
        mape_n = np.clip(mape_n, 0, 1)
        rmse_n = np.clip(rmse_n, 0, 1)
        r2_n   = np.clip(r2_n, 0, 1)
        
        return mape_n, rmse_n, r2_n

global_stats = GlobalStats(GLOBAL_STATS_PATH)

# ================= 4. Modified Trainer =================
class MTLTrainerOptuna(Trainer):
    def __init__(self, model, optimizer, device, scheduler, save_path, loss_weights):
        super().__init__(model, optimizer, None, device, scheduler=scheduler, save_path=save_path)
        self.loss_weights = loss_weights

    def calculate_loss(self, out, batch):
        w_chiN = self.loss_weights.get('w_chiN', 0)
        w_kstar = self.loss_weights.get('w_kstar', 0)
        w_cls = self.loss_weights.get('w_cls', 0)
        w_recon = self.loss_weights.get('w_recon', 0)
        
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()
        
        loss = 0
        if w_chiN > 0:
            loss += w_chiN * criterion_mse(out[0], batch.chiN_star)
        if w_kstar > 0:
            loss += w_kstar * criterion_mse(out[1], batch.k_star)
        if w_cls > 0:
            loss += w_cls * criterion_ce(out[2], batch.topo_type)
        if w_recon > 0:
            out_A, out_F, out_E = out[3]
            target_A = batch.dense_adj.view(-1, 32, 32)
            target_F = batch.dense_F.view(-1, 32, 3)
            target_E = batch.dense_edge_attr.view(-1, 32, 32, 2)
            loss_A = nn.functional.binary_cross_entropy(out_A, target_A)
            loss_F = criterion_mse(out_F, target_F)
            loss_E = criterion_mse(out_E, target_E)
            loss += w_recon * (loss_A + loss_F + loss_E)
            
        return loss

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

# ================= 5. Objective Function =================
def run_trial(trial, comb_name, active_tasks, loaders):
    train_loader, val_loader, test_loader = loaders
    
    # Suggest weights for active tasks
    loss_weights = {}
    for task in ['chiN', 'kstar', 'cls', 'recon']:
        if task in active_tasks:
            loss_weights[f'w_{task}'] = trial.suggest_float(f'w_{task}', 0.01, 100.0, log=True)
        else:
            loss_weights[f'w_{task}'] = 0.0

    # Model Init
    model = get_model(task_type='mtl', device=DEVICE, in_feats=3, edge_dim=2, hidden_dim=64, num_classes=5, max_nodes=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    save_dir = os.path.join(BASE_RESULT_ROOT, comb_name, f"trial_{trial.number}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pth")
    
    trainer = MTLTrainerOptuna(model, optimizer, DEVICE, scheduler, model_path, loss_weights)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS_PER_TRIAL)
    
    # Evaluation
    model.load_state_dict(torch.load(model_path))
    res = evaluate_and_plot_mtl(model, test_loader, DEVICE, save_dir)
    
    # Identify target tasks for scoring
    target_tasks = []
    if 'chiN' in active_tasks: target_tasks.append('chiN')
    if 'kstar' in active_tasks: target_tasks.append('kstar')
    
    if not target_tasks: # fallback if none (shouldn't happen)
        return res['chiN_mape'] 

    # Calculate Score
    scores = []
    for t in target_tasks:
        mape, rmse, r2 = res[f'{t}_mape'], res[f'{t}_rmse'], res[f'{t}_r2']
        global_stats.update(mape, rmse, r2)
        m_n, rs_n, r2_n = global_stats.normalize(mape, rmse, r2)
        score = np.sqrt(m_n**2 + rs_n**2 + r2_n**2)
        scores.append(score)
        
        # Log to trial
        trial.set_user_attr(f"{t}_mape", float(mape))
        trial.set_user_attr(f"{t}_rmse", float(rmse))
        trial.set_user_attr(f"{t}_r2", float(r2))
        trial.set_user_attr(f"{t}_score", float(score))

    final_score = np.mean(scores)
    return float(final_score)

# ================= 6. Main Execution =================
def main():
    print(f"🚀 Starting MTL Ablation & Optuna exploration on {DEVICE}")
    
    # Load Data
    full_dataset = load_and_process_dataset(DATA_DIR, task_type='mtl')
    tr_data, val_data, test_data = get_splits(full_dataset, DATA_DIR)
    loaders = (
        DataLoader(tr_data, batch_size=64, shuffle=True),
        DataLoader(val_data, batch_size=64),
        DataLoader(test_data, batch_size=64)
    )

    combinations = [
        {"name": "ALL", "tasks": ["chiN", "kstar", "cls", "recon"]},
        {"name": "Recon_2Reg", "tasks": ["chiN", "kstar", "recon"]},
        {"name": "Cls_2Reg", "tasks": ["chiN", "kstar", "cls"]},
        {"name": "Recon_Cls_ChiN", "tasks": ["chiN", "cls", "recon"]},
        {"name": "Recon_Cls_K", "tasks": ["kstar", "cls", "recon"]},
        {"name": "Recon_ChiN", "tasks": ["chiN", "recon"]},
        {"name": "Recon_K", "tasks": ["kstar", "recon"]},
        {"name": "Cls_ChiN", "tasks": ["chiN", "cls"]},
        {"name": "Cls_K", "tasks": ["kstar", "cls"]},
    ]

    summary_records = []

    for comb in combinations:
        print(f"\n" + "="*60)
        print(f"🧬 Evaluating Combination: {comb['name']} (Tasks: {comb['tasks']})")
        
        study_name = f"study_{comb['name']}"
        db_path = f"sqlite:///{os.path.join(BASE_RESULT_ROOT, study_name)}.db"
        study = optuna.create_study(
            study_name=study_name,
            storage=db_path,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(lambda trial: run_trial(trial, comb['name'], comb['tasks'], loaders), n_trials=TRIALS_PER_COMB)
        
        best_trial = study.best_trial
        print(f"🏆 Best trial for {comb['name']}: Score {best_trial.value:.4f}")
        
        record = {
            "Combination": comb["name"],
            "Best_Score": best_trial.value,
            "Best_Weights": str(best_trial.params)
        }
        for attr, val in best_trial.user_attrs.items():
            record[attr] = val
        summary_records.append(record)
        
        # Save incremental summary
        pd.DataFrame(summary_records).to_csv(os.path.join(BASE_RESULT_ROOT, "ablation_summary.csv"), index=False)

    print("\n" + "="*60)
    print("✅ All evaluations complete!")
    print(f"Results saved to {BASE_RESULT_ROOT}")

if __name__ == "__main__":
    main()
