import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

def get_splits(dataset, data_dir, random_state=42):
    """
    获取或保存固定的数据集切分。
    """
    split_info_path = os.path.join(os.path.dirname(data_dir), "fixed_dataset_splits_shared.pt")

    if os.path.exists(split_info_path):
        print(f"📥 加载统一的数据集划分文件: {split_info_path}")
        splits = torch.load(split_info_path, weights_only=False)
        train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    else:
        print(f"🔨 生成并保存固定的索引切分: {split_info_path}")
        indices = np.arange(len(dataset))
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)
        torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx}, split_info_path)
    
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, save_path="best_model.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.train_hist = []
        self.val_hist = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            # Basic regressor assumes y
            loss = self.criterion(out, batch.y)
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
                loss = self.criterion(out, batch.y)
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    def fit(self, train_loader, val_loader, epochs=200):
        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
            avg_t_loss = self.train_epoch(train_loader)
            avg_v_loss = self.validate(val_loader)
            
            self.train_hist.append(avg_t_loss)
            self.val_hist.append(avg_v_loss)
            
            if self.scheduler:
                self.scheduler.step(avg_v_loss)
                
            if avg_v_loss < self.best_val_loss:
                self.best_val_loss = avg_v_loss
                torch.save(self.model.state_dict(), self.save_path)
        
        return self.train_hist, self.val_hist
