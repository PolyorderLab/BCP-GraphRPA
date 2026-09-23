import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TAPD_DATA_ROOT", PACKAGE_ROOT / "data"))
sys.path.append(str(PACKAGE_ROOT))
from src.data_utils import load_and_process_dataset
from src.plotting import compute_metrics, plot_parity_log
from src.train_utils import Trainer
from src.viz_utils import save_loss_curve


IRA_RESULT_ROOT = str(PACKAGE_ROOT / "results" / "ira")
TRAIN_DATA_DIR = str(DATA_ROOT / "chiNlimited/n_train_100k")
TEST_DATA_DIR = str(DATA_ROOT / "chiNlimited/n_test_2k_leakage_free_seed42")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUM_LAYERS = 4
DEC_LAYERS = 1
EPOCHS = 200


def extract_predictions(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out[0].cpu().numpy() if isinstance(out, (tuple, list)) else out.cpu().numpy()
            trues = batch.y.cpu().numpy()
            all_preds.append(preds.reshape(-1))
            all_trues.append(trues.reshape(-1))
    y_pred_log = np.concatenate(all_preds)
    y_true_log = np.concatenate(all_trues)
    return 10 ** y_pred_log, 10 ** y_true_log


def clone_scalar_attrs(src, dst):
    for key, value in src:
        if key not in {"x", "edge_index", "edge_attr"}:
            setattr(dst, key, value.clone() if torch.is_tensor(value) else value)
    return dst


class ConstantNodeFeatureWrapper:
    def __init__(self, dataset, constant):
        self.dataset = dataset
        self.constant = float(constant)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        data.x = torch.full_like(data.x, self.constant)
        return data


class EdgeTypeOnlyWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        data.edge_attr = data.edge_attr[:, 1:2].clone()
        return data


def to_line_graph(data, use_edge_attr_as_x=True, use_node_attr_as_edge_attr=True):
    data = data.clone()
    undirected_edges = {}
    edge_index = data.edge_index.cpu()

    for col in range(edge_index.size(1)):
        u = int(edge_index[0, col])
        v = int(edge_index[1, col])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key not in undirected_edges:
            undirected_edges[key] = col

    edges = list(undirected_edges.keys())
    if not edges:
        x_dim = data.edge_attr.size(1) if use_edge_attr_as_x and data.edge_attr is not None else data.x.size(1)
        edge_dim = data.x.size(1)
        out = Data(
            x=torch.zeros((1, x_dim), dtype=data.x.dtype, device=data.x.device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device),
            edge_attr=torch.empty((0, edge_dim), dtype=data.x.dtype, device=data.x.device),
        )
        return clone_scalar_attrs(data, out)

    if use_edge_attr_as_x:
        new_x = torch.stack([data.edge_attr[undirected_edges[e]] for e in edges], dim=0)
    else:
        new_x = torch.ones((len(edges), 1), dtype=data.x.dtype, device=data.x.device)

    line_edges = []
    line_edge_attrs = []
    for i, (u1, v1) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]
            shared = set((u1, v1)).intersection((u2, v2))
            if not shared:
                continue
            shared_node = next(iter(shared))
            line_edges.extend([(i, j), (j, i)])
            if use_node_attr_as_edge_attr:
                shared_attr = data.x[shared_node].clone()
                line_edge_attrs.extend([shared_attr, shared_attr.clone()])

    if line_edges:
        new_edge_index = torch.tensor(line_edges, dtype=torch.long, device=data.edge_index.device).t().contiguous()
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device)

    if use_node_attr_as_edge_attr:
        if line_edge_attrs:
            new_edge_attr = torch.stack(line_edge_attrs, dim=0)
        else:
            new_edge_attr = torch.empty((0, data.x.size(1)), dtype=data.x.dtype, device=data.x.device)
    else:
        new_edge_attr = torch.empty((new_edge_index.size(1), 0), dtype=data.x.dtype, device=data.x.device)

    out = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    return clone_scalar_attrs(data, out)


class LineGraphWrapper:
    def __init__(self, dataset, include_line_edge_attr):
        self.dataset = dataset
        self.include_line_edge_attr = include_line_edge_attr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return to_line_graph(
            self.dataset[idx],
            use_edge_attr_as_x=True,
            use_node_attr_as_edge_attr=self.include_line_edge_attr,
        )


class GINEEncoder(nn.Module):
    def __init__(self, in_feats, edge_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.edge_lin = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            in_d = in_feats if i == 0 else hidden_dim
            self.edge_lin.append(nn.Linear(edge_dim, in_d))
            nn_module = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            self.convs.append(GINEConv(nn_module))
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for i, conv in enumerate(self.convs):
            edge_attr_mapped = self.edge_lin[i](edge_attr)
            x = conv(x, edge_index, edge_attr=edge_attr_mapped)
            if i != self.num_layers - 1:
                x = self.relu(x)
        return global_mean_pool(x, batch)


class LearnableNodeGINEEncoder(nn.Module):
    def __init__(self, max_nodes, edge_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(max_nodes, hidden_dim) * 0.01)
        self.gine = GINEEncoder(
            in_feats=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, data):
        node_counts = (data.ptr[1:] - data.ptr[:-1]).tolist()
        data = data.clone()
        data.x = torch.cat([self.node_embedding[:num_nodes] for num_nodes in node_counts], dim=0)
        return self.gine(data)


class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_dim=256, num_layers=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            in_d = in_feats if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_d, hidden_dim))
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.relu(x)
        return global_mean_pool(x, batch)


class SingleTaskRegressor(nn.Module):
    def __init__(self, encoder, hidden_dim=256, num_dec_layers=1):
        super().__init__()
        self.encoder = encoder
        layers = []
        curr_dim = hidden_dim
        for _ in range(num_dec_layers - 1):
            layers += [nn.Linear(curr_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, data):
        return self.decoder(self.encoder(data))


def load_splits():
    full_train_dataset = load_and_process_dataset(TRAIN_DATA_DIR, task_type="chiN")
    test_dataset = load_and_process_dataset(TEST_DATA_DIR, task_type="chiN")
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=0.2,
        random_state=42,
    )
    return Subset(full_train_dataset, train_idx), Subset(full_train_dataset, val_idx), test_dataset


def train_and_evaluate(model, train_dataset, val_dataset, test_dataset, result_dir, title, report_note):
    os.makedirs(result_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    model_path = os.path.join(result_dir, "best_model.pth")
    trainer = Trainer(model, optimizer, criterion, DEVICE, scheduler=scheduler, save_path=model_path)
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    save_loss_curve(train_hist, val_hist, os.path.join(result_dir, "loss_curve.png"))

    model.load_state_dict(torch.load(model_path, weights_only=True))
    y_pred, y_true = extract_predictions(model, test_loader, DEVICE)
    metrics = compute_metrics(y_true, y_pred)

    plot_parity_log(
        y_true,
        y_pred,
        os.path.join(result_dir, "parity_plot_nature"),
        title=title,
        xlabel=r"True $\chi N^*$",
        ylabel=r"Predicted $\chi N^*$",
        stats_metrics=metrics,
        color="#0F4D92",
        figsize=(5, 5),
    )

    report = (
        f"chiN IRA Experiment: {title}\n"
        f"{'='*45}\n"
        f"Hyperparameters:\n"
        f"  - hidden_dim: {HIDDEN_DIM}\n"
        f"  - learning_rate: {LEARNING_RATE}\n"
        f"  - batch_size: {BATCH_SIZE}\n"
        f"  - num_layers: {NUM_LAYERS}, dec_layers: {DEC_LAYERS}\n"
        f"  - epochs: {EPOCHS}\n"
        f"{'='*45}\n"
        f"Test Set Results:\n"
        f"  - MAPE : {metrics['MAPE']:.2f}%\n"
        f"  - RMSE : {metrics['RMSE']:.2f}\n"
        f"  - MAE  : {metrics['MAE']:.2f}\n"
        f"  - R2   : {metrics['R²']:.4f}\n"
        f"{'='*45}\n"
        f"{report_note}\n"
    )
    print(report)
    with open(os.path.join(result_dir, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(result_dir, "parity_plot_nature_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"{'='*40}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    return metrics
