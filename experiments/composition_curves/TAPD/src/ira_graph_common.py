import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

sys.path.append("/home/zxy/project/TAPD")
from src.data_utils import load_and_process_dataset
from src.plotting import compute_metrics, plot_parity_log
from src.train_utils import Trainer
from src.viz_utils import save_loss_curve


IRA_GRAPH_RESULT_ROOT = "/home/zxy/project/TAPD/outcome/IRA_graph"
TRAIN_DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/chiNlimited/n_train_100k"
TEST_DATA_DIR = "/home/zxy/project/TAPD/TrainData/data/chiNlimited/n_test_2k"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
LEARNING_RATE = 0.005
BATCH_SIZE = 32
NUM_LAYERS = 5
DEC_LAYERS = 1
EPOCHS = 200


def clone_graph_attrs(src, dst, skip_extra=None):
    skip = {"x", "edge_index", "edge_attr"}
    if skip_extra:
        skip.update(skip_extra)
    for key, value in src:
        if key not in skip:
            setattr(dst, key, value.clone() if torch.is_tensor(value) else value)
    return dst


def unique_undirected_edges(data):
    edge_index = data.edge_index.cpu()
    seen = {}
    for col in range(edge_index.size(1)):
        u = int(edge_index[0, col])
        v = int(edge_index[1, col])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key not in seen:
            seen[key] = col
    return list(seen.items())


def canonical_bidirectional_chain(num_nodes):
    edge_pairs = []
    for node_idx in range(max(0, num_nodes - 1)):
        edge_pairs.append((node_idx, node_idx + 1))
        edge_pairs.append((node_idx + 1, node_idx))
    if edge_pairs:
        return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return torch.empty((2, 0), dtype=torch.long)


def resize_edge_attr(edge_attr, num_edges):
    if edge_attr.size(0) == num_edges:
        return edge_attr.clone()
    if num_edges == 0:
        return torch.empty((0, edge_attr.size(1)), dtype=edge_attr.dtype)
    if edge_attr.size(0) == 0:
        return torch.zeros((num_edges, edge_attr.size(1)), dtype=edge_attr.dtype)
    repeats = int(np.ceil(num_edges / edge_attr.size(0)))
    return edge_attr.repeat((repeats, 1))[:num_edges].clone()


class VirtualNodeWrapper:
    def __init__(self, dataset, node_mode, edge_mode="full", graph_mode="original", shuffle_seed=20260609):
        self.dataset = dataset
        self.node_mode = node_mode
        self.edge_mode = edge_mode
        self.graph_mode = graph_mode
        self.shuffle_seed = shuffle_seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.node_mode == "zero":
            data.x = torch.zeros_like(data.x)
        elif self.node_mode == "one":
            data.x = torch.ones_like(data.x)
        elif self.node_mode == "degree":
            data.x = data.x[:, :1].clone()
        elif self.node_mode == "avg_neighbor_degree":
            data.x = data.x[:, 1:2].clone()
        elif self.node_mode == "degree_avg_neighbor_degree":
            data.x = data.x[:, :2].clone()
        elif self.node_mode != "original":
            raise ValueError(f"Unknown node_mode: {self.node_mode}")

        if self.graph_mode == "original":
            pass
        elif self.graph_mode == "fixed_chain":
            fixed_edge_index = canonical_bidirectional_chain(data.x.size(0))
            data.edge_attr = resize_edge_attr(data.edge_attr, fixed_edge_index.size(1))
            data.edge_index = fixed_edge_index
        else:
            raise ValueError(f"Unknown graph_mode: {self.graph_mode}")

        if self.edge_mode == "full":
            pass
        elif self.edge_mode == "f":
            data.edge_attr = data.edge_attr[:, :1].clone()
        elif self.edge_mode == "type":
            data.edge_attr = data.edge_attr[:, 1:2].clone()
        elif self.edge_mode == "zero":
            data.edge_attr = torch.zeros_like(data.edge_attr)
        elif self.edge_mode == "shuffle":
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed + int(idx))
            perm = torch.randperm(data.edge_attr.size(0), generator=generator)
            data.edge_attr = data.edge_attr[perm].clone()
        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

        num_nodes = data.x.size(0)
        virtual_idx = num_nodes
        virtual_x = torch.zeros((1, data.x.size(1)), dtype=data.x.dtype)
        x = torch.cat([data.x, virtual_x], dim=0)

        node_ids = torch.arange(num_nodes, dtype=torch.long)
        virtual_to_nodes = torch.stack(
            [torch.full((num_nodes,), virtual_idx, dtype=torch.long), node_ids],
            dim=0,
        )
        nodes_to_virtual = torch.stack(
            [node_ids, torch.full((num_nodes,), virtual_idx, dtype=torch.long)],
            dim=0,
        )
        virtual_edges = torch.cat([virtual_to_nodes, nodes_to_virtual], dim=1)
        edge_index = torch.cat([data.edge_index, virtual_edges], dim=1)

        virtual_edge_attr = torch.zeros((virtual_edges.size(1), data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
        edge_attr = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)

        out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return clone_graph_attrs(data, out)


class DirectedGraphWrapper:
    def __init__(self, dataset, node_mode):
        self.dataset = dataset
        self.node_mode = node_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.node_mode == "zero":
            data.x = torch.zeros_like(data.x)
        elif self.node_mode != "original":
            raise ValueError(f"Unknown node_mode: {self.node_mode}")

        edge_cols = []
        edge_pairs = []
        for (u, v), attr_col in unique_undirected_edges(data):
            edge_pairs.append((u, v))
            edge_cols.append(attr_col)

        if edge_pairs:
            data.edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            data.edge_attr = data.edge_attr[edge_cols].clone()
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = torch.empty((0, data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
        return data


class SelfLoopGraphWrapper:
    def __init__(self, dataset, node_mode):
        self.dataset = dataset
        self.node_mode = node_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.node_mode == "zero":
            data.x = torch.zeros_like(data.x)
        elif self.node_mode != "original":
            raise ValueError(f"Unknown node_mode: {self.node_mode}")

        num_nodes = data.x.size(0)
        node_ids = torch.arange(num_nodes, dtype=torch.long)
        self_loops = torch.stack([node_ids, node_ids], dim=0)
        data.edge_index = torch.cat([data.edge_index, self_loops], dim=1)
        loop_attr = torch.zeros((num_nodes, data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
        data.edge_attr = torch.cat([data.edge_attr, loop_attr], dim=0)
        return data


class BipartiteGraphWrapper:
    def __init__(self, dataset, junction_mode, block_mode="full"):
        self.dataset = dataset
        self.junction_mode = junction_mode
        self.block_mode = block_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        undirected = unique_undirected_edges(data)
        if undirected:
            num_junctions = int(data.edge_index.max().item()) + 1
        else:
            num_junctions = min(1, data.x.size(0))

        if self.junction_mode == "zero" or self.junction_mode == "learnable":
            junction_x = torch.zeros((num_junctions, 3), dtype=data.x.dtype)
        elif self.junction_mode == "degree":
            junction_x = data.x[:num_junctions, :3].clone()
        else:
            raise ValueError(f"Unknown junction_mode: {self.junction_mode}")

        block_x = []
        edge_pairs = []
        for block_idx, ((u, v), attr_col) in enumerate(undirected):
            block_node = num_junctions + block_idx
            attr = data.edge_attr[attr_col]
            if self.block_mode == "full":
                block_feat = torch.stack([attr[0], attr[1], torch.ones((), dtype=data.x.dtype)])
            elif self.block_mode == "f":
                block_feat = torch.stack([attr[0], torch.zeros((), dtype=data.x.dtype), torch.ones((), dtype=data.x.dtype)])
            elif self.block_mode == "type":
                block_feat = torch.stack([torch.zeros((), dtype=data.x.dtype), attr[1], torch.ones((), dtype=data.x.dtype)])
            else:
                raise ValueError(f"Unknown block_mode: {self.block_mode}")
            block_x.append(block_feat)
            edge_pairs.extend([(u, block_node), (block_node, u), (v, block_node), (block_node, v)])

        if block_x:
            x = torch.cat([junction_x, torch.stack(block_x, dim=0)], dim=0)
        else:
            x = junction_x

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((edge_index.size(1), 2), dtype=data.edge_attr.dtype)

        out = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            junction_count=torch.tensor([num_junctions], dtype=torch.long),
            block_count=torch.tensor([len(undirected)], dtype=torch.long),
        )
        return clone_graph_attrs(data, out, skip_extra={"junction_count", "block_count"})


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


class BipartiteLearnableJunctionEncoder(nn.Module):
    def __init__(self, max_junctions, block_in_feats=3, edge_dim=2, hidden_dim=256, num_layers=4):
        super().__init__()
        self.junction_embedding = nn.Parameter(torch.randn(max_junctions, hidden_dim) * 0.01)
        self.block_lin = nn.Linear(block_in_feats, hidden_dim)
        self.gine = GINEEncoder(hidden_dim, edge_dim, hidden_dim, num_layers)

    def forward(self, data):
        hidden_parts = []
        offset = 0
        for num_junctions, num_blocks in zip(data.junction_count.tolist(), data.block_count.tolist()):
            hidden_parts.append(self.junction_embedding[:num_junctions])
            block_x = data.x[offset + num_junctions:offset + num_junctions + num_blocks]
            hidden_parts.append(self.block_lin(block_x))
            offset += num_junctions + num_blocks

        data = data.clone()
        data.x = torch.cat(hidden_parts, dim=0)
        return self.gine(data)


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
        f"chiN IRA_graph Experiment: {title}\n"
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
