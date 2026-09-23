"""Minimal neural-network definitions required to load the final C2 weights."""

import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class GINEEncoder(nn.Module):
    """Five-layer edge-aware GIN encoder used by the saved checkpoints."""

    def __init__(self, in_feats, edge_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.edge_lin = nn.ModuleList()
        self.num_layers = num_layers
        for layer_index in range(num_layers):
            in_dim = in_feats if layer_index == 0 else hidden_dim
            self.edge_lin.append(nn.Linear(edge_dim, in_dim))
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            self.convs.append(GINEConv(mlp))
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x
        for layer_index, conv in enumerate(self.convs):
            edge_attr = self.edge_lin[layer_index](data.edge_attr)
            x = conv(x, data.edge_index, edge_attr=edge_attr)
            if layer_index != self.num_layers - 1:
                x = self.relu(x)
        return global_mean_pool(x, data.batch)


class SingleTaskRegressor(nn.Module):
    """Encoder plus the one-layer scalar decoder used for log10(chi N*)."""

    def __init__(self, encoder, hidden_dim=256, num_dec_layers=1):
        super().__init__()
        self.encoder = encoder
        layers = []
        current_dim = hidden_dim
        for _ in range(num_dec_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, data):
        return self.decoder(self.encoder(data))
