import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_feats=4, edge_dim=2, hidden_dim=64, num_layers=3):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.edge_lin = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            in_d = in_feats if i == 0 else hidden_dim
            out_d = hidden_dim 
            
            # 将边特征映射到与节点特征一致的维度
            self.edge_lin.append(nn.Linear(edge_dim, in_d))
            
            nn_module = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_d)
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
        x = global_mean_pool(x, batch) 
        return x

class GNNChiNRegressor(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden_dim=64, num_dec_layers=2):
        super().__init__()
        self.encoder = encoder 
        
        # 直接使用 hidden_dim 作为输入维度，避开对 encoder 内部结构的索引访问
        curr_dim = hidden_dim 
        
        layers = []
        for _ in range(num_dec_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        
        layers.append(nn.Linear(curr_dim, 1))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, data):
        graph_embed = self.encoder(data) 
        return self.decoder(graph_embed)

def get_model(in_feats=4, edge_dim=2, hidden_dim=64, num_enc_layers=3, num_dec_layers=2, device='cpu'):
    encoder = GNNEncoder(in_feats, edge_dim, hidden_dim, num_enc_layers)
    model = GNNChiNRegressor(encoder, hidden_dim=hidden_dim, num_dec_layers=num_dec_layers)
    return model.to(device)