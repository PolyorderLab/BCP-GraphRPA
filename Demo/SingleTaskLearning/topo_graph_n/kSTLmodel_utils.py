import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_feats=3, edge_dim=2, hidden_dim=64, num_layers=3):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.edge_lin = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            in_d = in_feats if i == 0 else hidden_dim
            out_d = hidden_dim 
            
            # 将边特征映射到与节点特征一致的维度
            self.edge_lin.append(nn.Linear(edge_dim, in_d))
            
            # 【优化】在 GINE 的 MLP 中加入了 BatchNorm1d，提升收敛稳定性和速度
            nn_module = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_d),
                nn.BatchNorm1d(out_d)
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

# 【修改点1】：重命名为 GNNKStarRegressor
class GNNKStarRegressor(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden_dim=64, num_dec_layers=2, use_global_features=True):
        super().__init__()
        self.encoder = encoder 
        self.use_global_features = use_global_features
        
        # 利用我们在 data_utils 提取的全局标量和分类标签
        if self.use_global_features:
            # topo_type 有 5 种分类 (0~4)，映射为一个 16 维的 Embedding 向量
            self.topo_embed = nn.Embedding(num_embeddings=5, embedding_dim=16)
            # 全连接层的输入维度 = 图向量(64) + 拓扑特征(16) + 嵌段数标量(1)
            curr_dim = hidden_dim + 16 + 1 
        else:
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
        # 1. 提取图特征向量 (Graph Embedding)
        graph_embed = self.encoder(data) 
        
        # 2. 融合全局物理特征 (Global Context Fusion)
        if self.use_global_features:
            # 提取拓扑 Embedding
            topo_vec = self.topo_embed(data.topo_type)
            # 提取嵌段数，转换为 float 并对齐形状
            n_b_vec = data.num_blocks.float().view(-1, 1)
            
            # 沿着特征维度拼接：[Batch_size, 64 + 16 + 1]
            graph_embed = torch.cat([graph_embed, topo_vec, n_b_vec], dim=-1)
            
        # 3. 通过 MLP 回归器输出最终预测值 (现在代表 k_star)
        return self.decoder(graph_embed)

def get_model(in_feats=3, edge_dim=2, hidden_dim=64, num_enc_layers=3, num_dec_layers=2, use_global_features=True, device='cpu'):
    encoder = GNNEncoder(in_feats, edge_dim, hidden_dim, num_enc_layers)
    # 【修改点2】：实例化新的 GNNKStarRegressor
    model = GNNKStarRegressor(encoder, hidden_dim=hidden_dim, num_dec_layers=num_dec_layers, use_global_features=use_global_features)
    return model.to(device)