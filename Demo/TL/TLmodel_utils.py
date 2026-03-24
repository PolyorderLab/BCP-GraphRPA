import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_feats=3, edge_dim=2, hidden_dim=64, latent_dim=32, num_layers=3):
        """
        注意：in_feats 已经修改为 3，以匹配你 (24, 3) 的数据结构。
        """
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            in_d = in_feats if i == 0 else hidden_dim
            out_d = hidden_dim 
            
            nn_module = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_d)
            )
            
            # 💡 核心修复：直接把 edge_dim 传给 GINEConv，让官方底层自动处理维度对齐！
            # 我们直接删除了笨重的 self.edge_lin 逻辑
            self.convs.append(GINEConv(nn_module, edge_dim=edge_dim))
            
        self.relu = nn.ReLU()
        
        # VAE 潜在空间映射头 (mu 和 log_var)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 消息传递
        for i, conv in enumerate(self.convs):
            # 现在只需要直接把原始的 edge_attr 扔进去就行了
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i != self.num_layers - 1:
                x = self.relu(x)
                
        # 全局池化，获得图级别嵌入
        x_pool = global_mean_pool(x, batch) 
        
        # 映射到潜在分布参数
        mu = self.fc_mu(x_pool)
        logvar = self.fc_logvar(x_pool)
        
        return mu, logvar



class GraphTopologyAndEdgeDecoder(nn.Module):
    def __init__(self, latent_dim=32, max_nodes=24, hidden_dim=128):
        """
        Decodes the latent vector into an Adjacency Matrix and an Edge Attribute Matrix.
        """
        super().__init__()
        self.max_nodes = max_nodes
        
        # 1. Topology Decoder (Predicts existence of an edge)
        self.adj_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, max_nodes * max_nodes * 1),
            nn.Sigmoid() # Bound probabilities between 0 and 1
        )
        
        # 2. Edge Attribute Decoder (Predicts volume fraction f and chemical type A/B)
        self.edge_attr_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, max_nodes * max_nodes * 2),
            nn.Sigmoid() # f and type are both bounded between 0 and 1
        )
        
    def forward(self, z):
        # Decode and reshape adjacency matrix: [B, N_max, N_max]
        adj_flat = self.adj_decoder(z)
        adj_matrix = adj_flat.view(-1, self.max_nodes, self.max_nodes)
        
        # Enforce physical symmetry for undirected polymer graphs
        adj_matrix = (adj_matrix + adj_matrix.transpose(1, 2)) / 2.0
        
        # Decode and reshape edge attributes: [B, N_max, N_max, 2]
        attr_flat = self.edge_attr_decoder(z)
        attr_matrix = attr_flat.view(-1, self.max_nodes, self.max_nodes, 2)
        
        # Enforce physical symmetry for edge attributes
        attr_matrix = (attr_matrix + attr_matrix.transpose(1, 2)) / 2.0
        
        return adj_matrix, attr_matrix


class GraphVAE(nn.Module):
    def __init__(self, encoder: GNNEncoder, decoder: GraphTopologyAndEdgeDecoder):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, data):
        # 1. Encode graph into latent distribution parameters
        mu, logvar = self.encoder(data)
        
        # 2. Sample latent vector z using reparameterization
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode z into structural and chemical matrices
        pred_adj, pred_edge_attr = self.decoder(z)
        
        return pred_adj, pred_edge_attr, mu, logvar


# ========= 注意同步修改工厂函数中的默认参数 =========

def get_model(in_feats=3, edge_dim=2, hidden_dim=64, latent_dim=32, max_nodes=24, num_enc_layers=3, device='cpu'):
    """
    初始化 Graph VAE 模型。默认 in_feats 改为 3。
    """
    encoder = GNNEncoder(in_feats, edge_dim, hidden_dim, latent_dim, num_enc_layers)
    decoder = GraphTopologyAndEdgeDecoder(latent_dim, max_nodes, hidden_dim=128)
    # 假设 GraphVAE 类的定义保持不变
    model = GraphVAE(encoder, decoder)
    
    return model.to(device)