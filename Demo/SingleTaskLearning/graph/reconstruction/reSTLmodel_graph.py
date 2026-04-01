import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_feats=3, edge_dim=2, hidden_dim=64, num_layers=4):
        super(GNNEncoder, self).__init__()
        
        # 统一映射到 hidden_dim 空间，彻底解决维度不对齐问题
        self.node_proj = nn.Linear(in_feats, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(num_layers):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.convs.append(GINEConv(nn_module))
            
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 将节点特征和边特征映射到同一维度
        x = self.node_proj(x)
        edge_attr_mapped = self.edge_proj(edge_attr)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr_mapped)
            if i != self.num_layers - 1:
                x = self.relu(x)
                
        # 获得全局图表征
        x_pooled = global_mean_pool(x, batch) 
        return x_pooled

# ================= 2. 图全息重构单任务模型 =================
class GNNReconstructionModel(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden_dim=64, max_nodes=32):
        super().__init__()
        self.encoder = encoder 
        self.max_nodes = max_nodes
        curr_dim = hidden_dim  
        
        # 2.1 重构邻接矩阵 A (预测图的连通性, 0或1)
        self.rec_A = nn.Sequential(
            nn.Linear(curr_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, max_nodes * max_nodes),
            nn.Sigmoid() # Sigmoid 将输出压缩到 0-1 之间，用于 BCE Loss
        )
        
        # 2.2 重构节点特征 F (预测节点的度数等物理量)
        self.rec_F = nn.Sequential(
            nn.Linear(curr_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, max_nodes * 3) 
        )
        
        # 2.3 重构边特征 E (预测连接两端嵌段的长短和化学属性)
        self.rec_E = nn.Sequential(
            nn.Linear(curr_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, max_nodes * max_nodes * 2) 
        )
    
    def forward(self, data):
        # 1. 编码：将变长的图压缩为一个固定长度的隐向量
        graph_embed = self.encoder(data) 
        
        # 2. 解码：从隐向量恢复底层的矩阵信息
        out_A = self.rec_A(graph_embed).view(-1, self.max_nodes, self.max_nodes)
        out_F = self.rec_F(graph_embed).view(-1, self.max_nodes, 3)
        out_E = self.rec_E(graph_embed).view(-1, self.max_nodes, self.max_nodes, 2)
        
        return out_A, out_F, out_E

# ================= 3. 工厂函数 =================
def get_model(in_feats=3, edge_dim=2, hidden_dim=64, num_enc_layers=4, max_nodes=32, device='cpu'):
    encoder = GNNEncoder(in_feats, edge_dim, hidden_dim, num_enc_layers)
    model = GNNReconstructionModel(encoder, hidden_dim=hidden_dim, max_nodes=max_nodes)
    return model.to(device)