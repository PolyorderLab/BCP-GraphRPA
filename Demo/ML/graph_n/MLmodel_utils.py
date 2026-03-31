import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_feats=3, edge_dim=2, hidden_dim=64, num_layers=4):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.edge_lin = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            in_d = in_feats if i == 0 else hidden_dim
            out_d = hidden_dim 
            
            self.edge_lin.append(nn.Linear(edge_dim, in_d))
            
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
                
        x_pooled = global_mean_pool(x, batch) 
        return x_pooled

class GNNMultiTaskModel(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden_dim=64, num_classes=5, max_nodes=32):
        super().__init__()
        self.encoder = encoder 
        self.max_nodes = max_nodes
        
        # 共享特征维度 = GNN图特征(64) + 嵌段数(1)
        curr_dim = hidden_dim + 1  
        
        # ================= Task 1-3: 回归与分类 =================
        self.head_chiN = nn.Sequential(
            nn.Linear(curr_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.head_kstar = nn.Sequential(
            nn.Linear(curr_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.head_cls = nn.Sequential(
            nn.Linear(curr_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes) 
        )
        
        # ================= Task 4: 真正的图全息重构 =================
        # 4.1 重构邻接矩阵 A (形状: 32 x 32) - 预测有无连接 (0或1)
        self.rec_A = nn.Sequential(
            nn.Linear(curr_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, max_nodes * max_nodes),
            nn.Sigmoid() 
        )
        
        # 4.2 重构节点特征 F (形状: 32 x 3) - 预测节点度数等连续值
        self.rec_F = nn.Sequential(
            nn.Linear(curr_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, max_nodes * 3) 
        )
        
        # 4.3 重构边特征 E (形状: 32 x 32 x 2) - 预测长度 f_val 和类型 type_val
        self.rec_E = nn.Sequential(
            nn.Linear(curr_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, max_nodes * max_nodes * 2) 
        )
    
    def forward(self, data):
        graph_embed = self.encoder(data) 
        n_b_vec = data.num_blocks.float().view(-1, 1)
        shared_rep = torch.cat([graph_embed, n_b_vec], dim=-1)
        
        # 前三个任务输出
        out_chiN = self.head_chiN(shared_rep)
        out_kstar = self.head_kstar(shared_rep)
        out_cls = self.head_cls(shared_rep)
        
        # 重构任务输出，Reshape 成真实的矩阵形状
        out_A = self.rec_A(shared_rep).view(-1, self.max_nodes, self.max_nodes)
        out_F = self.rec_F(shared_rep).view(-1, self.max_nodes, 3)
        out_E = self.rec_E(shared_rep).view(-1, self.max_nodes, self.max_nodes, 2)
        
        # 把重构结果打包成一个元组返回
        return out_chiN, out_kstar, out_cls, (out_A, out_F, out_E)

def get_model(in_feats=3, edge_dim=2, hidden_dim=64, num_enc_layers=4, max_nodes=32, device='cpu'):
    encoder = GNNEncoder(in_feats, edge_dim, hidden_dim, num_enc_layers)
    model = GNNMultiTaskModel(encoder, hidden_dim=hidden_dim, num_classes=5, max_nodes=max_nodes)
    return model.to(device)