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
        x = global_mean_pool(x, batch) 
        return x

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, use_sigmoid=False):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class SingleTaskRegressor(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_dec_layers=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = MLPHead(hidden_dim, hidden_dim, 1, num_layers=num_dec_layers)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        return self.decoder(graph_embed)

class SingleTaskClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.decoder = MLPHead(hidden_dim, hidden_dim, num_classes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        return self.decoder(graph_embed)

class ReconstructionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, max_nodes=32):
        super().__init__()
        self.max_nodes = max_nodes
        self.rec_A = MLPHead(input_dim, 256, max_nodes * max_nodes, use_sigmoid=True)
        self.rec_F = MLPHead(input_dim, 256, max_nodes * 3)
        self.rec_E = MLPHead(input_dim, 512, max_nodes * max_nodes * 2)
        
    def forward(self, x):
        out_A = self.rec_A(x).view(-1, self.max_nodes, self.max_nodes)
        out_F = self.rec_F(x).view(-1, self.max_nodes, 3)
        out_E = self.rec_E(x).view(-1, self.max_nodes, self.max_nodes, 2)
        return out_A, out_F, out_E

class DualRegressionModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_kstar = self.head_kstar(graph_embed)
        return out_chiN, out_kstar

class ChiNReconModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, max_nodes=32):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_rec = self.head_rec(graph_embed)
        return out_chiN, out_rec

class KReconModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, max_nodes=32):
        super().__init__()
        self.encoder = encoder
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_kstar = self.head_kstar(graph_embed)
        out_rec = self.head_rec(graph_embed)
        return out_kstar, out_rec

class ChiNClsModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_cls = self.head_cls(graph_embed)
        return out_chiN, out_cls

class KClsModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_kstar = self.head_kstar(graph_embed)
        out_cls = self.head_cls(graph_embed)
        return out_kstar, out_cls

class DualReconModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, max_nodes=32):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_kstar = self.head_kstar(graph_embed)
        out_rec = self.head_rec(graph_embed)
        return out_chiN, out_kstar, out_rec

class DualClsModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_kstar = self.head_kstar(graph_embed)
        out_cls = self.head_cls(graph_embed)
        return out_chiN, out_kstar, out_cls

class ChiNReconClsModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5, max_nodes=32):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_cls = self.head_cls(graph_embed)
        out_rec = self.head_rec(graph_embed)
        return out_chiN, out_cls, out_rec

class KReconClsModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5, max_nodes=32):
        super().__init__()
        self.encoder = encoder
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_kstar = self.head_kstar(graph_embed)
        out_cls = self.head_cls(graph_embed)
        out_rec = self.head_rec(graph_embed)
        return out_kstar, out_cls, out_rec

class MultiTaskModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_classes=5, max_nodes=32):
        super().__init__()
        self.encoder = encoder
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)
        
    def forward(self, data):
        graph_embed = self.encoder(data)
        out_chiN = self.head_chiN(graph_embed)
        out_kstar = self.head_kstar(graph_embed)
        out_cls = self.head_cls(graph_embed)
        out_rec = self.head_rec(graph_embed)
        return out_chiN, out_kstar, out_cls, out_rec

def get_model(task_type='chiN', **kwargs):
    device = kwargs.get('device', 'cpu')
    hidden_dim = kwargs.get('hidden_dim', 64)
    encoder = GNNEncoder(
        in_feats=kwargs.get('in_feats', 3),
        edge_dim=kwargs.get('edge_dim', 2),
        hidden_dim=hidden_dim,
        num_layers=kwargs.get('num_enc_layers', 3)
    )
    
    if task_type in ['chiN', 'k']:
        model = SingleTaskRegressor(encoder, hidden_dim, kwargs.get('num_dec_layers', 2))
    elif task_type == 'cls':
        model = SingleTaskClassifier(encoder, hidden_dim, kwargs.get('num_classes', 5))
    elif task_type == 'reconstruction':
        model = nn.ModuleList([encoder, ReconstructionHead(hidden_dim, hidden_dim, kwargs.get('max_nodes', 32))])
        # Note: ModuleList isn't quite right for a model with forward. 
        # Better to have a wrapper:
        class ReconModelWrapper(nn.Module):
            def __init__(self, encoder, head):
                super().__init__()
                self.encoder = encoder
                self.head = head
            def forward(self, data):
                return self.head(self.encoder(data))
        model = ReconModelWrapper(encoder, ReconstructionHead(hidden_dim, hidden_dim, kwargs.get('max_nodes', 32)))
    elif task_type == 'mtl':
        model = MultiTaskModel(encoder, hidden_dim, kwargs.get('num_classes', 5), kwargs.get('max_nodes', 32))
    elif task_type == 'ReconClsChiN':
        # Use MultiTaskModel (chiN, k*, cls, rec) and ignore k* during training/eval
        model = MultiTaskModel(encoder, hidden_dim, kwargs.get('num_classes', 5), kwargs.get('max_nodes', 32))
    elif task_type == 'ReconClsK':
        # Use MultiTaskModel as well; we'll ignore chiN output
        model = MultiTaskModel(encoder, hidden_dim, kwargs.get('num_classes', 5), kwargs.get('max_nodes', 32))
    elif task_type == 'dual_regression':
        model = DualRegressionModel(encoder, hidden_dim)
    elif task_type == 'chiN_recon':
        model = ChiNReconModel(encoder, hidden_dim, kwargs.get('max_nodes', 32))
    elif task_type == 'k_recon':
        model = KReconModel(encoder, hidden_dim, kwargs.get('max_nodes', 32))
    elif task_type == 'chiN_cls':
        model = ChiNClsModel(encoder, hidden_dim, kwargs.get('num_classes', 5))
    elif task_type == 'k_cls':
        model = KClsModel(encoder, hidden_dim, kwargs.get('num_classes', 5))
    elif task_type == 'dual_cls':
        model = DualClsModel(encoder, hidden_dim, kwargs.get('num_classes', 5))
    else:
        raise ValueError(f"Unknown task type: {task_type}")
        
    return model.to(device)


# ===================== GNN models with descriptors =====================

from desc_utils import DESCRIPTOR_DIM


class DescriptorEncoder(nn.Module):
    def __init__(self, input_dim=DESCRIPTOR_DIM, hidden_dim=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DescGNNEncoder(nn.Module):
    def __init__(
        self,
        in_feats=3,
        edge_dim=2,
        hidden_dim=64,
        num_layers=3,
        use_desc=True,
        desc_dim=DESCRIPTOR_DIM,
    ):
        super().__init__()
        self.use_desc = use_desc
        self.gnn_encoder = GNNEncoder(
            in_feats=in_feats,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        if use_desc:
            self.desc_encoder = DescriptorEncoder(
                input_dim=desc_dim, hidden_dim=hidden_dim, output_dim=hidden_dim
            )
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fusion = nn.Identity()

    def forward(self, data):
        graph_embed = self.gnn_encoder(data)
        if self.use_desc and hasattr(data, "desc"):
            desc = data.desc
            if desc.dim() == 3:
                desc = desc.squeeze(1)
            desc_embed = self.desc_encoder(desc)
            combined = torch.cat([graph_embed, desc_embed], dim=-1)
            output = self.fusion(combined)
        else:
            output = self.fusion(graph_embed)
        return output


class SingleTaskRegressorWithDesc(nn.Module):
    def __init__(self, encoder, hidden_dim=64, num_dec_layers=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = MLPHead(hidden_dim, hidden_dim, 1, num_layers=num_dec_layers)

    def forward(self, data):
        embed = self.encoder(data)
        return self.decoder(embed)


class DescGNNMultiTaskModel(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_classes=5,
        max_nodes=32,
        in_feats=3,
        edge_dim=2,
        use_desc=True,
        desc_dim=DESCRIPTOR_DIM,
    ):
        super().__init__()
        self.encoder = DescGNNEncoder(
            in_feats=in_feats, edge_dim=edge_dim, hidden_dim=hidden_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)
        self.max_nodes = max_nodes
        self.rec_A = MLPHead(hidden_dim, 256, max_nodes * max_nodes, use_sigmoid=True)
        self.rec_F = MLPHead(hidden_dim, 256, max_nodes * 3)
        self.rec_E = MLPHead(hidden_dim, 512, max_nodes * max_nodes * 2)

    def forward(self, data):
        z = self.encoder(data)
        out_chiN = self.head_chiN(z)
        out_kstar = self.head_kstar(z)
        out_cls = self.head_cls(z)
        out_A = self.rec_A(z).view(-1, self.max_nodes, self.max_nodes)
        out_F = self.rec_F(z).view(-1, self.max_nodes, 3)
        out_E = self.rec_E(z).view(-1, self.max_nodes, self.max_nodes, 2)
        return out_chiN, out_kstar, out_cls, (out_A, out_F, out_E)


class DescDualRegressionModel(nn.Module):
    def __init__(self, hidden_dim=64, in_feats=3, edge_dim=2, use_desc=True,
                 desc_dim=DESCRIPTOR_DIM):
        super().__init__()
        self.encoder = DescGNNEncoder(
            in_feats=in_feats, edge_dim=edge_dim, hidden_dim=hidden_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_kstar = MLPHead(hidden_dim, hidden_dim, 1)

    def forward(self, data):
        embed = self.encoder(data)
        return self.head_chiN(embed), self.head_kstar(embed)


class DescChiNClsModel(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=5, in_feats=3, edge_dim=2,
                 use_desc=True, desc_dim=DESCRIPTOR_DIM):
        super().__init__()
        self.encoder = DescGNNEncoder(
            in_feats=in_feats, edge_dim=edge_dim, hidden_dim=hidden_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
        self.head_chiN = MLPHead(hidden_dim, hidden_dim, 1)
        self.head_cls = MLPHead(hidden_dim, hidden_dim, num_classes)

    def forward(self, data):
        embed = self.encoder(data)
        return self.head_chiN(embed), self.head_cls(embed)


class DescReconstructionModel(nn.Module):
    def __init__(self, in_feats=3, edge_dim=2, hidden_dim=128, num_enc_layers=4,
                 max_nodes=32, use_desc=True, desc_dim=DESCRIPTOR_DIM):
        super().__init__()
        self.encoder = DescGNNEncoder(
            in_feats=in_feats, edge_dim=edge_dim, hidden_dim=hidden_dim,
            num_layers=num_enc_layers, use_desc=use_desc, desc_dim=desc_dim,
        )
        self.head_rec = ReconstructionHead(hidden_dim, hidden_dim, max_nodes)

    def forward(self, data):
        embed = self.encoder(data)
        return self.head_rec(embed)


def get_desc_model(task_type="chiN", **kwargs):
    device = kwargs.get("device", "cpu")
    hidden_dim = kwargs.get("hidden_dim", 64)
    use_desc = kwargs.get("use_desc", True)
    in_feats = kwargs.get("in_feats", 3)
    edge_dim = kwargs.get("edge_dim", 2)
    desc_dim = kwargs.get("desc_dim", DESCRIPTOR_DIM)

    if task_type in ["chiN", "k"]:
        encoder = DescGNNEncoder(
            in_feats=in_feats, edge_dim=edge_dim, hidden_dim=hidden_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
        model = SingleTaskRegressorWithDesc(
            encoder, hidden_dim, kwargs.get("num_dec_layers", 2)
        )
    elif task_type == "reconstruction":
        model = DescReconstructionModel(
            in_feats=in_feats, edge_dim=edge_dim, hidden_dim=hidden_dim,
            num_enc_layers=kwargs.get("num_enc_layers", 4),
            max_nodes=kwargs.get("max_nodes", 32),
            use_desc=use_desc, desc_dim=desc_dim,
        )
    elif task_type == "dual_regression":
        model = DescDualRegressionModel(
            hidden_dim=hidden_dim, in_feats=in_feats, edge_dim=edge_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
    elif task_type == "chiN_cls":
        model = DescChiNClsModel(
            hidden_dim=hidden_dim, num_classes=kwargs.get("num_classes", 5),
            in_feats=in_feats, edge_dim=edge_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
    elif task_type in ["mtl", "desc_gnn_mtl"]:
        model = DescGNNMultiTaskModel(
            hidden_dim=hidden_dim, num_classes=kwargs.get("num_classes", 5),
            max_nodes=kwargs.get("max_nodes", 32),
            in_feats=in_feats, edge_dim=edge_dim,
            use_desc=use_desc, desc_dim=desc_dim,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return model.to(device)
