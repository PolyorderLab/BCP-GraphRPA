import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import Optional, Tuple, Dict, List
import sys

sys.path.append("/home/zxy/project/TAPD/src")
from graph_builder import build_copolymer_graph, build_m1_graph, build_m2_graph
from model_utils import get_model


class ChiNPredictor:
    """
    Wrapper class for predicting chiN* using trained GNN model.
    Model config: hidden_dim=256, enc_layers=4, dec_layers=1
    """

    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        HIDDEN_DIM = 256
        ENC_LAYERS = 4
        DEC_LAYERS = 1

        self.model = get_model(
            task_type='chiN',
            device=self.device,
            in_feats=3,
            edge_dim=2,
            hidden_dim=HIDDEN_DIM,
            num_enc_layers=ENC_LAYERS,
            num_dec_layers=DEC_LAYERS
        )
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, topo_type: str, fA: float) -> float:
        """Predict chiN* for a single composition."""
        graph = build_copolymer_graph(topo_type, fA, N_max=32)
        graph = graph.to(self.device)

        with torch.no_grad():
            log_chiN_pred = self.model(graph)
            chiN_pred = 10 ** log_chiN_pred.cpu().item()

        return chiN_pred

    def predict_batch(self, topo_type: str, fA_values: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict chiN* for multiple compositions."""
        if topo_type == "M1":
            graphs = [build_m1_graph(fA, N_max=32) for fA in fA_values]
        elif topo_type == "M2":
            graphs = [build_m2_graph(fA, N_max=32) for fA in fA_values]
        else:
            graphs = [build_copolymer_graph(topo_type, fA, N_max=32) for fA in fA_values]

        predictions = []
        with torch.no_grad():
            for g in graphs:
                g = g.to(self.device)
                log_chiN = self.model(g)
                chiN = 10 ** log_chiN.cpu().item()
                predictions.append(chiN)
        return np.array(predictions)


def get_all_ml_predictions(model_path: str,
                           fA_range: Tuple[float, float] = (0.1, 0.9),
                           num_points: int = 100,
                           topologies: List[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get ML predictions for all topologies."""
    if topologies is None:
        topologies = ['AB', 'ABA', 'ABAB', 'ABABA', 'AB2', 'AB3', 'M1', 'M2']
    results = {}

    predictor = ChiNPredictor(model_path)
    fA_values = np.linspace(fA_range[0], fA_range[1], num_points)

    for topo in topologies:
        chiN_pred = predictor.predict_batch(topo, fA_values)
        results[topo] = (fA_values, chiN_pred)

    return results
