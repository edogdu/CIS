# Standard library imports
from datetime import datetime
import logging
import os
import time
from typing import Any, Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    HGTConv,
    Linear,
    global_max_pool,
)
import json
from torch_geometric.explain import Explainer, GNNExplainer, HeteroExplanation
from factories import data
from models.focal_loss import FocalLoss
from repositories.graphs.pyg_builder import get_hetero_column_names, visualize_features_distribution # for GNNExplainer feature names
import copy
from captum.attr import IntegratedGradients
from functools import partial
from repositories.graphs.pyg_builder import y_labels
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

class GNNHeteroEncoderModel(nn.Module):
    """GNN module to produce node embeddings for heterogeneous graphs."""
    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroEncoderModel, self).__init__()
        if metadata is None:
            raise ValueError("Metadata must be provided for heterogeneous graphs.")
        self.metadata = metadata
        self.config = config
        

        hd    = config.get("hidden_dim", 32)
        heads = config.get("num_heads", 4)

        # Which node types to pool over (you can override via config)
        self.pooled_types: List[str] = config.get("pooled_types", ["Pumps", "FlowSensors", "Tanks", "Valves", "Connections", "Endpoints"])

        # Per-node-type input projection to a shared hidden dim
        self.lin_dict = nn.ModuleDict()
        for node_type in self.metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hd)

        # HGT backbone
        self.num_layers = config.get("num_layers", 3)

        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(HGTConv(hd, hd, metadata=self.metadata, heads=heads))

        self.dropout = float(config.get("dropout", 0.5))
        
        # Projection after concatenating pooled node-type embeddings
        pooled_width = hd * max(1, len(self.pooled_types))
        self.lin1 = Linear(pooled_width, hd)

        self.to(DEVICE)


    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        # 1) Type-wise input projections
        
        x_dict = {
            ntype: self.lin_dict[ntype](x).relu()
            for ntype, x in data.x_dict.items()
        }

        # 2) HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 3) Graph-level pooling over selected node types (robust if missing)
        pools = []
        num_graphs = data.num_graphs
        for ntype in self.pooled_types:
            if ntype in x_dict and hasattr(data[ntype], "batch"):
                pools.append(global_max_pool(x_dict[ntype], data[ntype].batch, size=num_graphs))
            else:
                # keep dims aligned so concatenation works
                pools.append(torch.zeros((num_graphs, self.config.get("hidden_dim", 32)), device=x_dict[next(iter(x_dict))].device))

        h = torch.cat(pools, dim=1) if len(pools) > 1 else pools[0]

        # 4) Final MLP + Dropout
        h = F.relu(self.lin1(F.dropout(h, p=self.dropout, training=self.training)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h
