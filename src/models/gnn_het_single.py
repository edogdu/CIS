import copy
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import Subset
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader

from repositories.graphs.pyg_builder import (
    get_hetero_column_names,
    y_labels,
)

from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from models.losses.focal_loss import FocalLoss
from xai.captum_explainer import CaptumExplainer

# Local application/library specific imports
logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class GNNHeteroClassifierModel(nn.Module):

    def __init__(self, config, metadata):
        super().__init__()

        self.config = config
        self.metadata = metadata
        self.scalers = {}
        self.criterion = None

        self.encoder = GNNHeteroEncoderModel(
            config,
            metadata
        )

        hd = config.get("hidden_dim", 64)

        self.out = nn.Linear(
            hd,
            len(y_labels)
        )

        self.explainer = CaptumExplainer(self)

        self.to(DEVICE)

    def explain_snapshot(self, data):
        return self.explainer.explain(data)
