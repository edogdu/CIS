# Custom imports
from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from src.xai.captum_explainer import CaptumExplainer

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

from factories import data
from models.focal_loss import FocalLoss
from repositories.graphs.pyg_builder import get_hetero_column_names, visualize_features_distribution # for GNNExplainer feature names
import copy
from repositories.graphs.pyg_builder import y_labels
from sklearn.preprocessing import RobustScaler

# Local application/library specific imports
logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# only call IG here, because explanations are handled by the XAI module
self.explainer = CaptumExplainer(self)
explanation = self.explainer.explain(snapshot)
