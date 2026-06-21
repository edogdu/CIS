# Custom imports
from models.encoders.hetero_encoder import GNNHeteroEncoderModel

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
from functools import partial
from repositories.graphs.pyg_builder import y_labels
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# import moved file
from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from src.xai import captum_explainer.py

class GNNEarlyStopping:
    """Early stopping utility to stop training when 
    macro F1 score across all anomaly classes does not improve.
    We ignore the normal class (class 0) for early stopping as it is over-represented.    
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
    
    def step(self, val: float):
        if self.best_score is None:
            self.best_score = val
        elif self.mode == 'max' and val > self.best_score + self.min_delta:
            self.best_score = val
            self.counter = 0
        elif self.mode == 'min' and val < self.best_score - self.min_delta:
            self.best_score = val
            self.counter = 0
        # elif val == 0.0:
        #     # special case to avoid early stopping at beginning
        #     pass
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                    
        return self.early_stop
                    
