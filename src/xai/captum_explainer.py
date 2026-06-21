# captum_explainer.py
# tasks: prepare inputs, run IG, and return raw attributions

# imports
import logging
import json
import os
import time

from functools import partial
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from torch_geometric.data import Batch, HeteroData
from xai.graph_context import fast_model_forward_wrapper

from repositories.graphs.pyg_builder import (
    get_hetero_column_names,
    y_labels,
)

# logging information
logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class CaptumExplainer:

    def __init__(
        self,
        model,
        device=DEVICE,
        n_steps=50,
    ):
        self.model = model
        self.device = device
        self.n_steps = n_steps

    def explain(            # need to add explain_graph, and explain_snapshot
        self,
        data: HeteroData,
        target_class=None,
    ):
        self.model.eval()

        data = data.to(self.device)

        with torch.no_grad():
            logits = self.model(data)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        inputs = []
        node_types = []

        for ntype in data.x_dict:
            if hasattr(data[ntype], "x"):
                inputs.append(
                    data[ntype].x.detach().clone().requires_grad_(True)
                )
                node_types.append(ntype)

        inputs = tuple(inputs)
        baselines = tuple(torch.zeros_like(x) for x in inputs)

        forward_func = partial(
            fast_model_forward_wrapper,
            self.model,
            data,
            self.device,
        )

        ig = IntegratedGradients(forward_func)

        attributions = ig.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target_class,
            n_steps=self.n_steps,
            internal_batch_size=10,
        )

        return {
            "target_class": target_class,
            "confidence": confidence,
            "node_types": node_types,
            "attributions": attributions,
        }
