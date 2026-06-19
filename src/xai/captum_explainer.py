# captum_explainer.py
# only place that should contain _fast_model_forward_wrapper, IntegratedGradients, and attribution logic

# imports
import logging
from functools import partial
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from captum.attr import IntegratedGradients

# logging information
logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# wrapper function
def _fast_model_forward_wrapper(model: nn.Module, data: HeteroData, model_device: str, *node_inputs: torch.Tensor):
    """
    Optimized wrapper for Captum. Reconstructs the batched graph from interpolated inputs.
    """
    # Create a lightweight container
    temp_data = HeteroData()
    
    # Identify node types with features
    node_types_with_features = [ntype for ntype in data.x_dict.keys() 
                                if data[ntype].num_nodes > 0 and hasattr(data[ntype], "x")]
    
    # Determine Batching Params
    first_input = node_inputs[0]
    original_num_nodes = data[node_types_with_features[0]].num_nodes
    total_nodes = first_input.size(0)
    
    is_batched = total_nodes != original_num_nodes
    num_replications = total_nodes // original_num_nodes if is_batched else 1
    
    
    temp_data.num_graphs = num_replications
    

    # Reconstruct Node Features & Batch Vector
    for idx, ntype in enumerate(node_types_with_features):
        temp_data[ntype].x = node_inputs[idx]
        
        if is_batched:
            current_num_nodes = data[ntype].num_nodes
            # Create batch vector [0,0... 1,1...]
            batch_ids = torch.arange(num_replications, device=model_device)
            temp_data[ntype].batch = torch.repeat_interleave(batch_ids, current_num_nodes)
        else:
            temp_data[ntype].batch = torch.zeros(data[ntype].num_nodes, dtype=torch.long, device=model_device)

    # Reconstruct Edges (Vectorized)
    if not is_batched:
        temp_data.edge_index_dict = data.edge_index_dict
    else:
        new_edge_index_dict = {}
        for etype, edge_index in data.edge_index_dict.items():
            # Standard edge replication logic
            num_edges = edge_index.size(1)
            src_ntype, _, dst_ntype = etype
            src_count = data[src_ntype].num_nodes
            dst_count = data[dst_ntype].num_nodes
            
            # Create offsets
            offsets_src = (torch.arange(num_replications, device=model_device) * src_count).view(-1, 1)
            offsets_dst = (torch.arange(num_replications, device=model_device) * dst_count).view(-1, 1)
            
            # Expand edges [2, num_edges] -> [num_reps, 2, num_edges]
            edges_expanded = edge_index.unsqueeze(0).expand(num_replications, 2, num_edges).clone()
            
            # Add offsets
            edges_expanded[:, 0, :] += offsets_src
            edges_expanded[:, 1, :] += offsets_dst
            
            # Flatten to [2, total_edges]
            new_edge_index_dict[etype] = edges_expanded.permute(1, 0, 2).reshape(2, -1)
            
        temp_data.edge_index_dict = new_edge_index_dict

    # Run Model
    return model(temp_data)

# Captum Explainer class
class CaptumExplainer:

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        n_steps: int = 50
    ):
        self.model = model
        self.device = device
        self.n_steps = n_steps

# input extraction
def build_inputs(
    self,
    data: HeteroData
):
  
  # extract inputs_tuple, baselines_tuple, node_types from graph
inputs = []
baselines = []

for ntype in node_types:
    x = data[ntype].x

    inputs.append(x)

    baselines.append(
        torch.zeros_like(x)
    )

# attribution computation
def attribute(
    self,
    data: HeteroData,
    target: int
):
  forward_func = partial(
    _fast_model_forward_wrapper,
    self.model,
    data,
    self.device
)

ig = IntegratedGradients(
    forward_func=forward_func
)

attributions = ig.attribute(
    ...
)
{
    "node_types": node_types,
    "attributions": attributions
}

# explanation method
def explain(
    self,
    data: HeteroData,
    target: int
):

  # high-level API
  attr = self.attribute(
    data,
    target
)

return attr

# ------------------------------------------------------------------------------------------------- 

