# graph_context.py
# must map original_id back to asset_name, IP, endpoint, protocol, and PLC tag

import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

# Captum expects a plain callable

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

# extract snapshot metadata (graph-level metadata)
def extract_snapshot_metadata(
    data,
    y_labels,
):
    snapshot_id = "unknown"

    if hasattr(data, "snapshot_id"):
        raw = data.snapshot_id

        if torch.is_tensor(raw):
            if raw.numel() == 1:
                snapshot_id = str(raw.item())
            else:
                snapshot_id = str(raw.tolist())
        else:
            snapshot_id = str(raw)

    true_idx = -1
    true_name = "Unknown"

    if hasattr(data, "y"):
        if data.y.numel() == 1:
            true_idx = int(data.y.item())

            if (
                true_idx >= 0
                and true_idx < len(y_labels)
            ):
                true_name = y_labels[true_idx]

    return {
        "snapshot_id": snapshot_id,
        "true_label_idx": true_idx,
        "true_label_name": true_name,
    }

# extract node IDs (maps the attribution row index to the original_id)
def extract_node_ids(
    data,
    ntype,
    num_nodes,
):
    orig_ids = []

    if hasattr(data[ntype], "original_id"):
        raw = data[ntype].original_id

        orig_ids = (
            raw.tolist()
            if torch.is_tensor(raw)
            else raw
        )

    elif hasattr(data[ntype], "n_id"):
        orig_ids = data[ntype].n_id.tolist()

    if len(orig_ids) != num_nodes:
        orig_ids = [
            f"{ntype}_{i}"
            for i in range(num_nodes)
        ]

    return orig_ids

# extract feature names (maps feature index to feature name)
def extract_feature_names(
    ntype,
    num_features,
):
    try:
        key = ntype

        if ntype in [
            "TankMeasurements",
            "ValveMeasurements",
        ]:
            key = "Measurements"

        elif ntype in [
            "PumpMeasurements",
            "SensorMeasurements",
        ]:
            key = "StateMeasurements"

        elif not ntype.endswith("s"):
            key = ntype + "s"

        names = get_hetero_column_names(key)

        if len(names) == num_features:
            return names

    except Exception:
        pass

    return [
        f"feat_{i}"
        for i in range(num_features)
    ]

# maps original_id to asset_name, IP, endpoint, protocol, and PLC tag
def extract_node_context(data, ntype, original_id):
    metadata = getattr(data, "metadata", None)
    if metadata is None:
        return {}

    context_lookup = getattr(data, "context_lookup", {})
    return context_lookup.get(ntype, {}).get(original_id, {})

def build_node_context(data, raw_snapshot_single):
    """
    Build context from snapshot["nodes"] structure:
    each node has: id, labels, properties
    """

    context = defaultdict(dict)

    for node in raw_snapshot_single["nodes"]:
        # node type is stored in labels[0]
        ntype = node["labels"][0]
        oid = node["id"]
        props = node.get("properties", {})

        context[ntype][oid] = props

    data.context_lookup = context
    return context
