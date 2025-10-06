from torch_geometric.data import Data
from dataclasses import dataclass
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce, degree
import torch
from typing import Tuple, Dict, Any, List, Optional
import logging

data_builder_log = logging.getLogger("data_builder")
global_schema = {
    'property_keys': [
        # Network properties
        "avg_size", "destination_port", "destination_total_packets", "duration",
        "max_size", "min_size", "num_connections", "source_port", "source_total_packets",
        # Asset properties
        "avg_val", "max_val", "min_val", "num_measurements"
    ],
    # Node Labels - used for node type one-hot encoding
    'label_space': ["Endpoint", "Asset", "Connection", "Measurement"],
    # Categorical feature mappings - used for categorical feature one-hot encoding
    'categorical_mappings': {
        'protocol': ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "OTHER"],
        'asset_type': ["HMI", "PLC", "Pump", "Valv", "Flow Sensor", "Tank", "PressureSensor"],
        'measurement_type': ["state", "pressure", "val"],
    },
    # feature toggles
    "use_ip_features": True,
    "ip_hash_buckets": 0,
}

@dataclass
class FeatureLayout:
    property_keys: List[str]
    label_space: List[str]
    categorical_mappings: Dict[str, List[str]]
    use_ip_features: bool
    ip_hash_buckets: int
    include_masks: bool = True
    include_degree: bool = True

    #computed widths
    numeric_width: int = 0
    categorical_width: int = 0
    masks_width: int = 0
    labels_width: int = 0
    ip_width: int = 0
    mac_width: int = 12 # 6 src + 6 dst  
    total_width: int = 0

    #section offsets - start positions of each section in the feature vector
    numeric_offset: int = 0
    categorical_offset: int = 0
    masks_offset: int = 0
    labels_offset: int = 0
    ipmac_offset: int = 0

    #offsets for specific categorical features
    categorical_offsets: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        for key, vals in self.categorical_mappings.items():
            if "OTHER" not in vals:
                self.categorical_mappings[key] = vals + ["OTHER"]
        
        self.numeric_width = len(self.property_keys)
        self.masks_width = self.numeric_width if self.include_masks else 0
        self.labels_width = len(self.label_space)
        self.categorical_width = sum(len(v) for v in self.categorical_mappings.values())
        self.ip_width = self.ip_hash_buckets * 2 if (self.use_ip_features and self.ip_hash_buckets > 0) else 0
        self.total_width = (self.numeric_width + self.masks_width + self.labels_width +
                            self.categorical_width + self.ip_width + self.mac_width)
        #compute offsets
        self.numeric_offset = 0
        self.masks_offset = self.numeric_offset + self.numeric_width
        self.labels_offset = self.masks_offset + self.masks_width
        self.categorical_offset = self.labels_offset + self.labels_width
        self.ipmac_offset = self.categorical_offset + self.categorical_width

        #compute categorical offsets
        self.categorical_offsets = {}
        current_offset = self.categorical_offset
        for key, vals in self.categorical_mappings.items():
            self.categorical_offsets[key] = (current_offset, len(vals))
            current_offset += len(vals)


#helper functions

#handle IP address to numerical features
def ip_to_onehot_features(ip: str, buckets: int) -> list[float]:
    if ip is None or buckets <= 0:
        return []
    parts = ip.split('.')
    if len(parts) != 4:
        return [0.0] * buckets
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return [0.0] * buckets
    if any(num < 0 or num > 255 for num in nums):
        return [0.0] * buckets
    ip_num = (nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3]
    features = [0.0] * buckets
    index = ip_num % buckets
    features[index] = 1.0
    return features

def mac_to_features(mac: str) -> list[float]:
    if mac is None:
        return [0.0] * 6
    parts = mac.split(':')
    if len(parts) != 6:
        return [0.0] * 6
    try:
        nums = [int(part, 16) for part in parts]
    except ValueError:
        return [0.0] * 6
    if any(num < 0 or num > 255 for num in nums):
        return [0.0] * 6
    return [num / 255.0 for num in nums]

def to_float(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    elif isinstance(val, bool):
        return 1.0 if val else 0.0
    elif isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return 0.0
    else:
        return 0.0
    
def build_layout(schema: Dict[str, Any]=None) -> FeatureLayout:
    if schema is None:
        schema = global_schema
    return FeatureLayout(
        property_keys = list(schema['property_keys']),
        label_space = list(schema['label_space']),
        categorical_mappings = dict(schema['categorical_mappings']),
        use_ip_features = schema.get('use_ip_features', False),
        ip_hash_buckets = schema.get('ip_hash_buckets', 0),
        include_masks = schema.get('include_masks', True),
        include_degree = schema.get('include_degree', True)
    )

def fill_numeric_and_mask_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    for i, key in enumerate(layout.property_keys):
        if key in properties and properties[key] is not None:
            feature_tensor[tensor_row, layout.numeric_offset + i] = to_float(properties[key])
            if layout.include_masks:
                feature_tensor[tensor_row, layout.masks_offset + i] = 1.0

def fill_label_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, labels: List[str]):
    # one-hot encode node labels
    idx = {label: i for i, label in enumerate(layout.label_space)}
    for label in labels or []:
        # if label is in layout, set corresponding position to 1
        j = idx.get(label)
        if j is not None:
            feature_tensor[tensor_row, layout.labels_offset + j] = 1.0

def fill_categorical_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    for k, c in layout.categorical_mappings.items():
        offset, width = layout.categorical_offsets[k]
        val = properties.get(k)
        val = val if isinstance(val, str) and val in c else ("OTHER" if "OTHER" in c else None)
        if val is not None:
            j = c.index(val)
            feature_tensor[tensor_row, offset + j] = 1.0

def fill_ip_mac_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    # IP features
    ip_buckets = layout.ip_hash_buckets if (layout.use_ip_features and layout.ip_hash_buckets > 0) else 0
    offset = layout.ipmac_offset

    if ip_buckets > 0:
        src_ip = properties.get('source_ip')
        dst_ip = properties.get('destination_ip')
        src_ip_features = ip_to_onehot_features(src_ip, ip_buckets)
        dst_ip_features = ip_to_onehot_features(dst_ip, ip_buckets)
        if src_ip_features:
            feature_tensor[tensor_row, offset : offset + ip_buckets] = torch.tensor(src_ip_features, dtype=torch.float32)
        offset += ip_buckets
        if dst_ip_features:
            feature_tensor[tensor_row, offset : offset + ip_buckets] = torch.tensor(dst_ip_features, dtype=torch.float32)
        offset += ip_buckets
    # MAC features (6 bytes each)
    src_mac = properties.get('source_mac')
    dst_mac = properties.get('destination_mac')
    src_mac_feats = mac_to_features(src_mac)
    dst_mac_feats = mac_to_features(dst_mac)    
    feature_tensor[tensor_row, offset : offset + 6] = torch.tensor(src_mac_feats, dtype=torch.float32)
    feature_tensor[tensor_row, offset + 6 : offset + 12] = torch.tensor(dst_mac_feats, dtype=torch.float32)

def build_edge_index(num_nodes: int, edges: List[Dict[str, Any]], graphid_to_tensorid: Dict[int, int]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    source, target = [], []
    for edge in edges:
        src = edge.get('source')
        dst = edge.get('target')
        src_id = graphid_to_tensorid.get(src)
        dst_id = graphid_to_tensorid.get(dst)
        if src_id is None or dst_id is None:
            continue
        source.append(src_id)
        target.append(dst_id)
    edge_index = torch.tensor([source, target], dtype=torch.long)    
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes, reduce='add')
    return edge_index

def add_degree_column(feature_tensor: torch.Tensor, edge_index: torch.Tensor):
    deg = degree(edge_index[0], num_nodes=feature_tensor.size(0), dtype=torch.float32).unsqueeze(1)
    return torch.cat([feature_tensor, deg], dim=1)

def map_to_features(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        schema: Dict[str, Any]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    layout = build_layout(schema)

    #id mappings
    graphid_to_tensorid = {node['id']: i for i, node in enumerate(nodes)}
    tensorid_to_graphid = {i: node['id'] for i, node in enumerate(nodes)}    
       
    N = len(graphid_to_tensorid)

    #node features
    x = torch.zeros((N, layout.total_width), dtype=torch.float32)
    for node in nodes:
        i = graphid_to_tensorid[node['id']]
        props = node.get('properties', {})
        labels = node.get('labels', [])
        fill_numeric_and_mask_features(x, i, layout, props)
        fill_label_features(x, i, layout, labels)
        fill_categorical_features(x, i, layout, props)
        fill_ip_mac_features(x, i, layout, props)

    #edges
    edge_index = build_edge_index(N, edges, graphid_to_tensorid)
    if layout.include_degree:
        x = add_degree_column(x, edge_index)
        

    return x, edge_index


def to_pyg_data(snapshot: dict, schema: Dict[str, Any] = None) -> Data:
    # Convert the snapshot data into PyG Data format
    nodes = snapshot.get('nodes', [])
    edges = snapshot.get('relationships', [])

    x, edge_index = map_to_features(nodes, edges, schema)
    
    data_builder_log.info(f"Built features with shape: {x.shape}, edge_index shape: {edge_index.shape}")

    data = Data(x=x, edge_index=edge_index)
    #data.pos_edge_label_index = edge_index
    data.num_node_features = x.size(1)
    data.num_nodes = x.size(0)
    data.graphid_to_tensorid = {node['id']: i for i, node in enumerate(nodes)}
    data.tensorid_to_graphid = {i: node['id'] for i, node in enumerate(nodes)}
    

    # Copy snapshot-level properties
    for key in ('snapshot_id', 'system_id', 'start_time', 'end_time', 'duration'):
        val = snapshot.get(key)
        if val is not None:
            data[key] = val

    return data