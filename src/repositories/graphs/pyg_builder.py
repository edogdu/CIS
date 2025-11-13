from torch_geometric.data import Data
from dataclasses import dataclass
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce, degree
import torch
from typing import Tuple, Dict, Any, List, Optional
import logging
import math
#from torch_geometric.transforms import LocalClusteringCoefficient

logger = logging.getLogger("data_builder")

#global schema definition - can be overridden by passing a schema dict to build_layout
global_schema = {
    'property_keys': [
        # Network properties
        "avg_size", #, "destination_port", "destination_total_packets",
         "num_connections"  #, "source_port", "source_total_packets",
    ],
    'physical_property_keys': [
        "state",
        "pressure",
        "value"
    ],
    # Node Labels - used for node type one-hot encoding
    'label_space': ["Endpoint", "Asset", "Connection", "Measurement"],
    # Categorical feature mappings - used for categorical feature one-hot encoding
    'categorical_mappings': {
        'protocol': ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "OTHER"],
        'asset_type': ["HMI", "PLC", "Pump", "Valv", "Flow Sensor", "Tank", "PressureSensor"],
        'measurement_type': ["state", "pressure", "val"],
        'destination_port': ["well_known", "registered", "ephemeral", "other" ], # well-known: 0-1023, registered: 1024-49151, ephemeral: 49152-65535
        'source_port': ["well_known", "registered", "ephemeral", "other" ],
        'source_ip': ["PLC_1_IP", "PLC_2_IP", "PLC_3_IP", "PLC_4_IP", "HMI_1_IP","Flow_sensor_1_IP","Flow_sensor_2_IP", "OTHER"],
        'destination_ip': ["PLC_1_IP", "PLC_2_IP", "PLC_3_IP", "PLC_4_IP", "HMI_1_IP","Flow_sensor_1_IP","Flow_sensor_2_IP", "OTHER"],
        'ip': ["PLC_1_IP", "PLC_2_IP", "PLC_3_IP", "PLC_4_IP", "HMI_1_IP","Flow_sensor_1_IP","Flow_sensor_2_IP", "OTHER"]
        
    },
    "mac_width": 18
}

y_labels = ['normal', 'mitm', 'dos', 'scan', 'physical fault', 'anomaly']
y_bin_labels = ['normal', 'anomaly']



@dataclass
class FeatureLayout:
    """Data class to hold the layout of features for the graph."""
    property_keys: List[str]
    physical_property_keys: List[str]
    label_space: List[str]
    categorical_mappings: Dict[str, List[str]]
    include_masks: bool = True
    include_degree: bool = True

    #computed widths
    numeric_width: int = 0
    categorical_width: int = 0
    masks_width: int = 0
    labels_width: int = 0    
    mac_width: int = 18 # 6 src + 6 dst +  
    total_width: int = 0
    physical_width: int = 0  # reserved for physical measurement features

    #section offsets - start positions of each section in the feature vector
    numeric_offset: int = 0
    categorical_offset: int = 0
    masks_offset: int = 0
    labels_offset: int = 0
    ipmac_offset: int = 0
    physical_offset: int = 0

    #offsets for specific categorical features
    categorical_offsets: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        for key, vals in self.categorical_mappings.items():
            if "OTHER" not in vals:
                self.categorical_mappings[key] = vals + ["OTHER"]
        
        self.numeric_width = len(self.property_keys)
        self.physical_width = len(self.physical_property_keys)
        self.masks_width = self.numeric_width if self.include_masks else 0
        self.labels_width = len(self.label_space)
        self.categorical_width = sum(len(v) for v in self.categorical_mappings.values())        
        self.total_width = (self.numeric_width + self.masks_width + self.labels_width +
                            self.categorical_width + self.mac_width + self.physical_width)
        #compute offsets
        self.numeric_offset = 0
        self.physical_offset = self.numeric_offset + self.numeric_width
        self.masks_offset = self.physical_offset + self.physical_width
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




def ylabel_to_index(label: str, binary: bool = False) -> Optional[int]:
    """Convert a label string to its index in the label space."""
    if label is None or label.lower() not in (y_bin_labels if binary else y_labels):
        raise ValueError("Label cannot be None or unknown")
    i = (y_bin_labels if binary else y_labels).index(label.lower())
    return i

def index_to_ylabel(i, binary: bool = False) -> str:
    """Convert an index to its corresponding label string."""
    idx = int(i)
    if not binary and 0 <= idx < len(y_labels):
        return y_labels[int(idx)]
    elif binary and 0 <= idx < len(y_bin_labels):
        return y_bin_labels[int(idx)]
    else:
        raise ValueError("Index out of range for y_labels")

#handle IP address to numerical features
# def ip_to_onehot_features(ip: str, buckets: int) -> list[float]:
#     """Convert IP address string to one-hot encoded features based on hash buckets."""
#     if ip is None or buckets <= 0 or ip == "0":
#         return [0.0] * buckets
#     ip = ip.split('/')[0] # remove CIDR if present
#     parts = ip.split('.')
#     if len(parts) != 4:
#         return [0.0] * buckets
#     try:
#         nums = [int(part) for part in parts]
#     except ValueError:
#         return [0.0] * buckets
#     if any(num < 0 or num > 255 for num in nums):
#         return [0.0] * buckets
#     ip_num = (nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3]
#     features = [0.0] * buckets
#     index = ip_num % buckets
#     features[index] = 1.0
#     return features

def map_port_to_category(port: Optional[int]) -> Optional[str]:
    """Map port number to category string."""
    if port is None:
        return None
    if port == 0:
        return "other"
    elif 1 <= port <= 1023:
        return "well_known"
    elif 1024 <= port <= 49151:
        return "registered"
    elif 49152 <= port <= 65535:
        return "ephemeral"
    else:
        return "other"

def mac_to_features(mac: str) -> list[float]:
    """Convert MAC address string to 6 numerical features normalized between 0 and 1."""
    if mac is None or mac == "0":
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
    """Convert a value to float, handling different types."""
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
    """Build FeatureLayout from schema dictionary."""
    if schema is None:
        schema = global_schema
    return FeatureLayout(
        property_keys = list(schema['property_keys']),
        physical_property_keys = list(schema['physical_property_keys']),
        label_space = list(schema['label_space']),
        categorical_mappings = dict(schema['categorical_mappings']),
        include_masks = schema.get('include_masks', True),
        include_degree = schema.get('include_degree', True)
    )

def fill_physical_measurement_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    """Fill physical measurement features into the feature tensor."""
    measurement_type = properties.get('measurement_type')
    phys_measurement = properties.get('avg_value')
    #logging.info(f"Filling physical measurement features for measurement_type: {measurement_type}, value: {phys_measurement}")
    for i, phys_key in enumerate(layout.physical_property_keys):
        #logging.info(f"Processing physical property key: {phys_key}")
        if phys_key == 'state':
            if measurement_type == 'state':
                feature_tensor[tensor_row, layout.physical_offset + i] = to_float(phys_measurement)
            else:
                feature_tensor[tensor_row, layout.physical_offset + i] = 0.0
        elif phys_key == 'pressure':
            if measurement_type == 'pressure':
                feature_tensor[tensor_row, layout.physical_offset + i] = to_float(phys_measurement) / 1843.0
            else:
                feature_tensor[tensor_row, layout.physical_offset + i] = 0.0
        elif phys_key == 'value':
            if measurement_type == 'val':
                feature_tensor[tensor_row, layout.physical_offset + i] = to_float(phys_measurement) / 4683.0
            else:
                feature_tensor[tensor_row, layout.physical_offset + i] = 0.0
        else:
            feature_tensor[tensor_row, layout.physical_offset + i] = 0.0
    

def fill_numeric_and_mask_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    """Fill numeric and mask features into the feature tensor."""
    for i, key in enumerate(layout.property_keys):
        if key in properties and properties[key] is not None:            
            if key == 'avg_size':                
                feature_tensor[tensor_row, layout.numeric_offset + i] = (to_float(properties[key]) - 60.0) / (78.0 - 60.0)
            else:
                feature_tensor[tensor_row, layout.numeric_offset + i] = math.log1p(to_float(properties[key]))
            if layout.include_masks:
                feature_tensor[tensor_row, layout.masks_offset + i] = 1.0
        else:
            feature_tensor[tensor_row, layout.numeric_offset + i] = 0.0
            if layout.include_masks:
                feature_tensor[tensor_row, layout.masks_offset + i] = 0.0

def fill_label_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, labels: List[str]):
    """Fill label features into the feature tensor."""
    # one-hot encode node labels
    idx = {label: i for i, label in enumerate(layout.label_space)}
    for label in labels or []:
        # if label is in layout, set corresponding position to 1
        j = idx.get(label)
        if j is not None:
            feature_tensor[tensor_row, layout.labels_offset + j] = 1.0        

def fill_categorical_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    """Fill categorical features into the feature tensor."""
    for k, c in layout.categorical_mappings.items():
        offset, width = layout.categorical_offsets[k]
        val = properties.get(k)        
        # if k in ['source_port', 'destination_port'], map port number to category
        # if k in ['source_ip', 'destination_ip'], map IP to category based on predefined list
        # otherwise, use val directly
        if k in ['source_port', 'destination_port']:
            val = map_port_to_category(val)
        elif k in ['source_ip', 'destination_ip']:
            val = map_ip_to_category(val)
        elif k == 'ip':
            if val is not None:
                val = map_ip_to_category(val)
            else:
                asset_type = properties.get('asset_type')
                asset_name = properties.get('asset_name')
                if asset_type in ['PLC', 'HMI', 'Flow Sensor'] and asset_name is not None:
                    val = f"{asset_name}_IP"
        else:
            val = val if isinstance(val, str) and val in c else None        

        #val =  val if isinstance(val, str) and val in c else map_port_to_category(val) if k in ['source_port', 'destination_port'] else ("OTHER" if "OTHER" in c else None)
        if val is not None:
            j = c.index(val)
            feature_tensor[tensor_row, offset + j] = 1.0

def map_ip_to_category(ip: Optional[str]) -> str:
    """Map IP address to predefined categories based on known IPs."""
    if ip is None or ip == "0":
        return "OTHER"
    ip_map = {
        "84.3.251.18": "PLC_1_IP",
        "84.3.251.101": "PLC_2_IP",
        "84.3.251.102": "PLC_3_IP",
        "84.3.251.103": "PLC_4_IP",
        "84.3.251.20": "HMI_1_IP",
        "84.3.251.104": "Flow_sensor_1_IP",
        "84.3.251.105": "Flow_sensor_2_IP"
    }
    return ip_map.get(ip, "OTHER")

def fill_ip_mac_features(feature_tensor: torch.Tensor, tensor_row: int, layout: FeatureLayout, properties: Dict[str, Any]):
    """Fill IP and MAC address features into the feature tensor."""
    # IP features
    #ip_buckets = layout.ip_hash_buckets if (layout.use_ip_features and layout.ip_hash_buckets > 0) else 0
    current_offset = layout.ipmac_offset
    src_mac = properties.get('source_mac')
    dst_mac = properties.get('destination_mac')
    mac = properties.get('mac')
    src_mac_feats = mac_to_features(src_mac)
    dst_mac_feats = mac_to_features(dst_mac)    
    mac_feats = mac_to_features(mac)
    feature_tensor[tensor_row, current_offset : current_offset + 6] = torch.tensor(src_mac_feats, dtype=torch.float32)
    feature_tensor[tensor_row, current_offset + 6 : current_offset + 12] = torch.tensor(dst_mac_feats, dtype=torch.float32)
    feature_tensor[tensor_row, current_offset + 12 : current_offset + 18] = torch.tensor(mac_feats, dtype=torch.float32)
    current_offset += 18

def build_edge_index(num_nodes: int, edges: List[Dict[str, Any]], graphid_to_tensorid: Dict[int, int]) -> torch.Tensor:
    """Build edge index tensor from edges and graphid to tensorid mapping."""
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
    """Add node degree as an additional feature column.  Node degree is computed from the edge_index.  It is used to help the model learn structural properties of the graph."""
    deg = degree(edge_index[0], num_nodes=feature_tensor.size(0), dtype=torch.float32).unsqueeze(1)
    deg = torch.log1p(deg)  # log scale
    return torch.cat([feature_tensor, deg], dim=1)

def map_to_features(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: FeatureLayout
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map nodes and edges to feature tensor and edge index."""
    
    unique_nodes = {node['id']: node for node in nodes}
    unique_nodes = list(unique_nodes.values())

    # print(f"Nodes: {len(unique_nodes)}, Edges: {len(edges)}")
    # print(f"Node example: {unique_nodes[0] if unique_nodes else 'No nodes'}")
    # print(f"Edge example: {edges[0] if edges else 'No edges'}")
    #id mappings
    graphid_to_tensorid = {node['id']: i for i, node in enumerate(unique_nodes)}
    tensorid_to_graphid = {i: node['id'] for i, node in enumerate(unique_nodes)}
       
    N = len(graphid_to_tensorid)
    print("Number of nodes:", N)
    #node features
    x = torch.zeros((N, layout.total_width), dtype=torch.float32)
    for node in unique_nodes:
        # print(f"Processing node: {node['id']}")
        # print(f"Node properties: {node.get('properties', {})}")
        # print(f"Node labels: {node.get('labels', [])}")
        i = graphid_to_tensorid[node['id']]
        props = node.get('properties', {})
        labels = node.get('labels', [])
        fill_numeric_and_mask_features(x, i, layout, props)
        fill_physical_measurement_features(x, i, layout, props)
        fill_label_features(x, i, layout, labels)
        fill_categorical_features(x, i, layout, props)
        fill_ip_mac_features(x, i, layout, props)
        #print("Node processed.")

    #edges
    edge_index = build_edge_index(N, edges, graphid_to_tensorid)
    if layout.include_degree:
        x = add_degree_column(x, edge_index)
        

    return x, edge_index

def build_feature_names(layout: FeatureLayout) -> List[str]:
    """Build list of feature names based on the layout."""
    feature_names = []
    #numeric features
    feature_names.extend(layout.property_keys)
    feature_names.extend(f"avg_value_{key}" for key in layout.physical_property_keys)
    #mask features
    if layout.include_masks:
        feature_names.extend([f"{key}_mask" for key in layout.property_keys])
    #label features
    feature_names.extend([f"label_{label}" for label in layout.label_space])
    #categorical features
    for k, vals in layout.categorical_mappings.items():
        feature_names.extend([f"{k}_{val}" for val in vals])
    #IP features
    #if layout.ip_width > 0:
    #    feature_names.extend([f"source_ip_bucket_{i}" for i in range(layout.ip_hash_buckets)])
    #    feature_names.extend([f"destination_ip_bucket_{i}" for i in range(layout.ip_hash_buckets)])
    #MAC features
    feature_names.extend([f"source_mac_byte_{i}" for i in range(6)])
    feature_names.extend([f"destination_mac_byte_{i}" for i in range(6)])
    feature_names.extend([f"mac_byte_{i}" for i in range(6)])
    #degree feature
    if layout.include_degree:
        feature_names.append("node_degree")
    return feature_names


def to_pyg_data(snapshot: dict, schema: Dict[str, Any] = None, write_name: bool = False) -> Data:
    """Convert a snapshot dictionary to a PyG Data object."""
    # Convert the snapshot data into PyG Data format
    nodes = snapshot.get('nodes', [])
    edges = snapshot.get('relationships', [])

    logger.info(f"Converting snapshot {snapshot.get('snapshot_id')} with {len(nodes)} nodes and {len(edges)} edges to PyG Data.")
    layout = build_layout(schema)
    x, edge_index = map_to_features(nodes, edges, layout)

    logger.info(f"Built features with shape: {x.shape}, edge_index shape: {edge_index.shape}")

    data = Data(x=x, edge_index=edge_index)
    data.feature_names = build_feature_names(layout)

    #data.pos_edge_label_index = edge_index
    #data.num_node_features = x.size(1)
    #data.num_nodes = x.size(0)
    #data.graphid_to_tensorid = {node['id']: i for i, node in enumerate(nodes)}
    #data.tensorid_to_graphid = {i: node['id'] for i, node in enumerate(nodes)}
    
    # write feature names once only and tensor to log files for debugging
    if write_name:
        #overwrite existing file with new feature names
        with open(f"./logs/feature_names.txt", "w") as f:
            for name in data.feature_names:
                f.write(f"{name}\n")
    with open(f"./logs/feature_tensor_{snapshot.get('snapshot_id')}.txt", "w") as f:
        for i in range(data.x.size(0)):
            f.write(f"Node {i} features: {data.x[i].tolist()}\n")

    # Copy snapshot-level properties
    snap_id = snapshot.get('snapshot_id', 'unknown_snapshot')
    logging.info(f"Assigning snapshot_id: {snap_id} to data object.")
    data.snapshot_id = snap_id
    # for key in ('snapshot_id', 'system_id', 'start_time', 'end_time', 'duration'):
    #     val = snapshot.get(key)
    #     if val is not None:
    #         data[key] = val
        

    return data

# Heterogeneous GNN model data functions
def to_pyg_hetero_data(snapshot: dict, write_name: bool = False) -> Data:
    """Convert a snapshot dictionary to a PyG Hetero Data object."""
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
    from torch_geometric.transforms import ToUndirected

    nodes = snapshot.get('nodes', [])
    edges = snapshot.get('relationships', [])    
    data = HeteroData()

    node_id_to_index_map = {}

    node_features = {
        'Assets': [],
        'Connections': [],
        'Measurements': [],
        'Endpoints': []
    }

    node_ids = {
        'Assets': [],
        'Connections': [],
        'Measurements': [],
        'Endpoints': []
    }

    #build node features for each node type
    for i, node in enumerate(nodes):
        node_id = node['id']
        labels = node.get('labels', [])
        properties = node.get('properties', {})

        if 'Asset' in labels:
            features = map_hetero_asset_features(properties)
            node_features['Assets'].append(features)
            node_ids['Assets'].append(node_id)
        elif 'Connection' in labels:
            features = map_hetero_connection_features(properties)
            node_features['Connections'].append(features)
            node_ids['Connections'].append(node_id)
        elif 'Measurement' in labels:
            features = map_hetero_measurement_features(properties)
            node_features['Measurements'].append(features)
            node_ids['Measurements'].append(node_id)
        elif 'Endpoint' in labels:
            features = map_hetero_endpoint_features(properties)
            node_features['Endpoints'].append(features)
            node_ids['Endpoints'].append(node_id)

    # Convert lists to tensors
    logging.info("Converting node feature lists to tensors.")
    node_feature_tensors = {}
    for node_type in node_features:
        if node_features[node_type]:
            node_feature_tensors[node_type] = torch.tensor(node_features[node_type], dtype=torch.float)
        else:
            # Handle case with no nodes of this type - create empty tensor with correct feature size
            feature_size = len(get_hetero_column_names(node_type))
            node_feature_tensors[node_type] = torch.empty((0, feature_size), dtype=torch.float)

    logging.info("Node feature tensors created.")
    #populate node features for each type and build ID mappings
    for node_type, id_list in node_ids.items():        
        data[node_type].x = node_feature_tensors[node_type]
        data[node_type].num_nodes = len(id_list)

        for i, node_id in enumerate(id_list):
            node_id_to_index_map[node_id] = (node_type, i)

    # build edge indices for each node type
    edge_indices = {}
    logging.info("Building edge indices for each edge type.")
    for edge in edges:
        src_id = edge.get('source')
        dst_id = edge.get('target')
        edge_type = edge['type']

        if src_id not in node_id_to_index_map or dst_id not in node_id_to_index_map:
            continue

        src_type, src_idx = node_id_to_index_map[src_id]
        dst_type, dst_idx = node_id_to_index_map[dst_id]

        edge_key = (src_type, edge_type, dst_type)

        if edge_key not in edge_indices:
            edge_indices[edge_key] = ([], [])
        edge_indices[edge_key][0].append(src_idx)
        edge_indices[edge_key][1].append(dst_idx)

    # Convert edge indices to tensors
    logging.info("Converting edge index lists to tensors.")
    for edge_key, (src_list, dst_list) in edge_indices.items():
        data[edge_key].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    logging.info("Applying transformations to data.")
    # handle undirected edges
    data = ToUndirected()(data)
        
    logging.info("Adding feature names.")
    #Add feature names  
    data['Assets'].feature_names = get_hetero_column_names('Assets')
    data['Connections'].feature_names = get_hetero_column_names('Connections')
    data['Measurements'].feature_names = get_hetero_column_names('Measurements')
    data['Endpoints'].feature_names = get_hetero_column_names('Endpoints')

    # Copy snapshot-level properties
    logging.info("Copying snapshot-level properties to data object.")
    snap_id = snapshot.get('snapshot_id', 'unknown_snapshot')
    logging.info(f"Assigning snapshot_id: {snap_id} to data object.")
    data.snapshot_id = snap_id

    logging.info("Finished writing feature names and tensors to log files.")
    return data

def write_hetero_feature_mappings(data, snapshot_id, write_name: bool = False):
    logging.info("Writing feature names and tensors to log files for debugging.")
    
    # Feature names
    if write_name:
        with open(f"./logs/hetero_feature_names_asset.txt", "w") as f:
            for name in data['Assets'].feature_names:
                f.write(f"{name}\n")
                
        with open(f"./logs/hetero_feature_names_connection.txt", "w") as f:
            for name in data['Connections'].feature_names:
                f.write(f"{name}\n")
        with open(f"./logs/hetero_feature_names_measurement.txt", "w") as f:
            for name in data['Measurements'].feature_names:
                f.write(f"{name}\n")
        with open(f"./logs/hetero_feature_names_endpoint.txt", "w") as f:
            for name in data['Endpoints'].feature_names:
                f.write(f"{name}\n")
    #Assets
    with open(f"./logs/hetero_feature_tensor_asset_{snapshot_id}.txt", "w") as f:        
        for i in range(data['Assets'].x.size(0)):
            f.write(f"{data['Assets'].x[i].tolist()}\n")
    
    #Connections
    with open(f"./logs/hetero_feature_tensor_connection_{snapshot_id}.txt", "w") as f:
        for i in range(data['Connections'].x.size(0)):
            f.write(f"{data['Connections'].x[i].tolist()}\n")
    
    #Measurements
    with open(f"./logs/hetero_feature_tensor_measurement_{snapshot_id}.txt", "w") as f:
        for i in range(data['Measurements'].x.size(0)):
            f.write(f"{data['Measurements'].x[i].tolist()}\n")

    #Endpoints
    with open(f"./logs/hetero_endpoint_feature_tensor_{snapshot_id}.txt", "w") as f:
        for i in range(data['Endpoints'].x.size(0)):
            f.write(f"{data['Endpoints'].x[i].tolist()}\n")
    # asset_features = ['asset_type']
    # connection_features = ['avg_size', 'num_connections', 'protocol', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'source_mac', 'destination_mac']
    # measurement_features = ['measurement_type', 'avg_value']
    # endpoint_features = ['ip']

    # 'categorical_mappings': {
    #     'protocol': ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "OTHER"],
    #     'asset_type': ["HMI", "PLC", "Pump", "Valv", "Flow Sensor", "Tank", "PressureSensor"],
    #     'measurement_type': ["state", "pressure", "val"],
    #     'destination_port': ["well_known", "registered", "ephemeral", "other" ], # well-known: 0-1023, registered: 1024-49151, ephemeral: 49152-65535
    #     'source_port': ["well_known", "registered", "ephemeral", "other" ],
    #     'source_ip': ["PLC_1_IP", "PLC_2_IP", "PLC_3_IP", "PLC_4_IP", "HMI_1_IP","Flow_sensor_1_IP","Flow_sensor_2_IP", "OTHER"],
    #     'destination_ip': ["PLC_1_IP", "PLC_2_IP", "PLC_3_IP", "PLC_4_IP", "HMI_1_IP","Flow_sensor_1_IP","Flow_sensor_2_IP", "OTHER"],
    #     'ip': ["PLC_1_IP", "PLC_2_IP", "PLC_3_IP", "PLC_4_IP", "HMI_1_IP","Flow_sensor_1_IP","Flow_sensor_2_IP", "OTHER"]
        
    # }

def get_hetero_column_names(node_type: str) -> List[str]:
    """Get feature column names for a given heterogeneous node type."""
    if node_type == 'Assets':
        return [f"asset_type_{cat}" for cat in global_schema['categorical_mappings']['asset_type']]
    elif node_type == 'Connections':
        feature_names = []
        feature_names.extend([f"protocol_{cat}" for cat in global_schema['categorical_mappings']['protocol']])
        feature_names.extend([f"source_ip_{cat}" for cat in global_schema['categorical_mappings']['source_ip']])
        feature_names.extend([f"destination_ip_{cat}" for cat in global_schema['categorical_mappings']['destination_ip']])
        feature_names.extend([f"source_port_{cat}" for cat in global_schema['categorical_mappings']['source_port']])
        feature_names.extend([f"destination_port_{cat}" for cat in global_schema['categorical_mappings']['destination_port']])
        feature_names.extend(['avg_size', 'num_connections'])
        return feature_names
    elif node_type == 'Measurements':
        feature_names = []
        feature_names.extend([f"measurement_type_{cat}" for cat in global_schema['categorical_mappings']['measurement_type']])
        feature_names.extend(['avg_value_state', 'avg_value_pressure', 'avg_value_val'])
        return feature_names
    elif node_type == 'Endpoints':
        return [f"ip_{cat}" for cat in global_schema['categorical_mappings']['ip']]
    else:
        return []

def map_hetero_asset_features(properties: Dict[str, Any]):
    """Map asset node properties to feature tensor."""
    #logger.info("Mapping asset features.")

    # build one-hot encoding based on predefined categories
    categories = global_schema['categorical_mappings']['asset_type']
    feature_vector = [0.0] * len(categories)
    asset_type = properties.get('asset_type')
    if asset_type is not None:
        for i, category in enumerate(categories):
            if asset_type.lower() == category.lower():
                feature_vector[i] = 1.0
    return feature_vector

def map_hetero_connection_features(properties: Dict[str, Any]):
    """Map connection node properties to feature tensor."""
    #logger.info("Mapping connection features.")
     # build one-hot encoding based on predefined categories
    categories_protocol = global_schema['categorical_mappings']['protocol']
    categories_source_ip = global_schema['categorical_mappings']['source_ip']
    categories_destination_ip = global_schema['categorical_mappings']['destination_ip']
    categories_source_port = global_schema['categorical_mappings']['source_port']
    categories_destination_port = global_schema['categorical_mappings']['destination_port']

    avg_size = properties.get('avg_size')
    num_connections = properties.get('num_connections')
    
    #logger.info("Mapping protocol feature.")
    #protocol one-hot encoding
    feature_vector = [0.0] * len(categories_protocol)
    protocol = properties.get('protocol')
    if protocol is not None:
        for i, category in enumerate(categories_protocol):
            if protocol.upper() == category.upper():
                feature_vector[i] = 1.0

    #logger.info("Mapping source IP feature.")
    # source ip one-hot encoding
    source_ip_vector = [0.0] * len(categories_source_ip)
    source_ip = properties.get('source_ip')
    source_ip_mapped = map_ip_to_category(source_ip)
    if source_ip_mapped is not None:
        for i, category in enumerate(categories_source_ip):
            if source_ip_mapped == category:
                source_ip_vector[i] = 1.0
    feature_vector.extend(source_ip_vector)

    #logger.info("Mapping destination IP feature.")
    # destination ip one-hot encoding
    destination_ip_vector = [0.0] * len(categories_destination_ip)
    destination_ip = properties.get('destination_ip')
    destination_ip_mapped = map_ip_to_category(destination_ip)
    if destination_ip_mapped is not None:
        for i, category in enumerate(categories_destination_ip):
            if destination_ip_mapped == category:
                destination_ip_vector[i] = 1.0
    feature_vector.extend(destination_ip_vector)

    #logger.info("Mapping source port feature.")
    # source port one-hot encoding
    source_port_vector = [0.0] * len(categories_source_port)
    source_port = properties.get('source_port')
    source_port_mapped = map_port_to_category(source_port)
    if source_port_mapped is not None:
        for i, category in enumerate(categories_source_port):
            if source_port_mapped == category:
                source_port_vector[i] = 1.0
    feature_vector.extend(source_port_vector)

    #logger.info("Mapping destination port feature.")
    # destination port one-hot encoding
    destination_port_vector = [0.0] * len(categories_destination_port)
    destination_port = properties.get('destination_port')
    destination_port_mapped = map_port_to_category(destination_port)
    if destination_port_mapped is not None:
        for i, category in enumerate(categories_destination_port):
            if destination_port_mapped == category:
                destination_port_vector[i] = 1.0
    feature_vector.extend(destination_port_vector)

    #logger.info("Mapping numeric features.")
    # numeric features
    feature_vector.extend([
        (to_float(avg_size) - 60.0) / (78.0 - 60.0) if avg_size is not None else 0.0,
        math.log1p(to_float(num_connections)) if num_connections is not None else 0.0
    ])
    
    return feature_vector

def map_hetero_measurement_features(properties: Dict[str, Any]):
    """Map measurement node properties to feature tensor."""
    #logger.info("Mapping measurement features.")
    categories = global_schema['categorical_mappings']['measurement_type']
    feature_vector = [0.0] * len(categories)
    feature_vector_numeric = [0.0] * 3  # for avg_value
    measure_type = properties.get('measurement_type')
    val = properties.get('avg_value')
    if measure_type is not None:
        for i, category in enumerate(categories):
            if measure_type.lower() == category.lower():
                feature_vector[i] = 1.0
    if val is not None:
        if measure_type == 'state':
            feature_vector_numeric[0] = to_float(val)
        elif measure_type == 'pressure':
            feature_vector_numeric[1] = to_float(val) / 1843.0
        elif measure_type == 'val':
            feature_vector_numeric[2] = to_float(val) / 4683.0
        
    return feature_vector + feature_vector_numeric


def map_hetero_endpoint_features(properties: Dict[str, Any]):
    """Map endpoint node properties to feature tensor."""
    #logger.info("Mapping endpoint features.")
    # build one-hot encoding based on predefined categories
    categories = global_schema['categorical_mappings']['ip']
    feature_vector = [0.0] * len(categories)
    
    val = properties.get('ip')
    ip_mapped = map_ip_to_category(val)
    if ip_mapped is not None:
        for i, category in enumerate(categories):
            if ip_mapped == category:
                feature_vector[i] = 1.0
    return feature_vector
            