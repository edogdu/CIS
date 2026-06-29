# repositories -> graphs -> pyg_builder.py
import csv
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from collections import defaultdict
from dataclasses import dataclass
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce, degree
import torch
from typing import Tuple, Dict, Any, List, Optional
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
from torch_geometric.transforms import ToUndirected
import pandas as pd
#from torch_geometric.transforms import LocalClusteringCoefficient

logger = logging.getLogger("data_builder")

#global schema definition - can be overridden by passing a schema dict to build_layout
global_schema = {
    'property_keys': [
        # Network properties
        "avg_size", #, "destination_port", "destination_total_packets",
         "num_connections",  #, "source_port", "source_total_packets",
         "tcp_cwr_count", "tcp_ece_count", "tcp_urg_count", "tcp_ack_count",
         "tcp_psh_count", "tcp_rst_count", "tcp_syn_count", "tcp_fin_count",
         "tcp_ack_ratio",
         "modbus_response_count", "modbus_response_ratio", "avg_modbus_response_code"
    ],
    'physical_property_keys': [
        "state",
        "pressure",
        "value",
        #"min_value",
        #"max_value",
        "stddev_value"
    ],
    # Node Labels - used for node type one-hot encoding
    'label_space': ["Endpoint", "Asset", "Connection", "Measurement"],
    # Categorical feature mappings - used for categorical feature one-hot encoding
    'categorical_mappings': {
        'protocol': ["TCP", "ICMP", "IP", "Modbus", "ARP", "OTHER"],
        'asset_type': ["HMI", "PLC", "Pump", "Valv", "Flow Sensor", "Tank", "PressureSensor","External"],
        'measurement_type': ["state", "pressure", "val"],

    },
    "mac_width": 18
}
# Label spaces
#y_labels = ['normal', 'anomaly', 'scan', 'dos', 'mitm', 'physical fault']
y_labels = ['normal', 'scan', 'dos', 'mitm', 'physical fault']
y_bin_labels = ['normal', 'anomaly']
y_anomaly_labels = ['scan', 'dos', 'mitm', 'physical fault']
y_anomaly_Index_mapback = [1,2,3,4]

# physical asset types
physical_asset_types = ['Pump', 'Valve', 'Tank', 'FlowSensor']

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
def ip_to_onehot_features(ip: str, buckets: int=4) -> list[float]:
    """Convert IP address string to one-hot encoded features based on hash buckets."""
    if ip is None or buckets <= 0 or ip == "0" or ip == "" or type(ip) is not str:
        return [0.0] * buckets
    ip = ip.split('/')[0] # remove CIDR if present
    parts = ip.split('.')
    features = [0.0] * buckets
    if len(parts) != 4:
        return features
    for i in range(len(parts)):
        try:
            part_int = int(parts[i])
            features[i] = part_int / 255.0
        except ValueError:
            features[i] = 0.0

    return features

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
    if mac is None or mac == "0" or mac == "" or type(mac) is not str:
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
            v = float(val)
            return 0.0 if math.isnan(v) or math.isinf(v) else v
        except ValueError:
            return 0.0
    else:
        return 0.0

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

def calculate_endpoint_stats(nodes, snapshot_id=None) -> None:
    """Calculate the following features foreach endpoint node, using connection nodes:
        - endpoint_num_unique_ports: number of unique ports connected to the endpoint
        - endpoint_port_entropy: entropy of the port distribution
        - endpoint_unique_peer_count: number of unique peers connected to the endpoint
        - endpoint_in_out_ratio: ratio of incoming to outgoing connections
        - endpoint_num_unique_protocols: number of unique protocols used by the endpoint
        - endpoint_protocol_entropy: entropy of the protocol distribution
    """
    endpoint_metadata = {}
    connection_props = {}

    node_context = {
        "Asset": {},
        "Endpoint": {},
        "Connection": {},
        "Measurement": {},
    }

    for node in nodes:
        node_id = node['id']
        labels = node.get('labels', [])
        props = node.get('properties', {})

        if 'Endpoint' in labels:
            endpoint_metadata[node_id] = {
                'ip': props.get('ip'),
                'mac': props.get('mac'),
            }
        elif 'Connection' in labels:
            connection_props[node_id] = props

        elif "Asset" in labels:
          node_context["Asset"][node_id] = {
            "asset_name": props.get("name"),
            "asset_type": props.get("asset_type"),
            "plc_tag": props.get("tag"),
            "ip": props.get("ip"),
        }
        elif "Endpoint" in labels:
          node_context["Endpoint"][node_id] = {
            "ip": props.get("ip"),
            "mac": props.get("mac"),
        }
        elif "Connection" in labels:
          # every connection node needs to get full raw metadata
          # can add engineered features for connections, especially empty connections too
            node_context["Connection"][node_id] = props.copy()

        elif "Measurement" in labels:
          node_context["Measurement"][node_id] = {
            "measurement_type": props.get("measurement_type"),
            "asset_id": props.get("asset_id"),
            "avg_value": props.get("avg_value"),
        }

    # Lookup helpers
    ip_to_endpoint = {}
    mac_to_endpoint = {}
    for endpoint_id, meta in endpoint_metadata.items():
        ip = meta.get('ip')
        mac = meta.get('mac')
        if ip:
            ip_to_endpoint.setdefault(ip, set()).add(endpoint_id)
        if mac:
            mac_to_endpoint.setdefault(mac, set()).add(endpoint_id)

    endpoint_stats: Dict[str, Dict[str, Any]] = {}
    for conn_id, props in connection_props.items():
        src_ip = props.get('source_ip')
        dst_ip = props.get('destination_ip')
        src_mac = props.get('source_mac')
        dst_mac = props.get('destination_mac')
        protocol = props.get('protocol')
        #logging.info(f"Processing connection {conn_id}: src_ip={src_ip}, dst_ip={dst_ip}, src_mac={src_mac}, dst_mac={dst_mac}, protocol={protocol}")
        if src_ip is not None:
            src_ip = str(src_ip).split('/')[0]
        if dst_ip is not None:
            dst_ip = str(dst_ip).split('/')[0]

        src_endpoints = ip_to_endpoint.get(src_ip, set()).union(mac_to_endpoint.get(src_mac, set()))
        dst_endpoints = ip_to_endpoint.get(dst_ip, set()).union(mac_to_endpoint.get(dst_mac, set()))

        for endpoint_id in src_endpoints:
            stats = endpoint_stats.setdefault(endpoint_id, {
                'ports': {},
                'peers': set(),
                'incoming': 0,
                'outgoing': 0,
                'protocols': {}
            })
            port = props.get('destination_port')
            if port is not None:
                stats['ports'][port] = stats['ports'].get(port, 0) + 1
            peer = dst_ip or dst_mac
            if peer:
                stats['peers'].add(peer)
            stats['outgoing'] += 1
            if protocol:
                stats['protocols'][protocol] = stats['protocols'].get(protocol, 0) + 1

        for endpoint_id in dst_endpoints:
            stats = endpoint_stats.setdefault(endpoint_id, {
                'ports': {},
                'peers': set(),
                'incoming': 0,
                'outgoing': 0,
                'protocols': {}
            })
            port = props.get('source_port')
            if port is not None:
                stats['ports'][port] = stats['ports'].get(port, 0) + 1
            peer = src_ip or src_mac
            if peer:
                stats['peers'].add(peer)
            stats['incoming'] += 1
            if protocol:
                stats['protocols'][protocol] = stats['protocols'].get(protocol, 0) + 1

    # Compute final stats
    final_stats = {}
    for endpoint_id, stats in endpoint_stats.items():
        ports = stats['ports']
        total_ports = sum(ports.values())
        port_entropy = -sum((count / total_ports) * math.log2(count / total_ports) for count in ports.values()) if total_ports > 0 else 0.0

        protocols = stats['protocols']
        total_protocols = sum(protocols.values())
        protocol_entropy = -sum((count / total_protocols) * math.log2(count / total_protocols) for count in protocols.values()) if total_protocols > 0 else 0.0

        incoming = stats['incoming']
        outgoing = stats['outgoing']
        in_out_ratio = incoming / (outgoing + 1e-6)  # avoid division by zero

        logging.info(f'port count: {len(ports)}')
        # append port count to csv file for data analysis
        os.makedirs("./logs", exist_ok=True)
        with open(f'./logs/port_counts.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([snapshot_id, endpoint_id, len(ports)])
        final_stats[endpoint_id] = {
            'endpoint_ports_count_bucket_1': 1.0 if len(ports) == 1 else 0.0,
            'endpoint_ports_count_bucket_2_10': 1.0 if 2 <= len(ports) <= 50 else 0.0,
            'endpoint_ports_count_bucket_11_100': 1.0 if 51 <= len(ports) <= 100 else 0.0,
            'endpoint_ports_count_bucket_high': 1.0 if len(ports) > 100 else 0.0,
            'endpoint_port_entropy': port_entropy,
            'endpoint_unique_peer_count': len(stats['peers']),
            'endpoint_in_out_ratio': in_out_ratio,
            'endpoint_num_unique_protocols': len(protocols),
            'endpoint_protocol_entropy': protocol_entropy
        }

    return final_stats, node_context

def merge_physical_asset_info(nodes, snapshot_id):
    """Merge physical asset information from asset and measurement nodes."""
    physical_assets = {}
    for node in nodes:
        labels = node.get('labels', [])
        props = node.get('properties', {})
        node_id = node['id']

        #if any of physical_asset_types match on labels, process as physical asset
        if any(patype in labels for patype in physical_asset_types):


            asset_type = props.get('asset_type', '').lower()
            asset_id = props.get('asset_id')
            #must have asset_id property
            if not asset_id:
                continue

            # look for asset_id property in pysical_assets
            if asset_id not in physical_assets:
                physical_assets[asset_id] = {
                    'measurements': {}
                }
            physical_assets[asset_id]['asset_type'] = asset_type

        elif 'Measurement' in labels:
            asset_id = props.get('asset_id')
            #must have asset_id property
            if not asset_id:
                continue
            if asset_id not in physical_assets:
                physical_assets[asset_id] = {
                    'asset_type': 'unknown',
                    'measurements': {}
                }
            avg_value = to_float(props.get('avg_value'))
            physical_assets[asset_id]['measurements']['avg_value'] = avg_value
            physical_assets[asset_id]['measurements']['stddev_value'] = to_float(props.get('stddev_value'))

    return physical_assets

# Heterogeneous GNN model data functions
def to_pyg_hetero_data(snapshot: dict, write_name: bool = False) -> Data:
    """Convert a snapshot dictionary to a PyG Hetero Data object."""
    nodes = snapshot.get('nodes', [])
    edges = snapshot.get('relationships', [])
    data = HeteroData()
    node_id_to_index_map = {}

    node_features = {
        'Connections': [],
        'Endpoints': [],
        'Asset': [],
        'Measurement': [],
        'Pumps': [],
        'FlowSensors': [],
        'Tanks': [],
        'Valves': []
    }

    node_ids = {
        'Connections': [],
        'Endpoints': [],
        'Asset': [],
        'Measurement': [],
        'Pumps': [],
        'FlowSensors': [],
        'Tanks': [],
        'Valves': []
    }

    snap_id = snapshot.get('snapshot_id', 'unknown_snapshot')
    endpoint_stats, node_context = calculate_endpoint_stats(nodes, snapshot_id=snap_id)
    physical_assets = merge_physical_asset_info(nodes, snapshot_id=snap_id)

    #build node features for each node type
    for i, node in enumerate(nodes):
        node_id = node['id']
        labels = node.get('labels', [])
        properties = node.get('properties', {})

        if any(patype in labels for patype in physical_asset_types):
            l = set(labels).intersection(physical_asset_types)
            node_type = f'{l.pop()}s'

            asset_id = properties.get('asset_id')
            asset_info = physical_assets.get(asset_id, {})
            if not asset_info:
                logging.warning(f'No physical asset info found for asset_id: {asset_id} in node_id: {node_id}')
                continue
            features = [0.0] * 2
            features[0] = asset_info.get('measurements', {}).get('avg_value', 0.0)
            features[1] = asset_info.get('measurements', {}).get('stddev_value', 0.0)
            node_features[node_type].append(features)
            node_ids[node_type].append(node_id)

        elif 'Connection' in labels:
            features = map_hetero_connection_features(properties)
            node_features['Connections'].append(features)
            node_ids['Connections'].append(node_id)

        elif 'Endpoint' in labels:
            features = map_hetero_endpoint_features(properties)
            stat_features = [0.0] * 6
            stats = endpoint_stats.get(node_id, {})
            stat_features[0] = float(len(stats.get('ports', {})))

            stat_features[1] = float(stats.get('endpoint_port_entropy', 0.0))
            stat_features[2] = float(stats.get('endpoint_unique_peer_count', 0))
            stat_features[3] = float(stats.get('endpoint_in_out_ratio', 0.0))
            stat_features[4] = float(stats.get('endpoint_num_unique_protocols', 0))
            stat_features[5] = float(stats.get('endpoint_protocol_entropy', 0.0))
            features.extend(stat_features)
            node_features['Endpoints'].append(features)
            node_ids['Endpoints'].append(node_id)

        elif 'Asset' in labels:
          features = map_hetero_asset_features(properties)
          node_features['Asset'].append(features)
          node_ids['Asset'].append(node_id)

        elif 'Measurement' in labels:
          features = map_hetero_measurement_features(properties)
          node_features['Measurement'].append(features)
          node_ids['Measurement'].append(node_id)

    # Convert lists to tensors
    #logging.info("Converting node feature lists to tensors.")
    node_feature_tensors = {}
    for node_type in node_features:
        if node_features[node_type]:
            node_feature_tensors[node_type] = torch.tensor(node_features[node_type], dtype=torch.float)
        else:
            # Handle case with no nodes of this type - create empty tensor with correct feature size
            feature_size = len(get_hetero_column_names(node_type)) # -1 for degree feature to be added later
            node_feature_tensors[node_type] = torch.empty((0, feature_size), dtype=torch.float)

    #logging.info("Node feature tensors created.")
    #populate node features for each type and build ID mappings
    for node_type, id_list in node_ids.items():
        data[node_type].x = node_feature_tensors[node_type]
        # graph to schema
        data['Connection'].feature_names = get_hetero_column_names('Connections')
        data['Endpoint'].feature_names = get_hetero_column_names('Endpoints')

        data['Asset'].feature_names = get_hetero_column_names('Assets')
        data['Measurement'].feature_names = get_hetero_column_names('Measurements')

        data[node_type].num_nodes = len(id_list)
        data[node_type].original_id = id_list

        for i, node_id in enumerate(id_list):
            node_id_to_index_map[node_id] = (node_type, i)

    # build edge indices for each node type
    edge_indices = {}
    #logging.info("Building edge indices for each edge type.")
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
    #logging.info("Converting edge index lists to tensors.")
    for edge_key, (src_list, dst_list) in edge_indices.items():
        data[edge_key].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    #logging.info("Adding feature names.")
    #Add feature names
    #data['Assets'].feature_names = get_hetero_column_names('Assets')
    #FlowSensor = Asset{asset_type:'Flow_sensor'}
    data['Connections'].feature_names = get_hetero_column_names('Connections')
    data['Pumps'].feature_names = get_hetero_column_names('Pumps')
    data['Valves'].feature_names = get_hetero_column_names('Valves')
    data['Tanks'].feature_names = get_hetero_column_names('Tanks')
    data['FlowSensors'].feature_names = get_hetero_column_names('FlowSensors')
    data['Endpoints'].feature_names = get_hetero_column_names('Endpoints')

    # add node degree
    #logging.info("Adding node degree features to hetero data.")
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes

        # Skip if no nodes of this type
        if num_nodes == 0:
            # We still need to add a "degree" column of size 0 to match feature_names
            deg_column = torch.empty((0, 1), dtype=torch.float32)
            data[node_type].x = torch.cat([data[node_type].x, deg_column], dim=1)
            continue

        # Collect all edge indices where this node_type is either source or destination
        all_edge_indices_for_node_type = []
        for edge_type in data.edge_types:
            # If this node type is the source
            if edge_type[0] == node_type:
                all_edge_indices_for_node_type.append(data[edge_type].edge_index[0])
            # If this node type is the destination
            if edge_type[2] == node_type:
                 all_edge_indices_for_node_type.append(data[edge_type].edge_index[1])

    #logging.info(f"Assigning snapshot_id: {snap_id} to data object.")
    data = ToUndirected()(data)

    # Re-attach custom attributes AFTER ToUndirected()
    for node_type, id_list in node_ids.items():
        data[node_type].original_id = id_list

    data.snapshot_id = snap_id
    data.context_lookup = node_context

    #logging.info("Finished writing feature names and tensors to log files.")
    return data

def calculate_entropy_from_counts(counts: dict) -> float:
    """Calculate entropy given a dictionary of counts."""
    if not counts:
        return 0.0
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p + 1e-12)  # add small value to avoid log(0)
    return entropy

def visualize_features_distribution(data: list[HeteroData]):
    """Visualize the distribution of features for each node type in the
    entire heterogeneous dataset."""
    os.makedirs("./exports/data/feature_distributions", exist_ok=True)
    #combine node information from all graphs
    combined_data = []
    # create holders with key (node_type,feature_name)
    for graph in data:
        for node_type in graph.node_types:
            feature_names = graph[node_type].feature_names
            x = graph[node_type].x.numpy()
            for i, feature_name in enumerate(feature_names):
                combined_data.append({
                    'node_type': node_type,
                    'feature_name': feature_name,
                    'values': x[:, i]
                })
    df = pd.DataFrame(combined_data)
    # plot distribution for each node type and feature
    for (node_type, feature_name), group in df.groupby(['node_type', 'feature_name']):
        plt.figure(figsize=(10, 6))
        plt.hist(group['values'].explode(), bins=50, alpha=0.7)
        plt.title(f'Distribution of {feature_name} for node type {node_type}')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'./exports/data/feature_distributions/{node_type}_{feature_name}_distribution.png')
        plt.close()

def write_hetero_feature_mappings(data, snapshot_id, write_name: bool = False):
    logging.info("Writing feature names and tensors to log files for debugging.")

    # Feature names
    if write_name:
        with open(f"./logs/hetero_feature_names_connection.txt", "w") as f:
            for name in get_hetero_column_names('Connections'):
                f.write(f"{name}\n")

        with open(f"./logs/hetero_feature_names_endpoint.txt", "w") as f:
            for name in get_hetero_column_names('Endpoints'):
                f.write(f"{name}\n")
    #Assets
    for asset_type in physical_asset_types:
        with open(f"./logs/hetero_feature_names_{asset_type.lower()}_{snapshot_id}.txt", "w") as f:
            for name in get_hetero_column_names(f'{asset_type}s'):
                f.write(f"{name}\n")

    #Connections
    with open(f"./logs/hetero_feature_tensor_connection_{snapshot_id}.txt", "w") as f:
        for i in range(data['Connections'].x.size(0)):
            f.write(f"{data['Connections'].x[i].tolist()}\n")

    #Endpoints
    with open(f"./logs/hetero_endpoint_feature_tensor_{snapshot_id}.txt", "w") as f:
        for i in range(data['Endpoints'].x.size(0)):
            f.write(f"{data['Endpoints'].x[i].tolist()}\n")

def get_hetero_column_names(node_type: str):
    if node_type in ('Tanks','FlowSensors','Pumps','Valves'):
      return ['avg_value','stddev_value']
    elif node_type == 'Assets':
      return [f"asset_type_{cat}" for cat in global_schema['categorical_mappings']['asset_type']]
    elif node_type == 'Measurements':
      return ['avg_value', 'stddev_value']

    elif node_type == 'Connections':
        names = []
        names += [f"protocol_{cat}" for cat in global_schema['categorical_mappings']['protocol']]
        names += [f"source_mac_byte_{i}" for i in range(6)]
        names += [f"destination_mac_byte_{i}" for i in range(6)]
        names += ['avg_size','num_connections']
        names += ['tcp_cwr_count','tcp_ece_count','tcp_urg_count','tcp_ack_count',
                  'tcp_psh_count','tcp_rst_count','tcp_syn_count','tcp_fin_count']
        names += ['tcp_syn_ratio','tcp_ack_ratio','tcp_rst_ratio']
        names += ['modbus_response_count','modbus_response_ratio',
                  'avg_modbus_response_code','modbus_response_present']
        return names

    elif node_type == 'Endpoints':
        names = [f"mac_byte_{i}" for i in range(6)]
        names += [f"ip_part_{i}" for i in range(4)]
        names += ['endpoint_num_unique_ports','endpoint_port_entropy',
                  'endpoint_unique_peer_count','endpoint_in_out_ratio',
                  'endpoint_num_unique_protocols','endpoint_protocol_entropy']
        return names

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

def map_hetero_measurement_features(properties):
    return [
        to_float(properties.get('avg_value')),
        to_float(properties.get('stddev_value'))
    ]

def map_hetero_connection_features(properties: Dict[str, Any]):
    """Map connection node properties to feature tensor."""
    #logger.info("Mapping connection features.")
     # build one-hot encoding based on predefined categories
    categories_protocol = global_schema['categorical_mappings']['protocol']

    source_port = properties.get('source_port')
    destination_port = properties.get('destination_port')
    avg_size = properties.get('avg_size')
    num_connections = properties.get('num_connections')

    tcp_cwr_count = properties.get('tcp_cwr_count')
    tcp_ece_count = properties.get('tcp_ece_count')
    tcp_urg_count = properties.get('tcp_urg_count')
    tcp_ack_count = properties.get('tcp_ack_count')
    tcp_psh_count = properties.get('tcp_psh_count')
    tcp_rst_count = properties.get('tcp_rst_count')
    tcp_syn_count = properties.get('tcp_syn_count')
    tcp_fin_count = properties.get('tcp_fin_count')
    tcp_syn_ratio = properties.get('tcp_syn_ratio')
    tcp_ack_ratio = properties.get('tcp_ack_ratio')

    # calculate tcp_rst_ratio, since it's missing in the properties
    tcp_rst_ratio = (tcp_rst_count / num_connections) if tcp_rst_count is not None and num_connections is not None and num_connections != 0 else 0.0
    modbus_response_count = properties.get('modbus_response_count')
    modbus_response_ratio = properties.get('modbus_response_ratio')
    avg_modbus_response_code = properties.get('avg_modbus_response_code')
    modbus_response_present = properties.get('modbus_response_present')

    #logger.info("Mapping protocol feature.")
    #protocol one-hot encoding
    feature_vector = [0.0] * len(categories_protocol)
    protocol = properties.get('protocol')
    if protocol is not None:
        for i, category in enumerate(categories_protocol):
            if protocol.upper() == category.upper():
                feature_vector[i] = 1.0

    #logger.info("Mapping source IP feature.")
    source_mac = properties.get('source_mac')
    source_mac_vector = mac_to_features(source_mac)
    feature_vector.extend(source_mac_vector)

    #logger.info("Mapping destination IP feature.")
    destination_mac = properties.get('destination_mac')
    destination_mac_vector = mac_to_features(destination_mac)
    feature_vector.extend(destination_mac_vector)

    #logger.info("Mapping numeric features.")
    # numeric features
    feature_vector.extend([
        to_float(avg_size) if avg_size is not None else 0.0,
        to_float(num_connections) if num_connections is not None else 0.0
    ])

    # add tcp flag counts and ratios
    feature_vector.extend([to_float(tcp_cwr_count) if tcp_cwr_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_ece_count) if tcp_ece_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_urg_count) if tcp_urg_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_ack_count) if tcp_ack_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_psh_count) if tcp_psh_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_rst_count) if tcp_rst_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_syn_count) if tcp_syn_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_fin_count) if tcp_fin_count is not None else 0.0])
    feature_vector.extend([to_float(tcp_syn_ratio) if tcp_syn_ratio is not None else 0.0])
    feature_vector.extend([to_float(tcp_ack_ratio) if tcp_ack_ratio is not None else 0.0])
    feature_vector.extend([to_float(tcp_rst_ratio) if tcp_rst_ratio is not None else 0.0])

    # add modbus response features
    feature_vector.extend([to_float(modbus_response_count) if modbus_response_count is not None else 0.0])
    feature_vector.extend([to_float(modbus_response_ratio) if modbus_response_ratio is not None else 0.0])
    feature_vector.extend([to_float(avg_modbus_response_code) if avg_modbus_response_code is not None else 0.0])
    feature_vector.extend([to_float(modbus_response_present) if modbus_response_present is not None else 0.0])

    return feature_vector

def map_hetero_measurement_features(properties: Dict[str, Any]):
    """Map measurement node properties to feature tensor."""
    #logger.info("Mapping measurement features.")
    #categories = global_schema['categorical_mappings']['measurement_type']
    #feature_vector = [0.0] * len(categories)

    feature_vector_numeric = [0.0] * 2  # for avg_value and stddev_value
    measure_type = properties.get('measurement_type')
    val = properties.get('avg_value')
    stddev_val = properties.get('stddev_value')
    min_val = properties.get('min_value')
    max_val = properties.get('max_value')

    if measure_type != 'state':
        feature_vector_numeric[0] = to_float(val) if val is not None else 0.0
        feature_vector_numeric[1] = to_float(stddev_val) if stddev_val is not None and stddev_val > 0.01 else 0.0

    else:
        feature_vector_numeric[0] = to_float(val) if val is not None else 0.0
        # for 'state' type, we use a binary feature indicating if state changed
        feature_vector_numeric[1] = to_float(stddev_val) if stddev_val is not None and stddev_val > 0.01 else 0.0

    return feature_vector_numeric


def map_hetero_endpoint_features(properties: Dict[str, Any]):
    """Map endpoint node properties to feature tensor."""
    #logger.info("Mapping endpoint features.")
    # build one-hot encoding based on predefined categories
    #categories = global_schema['categorical_mappings']['ip']
    #feature_vector = [0.0] * len(categories)

    mac = properties.get('mac')
    feature_vector = mac_to_features(mac)
    val = properties.get('ip')
    feature_vector.extend(ip_to_onehot_features(val))

    return feature_vector
