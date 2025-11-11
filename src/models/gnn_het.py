from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import numpy as np
import pandas as pd
from repositories.graphs.pyg_builder import y_labels
import logging
from torch_geometric.nn import HGTConv, Linear, global_max_pool
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
import os

logging.info("Imported y_labels in gnn.py: %s", y_labels)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
y_anomaly_labels = y_labels[1:]  # Exclude normal class (0)

class GNNHeteroClassifierModel(nn.Module):
    """GNN model for anomaly detection.  It is supervised model,
    which classifies each graph as normal, MITM, DoS, scan, physical fault, anomaly
    This will allow for heterogeneous graphs with different node types.
    """

    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroClassifierModel, self).__init__()
        if metadata is None:
            raise ValueError("Metadata must be provided for heterogeneous graphs.")
        self.metadata = metadata
        self.config = config
        hd = config.get("hidden_dim", 32)
        heads = config.get("num_heads", 4)

        self.lin_dict = nn.ModuleDict()
        for node_type in self.metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hd)


        self.conv1 = HGTConv(hd, hd, metadata=self.metadata, heads=heads)
        self.conv2 = HGTConv(hd, hd, metadata=self.metadata, heads=1)

        self.dropout = config.get("dropout", 0.5)
        self.lin1 = Linear(hd * 2, hd)
        self.out = Linear(hd, 6)  # 6 classes: normal, mitm, dos, scan, physical fault, anomaly
        
        self.to(DEVICE)


    def forward(self, data: HeteroData) -> torch.Tensor:
        #x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        #logging.info(f"Node types in data: {data.node_types}")
        x_dict = {
            node_type: self.lin_dict[node_type](data[node_type].x).relu()
            for node_type in data.node_types
        }
        
        #x_dict = {key: x.to(DEVICE) for key, x in x_dict.items()}
        edge_index_dict = data.edge_index_dict
        num_graphs = data.num_graphs
        x_dict = self.conv1(x_dict, edge_index_dict)        
        x_dict = self.conv2(x_dict, edge_index_dict)

        if 'Measurements' in x_dict:
            #logging.info("Creating measurement pool")
            measurement_pool = global_max_pool(x_dict['Measurements'], data['Measurements'].batch)
        else:
            logging.warning("No 'Measurements' node type found in the graph data.")
            measurement_pool = torch.zeros((num_graphs, self.config.get("hidden_dim", 32)), device=DEVICE)
        if 'Connections' in x_dict:
            #logging.info("Creating connection pool")
            connection_pool = global_max_pool(x_dict['Connections'], data['Connections'].batch)

        else:
            logging.warning("No 'Connections' node type found in the graph data.")
            connection_pool = torch.zeros((num_graphs, self.config.get("hidden_dim", 32)), device=DEVICE)

        x = torch.cat([measurement_pool, connection_pool], dim=1)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x

def build_data_loaders(dataset: HeteroData):
    """Stratified split of dataset into train and test sets based on graph labels."""
    logging.info("Building data loaders with stratified split for dataset with %d samples...", len(dataset))
    
    # Get labels for stratification
    labels = [data.y.item() for data in dataset]
    class_counts = np.bincount(labels, minlength=len(y_labels))
    # get minimum class count needed for 0.2 test split with stratification and test count atleast 3
    min_class_count = []
    for i in range(len(y_labels)):
        required_count = int((3 - 0.2 * class_counts[i]) / 0.2) if class_counts[i] > 0 else 0 # at least 3 samples in test set, except when class count is 0
        min_class_count.append(int(np.ceil(required_count)))
    logging.info("Minimum samples needed per class for stratified splitting: " + ", ".join([f"{y_labels[i]}: {min_class_count[i]}" for i in range(len(y_labels))]))
    logging.info("Overall class distribution in dataset: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(class_counts)]))
    
    # Ensure each class has at least 15 samples for stratified splitting
    need_oversampling = False
    for i, count in enumerate(class_counts):
        if count < min_class_count[i]:
            need_oversampling = True
            logging.warning("Class '%s' has only %d samples. Stratified splitting may not be reliable.", y_labels[i], count)
            # manually oversample this class in the dataset
            samples_to_add = min_class_count[i] - count
            class_samples = [data for data in dataset if data.y.item() == i]
            for _ in range(samples_to_add):
                dataset.append(class_samples[np.random.randint(0, len(class_samples))])
            logging.info("After oversampling, dataset size is %d samples.", len(dataset))
    if need_oversampling:
        # Recompute labels after oversampling
        labels = [data.y.item() for data in dataset]
        class_counts = np.bincount(labels, minlength=len(y_labels))
        logging.info("New Overall class distribution in dataset: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(class_counts)]))

    # split dataset for 60% train, 40% validation/final test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)

    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))

    # further split test into 20% validation and 20% final test
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    test_labels = [labels[i] for i in test_idx]
    relative_val_idx, relative_final_test_idx = next(splitter2.split(np.zeros(len(test_idx)), test_labels))

    # Map relative indices back to original test indices
    val_idx = [test_idx[i] for i in relative_val_idx]
    final_test_idx = [test_idx[i] for i in relative_final_test_idx]

    # check distribution in each split
    test_counts = np.bincount(test_labels, minlength=len(y_labels))
    val_labels = [labels[i] for i in val_idx]
    val_counts = np.bincount(val_labels, minlength=len(y_labels))
    final_test_labels = [labels[i] for i in final_test_idx]
    final_test_counts = np.bincount(final_test_labels, minlength=len(y_labels))

    logging.info("Overall label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(np.bincount(labels, minlength=len(y_labels)))]))
    logging.info("Train label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(np.bincount([labels[i] for i in train_idx], minlength=len(y_labels)))]))
    logging.info("Validation label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(val_counts)]))
    logging.info("Final test label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(final_test_counts)]))

    # Handling class imbalance with weighted random sampler did not help much,
    # will oversample minority classes manually instead.
    # This is for training set only
    #train_labels = [np.array(labels)[train_idx][i] for i in range(len(train_idx))]
    #class_counts = np.bincount(train_labels, minlength=len(y_labels))
    #max_count = class_counts.max()
    # oversampled_train_idx = []
    # for i in range(len(y_labels)):
    #     if class_counts[i] == 0:
    #         continue
    #     # Find all indices of this class in the training set
    #     class_indices = [idx for idx, label in zip(train_idx, train_labels) if label == i]
    #     # Oversample to match max_count
    #     oversampled = np.random.choice(class_indices, max_count, replace=True)
    #     oversampled_train_idx.extend(oversampled.tolist())
    # np.random.shuffle(oversampled_train_idx)

    logging.info("Train set size: %d, Validation set size: %d, Final test set size: %d", len(train_idx), len(val_idx), len(final_test_idx))
    #train_set = Subset(dataset, oversampled_train_idx)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    final_test_set = Subset(dataset, final_test_idx)

    # ---------- commented out weighted random sampler -------------
    #class_sample_count = np.array([len(np.where(np.array(train_labels) == t)[0]) for t in np.unique(train_labels)])
    


    # Compute class weights
    #weights = get_weights(train_labels, min_num_classes=len(y_labels))
    #sampler_weights = torch.tensor([weights[int(t)] for t in train_labels], dtype=torch.double)

    #sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
    # ---------------------------------------------------------------

    # create loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    final_test_loader = DataLoader(final_test_set, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, final_test_loader