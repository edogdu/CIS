# Standard library imports
import logging
import os
import time
from typing import Any, Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    HGTConv,
    Linear,
    global_max_pool,
)
from models.focal_loss import FocalLoss
from matplotlib import pyplot as plt
import seaborn as sns
import copy
from captum.attr import IntegratedGradients, Saliency, DeepLift
from functools import partial
from sklearn.preprocessing import RobustScaler

# Local application/library specific imports
from models.gnn import get_label_metrics
from repositories.graphs.pyg_builder import y_labels, y_bin_labels, get_hetero_column_names
import json

logging.info("Imported y_labels in gnn.py: %s", y_labels)
logging.info("Imported y_bin_labels in gnn.py: %s", y_bin_labels)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
y_anomaly_labels = y_labels[1:]  # Exclude normal class (0)

logging.info("Imported y_labels in gnn.py: %s", y_labels)
logging.info("Imported y_bin_labels in gnn.py: %s", y_bin_labels)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
y_anomaly_labels = y_labels[1:]  # Exclude normal class (0)

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

class GNNHeteroBinEncoderModel(nn.Module):
    """GNN module to produce node embeddings for heterogeneous graphs."""
    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroBinEncoderModel, self).__init__()
        if metadata is None:
            raise ValueError("Metadata must be provided for heterogeneous graphs.")
        self.metadata = metadata
        self.config = config
        

        hd    = config.get("hidden_dim", 32)
        heads = config.get("num_heads", 4)
        self.scalers = {

        }

        # Which node types to pool over (you can override via config)
        self.pooled_types: List[str] = config.get("pooled_types", ["Pumps", "FlowSensors", "Tanks", "Valves", "Connections", "Endpoints"])

        # Per-node-type input projection to a shared hidden dim
        self.lin_dict = nn.ModuleDict()
        for node_type in self.metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hd)

        # HGT backbone
        self.conv_layers = config.get("num_layers", 2)
        self.convs = nn.ModuleList()
        for i in range(self.conv_layers):
            self.convs.append(HGTConv(hd, hd, metadata=self.metadata, heads=heads))

        self.dropout = float(config.get("dropout", 0.5))

        # Projection after concatenating pooled node-type embeddings
        pooled_width = hd * max(1, len(self.pooled_types))
        self.lin1 = Linear(pooled_width, hd)

        self.to(DEVICE)


    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        # 1) Type-wise input projections
        x_dict = {
            ntype: self.lin_dict[ntype](x).relu()
            for ntype, x in data.x_dict.items()
        }

        # 2) HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 3) Graph-level pooling over selected node types (robust if missing)
        pools = []
        num_graphs = data.num_graphs
        for ntype in self.pooled_types:
            if ntype in x_dict and hasattr(data[ntype], "batch"):
                pools.append(global_max_pool(x_dict[ntype], data[ntype].batch, size=num_graphs))
            else:
                # keep dims aligned so concatenation works
                pools.append(torch.zeros((num_graphs, self.config.get("hidden_dim", 32)), device=x_dict[next(iter(x_dict))].device))

        h = torch.cat(pools, dim=1) if len(pools) > 1 else pools[0]

        # 4) Final MLP + Dropout
        h = F.relu(self.lin1(F.dropout(h, p=self.dropout, training=self.training)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h

class GNNHeteroAnomalyDetectionModel(nn.Module):
    """GNN model for anomaly detection.  It is supervised model,
    which classifies each graph as normal, MITM, DoS, scan, physical fault, anomaly
    This will allow for heterogeneous graphs with different node types.
    """

    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroAnomalyDetectionModel, self).__init__()
        if metadata is None:
            raise ValueError("Metadata must be provided for heterogeneous graphs.")
        self.metadata = metadata
        self.config = config
        self.criterion = None
        self.scalers = {
            
        }
        self.bin_threshold = config.get("bin_threshold", 0.5)
        
        hd = config.get("hidden_dim", 32)        

        self.encoder = GNNHeteroBinEncoderModel(config, metadata)
        
        self.out = nn.Linear(hd, 1)   # -> [B,1] then squeeze to [B]

        self.to(DEVICE)


    def forward(self, data: HeteroData) -> torch.Tensor:
        h = self.encoder(data)   # [B, hd]
        logits = self.out(h).squeeze(dim=-1)   # [B]
        return logits


    def build_data_loaders(self, dataset: HeteroData):
        """Stratified split of dataset into train and test sets based on graph labels."""
        logging.info("Building data loaders with stratified split for dataset with %d samples...", len(dataset))
        
        logging.info("Creating  binary dataset...")
        # Get labels for stratification
        labels = [data.y.item() for data in dataset]
        class_counts = np.bincount(labels, minlength=len(y_labels))
        # min_samples = 8
        # for i, count in enumerate(class_counts):
        #     if 0 < count < min_samples:
        #         logging.warning(f"Class {y_labels[i]} has only {count} samples, which is less than the minimum required {min_samples}.")
        #         # Over-sample this class by duplicating samples
        #         needed = min_samples - count
        #         # copy needed random samples from this class
        #         class_samples = [idx for idx, label in enumerate(labels) if label == i]
        #         for _ in range(needed):
        #             sample_idx = np.random.choice(class_samples)
        #             dataset.append(copy.deepcopy(dataset[sample_idx]))
        #             labels.append(i)

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

        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Connections', feature_column_key='Connections')
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Endpoints', feature_column_key='Endpoints')
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='Tanks', feature_column_key='Tanks')
        dataset = self.scale_features(dataset, train_idx, val_idx, final_test_idx, node_type='FlowSensors', feature_column_key='FlowSensors')
        
        bin_dataset = copy.deepcopy(dataset)
        for data in bin_dataset:
            data.y = torch.tensor(0 if data.y.item() == 0 else 1, dtype=torch.long)
        logging.info("Binary dataset created.")

        anom_dataset = copy.deepcopy(dataset)
        # shift labels down by 1
        anom_dataset = [data for data in anom_dataset]
        for data in anom_dataset:
                data.y = torch.tensor(data.y.item() - 1, dtype=torch.long)

        



        # get anomaly indices in train and val sets, exclude normal class (0), shift by -1 for anomaly classes
        anomaly_train_idx = [i for i in train_idx if anom_dataset[i].y.item() > -1]
        anomaly_val_idx = [i for i in val_idx if anom_dataset[i].y.item() > -1]
        logging.info("Anomaly indices in training set: %s", anomaly_train_idx)
        logging.info("Anomaly indices in validation set: %s", anomaly_val_idx)

        logging.info("Train set size: %d, Validation set size: %d, Final test set size: %d", len(train_idx), len(val_idx), len(final_test_idx))
        #train_set = Subset(dataset, oversampled_train_idx)
        train_set = Subset(bin_dataset, train_idx)
        anomaly_train_set = Subset(anom_dataset, [i for i in anomaly_train_idx])
        logging.info("Anomaly train set y values: %s", [dataset[i].y.item() for i in anomaly_train_idx])
        val_set = Subset(bin_dataset, val_idx)
        anomaly_val_set = Subset(anom_dataset, [i for i in anomaly_val_idx])
        final_test_set = Subset(dataset, final_test_idx)

        # create loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
        anom_train_loader = DataLoader(anomaly_train_set, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
        anom_val_loader = DataLoader(anomaly_val_set, batch_size=32, shuffle=False, num_workers=0)
        final_test_loader = DataLoader(final_test_set, batch_size=32, shuffle=False, num_workers=0)

        # Create export csv of split counts for each class for each subset
        logging.info("Exporting data split counts to ./exports/results/gnn_het_data_split_counts.csv")
        original_counts = {
            'Overall': np.bincount(labels, minlength=len(y_labels)),
            'Train': np.bincount([labels[i] for i in train_idx], minlength=len(y_labels)),
            'Validation': np.bincount([labels[i] for i in val_idx], minlength=len(y_labels)),
            'Final Test': np.bincount([labels[i] for i in final_test_idx], minlength=len(y_labels)),
        }
        logging.info("Original counts: %s", original_counts)
        bin_test_train_counts = {
            'Overall Binary': np.bincount([0 if label == 0 else 1 for label in labels], minlength=2),
            'Train': np.bincount([0 if labels[i] == 0 else 1 for i in train_idx], minlength=2),
            'Validation': np.bincount([0 if labels[i] == 0 else 1 for i in val_idx], minlength=2)            
        }
        logging.info("Binary counts: %s", bin_test_train_counts)

        anom_test_train_counts = {
            # 1-5 for anomaly classes only on test_labels
            'Overall Anomaly': np.bincount([labels[i] for i in range(len(labels)) if labels[i] > 0], minlength=len(y_labels)),
            'Train': np.bincount([labels[i] for i in train_idx if labels[i] > 0], minlength=len(y_labels)),
            'Validation': np.bincount([labels[i] for i in val_idx if labels[i] > 0], minlength=len(y_labels)),
            'Final Test': np.bincount([labels[i] for i in final_test_idx if labels[i] > 0], minlength=len(y_labels))
        }
        logging.info("Anomaly counts: %s", anom_test_train_counts)
        with open("./exports/results/gnn_het_data_split_counts.csv", "w") as f:
            f.write("Class," + ",".join(y_labels) + "\n")
            for split, counts in original_counts.items():
                f.write(split + "," + ",".join(str(count) for count in counts) + "\n")
            f.write("\nBinary Class Counts (0: normal, 1: anomaly):\n")
            f.write("Class,0,1\n")
            for split, counts in bin_test_train_counts.items():
                f.write(split + "," + ",".join(str(count) for count in counts) + "\n")
            f.write("\nAnomaly Class Counts (1-5):\n")
            f.write("Class," + ",".join(y_anomaly_labels) + "\n")
            for split, counts in anom_test_train_counts.items():
                f.write(split + "," + ",".join(str(count) for count in counts[1:]) + "\n")
        logging.info("Data split counts exported.")
        return (train_loader, val_loader), (anom_train_loader, anom_val_loader), final_test_loader
    def scale_features(self, dataset, train_idx, val_idx, final_test_idx, node_type,feature_column_key):
        ignore_scaling_keywords = ['_bucket_', 'asset_type_', 'protocol_', 'mac_byte_', 'ip_part_',
                                   'response_present']
        log_candidates = [
            'avg_size', 'avg_value','stddev_value', 'min_value','max_value', 'num_connections', 'modbus_response_count',
            'endpoint_unique_peer_count', 'endpoint_num_unique_protocols',
            'tcp_cwr_count', 'tcp_ece_count', 'tcp_urg_count', 'tcp_ack_count', 
            'tcp_psh_count', 'tcp_rst_count', 'tcp_syn_count', 'tcp_fin_count',
            'endpoint_in_out_ratio','endpoint_num_unique_ports', 'endpoint_port_entropy',
        ]

        if node_type not in self.scalers:
            self.scalers[node_type] = {}

        connection_features = get_hetero_column_names(feature_column_key)
        for feature_name in [f for f in connection_features if not any(keyword in f for keyword in ignore_scaling_keywords)]:
            feature_idx = connection_features.index(feature_name)
                        
            # gather all values for this feature from training set
            values = []
            for idx in train_idx:
                data = dataset[idx]
                if node_type in data.x_dict:
                    values.append(data.x_dict[node_type][:, feature_idx].cpu().numpy())
            values = np.concatenate(values)
            # apply log1p scaling to candidates
            if feature_name in log_candidates:
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                values = np.log1p(np.maximum(values, 0))
            scaler = RobustScaler()
            scaler.fit(values.reshape(-1, 1))
            self.scalers[node_type][feature_name] = scaler

            for idx in train_idx:
                data = dataset[idx]
                if node_type in data.x_dict:
                    
                    data.x_dict[node_type][:, feature_idx] = torch.where(data.x_dict[node_type][:, feature_idx] < 0.01, torch.tensor(0.0, device=data.x_dict[node_type].device), data.x_dict[node_type][:, feature_idx])
                    data.x_dict[node_type][:, feature_idx] = torch.log1p(torch.maximum(data.x_dict[node_type][:, feature_idx], torch.tensor(0.0, device=data.x_dict[node_type].device)))
                    data.x_dict[node_type][:, feature_idx] = torch.from_numpy(scaler.transform(data.x_dict[node_type][:, feature_idx].cpu().numpy().reshape(-1, 1))).to(data.x_dict[node_type].device).float().to(data.x_dict[node_type].device).view(-1)
                    
            # apply scaling to val and final test sets
            for idx in val_idx + final_test_idx:
                data = dataset[idx]
                if node_type in data.x_dict:                    
                    data.x_dict[node_type][:, feature_idx] = torch.where(data.x_dict[node_type][:, feature_idx] < 0.01, torch.tensor(0.0, device=data.x_dict[node_type].device), data.x_dict[node_type][:, feature_idx])
                    data.x_dict[node_type][:, feature_idx] = torch.log1p(torch.maximum(data.x_dict[node_type][:, feature_idx], torch.tensor(0.0, device=data.x_dict[node_type].device)))
                    data.x_dict[node_type][:, feature_idx] = torch.from_numpy(scaler.transform(data.x_dict[node_type][:, feature_idx].cpu().numpy().reshape(-1, 1))).to(data.x_dict[node_type].device).float().to(data.x_dict[node_type].device).view(-1)
                    
        return dataset

    def get_criterion(self, data_loader: DataLoader) -> nn.Module:
        """Get loss function with class weights to handle class imbalance."""
        
        # ---------- Binary head ----------
        # Count positives (y>0) and negatives (y==0) on this loader's dataset
        ys = torch.tensor([d.y.item() for d in data_loader.dataset], dtype=torch.long)
        pos = (ys > 0).sum().item()
        neg = (ys == 0).sum().item()
        # Avoid div-by-zero; if no positives, fall back to 1.0
        pos_weight_scalar = float(neg / pos) if pos > 0 else 1.0
        pos_weight_scalar = min(pos_weight_scalar, 5.0)  # cap at 5.0 to avoid extreme weights
        bin_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_scalar, dtype=torch.float32, device=DEVICE)
        )
        self.criterion = bin_criterion

        return bin_criterion

        
    def predict(self, data_loader: DataLoader) -> List[int]:
        """Predict anomaly classes for the given data HeteroData."""
        self.eval()        
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        bin_logits = self(batch)      # [B], [B,5]
        bin_preds = (torch.sigmoid(bin_logits) >= self.bin_threshold).long()
        return bin_preds.detach().cpu().tolist()
    
    # Training and evaluation functions

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer):
        self.train()
        total_loss = 0
        #bin_criterion, anom_criterion = criterion
        try:
            total_num = 0
            y_all = []
            y_pred_all = []
            
            for batch in loader:
                batch = batch.to(DEVICE)
                y = batch.y.view(-1).long()
                optimizer.zero_grad()
                bin_logits= self(batch)      # [B], [B,5]                
                y_bin = (y != 0).long()
                
                loss = self.criterion(bin_logits, y_bin.float())
                # Add L1 regularization to the loss 
                # l1_lambda = 1e-4
                # l1_norm = sum(p.abs().sum() for p in self.parameters())
                # loss = loss + l1_lambda * l1_norm

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.size(0)
                total_num += batch.size(0)

                bin_preds = (torch.sigmoid(bin_logits) >= self.bin_threshold).long()
                pred = bin_preds.clone()                
                y_all.extend(y_bin.detach().cpu().tolist())
                y_pred_all.extend(pred.detach().cpu().tolist())

            mean_loss = total_loss / max(1, total_num)
            return self.get_label_metrics(y_all, y_pred_all, mean_loss)
        except Exception as e:
            logging.error("Error during training epoch: %s", str(e))
            raise e
        
        #logging.info("Epoch Train Loss: %.4f, Mean Anomaly Macro F1: %.4f, Mean Macro F1: %.4f, Mean Balanced Acc: %.4f",
        #             mean_loss, mean_anomaly_macro_f1, mean_macro_f1, mean_balanced_acc)
    def get_label_metrics(self, y_true, y_pred, mean_loss, export_results: bool = False, is_final_test: bool = False):
        """Calculate classification metrics for labels."""        

        
        precision_val = precision_score(y_true, y_pred, zero_division=0)
        recall_val = recall_score(y_true, y_pred, zero_division=0)
        f1_score_val = f1_score(y_true, y_pred, zero_division=0)        
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        if export_results:
            with open(f"./exports/results/{'final_' if is_final_test else ''}gnn_het_detection_classification_perf_scores.csv", "w") as f:
                f.write("Metric,Value\n")
                f.write(f"loss,{mean_loss}\n")
                f.write(f"precision,{precision_val}\n")
                f.write(f"recall,{recall_val}\n")
                f.write(f"f1_score,{f1_score_val}\n")
                f.write(f"accuracy,{accuracy}\n")
                f.write(f"balanced_accuracy,{balanced_accuracy}\n")

        

        return {
            "loss": mean_loss,
            "precision": precision_val,
            "recall": recall_val,
            "f1_score": f1_score_val,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy
        } 

    @torch.no_grad()
    def evaluate_model(self, loader: DataLoader):
        self.eval()        
        total_loss = 0
        total_num = 0
        y_all = []
        y_pred_all = []
        for batch in loader:
            batch = batch.to(DEVICE)
            bin_logits = self(batch)      # [B], [B,5]
            y = batch.y.view(-1).long()
            y_bin = (y != 0).long()
            

            loss = 0.0

            lossA = self.criterion(bin_logits, y_bin.float())
            loss = loss + lossA

            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            preds = (torch.sigmoid(bin_logits) >= self.bin_threshold).long()
            y_all.extend(y_bin.detach().cpu().tolist())
            y_pred_all.extend(preds.detach().cpu().tolist())
        mean_loss = total_loss / max(1, total_num)
        return y_pred_all, self.get_label_metrics(y_all, y_pred_all, mean_loss)



    def fit_model(self, 
                train_loader: DataLoader,
                val_loader: DataLoader, 
                config: Dict[str, Any]):
        """Train the GNN model with early stopping based on validation anomaly macro F1 score."""
        num_epochs = config.get("max_epochs", 100)
        learning_rate = config.get("learning_rate", 0.001)
        patience = config.get("early_stopping_patience", 10)
        min_delta = config.get("early_stopping_min_delta", 0.0001)
        weight_decay = config.get("weight_decay", 1e-5)

        criterion = self.get_criterion(train_loader)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        early_stop_mode = 'min'  # We want to maximize F1 score
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,mode=early_stop_mode, factor=0.5, patience=10,min_lr=1e-6)
        
        early_stopper = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
        best_val_metrics = None
        best_model_state = None
        early_stop_metric = 'loss'
        

        logging.info("Stage 1: Training with binary head only for %d epochs...", config.get("stage1_epochs", 10))
        for epoch in range(config.get("stage1_epochs", 100)):
            train_metrics = self.train_epoch(train_loader, optimizer)
            _, val_metrics = self.evaluate_model(val_loader)
            logging.info("Stage 1 Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Val F1: %.4f",
                        epoch+1, train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score"])
            scheduler.step(val_metrics[early_stop_metric])

            # Check for early stopping
            if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
                best_val_metrics = val_metrics
                best_model_state = self.state_dict()
                logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics[early_stop_metric])

            early_stopper.step(val_metrics[early_stop_metric])
            if early_stopper.early_stop:
                logging.info("Early stopping triggered at epoch %d during Stage 1.", epoch+1)
                break
        # Load best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        

        # Calculate final training metrics
        _, final_train_metrics = self.evaluate_model(train_loader)        
        logging.info("Final Training Metrics - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                    final_train_metrics["loss"], final_train_metrics["f1_score"], final_train_metrics["recall"],
                    final_train_metrics["precision"], final_train_metrics["balanced_accuracy"], final_train_metrics["accuracy"])

        return self, criterion, final_train_metrics

    def test_model(self, test_loader: DataLoader, test_description: str="Final Test Set"):
        """Evaluate the trained model on the final test set and print classification report."""
        
        _, test_metrics = self.evaluate_model(test_loader)
        logging.info("%s Results - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                    test_description, test_metrics["loss"], test_metrics["f1_score"], test_metrics["recall"],
                    test_metrics["precision"], test_metrics["balanced_accuracy"], test_metrics["accuracy"])
        
        # Get detailed classification report
        y_all = []
        y_pred_all = []
        total_loss = 0
        total_num = 0
        for batch in test_loader:
            batch = batch.to(DEVICE)
            bin_logits = self(batch)      # [B], [B,5]
            y = batch.y.view(-1).long()
            y_bin = (y != 0).long()

            loss = 0.0
            
            lossA = self.criterion(bin_logits, y_bin.float())
            loss = loss + lossA
            
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            bin_preds = (torch.sigmoid(bin_logits) >= self.bin_threshold).long()
            
            logging.info("Batch true labels (binary): %s", y_bin.detach().cpu().tolist())
            y_all.extend(y_bin.detach().cpu().tolist())
            y_pred_all.extend(bin_preds.detach().cpu().tolist())

        # Get Explainability for all anomaly predictions
        for i in range(len(y_pred_all)):
            if y_pred_all[i] > 0:
                logging.info("Generating explanation for test sample %d with predicted class %d (%s)", i, y_pred_all[i], y_bin_labels[y_pred_all[i]])
                data = test_loader.dataset[i]
                explainer_results = self.explain_with_captum(data)
        target_names = ['Normal', 'Anomaly']
        logging.info(f"yall: {y_all}, ypredall: {y_pred_all}")
        self.get_label_metrics(y_all, y_pred_all, total_loss / max(1, total_num), export_results=True)
        report_df = pd.DataFrame(classification_report(y_all, y_pred_all, target_names=target_names, output_dict=True)).transpose()
        report_df.to_csv(f"./exports/results/classification_report_detection_{test_description}.csv")
        
        # save confusion matrix image
        cm = confusion_matrix(y_all, y_pred_all)
        # add timestamp to filename to avoid overwriting
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"./exports/images/gnn_het_classification_detection_confusion_matrix_{test_description}_{int(time.time())}.png")
        plt.close()
        # explain test with captum
        # for i, batch in enumerate(test_loader):
        #     explain_with_captum(model, batch, i, use_anomaly=use_anomaly, test_description=test_description)
        return y_all, y_pred_all

    # Helper functions for training and evaluation
    def get_weights(labels, min_num_classes, epsilon=1e-6):
        """Compute class weights to handle class imbalance."""
        counts = np.bincount(labels, minlength=min_num_classes).astype(np.float32)
        counts[counts == 0] = epsilon  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * len(counts)  # normalize
        weights[~np.isfinite(weights)] = epsilon  # handle any inf or nan
        return weights

    def explain_with_captum(self, data: HeteroData, save_dir="./exports/explanations"):
        """
        Generates explanations with Snapshot ID, True/Pred Y, and Global Feature Plots.
        """
        self.eval()
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())

        # Extract Metadata (Snapshot, True Y, etc.)
        # Handle Snapshot ID (support string, int, or tensor)
        snapshot_id = "unknown"
        if hasattr(data, 'snapshot_id'):
            s_id = data.snapshot_id
            if torch.is_tensor(s_id):
                snapshot_id = str(s_id.item()) if s_id.numel() == 1 else str(s_id.tolist())
            else:
                snapshot_id = str(s_id)
        
        # Handle True Label
        true_label_idx = data.y.item() if hasattr(data, 'y') and data.y.numel() == 1 else -1
        true_label_name = y_labels[true_label_idx] if 0 <= true_label_idx < len(y_labels) else "Unknown"

        # Get Model Prediction
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        
        with torch.no_grad():
            logits = self(batch)
            pred_class = (torch.sigmoid(logits) >= self.bin_threshold).long().item()
            pred_prob = torch.sigmoid(logits).max().item()
            pred_label_name = y_labels[pred_class]

        # Filter: Only explain anomalies (Optional: remove if you want to explain everything)
        if pred_class == 0:
            logging.info(f"Skipping explanation for Normal traffic (Snapshot: {snapshot_id}).")
            return None

        logging.info(f"Explaining Snapshot {snapshot_id}: True: {true_label_name} -> Pred: {pred_label_name} ({pred_prob:.4f})")

        # Prepare Inputs for Captum
        inputs_list = []
        node_types = []
        for ntype in data.x_dict.keys():
            if data[ntype].num_nodes > 0 and hasattr(data[ntype], "x"):
                inputs_list.append(data[ntype].x.clone().detach().requires_grad_(True))
                node_types.append(ntype)

        inputs_tuple = tuple(inputs_list)
        baselines_tuple = tuple(torch.zeros_like(t) for t in inputs_tuple)

        # Run Attribution
        # Uses the static wrapper defined previously
        forward_func = partial(_fast_model_forward_wrapper, self, data, DEVICE)
        ig = IntegratedGradients(forward_func=forward_func)

        try:
            attributions = ig.attribute(
                inputs=inputs_tuple,
                baselines=baselines_tuple,
                target=pred_class,
                n_steps=50,
                internal_batch_size=10
            )
        except Exception as e:
            logging.error(f"Error during IG attribution for Snapshot {snapshot_id}: {e}")
            return None

        # Process Results ---
        explanation_data = {
            "meta": {
                "timestamp": timestamp,
                "snapshot_id": snapshot_id,
                "true_y": true_label_idx,
                "true_label": true_label_name,
                "predicted_y": pred_class,
                "predicted_label": pred_label_name,
                "confidence": pred_prob
            },
            "node_importances": [],
            "feature_importances": {}
        }
        
        all_node_rankings = []
        global_feature_records = [] # For the Bar Chart

        for idx, ntype in enumerate(node_types):
            attr_tensor = attributions[idx].detach().cpu()
            
            # Feature Importance
            feat_imp = attr_tensor.abs().mean(dim=0).numpy()
            num_features = len(feat_imp)
            
            # Safe Name Mapping
            col_names = []
            try:
                col_name_key = ntype
                if ntype in ["TankMeasurements", "ValveMeasurements"]:
                    col_name_key = "Measurements"
                elif ntype in ["PumpMeasurements", "SensorMeasurements"]:
                    col_name_key = "StateMeasurements"
                else:
                    if not ntype.endswith("s"):
                        col_name_key = ntype + "s"
                col_names = get_hetero_column_names(col_name_key)
            except Exception:
                pass

            if len(col_names) != num_features:
                col_names = [f"feat_{i}" for i in range(num_features)]

            # Store in JSON
            explanation_data["feature_importances"][ntype] = {
                k: float(v) for k, v in zip(col_names, feat_imp)
            }

            # Collect for Global Plot
            for fname, fscore in zip(col_names, feat_imp):
                global_feature_records.append({
                    "node_type": ntype,
                    "feature": fname,
                    "full_name": f"{ntype}: {fname}",
                    "score": float(fscore)
                })

            # Node Importance & Mapping
            node_imp = attr_tensor.abs().sum(dim=1).numpy()
            num_nodes = len(node_imp)
            
            # Safe ID Mapping
            orig_ids = []
            if hasattr(data[ntype], 'original_id'):
                raw = data[ntype].original_id
                orig_ids = raw.tolist() if torch.is_tensor(raw) else raw
            elif hasattr(data[ntype], 'n_id'):
                orig_ids = data[ntype].n_id.tolist()
            
            if len(orig_ids) != num_nodes:
                orig_ids = [f"{ntype}_{i}" for i in range(num_nodes)]

            # Create Records
            for i, score in enumerate(node_imp):
                if score > 1e-4:
                    record = {
                        "snapshot_id": snapshot_id, # Added to record
                        "node_type": ntype,
                        "pyg_index": i,
                        "original_id": str(orig_ids[i]),
                        "importance_score": float(score),
                        "true_label": true_label_name,
                        "pred_label": pred_label_name
                    }
                    explanation_data["node_importances"].append(record)
                    all_node_rankings.append(record)

        # Generate Global Top 5 Feature Plot
        if global_feature_records:
            # Sort by score descending
            global_feature_records.sort(key=lambda x: x['score'], reverse=True)
            top_5_features = global_feature_records[:5]
            
            # Extract data for plotting
            plot_names = [x['full_name'] for x in top_5_features]
            plot_scores = [x['score'] for x in top_5_features]
            
            # Plotting
            plt.figure(figsize=(10, 6))
            sns.barplot(x=plot_scores, y=plot_names, palette="viridis")
            plt.title(f"Top 5 Features (Snapshot {snapshot_id})\nTrue: {true_label_name} | Pred: {pred_label_name}")
            plt.xlabel("Mean Absolute Attribution")
            plt.tight_layout()
            
            # Save Plot
            plot_path = f"{save_dir}/bin_plot_top5_feats_snap{snapshot_id}_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close() 
            
            # Add top 5 to explanation data for easy access
            explanation_data["top_5_global_features"] = top_5_features

        # Export Files
        all_node_rankings.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Save JSON
        json_path = f"{save_dir}/bin_explanation_snap{snapshot_id}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(explanation_data, f, indent=2)
            
        # Save CSV
        node_df = pd.DataFrame(all_node_rankings)
        if not node_df.empty:
            node_csv_path = f"{save_dir}/bin_ranking_snap{snapshot_id}_{timestamp}.csv"
            node_df.head(50).to_csv(node_csv_path, index=False)
            
        logging.info(f"Saved explanations and plot for snapshot {snapshot_id} to {save_dir}")
        return explanation_data       

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

