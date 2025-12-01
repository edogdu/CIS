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

from matplotlib import pyplot as plt
import seaborn as sns
import copy
from captum.attr import IntegratedGradients, Saliency, DeepLift
from functools import partial

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

def _model_forward_wrapper(model: nn.Module, data: HeteroData, model_device: str, *node_inputs: torch.Tensor):
    """
    Wrapper for Captum. Reconstructs the batched graph from the interpolated inputs.
    Args:
        model (nn.Module): The GNN model.
        data (HeteroData): The original single graph data.
        model_device (str): Device to run the model on.
        *node_inputs (torch.Tensor): Interpolated node features for each node type.
                                     This is from Captum during attribution.
    """
    # Create a new HeteroData object to avoid modifying the original
    temp_data = HeteroData()
    
    # Copy edge indices
    temp_data.edge_index_dict = data.edge_index_dict
    
    # Get node types that have features
    node_types_with_features = [ntype for ntype in data.x_dict.keys() 
                                if data[ntype].num_nodes > 0 and hasattr(data[ntype], "x")]
    
    # Determine if we're dealing with batched data
    first_input = node_inputs[0]
    original_num_nodes = data[node_types_with_features[0]].num_nodes
    new_num_nodes = first_input.size(0)
    
    is_batched = new_num_nodes != original_num_nodes
    num_replications = new_num_nodes // original_num_nodes if is_batched else 1
    
    # Update node features and batch vectors
    for idx, ntype in enumerate(node_types_with_features):
        temp_data[ntype].x = node_inputs[idx]
        temp_data[ntype].num_nodes = node_inputs[idx].size(0)
        
        if is_batched:
            # Create batch vector for batched graphs
            original_num = data[ntype].num_nodes
            batch_ids = torch.arange(num_replications, device=model_device)
            temp_data[ntype].batch = torch.repeat_interleave(batch_ids, original_num)
        else:
            # Single graph case
            temp_data[ntype].batch = torch.zeros(data[ntype].num_nodes, dtype=torch.long, device=model_device)
    
    # Update edge indices if batched
    if is_batched:
        new_edge_index_dict = {}
        for etype in data.edge_index_dict.keys():
            src_ntype, rel, dst_ntype = etype
            original_edge_index = data.edge_index_dict[etype]
            
            # Get node counts for source and destination
            src_node_count = data[src_ntype].num_nodes
            dst_node_count = data[dst_ntype].num_nodes
            
            replicated_edges = []
            for i in range(num_replications):
                src_offset = i * src_node_count
                dst_offset = i * dst_node_count
                offset = torch.tensor([[src_offset], [dst_offset]], 
                                     device=model_device, dtype=torch.long)
                replicated_edges.append(original_edge_index + offset)
            
            new_edge_index_dict[etype] = torch.cat(replicated_edges, dim=1)
        
        temp_data.edge_index_dict = new_edge_index_dict
    
    # Set num_graphs for proper pooling
    temp_data.num_graphs = num_replications
    
    # Move to device and run model
    temp_data = temp_data.to(model_device)
    logits = model(temp_data)
    
    return logits

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

        # Which node types to pool over (you can override via config)
        self.pooled_types: List[str] = config.get("pooled_types", ["Measurements", "Connections", "Endpoints", "Assets"])

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
        min_samples = 8
        for i, count in enumerate(class_counts):
            if 0 < count < min_samples:
                logging.warning(f"Class {y_labels[i]} has only {count} samples, which is less than the minimum required {min_samples}.")
                # Over-sample this class by duplicating samples
                needed = min_samples - count
                # copy needed random samples from this class
                class_samples = [idx for idx, label in enumerate(labels) if label == i]
                for _ in range(needed):
                    sample_idx = np.random.choice(class_samples)
                    dataset.append(copy.deepcopy(dataset[sample_idx]))
                    labels.append(i)
        bin_dataset = copy.deepcopy(dataset)
        for data in bin_dataset:
            data.y = torch.tensor(0 if data.y.item() == 0 else 1, dtype=torch.long)
        logging.info("Binary dataset created.")

        anom_dataset = copy.deepcopy(dataset)
        # shift labels down by 1
        anom_dataset = [data for data in anom_dataset]
        for data in anom_dataset:
                data.y = torch.tensor(data.y.item() - 1, dtype=torch.long)

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


    def get_criterion(self, data_loader: DataLoader) -> nn.Module:
        """Get loss function with class weights to handle class imbalance."""
        
        # ---------- Binary head ----------
        # Count positives (y>0) and negatives (y==0) on this loader's dataset
        ys = torch.tensor([d.y.item() for d in data_loader.dataset], dtype=torch.long)
        pos = (ys > 0).sum().item()
        neg = (ys == 0).sum().item()
        # Avoid div-by-zero; if no positives, fall back to 1.0
        pos_weight_scalar = float(neg / pos) if pos > 0 else 1.0
        pos_weight_scalar = min(pos_weight_scalar, 10.0)  # cap at 10.0 to avoid extreme weights
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
        bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
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
                loss = 0.0
                lossA = self.criterion(bin_logits, y_bin.float())
                loss = loss + lossA

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.size(0)
                total_num += batch.size(0)

                bin_preds = (torch.sigmoid(bin_logits) >= 0.4).long()
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
            preds = (torch.sigmoid(bin_logits) >= 0.5).long()
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
        weight_decay = config.get("weight_decay", 0.0001)

        criterion = self.get_criterion(train_loader)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-5)
        early_stop_mode = 'max'
        early_stopper = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
        best_val_metrics = None
        best_model_state = None
        early_stop_metric = 'f1_score'
        

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
            bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
            
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
    
    def explain_with_captum(self, data: HeteroData):
        """Generate explanations for the model's predictions using Captum.
        We assume prediction has already been made and is an anomaly.
        Focusing on node and node feature importance."""
        self.eval()

        # Get model prediction
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        with torch.no_grad():
            bin_logits = self(batch)
            pred = (torch.sigmoid(bin_logits) >= 0.5).long().item()
        
        logging.info("Model prediction for explanation: class %d (%s)", pred, y_bin_labels[pred])

        # Prepare inputs for Captum
        # Use the original single graph data (not batch) for the wrapper
        # This avoids issues with batch vectors
        inputs_list = [] 
        node_types_with_features = []
        
        for ntype in data.x_dict.keys():
            if data[ntype].num_nodes > 0 and hasattr(data[ntype], "x"):
                # Clone and enable gradients
                tensor = data[ntype].x.clone().detach().requires_grad_(True)
                inputs_list.append(tensor)
                node_types_with_features.append(ntype)

        inputs_tuple = tuple(inputs_list)
        
        # Create baselines from the new tuple
        baselines_tuple = tuple(torch.zeros_like(tensor) for tensor in inputs_tuple)

        # Create the forward_func for Captum
        # Be sure to pass the original single-graph data, not the batch
        forward_func = partial(
            _model_forward_wrapper,
            self,
            data,  # Use original data, not batch
            DEVICE
        )

        # Initialize and run Integrated Gradients
        ig = IntegratedGradients(forward_func=forward_func)
        
        try:
            attributions_tuple = ig.attribute(
                inputs=inputs_tuple,
                baselines=baselines_tuple,
                target=None,
                n_steps=50
            )
        except Exception as e:
            logging.error("Error during Captum attribution: %s", e)
            logging.error("Traceback:", exc_info=True)
            return None

        # Process and save the results
        explanation_results = {
            "predicted_class": pred,
            "predicted_label": y_bin_labels[pred],
            "node_feat_mask": {},  # For feature importance
            "node_importance": {}  # For node importance
        }

        for idx, ntype in enumerate(node_types_with_features):
            # Get attributions for this node type
            node_attr = attributions_tuple[idx]
            
            # Calculate Node Importance
            # Sum absolute attributions across features for each node            
            node_importance_summary = node_attr.abs().sum(dim=1)
            explanation_results["node_importance"][ntype] = node_importance_summary.detach().cpu().numpy().tolist()
            
            # Log top 3 most important nodes for this type
            top_k = min(3, node_importance_summary.shape[0])
            if top_k > 0:
                top_nodes = torch.topk(node_importance_summary, k=top_k)
                logging.info(f"  Top {top_k} important nodes for '{ntype}':")
                for i in range(top_k):
                    logging.info(f"    - Node Index: {top_nodes.indices[i].item()}, "
                            f"Importance: {top_nodes.values[i].item():.4f}")

            # Calculate Feature Importance
            # Average attributions across nodes for each feature            
            feature_importance_summary = node_attr.abs().mean(dim=0)
            explanation_results["node_feat_mask"][ntype] = feature_importance_summary.detach().cpu().numpy().tolist()

            # Get feature names for this specific node type
            try:
                feature_names = get_hetero_column_names(ntype)
            except Exception as e:
                logging.warning(f"Could not get feature names for {ntype}: {e}")
                feature_names = [f"feat_{i}" for i in range(len(feature_importance_summary))]
            
            # Only plot if there are features
            if len(feature_importance_summary) > 0:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(feature_importance_summary)), feature_importance_summary.detach().cpu().numpy())
                plt.xticks(range(len(feature_importance_summary)), feature_names, rotation=45, ha='right')
                plt.xlabel('Features')
                plt.ylabel('Importance (Average Absolute Attribution)')
                plt.title(f'Feature Importance for {ntype} (Prediction: {y_bin_labels[pred]})')
                plt.tight_layout()
                plt.savefig(f'./exports/images/gnn_het_captum_detection_node_feat_importance_{ntype}_{int(time.time())}.png', dpi=150)
                plt.close()
                
                # Log top 5 important features
                top_feat_k = min(5, len(feature_importance_summary))
                if top_feat_k > 0:
                    top_feats = torch.topk(feature_importance_summary, k=top_feat_k)
                    logging.info(f"  Top {top_feat_k} important features for '{ntype}':")
                    for i in range(top_feat_k):
                        feat_idx = top_feats.indices[i].item()
                        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
                        logging.info(f"    - {feat_name}: {top_feats.values[i].item():.4f}")

        # Save explanation results to JSON
        output_path = f'./exports/results/gnn_het_captum_detection_explanation_{int(time.time())}.json'
        with open(output_path, 'w') as f:
            json.dump(explanation_results, f, indent=2)
        
        logging.info(f"Explanation results saved to {output_path}")
        
        return explanation_results

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
        elif val == 0.0:
            # special case to avoid early stopping at beginning
            pass
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                    
        return self.early_stop

