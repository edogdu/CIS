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
import copy
from captum.attr import IntegratedGradients, Saliency, DeepLift
from functools import partial
from repositories.graphs.pyg_builder import y_labels, get_hetero_column_names
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Local application/library specific imports


logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
y_anomaly_labels = y_labels[1:]  # Exclude normal class (0)

logging.info("Imported y_anomaly_labels in gnn_het.py: %s", y_anomaly_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

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

class GNNHeteroEncoderModel(nn.Module):
    """GNN module to produce node embeddings for heterogeneous graphs."""
    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroEncoderModel, self).__init__()
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
        self.bin_thres = config.get("binary_threshold", 0.35)
        hd = config.get("hidden_dim", 32)
        self.criterion = None

        self.encoder = GNNHeteroEncoderModel(config, metadata)


        self.out = nn.Linear(hd, 5)   # -> [B,5] (classes: 1..5 shifted to 0..4 in loss)

        self.to(DEVICE)


    def forward(self, data: HeteroData) -> torch.Tensor:
        h = self.encoder(data)
        logits = self.out(h)                   # [B, 5]
        return logits
        
        

    def get_criterion(self, data_loader: DataLoader) -> nn.Module:
        """Get loss function with class weights to handle class imbalance."""
        
        # Gather all labels from the dataset
        labels = []
        for batch in data_loader:
            logging.info("Processing batch with %d graphs for criterion calculation.", batch.num_graphs)
            logging.info("Batch labels: %s", batch.y.view(-1).long().cpu().numpy().tolist())
            # Map labels to their corresponding anomaly classes, offset by -1 for CrossEntropyLoss
            labels.extend(batch.y.view(-1).long().cpu().numpy().tolist())
        
        weights = self.get_weights(labels, min_num_classes=5)
        logging.info("Anomaly Classifier Class Weights: %s", weights)
        anom_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float, device=DEVICE)
        )
        self.criterion = anom_criterion
        return anom_criterion

        

    # Training and evaluation functions

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer):
        self.train()
        total_loss = 0
        
        try:
            total_num = 0
            y_all = []
            y_pred_all = []
            for batch in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                anom_logits = self(batch)      # [B,5]
                y = batch.y.view(-1).long()
                loss = self.criterion(anom_logits, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_num += y.size(0)

                pred = anom_logits.argmax(dim=1)


                y_all.extend(y.detach().cpu().tolist())
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

        precision_val = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_val = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_score_val = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

        if export_results:
            with open(f"./exports/results/{'final_' if is_final_test else ''}gnn_het_multi_classify_perf_scores.csv", "w") as f:
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
            anom_logits = self(batch)      # [B,5]
            y = batch.y.view(-1).long()
            loss = self.criterion(anom_logits, y)
            
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            pred = anom_logits.argmax(dim=1)
            y_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(pred.detach().cpu().tolist())
        mean_loss = total_loss / max(1, total_num)
        return self.get_label_metrics(y_all, y_pred_all, mean_loss)

    @torch.no_grad()
    def predict(self, data: HeteroData) -> List[int]:
        """Predict anomaly classes for the given data HeteroData."""
        self.eval()
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        anom_logits = self(batch)      # [B,5]
        pred = anom_logits.argmax(dim=1)
        return pred.detach().cpu().tolist()
    
    def _verify_labels_in_loader(self, loader: DataLoader):
        """Verify that all anomaly classes are present in the DataLoader."""
        present_labels = set()
        for batch in loader:
            present_labels.update(batch.y.view(-1).long().cpu().numpy().tolist())
        
        missing_labels = set(range(4)) - present_labels
        if missing_labels:
            logging.warning("Warning: The following anomaly classes are missing in the DataLoader: %s", missing_labels)
        else:
            logging.info("All anomaly classes are present in the DataLoader.")

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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10,min_lr=1e-6)
        early_stop_mode = 'min'  # We want to maximize F1 score
        early_stopper = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
        best_val_metrics = None
        best_model_state = None
        early_stop_metric = 'loss'
        

        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate_model(val_loader)
            logging.info("Stage 2 Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Val F1: %.4f",
                        epoch+1, train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score"])
            scheduler.step(val_metrics[early_stop_metric])
            # Check for early stopping
            if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
                best_val_metrics = val_metrics
                best_model_state = self.state_dict()
                logging.info("New best model found at epoch %d with %s: %.4f", epoch, early_stop_metric, val_metrics[early_stop_metric])

            early_stopper.step(val_metrics[early_stop_metric])
            if early_stopper.early_stop:
                logging.info("Early stopping triggered at epoch %d during Stage 2.", epoch+1)
                break
        # Load best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        # Calculate final training metrics
        final_train_metrics = self.evaluate_model(train_loader)
        logging.info("Final Training Metrics - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                    final_train_metrics["loss"], final_train_metrics["f1_score"], final_train_metrics["recall"],
                    final_train_metrics["precision"], final_train_metrics["balanced_accuracy"], final_train_metrics["accuracy"])

        return self, criterion, final_train_metrics

    def test_model(self, test_loader: DataLoader, test_description: str="Final Test Set"):
        """Evaluate the trained model on the final test set and print classification report."""
        criterion = self.get_criterion(test_loader)
        test_metrics = self.evaluate_model(test_loader)
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
            anom_logits = self(batch)      # [B], [B,4]
            y = batch.y.view(-1).long()

            loss = self.criterion(anom_logits, y)
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            pred = anom_logits.argmax(dim=1)
            y_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(pred.detach().cpu().tolist())
        
        # Get Explainability for all anomaly predictions
        for i in range(len(y_pred_all)):
            # These are all anomaly predictions, so explain all
            logging.info("Generating explanation for test sample %d with true label %d and predicted label %d.", i, y_all[i], y_pred_all[i])
            data = test_loader.dataset[i]
            explainer_results = self.explain_with_captum(data)

        # shift back y_all and y_pred_all to original labels (1-4)   
        logging.info("y_all before shifting: %s", y_all)
        logging.info("y_pred_all before shifting: %s", y_pred_all)           
        self.get_label_metrics(y_all, y_pred_all, total_loss / max(1, total_num), export_results=True)
        logging.info("Generated classification report and saved to CSV.")
        expected_labels = list(range(len(y_anomaly_labels)))
        logging.info("Expected labels for classification report: %s", expected_labels)
        report = classification_report(y_all, y_pred_all, labels=expected_labels, target_names=y_anomaly_labels, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"./exports/results/classification_report_classify_{test_description}.csv")
        logging.info("Classification Report complete and saved.")
                # Confusion Matrix
        
        logging.info("Generating confusion matrix for %s.", test_description)
        # save confusion matrix image
        cm = confusion_matrix(y_all, y_pred_all)
        # add timestamp to filename to avoid overwriting
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"./exports/images/gnn_het_classification_classify_confusion_matrix{int(time.time())}.png")
        plt.close()
        logging.info("Confusion matrix saved.")
        

        return y_all, y_pred_all

    # Helper functions for training and evaluation
    def get_weights(self, labels, min_num_classes, epsilon=1e-6):
        """Compute class weights to handle class imbalance."""
        counts = np.bincount(labels, minlength=min_num_classes).astype(np.float32)
        counts[counts == 0] = epsilon  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * len(counts)  # normalize
        weights[~np.isfinite(weights)] = epsilon  # handle any inf or nan
        # square root to reduce extreme weights
        weights = np.sqrt(weights)
        
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
            pred_class = logits.argmax(dim=1).item()
            pred_prob = F.softmax(logits, dim=1).max().item()
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
            plot_path = f"{save_dir}/classify_plot_top5_feats_snap{snapshot_id}_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close() 
            
            # Add top 5 to explanation data for easy access
            explanation_data["top_5_global_features"] = top_5_features

        # Export Files
        all_node_rankings.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Save JSON
        json_path = f"{save_dir}/classify_explanation_snap{snapshot_id}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(explanation_data, f, indent=2)
            
        # Save CSV
        node_df = pd.DataFrame(all_node_rankings)
        if not node_df.empty:
            node_csv_path = f"{save_dir}/classify_ranking_snap{snapshot_id}_{timestamp}.csv"
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

# -------- Explanation functions --------
 