from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
#from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from repositories.graphs.pyg_builder import y_labels
import logging
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, GINConv
from torch_geometric.data import Batch

import os

logging.info("Imported y_labels in gnn.py: %s", y_labels)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
y_anomaly_labels = y_labels[1:]  # Exclude normal class (0)

# class GNNClassifierModel(nn.Module):
#     """GNN model for anomaly detection.  It is supervised model,
#     which classifies each graph as normal, MITM, DoS, scan, physical fault, anomaly"""

#     def __init__(self, config: Dict[str, Any], in_channels: int=51, out_channels: int=1):
#         super(GNNClassifierModel, self).__init__()
#         self.config = config
#         hd = config.get("hidden_dim", 32)
#         # self.conv1 = SAGEConv(in_channels, hd)
#         # self.bn1 = nn.BatchNorm1d(hd)
#         # self.conv2 = SAGEConv(hd, hd)
#         # self.bn2 = nn.BatchNorm1d(hd)
#         # self.dropout = config.get("dropout", 0.5)
#         # self.out = nn.Linear(hd, out_channels)
#         # self.lin1 = nn.Linear(hd, hd)

#         mlp1 = nn.Sequential(
#             nn.Linear(in_channels, hd),
#             nn.ReLU(),
#             nn.Linear(hd, hd)
#         )

#         mlp2 = nn.Sequential(
#             nn.Linear(hd, hd),
#             nn.ReLU(),
#             nn.Linear(hd, hd)
#         )
#         self.conv1 = GINConv(mlp1)
#         self.bn1 = nn.BatchNorm1d(hd)

#         self.conv2 = GINConv(mlp2)
#         self.bn2 = nn.BatchNorm1d(hd)

#         self.dropout = config.get("dropout", 0.5)
#         self.lin1 = nn.Linear(hd, hd)
#         self.out = nn.Linear(hd, out_channels)
        
#         self.to(DEVICE)


#     def forward(self, data: Data) -> torch.Tensor:
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # Global pooling (mean) over all nodes in the graph
#         x = global_max_pool(x, data.batch) 
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.out(x)        

#         return x

# Data splitting and sampling
# def build_data_loaders(dataset):
#     """Stratified split of dataset into train and test sets based on graph labels."""
#     logging.info("Building data loaders with stratified split for dataset with %d samples...", len(dataset))
    
#     # Get labels for stratification
#     labels = [data.y.item() for data in dataset]
#     class_counts = np.bincount(labels, minlength=len(y_labels))
#     logging.info("Overall class distribution in dataset: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(class_counts)]))

#     # Ensure each class has at least 15 samples for stratified splitting
#     need_oversampling = False
#     for i, count in enumerate(class_counts):
#         if count < 15:
#             need_oversampling = True
#             logging.warning("Class '%s' has only %d samples. Stratified splitting may not be reliable.", y_labels[i], count)
#             # manually oversample this class in the dataset
#             samples_to_add = 15 - count
#             class_samples = [data for data in dataset if data.y.item() == i]
#             for _ in range(samples_to_add):
#                 dataset.append(class_samples[np.random.randint(0, len(class_samples))])
#             logging.info("After oversampling, dataset size is %d samples.", len(dataset))
#     if need_oversampling:
#         # Recompute labels after oversampling
#         labels = [data.y.item() for data in dataset]
#         class_counts = np.bincount(labels, minlength=len(y_labels))
#         logging.info("New Overall class distribution in dataset: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(class_counts)]))

#     # split dataset for 60% train, 40% validation/final test
#     splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)

#     train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))

#     # further split test into 20% validation and 20% final test
#     splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
#     test_labels = [labels[i] for i in test_idx]
#     relative_val_idx, relative_final_test_idx = next(splitter2.split(np.zeros(len(test_idx)), test_labels))

#     # Map relative indices back to original test indices
#     val_idx = [test_idx[i] for i in relative_val_idx]
#     final_test_idx = [test_idx[i] for i in relative_final_test_idx]

#     # check distribution in each split
#     test_counts = np.bincount(test_labels, minlength=len(y_labels))
#     val_labels = [labels[i] for i in val_idx]
#     val_counts = np.bincount(val_labels, minlength=len(y_labels))
#     final_test_labels = [labels[i] for i in final_test_idx]
#     final_test_counts = np.bincount(final_test_labels, minlength=len(y_labels))

#     logging.info("Overall label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(np.bincount(labels, minlength=len(y_labels)))]))
#     logging.info("Train label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(np.bincount([labels[i] for i in train_idx], minlength=len(y_labels)))]))
#     logging.info("Validation label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(val_counts)]))
#     logging.info("Final test label distribution: " + ", ".join([f"{y_labels[i]}: {count}" for i, count in enumerate(final_test_counts)]))

#     # Handling class imbalance with weighted random sampler did not help much,
#     # will oversample minority classes manually instead.
#     # This is for training set only
#     #train_labels = [np.array(labels)[train_idx][i] for i in range(len(train_idx))]
#     #class_counts = np.bincount(train_labels, minlength=len(y_labels))
#     #max_count = class_counts.max()
#     # oversampled_train_idx = []
#     # for i in range(len(y_labels)):
#     #     if class_counts[i] == 0:
#     #         continue
#     #     # Find all indices of this class in the training set
#     #     class_indices = [idx for idx, label in zip(train_idx, train_labels) if label == i]
#     #     # Oversample to match max_count
#     #     oversampled = np.random.choice(class_indices, max_count, replace=True)
#     #     oversampled_train_idx.extend(oversampled.tolist())
#     # np.random.shuffle(oversampled_train_idx)

#     logging.info("Train set size: %d, Validation set size: %d, Final test set size: %d", len(train_idx), len(val_idx), len(final_test_idx))
#     #train_set = Subset(dataset, oversampled_train_idx)
#     train_set = Subset(dataset, train_idx)
#     val_set = Subset(dataset, val_idx)
#     final_test_set = Subset(dataset, final_test_idx)

#     # ---------- commented out weighted random sampler -------------
#     #class_sample_count = np.array([len(np.where(np.array(train_labels) == t)[0]) for t in np.unique(train_labels)])
    


#     # Compute class weights
#     #weights = get_weights(train_labels, min_num_classes=len(y_labels))
#     #sampler_weights = torch.tensor([weights[int(t)] for t in train_labels], dtype=torch.double)

#     #sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
#     # ---------------------------------------------------------------

#     # create loaders
#     train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
#     final_test_loader = DataLoader(final_test_set, batch_size=32, shuffle=False, num_workers=0)

#     return train_loader, val_loader, final_test_loader

def get_criterion(data_loader: DataLoader) -> nn.Module:
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
        pos_weight=torch.tensor(pos_weight_scalar, device=DEVICE)
    )

    # ---------- Anomaly head (5-way CE) ----------
    anom_labels = [d.y.item()-1 for d in data_loader.dataset if d.y.item() > 0]
    # ensure at least length 5 for bincount
    counts = np.bincount(anom_labels, minlength=5).astype(np.float32)
    counts[counts == 0] = 1e-6
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    anom_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float, device=DEVICE)
    )
    return bin_criterion, anom_criterion

    

# Training and evaluation functions

def train_epoch(model: nn.Module, loader: DataLoader, criterion: Tuple[nn.Module, nn.Module], optimizer: torch.optim.Optimizer, use_binary = True, use_anomaly = True):
    model.train()
    total_loss = 0
    bin_criterion, anom_criterion = criterion
    try:
        total_num = 0
        y_all = []
        y_pred_all = []
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            bin_logits, anom_logits = model(batch)      # [B], [B,5]
            y = batch.y.view(-1).long()
            y_bin = (y != 0).long()
            mask = (y_bin == 1)
            loss = 0.0
            if use_binary:
                lossA = bin_criterion(bin_logits, y_bin.float())
                loss = loss + lossA
            if use_anomaly and mask.any():
                y5 = (y[mask] - 1).long()
                lossB = anom_criterion(anom_logits[mask], y5)
                loss = loss + 2 * lossB
            
            if not use_binary and not use_anomaly:
                raise ValueError("At least one of use_binary or use_anomaly must be True.")
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)

            bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
            pred = bin_preds.clone()
            if mask.any():
                anom_preds = anom_logits[mask].argmax(dim=1) + 1
                pred[mask] = anom_preds

            y_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(pred.detach().cpu().tolist())

        mean_loss = total_loss / max(1, total_num)
        return get_label_metrics(y_all, y_pred_all, mean_loss)
    except Exception as e:
        logging.error("Error during training epoch: %s", str(e))
        raise e
    
    #logging.info("Epoch Train Loss: %.4f, Mean Anomaly Macro F1: %.4f, Mean Macro F1: %.4f, Mean Balanced Acc: %.4f",
    #             mean_loss, mean_anomaly_macro_f1, mean_macro_f1, mean_balanced_acc)
def get_label_metrics(y_true, y_pred, mean_loss):
    
    average_type = 'macro'
    anomaly_labels = y_labels[1:]  # Exclude normal class (0)
    logging.info("Calculating label metrics with average type: %s", average_type)
    precision_val = precision_score(y_true, y_pred, average=average_type, zero_division=0)
    recall_val = recall_score(y_true, y_pred, average=average_type, zero_division=0)
    f1_score_val = f1_score(y_true, y_pred, average=average_type, zero_division=0)
    f1_score_anomaly = f1_score(y_true, y_pred, labels=list(range(1, len(y_labels))), average=average_type, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    return {
        "loss": mean_loss,
        "precision": precision_val,
        "recall": recall_val,
        "f1_score": f1_score_val,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_score_anomaly": f1_score_anomaly
    } 

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: Tuple[nn.Module, nn.Module], use_binary = True, use_anomaly = True):
    model.eval()
    bin_criterion, anom_criterion = criterion
    total_loss = 0
    total_num = 0
    y_all = []
    y_pred_all = []
    for batch in loader:
        batch = batch.to(DEVICE)
        bin_logits, anom_logits = model(batch)      # [B], [B,5]
        y = batch.y.view(-1).long()
        y_bin = (y != 0).long()
        mask = (y_bin == 1)

        loss = 0.0
        if use_binary:
            lossA = bin_criterion(bin_logits, y_bin.float())
            loss = loss + lossA
        if use_anomaly and mask.any():
            y5 = (y[mask] - 1).long()
            lossB = anom_criterion(anom_logits[mask], y5)
            loss = loss + lossB
        if not use_binary and not use_anomaly:
            raise ValueError("At least one of use_binary or use_anomaly must be True.")
        
        total_loss += loss.item() * y.size(0)
        total_num += y.size(0)
        bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
        pred = bin_preds.clone()
        if mask.any():
            anom_preds = anom_logits[mask].argmax(dim=1) + 1
            pred[mask] = anom_preds
        y_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(pred.detach().cpu().tolist())
    mean_loss = total_loss / max(1, total_num)
    return get_label_metrics(y_all, y_pred_all, mean_loss)



def fit_model(model: nn.Module, 
              train_loader: DataLoader, 
              anom_train_loader: DataLoader, 
              val_loader: DataLoader, 
              anom_val_loader: DataLoader, 
              config: Dict[str, Any],
              use_anomaly: bool = True):
    """Train the GNN model with early stopping based on validation anomaly macro F1 score."""
    num_epochs = config.get("max_epochs", 100)
    learning_rate = config.get("learning_rate", 0.001)
    patience = config.get("early_stopping_patience", 10)
    min_delta = config.get("early_stopping_min_delta", 0.0001)

    bin_criterion, anom_criterion = get_criterion(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)
    early_stop_mode = 'min'
    early_stopper = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
    early_stopper2 = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
    early_stopper3 = GNNEarlyStopping(patience=patience, min_delta=min_delta, mode=early_stop_mode)
    best_val_metrics = None
    best_model_state = None
    early_stop_metric = 'loss'
    

    # Stage 1: Train with binary head only for initial epochs
    # freeze anomaly head    
    for param in model.anom_head.parameters():
        param.requires_grad = False
    logging.info("Stage 1: Training with binary head only for %d epochs...", config.get("stage1_epochs", 10))
    for epoch in range(config.get("stage1_epochs", 100)):
        train_metrics = train_epoch(model, train_loader, (bin_criterion, anom_criterion), optimizer, use_binary=True, use_anomaly=False)
        val_metrics = evaluate_model(model, val_loader, (bin_criterion, anom_criterion), use_binary=True, use_anomaly=False)
        logging.info("Stage 1 Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Val Anomaly F1: %.4f",
                     epoch+1, train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score_anomaly"])
        #scheduler.step(val_metrics[early_stop_metric])

        # Check for early stopping
        if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
            best_val_metrics = val_metrics
            best_model_state = model.state_dict()
            logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics[early_stop_metric])

        early_stopper.step(val_metrics[early_stop_metric])
        if early_stopper.early_stop:
            logging.info("Early stopping triggered at epoch %d during Stage 1.", epoch+1)
            break
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Stage 2: Train with anomaly head only for anomaly samples, freeze binary head
    # unfreeze anomaly head, freeze binary head and encoder
    if use_anomaly:
        logging.info("Stage 2: Training with anomaly head only for %d epochs...", config.get("stage2_epochs", 10))
        for param in model.bin_head.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.anom_head.parameters():
            param.requires_grad = True

        for epoch in range(config.get("stage2_epochs", 100)):
            train_metrics = train_epoch(model, anom_train_loader, (bin_criterion, anom_criterion), optimizer, use_binary=False, use_anomaly=True)
            val_metrics = evaluate_model(model, anom_val_loader, (bin_criterion, anom_criterion), use_binary=False, use_anomaly=True)
            logging.info("Stage 2 Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Val Anomaly F1: %.4f",
                        epoch+1, train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score_anomaly"])
            #scheduler.step(val_metrics[early_stop_metric])
            # Check for early stopping
            if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
                best_val_metrics = val_metrics
                best_model_state = model.state_dict()
                logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics["f1_score"])

            early_stopper2.step(val_metrics[early_stop_metric])
            if early_stopper2.early_stop:
                logging.info("Early stopping triggered at epoch %d during Stage 2.", epoch+1)
                break
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Stage 3: Joint training with both heads, unfreeze all
        for param in model.bin_head.parameters():
            param.requires_grad = True
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.anom_head.parameters():
            param.requires_grad = True
        logging.info("Stage 3: Joint training with both heads for %d epochs...", num_epochs)
        
        for epoch in range(20):
            train_metrics = train_epoch(model, train_loader, (bin_criterion, anom_criterion), optimizer, use_binary=True, use_anomaly=True)
            val_metrics = evaluate_model(model, val_loader, (bin_criterion, anom_criterion), use_binary=True, use_anomaly=True)
            logging.info("Stage 3 Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Val Anomaly F1: %.4f",
                        epoch+1, train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score_anomaly"])
            
            scheduler.step(val_metrics[early_stop_metric])
            # Check for early stopping
            if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
                best_val_metrics = val_metrics
                best_model_state = model.state_dict()
                logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics["f1_score"])



            early_stopper3.step(val_metrics[early_stop_metric])
            if early_stopper3.early_stop:
                logging.info("Early stopping triggered at epoch %d during Stage 3.", epoch+1)
                break
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    # Calculate final training metrics
    final_train_metrics = evaluate_model(model, train_loader, (bin_criterion, anom_criterion),use_anomaly=use_anomaly)
    logging.info("Final Training Metrics - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                 final_train_metrics["loss"], final_train_metrics["f1_score"], final_train_metrics["recall"],
                 final_train_metrics["precision"], final_train_metrics["balanced_accuracy"], final_train_metrics["accuracy"])

    return model, (bin_criterion, anom_criterion), final_train_metrics

def test_model(model: nn.Module, test_loader: DataLoader, criteria: Tuple[nn.Module, nn.Module], test_description: str="Final Test Set"):
    """Evaluate the trained model on the final test set and print classification report."""
    bin_criterion, anom_criterion = criteria
    test_metrics = evaluate_model(model, test_loader, criteria)
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
        bin_logits, anom_logits = model(batch)      # [B], [B,5]
        y = batch.y.view(-1).long()
        y_bin = (y != 0).long()
        mask = (y_bin == 1)

        loss = 0.0
        if bin_criterion is not None:
            lossA = bin_criterion(bin_logits, y_bin.float())
            loss = loss + lossA
        if anom_criterion is not None and mask.any():
            y5 = (y[mask] - 1).long()
            lossB = anom_criterion(anom_logits[mask], y5)
            loss = loss + lossB
        
        total_loss += loss.item() * y.size(0)
        total_num += y.size(0)
        bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
        pred = bin_preds.clone()
        if use_anomaly and mask.any():
            anom_preds = anom_logits[mask].argmax(dim=1) + 1
            pred[mask] = anom_preds
        y_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(pred.detach().cpu().tolist())

    report_df = pd.DataFrame(classification_report(y_all, y_pred_all, output_dict=True)).transpose()
    report_df.to_csv(f"./exports/results/classification_report_{test_description}.csv")

    return test_metrics

# Helper functions for training and evaluation
def get_weights(labels, min_num_classes, epsilon=1e-6):
    """Compute class weights to handle class imbalance."""
    counts = np.bincount(labels, minlength=min_num_classes).astype(np.float32)
    counts[counts == 0] = epsilon  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / np.sum(weights) * len(counts)  # normalize
    weights[~np.isfinite(weights)] = epsilon  # handle any inf or nan
    return weights

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
        elif self.mode == 'max' and val < self.best_score - self.min_delta:
            self.best_score = val
            self.counter = 0
        elif self.mode == 'min' and val > self.best_score + self.min_delta:
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

# @torch.no_grad()
# def compute_metrics(logits: torch.Tensor, y: torch.Tensor):
#     #y_pred = logits.argmax(dim=1).detach().cpu().numpy()
#     y_actual = y.detach().cpu().numpy().astype(np.int32)
    
#     probs = torch.sigmoid(logits).squeeze(1)
#     y_pred = (probs >= 0.5).long().detach().cpu().numpy()

#     # Overall F1 score for anomaly classes only (exclude normal class 0)
    
#     balanced_accuracy = balanced_accuracy_score(y_actual, y_pred)

#     # additional metrics for comparing with previous works
#     accuracy_val = accuracy_score(y_actual, y_pred)
#     precision_val = precision_score(y_actual, y_pred, average='binary', zero_division=0)
#     recall_val = recall_score(y_actual, y_pred, average='binary', zero_division=0)
#     f1_score_val = f1_score(y_actual, y_pred, average='binary', zero_division=0)

#     return {        
#         "balanced_accuracy": balanced_accuracy,
#         "accuracy": accuracy_val,
#         "precision": precision_val,
#         "recall": recall_val,
#         "f1_score": f1_score_val
#     }

# @torch.no_grad()
# def validate_dataset_integrity(dataset: torch.utils.data.Dataset):
#     """Validate that all graphs in the dataset have valid labels."""
#     invalid_count = 0
#     logging.info("Validating dataset integrity for %d graphs...", len(dataset))
#     for i, data in enumerate(dataset):
#         if data is None:
#             logging.warning("Data at index %d is None.", i)
#             invalid_count += 1
#             continue
#         if not hasattr(data, 'x') or data.x is None:
#             logging.warning("Data at index %d has no node features.", i)
#             invalid_count += 1
#             continue
#         if not hasattr(data, 'y') or data.y is None:
#             logging.warning("Data at index %d has no label.", i)
#             invalid_count += 1
#             continue
#         if not hasattr(data, 'edge_index') or data.edge_index is None:
#             logging.warning("Data at index %d has no edge_index.", i)
#             invalid_count += 1
#             continue
#         if data.x.shape[0] == 0 and data.num_nodes == 0:
#             logging.warning("Data at index %d has zero nodes.", i)
#             invalid_count += 1
#             continue
        
#         # check if data can load into batch
#         try:
#             Batch.from_data_list([data])
#         except Exception as e:
#             logging.warning("Data at index %d errors on Batch load with device %s: %s", i, DEVICE, str(e))
#             invalid_count += 1
#             continue

#     if invalid_count == 0:
#         logging.info("All graphs in the dataset have valid labels.")
#     else:
#         logging.warning("%d graphs with invalid labels found in the dataset.", invalid_count)
