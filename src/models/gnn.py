from typing import Any, Dict
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
DEVICE = "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
y_anomaly_labels = y_labels[1:]  # Exclude normal class (0)

class GNNClassifierModel(nn.Module):
    """GNN model for anomaly detection.  It is supervised model,
    which classifies each graph as normal, MITM, DoS, scan, physical fault, anomaly"""

    def __init__(self, config: Dict[str, Any], in_channels: int=51, out_channels: int=1):
        super(GNNClassifierModel, self).__init__()
        self.config = config
        hd = config.get("hidden_dim", 32)
        # self.conv1 = SAGEConv(in_channels, hd)
        # self.bn1 = nn.BatchNorm1d(hd)
        # self.conv2 = SAGEConv(hd, hd)
        # self.bn2 = nn.BatchNorm1d(hd)
        # self.dropout = config.get("dropout", 0.5)
        # self.out = nn.Linear(hd, out_channels)
        # self.lin1 = nn.Linear(hd, hd)

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hd),
            nn.ReLU(),
            nn.Linear(hd, hd)
        )

        mlp2 = nn.Sequential(
            nn.Linear(hd, hd),
            nn.ReLU(),
            nn.Linear(hd, hd)
        )
        self.conv1 = GINConv(mlp1)
        self.bn1 = nn.BatchNorm1d(hd)

        self.conv2 = GINConv(mlp2)
        self.bn2 = nn.BatchNorm1d(hd)

        self.dropout = config.get("dropout", 0.5)
        self.lin1 = nn.Linear(hd, hd)
        self.out = nn.Linear(hd, out_channels)
        
        self.to(DEVICE)


    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (mean) over all nodes in the graph
        x = global_max_pool(x, data.batch) 
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)        

        return x

# Data splitting and sampling
def build_data_loaders(dataset: torch.utils.data.Dataset):
    """Stratified split of dataset into train and test sets based on graph labels."""
    logging.info("Building data loaders with stratified split for dataset with %d samples...", len(dataset))
    
    # Get labels for stratification
    labels = [data.y.item() for data in dataset]

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

def get_criterion(data_loader: DataLoader) -> nn.Module:
    """Get loss function with class weights to handle class imbalance."""
    # class_weights = get_weights([data.y.item() for data in data_loader.dataset], min_num_classes=len(y_labels))
    # crit_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    # logging.info("Criterion class weights: %s", class_weights.tolist())
    # criterion = nn.CrossEntropyLoss(weight=crit_weights)
    # criterion.to(DEVICE)

    labels = [data.y.item() for data in data_loader.dataset]
    pos_weight = torch.tensor([(len(labels) - sum(labels)) / (sum(labels) + 1e-6)], dtype=torch.float)    
    #pos_weight = pos_weight * 0.9 # scale down to avoid too high weight

    logging.info("Criterion positive class weight: %.4f", pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion.to(DEVICE)
    return criterion

# Training and evaluation functions

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    total_loss = 0
    metrics_hist = []
    #logging.info("Training on %d batches...", len(loader))
    try:
        epoch_y_true = []
        epoch_y_pred = []
        for batch in loader:
            #logging.info("Processing batch...")
            batch = batch.to(DEVICE)           
            #logging.info("Batch moved to device %s.", DEVICE) 
            optimizer.zero_grad()            
            #logging.info("Optimizer zeroed.")
            logits = model(batch)            
            #logging.info("Model forward pass completed.")
            #logging.info("criterion being used: %s", str(criterion))
            #logging.info("y type: %s, y shape: %s", str(batch.y.dtype), str(batch.y.shape))
            loss = criterion(logits.squeeze(1), batch.y) 
            #logging.info("Loss calculation completed.")
            loss.backward()            
            #logging.info("Loss backward pass completed.")
            optimizer.step()            
            #logging.info("Optimizer step completed.")
            total_loss += loss.item() * batch.y.size(0)
            #logging.info("Total loss updated.")
            probs = torch.sigmoid(logits).squeeze(1)
            
            # get predicted labels
            #logging.info("Probs calculation completed.")
            y_pred_batch = (probs >= 0.5).long().detach().cpu().numpy()
            #logging.info("Preds calculation completed.")

            epoch_y_true.extend(batch.y.detach().cpu().numpy().astype(np.int32).tolist())
            epoch_y_pred.extend(y_pred_batch.tolist())
            #logging.info("Batch Loss: %.4f", loss.item())

        mean_loss = total_loss / len(loader.dataset)
        # Compute overall metrics for the epoch
        f1 = f1_score(epoch_y_true, epoch_y_pred, average='binary', zero_division=0)
        balanced_acc = balanced_accuracy_score(epoch_y_true, epoch_y_pred)
        accuracy = accuracy_score(epoch_y_true, epoch_y_pred)
        recall = recall_score(epoch_y_true, epoch_y_pred, average='binary', zero_division=0)
        precision = precision_score(epoch_y_true, epoch_y_pred, average='binary', zero_division=0)  
        return {
            "loss": mean_loss,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision
        }

    except Exception as e:
        logging.error("Error during training epoch: %s", str(e))
        raise e
    
    #logging.info("Epoch Train Loss: %.4f, Mean Anomaly Macro F1: %.4f, Mean Macro F1: %.4f, Mean Balanced Acc: %.4f",
    #             mean_loss, mean_anomaly_macro_f1, mean_macro_f1, mean_balanced_acc)
    

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0
    total_num = 0
    y_all = []
    y_pred_all = []
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data)
        loss = criterion(logits.squeeze(1), data.y)
        total_loss += loss.item() * data.y.size(0)
        
        y_all.extend(data.y.detach().cpu().numpy().astype(np.int32).tolist())
        probs = torch.sigmoid(logits).squeeze(1)
        y_pred = (probs >= 0.5).long().detach().cpu().numpy()
        y_pred_all.extend(y_pred.tolist())
        total_num += data.y.size(0)
        
    
    mean_loss = total_loss / total_num
    precision_val = precision_score(y_all, y_pred_all, average='binary', zero_division=0)
    recall_val = recall_score(y_all, y_pred_all, average='binary', zero_division=0)
    f1_score_val = f1_score(y_all, y_pred_all, average='binary', zero_division=0)
    accuracy = accuracy_score(y_all, y_pred_all)
    balanced_accuracy = balanced_accuracy_score(y_all, y_pred_all)

    return {
        "loss": mean_loss,
        "f1_score": f1_score_val,
        "precision": precision_val,
        "recall": recall_val,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy
    }

def fit_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]):
    """Train the GNN model with early stopping based on validation anomaly macro F1 score."""
    num_epochs = config.get("max_epochs", 100)
    learning_rate = config.get("learning_rate", 0.001)
    patience = config.get("early_stopping_patience", 10)
    min_delta = config.get("early_stopping_min_delta", 0.0001)

    criterion = get_criterion(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-6)

    early_stopper = GNNEarlyStopping(patience=patience, min_delta=min_delta)

    best_val_metrics = None
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        logging.info("Epoch %d/%d", epoch, num_epochs)
        train_metrics = train_epoch(model, train_loader, criterion, optimizer)
        logging.info("Train Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                     train_metrics["loss"], train_metrics["f1_score"], train_metrics["recall"], train_metrics["precision"],
                     train_metrics["balanced_accuracy"], train_metrics["accuracy"])
        val_metrics = evaluate_model(model, val_loader, criterion)
        logging.info("Validation Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                     val_metrics["loss"], val_metrics["f1_score"], val_metrics["recall"], val_metrics["precision"],
                     val_metrics["balanced_accuracy"], val_metrics["accuracy"])
        
        scheduler.step(val_metrics["f1_score"])

        # Check for early stopping
        if best_val_metrics is None or val_metrics["f1_score"] > best_val_metrics["f1_score"]:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict()
            logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics["f1_score"])

        if early_stopper.step(val_metrics["f1_score"]):
            logging.info("Early stopping triggered at epoch %d", epoch)
            logging.info("Ignoring early stopping for debugging purposes.")
            #break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, criterion, best_val_metrics

def test_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, test_description: str="Final Test Set"):
    """Evaluate the trained model on the final test set and print classification report."""
    #criterion = get_criterion(test_loader)
    test_metrics = evaluate_model(model, test_loader, criterion)
    logging.info("%s Results - Loss: %.4f, F1: %.4f, Recall: %.4f, Precision: %.4f, Balanced Acc: %.4f, Accuracy: %.4f",
                 test_description, test_metrics["loss"], test_metrics["f1_score"], test_metrics["recall"],
                 test_metrics["precision"], test_metrics["balanced_accuracy"], test_metrics["accuracy"])
    
    # Get detailed classification report
    y_all = []
    y_pred_all = []
    for data in test_loader:
        data = data.to(DEVICE)
        logits = model(data)
        probs = torch.sigmoid(logits).squeeze(1)
        y_pred = (probs >= 0.5).long().detach().cpu().numpy()
        y_all.extend(data.y.detach().cpu().numpy().astype(np.int32).tolist())
        y_pred_all.extend(y_pred.tolist())
    

    # export classification report to dataframe and save as csv
    report_dict = classification_report(y_all, y_pred_all, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    
    report_df.to_csv(f"./exports/results/classification_report_{test_description}.csv")

    return test_metrics

# Helper functions for training and evaluation
# def get_weights(labels, min_num_classes, epsilon=1e-6):
#     """Compute class weights to handle class imbalance."""
#     counts = np.bincount(labels, minlength=min_num_classes).astype(np.float32)
#     counts[counts == 0] = epsilon  # avoid division by zero
#     weights = 1.0 / counts
#     weights = weights / np.sum(weights) * len(counts)  # normalize
#     weights[~np.isfinite(weights)] = epsilon  # handle any inf or nan
#     return weights

class GNNEarlyStopping:
    """Early stopping utility to stop training when 
    macro F1 score across all anomaly classes does not improve.
    We ignore the normal class (class 0) for early stopping as it is over-represented.    
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False    
    
    def step(self, val: float):
        if self.best_score is None:
            self.best_score = val
        elif val > self.best_score + self.min_delta:
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

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor):
    #y_pred = logits.argmax(dim=1).detach().cpu().numpy()
    y_actual = y.detach().cpu().numpy().astype(np.int32)
    
    probs = torch.sigmoid(logits).squeeze(1)
    y_pred = (probs >= 0.5).long().detach().cpu().numpy()

    # Overall F1 score for anomaly classes only (exclude normal class 0)
    
    balanced_accuracy = balanced_accuracy_score(y_actual, y_pred)

    # additional metrics for comparing with previous works
    accuracy_val = accuracy_score(y_actual, y_pred)
    precision_val = precision_score(y_actual, y_pred, average='binary', zero_division=0)
    recall_val = recall_score(y_actual, y_pred, average='binary', zero_division=0)
    f1_score_val = f1_score(y_actual, y_pred, average='binary', zero_division=0)

    return {        
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy_val,
        "precision": precision_val,
        "recall": recall_val,
        "f1_score": f1_score_val
    }

@torch.no_grad()
def validate_dataset_integrity(dataset: torch.utils.data.Dataset):
    """Validate that all graphs in the dataset have valid labels."""
    invalid_count = 0
    logging.info("Validating dataset integrity for %d graphs...", len(dataset))
    for i, data in enumerate(dataset):
        if data is None:
            logging.warning("Data at index %d is None.", i)
            invalid_count += 1
            continue
        if not hasattr(data, 'x') or data.x is None:
            logging.warning("Data at index %d has no node features.", i)
            invalid_count += 1
            continue
        if not hasattr(data, 'y') or data.y is None:
            logging.warning("Data at index %d has no label.", i)
            invalid_count += 1
            continue
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            logging.warning("Data at index %d has no edge_index.", i)
            invalid_count += 1
            continue
        if data.x.shape[0] == 0 and data.num_nodes == 0:
            logging.warning("Data at index %d has zero nodes.", i)
            invalid_count += 1
            continue
        
        # check if data can load into batch
        try:
            Batch.from_data_list([data])
        except Exception as e:
            logging.warning("Data at index %d errors on Batch load with device %s: %s", i, DEVICE, str(e))
            invalid_count += 1
            continue

    if invalid_count == 0:
        logging.info("All graphs in the dataset have valid labels.")
    else:
        logging.warning("%d graphs with invalid labels found in the dataset.", invalid_count)
