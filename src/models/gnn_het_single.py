# Custom imports
from models.encoders.hetero_encoder import GNNHeteroEncoderModel

# Standard library imports
from datetime import datetime
import logging
import os
import time
from typing import Any, Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import re
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    HGTConv,
    Linear,
    global_max_pool,
)
import json
from torch_geometric.explain import Explainer, GNNExplainer, HeteroExplanation
from factories import data
from models.focal_loss import FocalLoss
from repositories.graphs.pyg_builder import get_hetero_column_names, visualize_features_distribution # for GNNExplainer feature names
import copy
from functools import partial
from repositories.graphs.pyg_builder import y_labels
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# import moved file
from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from src.xai import captum_explainer.py

# Local application/library specific imports


logging.info("Imported y_labels in gnn_het.py: %s", y_labels)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
    
        

    @torch.no_grad()
    def predict(self, data: HeteroData) -> List[int]:
        """Predict anomaly classes for the given data HeteroData."""
        self.eval()
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        anom_logits = self(batch)      # [B,5]
        pred = anom_logits.argmax(dim=1)
        return pred.detach().cpu().tolist()

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
            logging.info("Stage 2 Epoch %d: stop_score: %.4f Train Loss: %.4f, Val Loss: %.4f, f1 Score: %.4f, anomaly f1 Score: %.4f, binary f1 Score: %.4f",
                        epoch+1, val_metrics[early_stop_metric], train_metrics["loss"], val_metrics["loss"], val_metrics["f1_score"], val_metrics["anomaly_f1_score"], val_metrics["binary_f1_score"])
            
            scheduler.step(val_metrics[early_stop_metric])
            # Check for early stopping
            if best_val_metrics is None or (early_stop_mode == 'max' and val_metrics[early_stop_metric] > best_val_metrics[early_stop_metric]) or (early_stop_mode == 'min' and val_metrics[early_stop_metric] < best_val_metrics[early_stop_metric]):
                best_val_metrics = val_metrics
                best_model_state = self.state_dict()
                logging.info("New best model found at epoch %d with F1 Score: %.4f", epoch, val_metrics[early_stop_metric])

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
        #criterion = self.get_criterion(test_loader)
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
            anom_logits = self(batch)      # [B], [B,5]
            y = batch.y.view(-1).long()

            #loss = self.criterion(anom_logits, y)
            loss = self.criterion(anom_logits, y)
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)
            pred = anom_logits.argmax(dim=1)
            y_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(pred.detach().cpu().tolist())
        
        

        self.get_label_metrics(y_all, y_pred_all, total_loss / max(1, total_num), export_results=True)
        report = classification_report(y_all, y_pred_all, target_names=y_labels, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"./exports/results/classification_report_classify_{test_description}.csv")

                # Confusion Matrix
        
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

        #Get Explainability for all anomaly predictions
        for i in range(len(y_pred_all)):
            if y_pred_all[i] > 0:
                logging.info("Generating explanation for test sample %d with predicted class %d (%s)", i, y_pred_all[i], y_labels[y_pred_all[i]])
                data = test_loader.dataset[i]
                explainer_results = self.explain_with_captum(data)
                #Save or log explainer_results as needed
        

        return y_all, y_pred_all

    # Helper functions for training and evaluation
    def get_weights(self, labels, min_num_classes, epsilon=1e-6):
        """Compute class weights to handle class imbalance."""
        counts = np.bincount(labels, minlength=min_num_classes).astype(np.float32)
        counts[counts == 0] = epsilon  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * len(counts)  # normalize
        weights[~np.isfinite(weights)] = epsilon  # handle any inf or nan
        
        
        
        return weights   
        return self.early_stop 
