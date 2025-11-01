import logging
from models.ModelRunner import ModelRunner
from xai.XaiExplainer import XaiExplainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GAE
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from typing import List, Dict, Any, Tuple
import os
import pandas as pd        
import datetime

modelrunner_log = logging.getLogger("models_runner")

class GNNEncoderModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        if dropout is None:
            self.dropout = 0.5
        if dropout < 0 or dropout >= 1:
            raise ValueError("Dropout must be in the range [0, 1).")
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)        
        return x

class GNNAEModelRunner(ModelRunner):
    """Graph Neural Network Autoencoder model implementation."""
    

    def __init__(self, xai_runner: XaiExplainer, input_dim:int, hidden_dim:int, output_dim:int, config:dict, device: torch.device | str | None = None, edge_threshold: float | None = None, snapshot_threshold: float | None = None):        
        self.xai_runner = xai_runner
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.optimizer = None
        self.scheduler = None
        #self.set_seed(42)
        self.edge_threshold = edge_threshold
        self.snapshot_threshold = snapshot_threshold
        modelrunner_log.info(f"Using device: {self.device}")    
        modelrunner_log.info(f"Model config: {config}")
        modelrunner_log.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
        self.config.update({
            "learning_rate": config.get("learning_rate", 0.01),
            "weight_decay": config.get("weight_decay", 5e-4),
            "epochs": config.get("epochs", 200),
            "dropout": config.get("dropout", 0.5),
            "xai_topk": config.get("xai_topk", 20),
            "xai_loss_min": config.get("xai_loss_min", None)
        })

        self.init_model(input_dim, hidden_dim, output_dim)
        

        # define datasets
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.input_dim = input_dim
    
    def init_model(self, input_dim:int, hidden_dim:int=16, output_dim:int=16):
        if(input_dim == self.input_dim and self.model is not None):
            return
        self.input_dim = input_dim
        self.model = GAE(GNNEncoderModel(input_dim, hidden_dim, output_dim, self.config["dropout"])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.config["learning_rate"],
                                    weight_decay=self.config["weight_decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
        
        if hasattr(self.xai_runner, 'set_encoder'):
            self.xai_runner.set_encoder(self.model.encoder)
    
    def prepare_splits(self, dataset, val_fraction=0.1, test_fraction=0.1):
        """Prepare training, validation, and test splits."""
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("Validation and test fractions must sum to less than 1.")
        if not dataset:
            raise ValueError("Dataset is empty or None.")
        modelrunner_log.info(f"Preparing data splits with val_fraction={val_fraction}, test_fraction={test_fraction}")
        modelrunner_log.info(f"Dataset size: {len(dataset)}")
        
        splitter = RandomLinkSplit(num_val=val_fraction, 
                                   num_test=test_fraction, 
                                   is_undirected=True, 
                                   add_negative_train_samples=False)
        
        self.train_set, self.val_set, self.test_set = [], [], []

        for graph in dataset:
            tr, va, te = splitter(graph)
            self.train_set.append(tr)
            self.val_set.append(va)
            self.test_set.append(te)

    def _loss_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        z = self.model.encode(x, edge_index)

        pos_edge_index = getattr(data, 'pos_edge_label_index', None)
        if pos_edge_index is None:
            pos_edge_index = edge_index
        loss, pos_loss, neg_loss, neg_edge_index = self.compute_loss_per_edge(z, pos_edge_index.to(self.device), data.num_nodes)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
    
    @staticmethod
    def _bce_none():
        return nn.BCEWithLogitsLoss(reduction='none')

    def set_seed(self, seed: int=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def compute_loss_per_edge(self,
                              z: torch.Tensor,
                              pos_edge_index: torch.Tensor,
                              num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = nn.BCEWithLogitsLoss(reduction='none')

        # Positive edges
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        pos_loss = bce(pos_out, torch.ones_like(pos_out))

        # Negative edges
        neg_edge_index = self.generate_neg_edge_index(pos_edge_index, num_nodes)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        neg_loss = bce(neg_out, torch.zeros_like(neg_out))

        loss = pos_loss.mean() + neg_loss.mean()
        return loss, pos_loss.detach(), neg_loss.detach(), neg_edge_index

    def determine_anomaly_thresholds(self, edge_percentile: float=0.05, snapshot_percentile: float=0.05):
        """Determine anomaly detection threshold based on reconstruction errors.  We will use percentile to determine where
        to set the threshold. 0.05 means we set the threshold so that 5% of the edges with the lowest reconstruction scores are considered anomalies."""
        logging.info(f"Determining anomaly detection threshold at edge:{edge_percentile}, snapshot:{snapshot_percentile}...")
        all_probs = []
        edge_counts = []
        for i, d in enumerate(self.train_set):
            probs = self.predict(d)            
            all_probs.append(probs)
            edge_counts.append(probs.numel())
            logging.info(f"Processed training graph #{i} with {probs.numel()} edges for threshold determination.")
        all_probs_tensor = torch.cat(all_probs)
        edge_threshold_value = torch.quantile(all_probs_tensor, edge_percentile)
        self.edge_threshold = edge_threshold_value.mean().item()
        modelrunner_log.info(f"Determined anomaly detection edge threshold at {edge_percentile}: {self.edge_threshold}")

        # Now calculate snapshot threshold as well
        snapshot_thresholds = []
        for probs in all_probs:
            anomaly_indices = (probs < self.edge_threshold).to(torch.int32)
            snapshot_anomaly_ratio = float(anomaly_indices / probs.numel())
            snapshot_thresholds.append(snapshot_anomaly_ratio)
        
        all_snapshot_thresholds_tensor = torch.tensor(snapshot_thresholds)
        snapshot_threshold = torch.quantile(all_snapshot_thresholds_tensor, snapshot_percentile)
        self.snapshot_threshold = snapshot_threshold.mean().item()
        modelrunner_log.info(f"Determined anomaly detection snapshot threshold at {snapshot_percentile}: {self.snapshot_threshold}")

        return self.edge_threshold, self.snapshot_threshold

    def generate_neg_edge_index(self, pos_edge_index, num_nodes):
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, 
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1),
            method='sparse')
            
        return neg_edge_index

    def train_epochs(self, epochs: int=None) -> List[Dict[str, Any]]:
        """Train the GNN Autoencoder for a specified number of epochs."""
        if not self.train_set:
            raise ValueError("Training set is empty. Please prepare splits before training.")
        if epochs is None:
            epochs = self.config.get("epochs", 200)

        history = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for g in self.train_set:
                loss = self._loss_step(g)
                epoch_loss += loss
            train_avg_loss = epoch_loss / max(1, len(self.train_set))
            val_metrics = self.evaluate_set(self.val_set) if self.val_set else (None, None)
            modelrunner_log.info(f"Epoch {epoch:03d}, Train Loss: {train_avg_loss:.4f}, Val AUC: {val_metrics[0]:.4f}, Val AP: {val_metrics[1]:.4f}")

            history.append({
                'epoch': epoch,
                'train_loss': train_avg_loss,
                'val_auc': val_metrics[0] if val_metrics else None,
                'val_ap': val_metrics[1] if val_metrics else None
            })
        return history

    def evaluate_set(self, eval_set: List) -> Tuple[float, float]:
        """Evaluate the model on a given dataset."""
        if not eval_set:
            raise ValueError("Evaluation set is empty.")
        self.model.eval()
        total_auc, total_ap = [], []
        with torch.no_grad():
            for data in eval_set:
                x = data.x.to(self.device)
                z = self.model.encode(x, data.edge_index.to(self.device))
                pos = data.edge_index.to(self.device)
                #neg = data.neg_edge_label_index.to(self.device)
                # we dont have neg in val/test
                auc, ap = self.model.test(z, pos, self.generate_neg_edge_index(pos, data.num_nodes))
                total_auc.append(auc)
                total_ap.append(ap)
        avg_auc = sum(total_auc) / len(total_auc)
        avg_ap = sum(total_ap) / len(total_ap)
        return avg_auc, avg_ap
    
    def evaluate_test(self) -> Tuple[float, float]:
        """Evaluate the model on the test set."""
        if not self.test_set:
            raise ValueError("Test set is empty. Please prepare splits before evaluation.")
        return self.evaluate_set(self.test_set)

    def train(self, data):
        """Train the GNN Autoencoder with the provided graph data."""        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        pos_edge_index = getattr(data, 'pos_edge_label_index', None)
        if pos_edge_index is None:
            pos_edge_index = edge_index
        pos_edge_index = pos_edge_index.to(self.device)

        self.optimizer.zero_grad()
        z = self.model.encode(x, edge_index)
        loss, pos_loss, neg_loss, neg_edge_index = self.compute_loss_per_edge(z, pos_edge_index, data.num_nodes)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss.item())
        
        # Update the encoder in the XAI runner if applicable
        if self.xai_runner and hasattr(self.xai_runner, 'set_encoder'):
            self.xai_runner.set_encoder(self.model.encoder)
            
        return loss.item()
    
    def predict(self, data) -> torch.Tensor:
        """Make predictions using the trained GNN Autoencoder."""
        logging.info(f"Starting Prediction...")
        self.model.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            z = self.model.encode(x, edge_index)
            edge_scores = self.model.decoder(z, edge_index)
            #logging.info(f"Edge scores: {edge_scores}")
            a = torch.sigmoid(edge_scores).cpu()
            #logging.info(f"Processed edge scores: {a}")
            return a
    
    def detect_anomalies(self, data, threshold: float=0.65) -> List[Tuple[int, int]]:
        """Detect anomalies in the graph based on edge reconstruction scores."""
        scores = self.predict(data)
        anomaly_indices = (scores < self.edge_threshold).nonzero(as_tuple=False).view(-1)
        snapshot_anomaly_ratio = float(anomaly_indices.numel() / scores.numel())
        modelrunner_log.info(f"Snapshot anomaly ratio: {snapshot_anomaly_ratio}, Snapshot threshold: {self.snapshot_threshold}")
        if snapshot_anomaly_ratio < self.snapshot_threshold:
            return []
        anomalies = []        

        for idx in anomaly_indices.tolist():
            u = data.edge_index[0, idx].item()
            v = data.edge_index[1, idx].item()
            anomaly_score = scores[idx].item()
            modelrunner_log.info(f"Detected anomaly on edge ({u}, {v}) with score {anomaly_score}")
            anomalies.append({
                'src_tensorid': u,
                'dst_tensorid': v,
                'anomaly_score': anomaly_score
            })
        return anomalies

    def save_anomalies_csv(self, system_id, model_name, anomalies: List[Tuple[int, int]], path: str):
        """Save detected anomalies to a CSV file."""
        df = pd.DataFrame(anomalies)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(path, f"{system_id}_{model_name}_anomalies_{ts}.csv"), index=False)

    def explain(self, data, topk: int | None = None, z:torch.Tensor | None =None):
        """Explain the model's predictions using XAI techniques."""
        if not self.xai_runner:
            raise ValueError("XAI runner is not set.")
        if not hasattr(self.xai_runner, 'explain'):
            raise ValueError("XAI runner does not have an 'explain' method.")
        
        self.model.eval()
        if z is None:
            with torch.no_grad():
                x = data.x.to(self.device)
                edge_index = data.edge_index.to(self.device)
                z = self.model.encode(x, edge_index)
        
        pos_edge_index = getattr(data, 'pos_edge_label_index', None)
        if pos_edge_index is None:
            pos_edge_index = data.edge_index

        loss, pos_loss, neg_loss, neg_edge_index = self.compute_loss_per_edge(z, pos_edge_index.to(self.device), data.num_nodes)
        
        keep_indices = torch.arange(pos_loss.numel(), device=pos_loss.device)
        if self.config.get("xai_loss_min", None) is not None:
            keep_indices = (pos_loss >= self.config["xai_loss_min"]).nonzero(as_tuple=True)[0]
            if keep_indices.numel() == 0:
                return {
                    'edge_anomalies': torch.tensor([], dtype=torch.long),
                    'node_anomaly_scores': torch.zeros(data.num_nodes, dtype=torch.float)
                }
        k = topk if topk is not None else self.config.get("xai_topk", 20)
        k = max(0, min(k, keep_indices.numel()))

        if k == 0:
            node_scores = torch.zeros(data.num_nodes, dtype=torch.float, device=pos_loss.device)
            src_all, dst_all = pos_edge_index
            node_scores.scatter_add_(0, src_all, pos_loss)
            node_scores.scatter_add_(0, dst_all, pos_loss)
            return {
                'edge_anomalies': torch.tensor([], dtype=torch.long),
                'node_anomaly_scores': node_scores.detach().cpu()
            }
        
        top_local = torch.topk(pos_loss[keep_indices], k=k).indices
        top_indices = keep_indices[top_local]

        edge_anomalies = []
        for idx in top_indices.tolist():
            u = pos_edge_index[0, idx].item()
            v = pos_edge_index[1, idx].item()
            modelrunner_log.info(f"Explaining edge ({u}, {v}) with anomaly score {pos_loss[idx].item()}")
            modelrunner_log.info(data.x)
            modelrunner_log.info(data.edge_index)
            explanation = self.xai_runner.explain(data.x, data.edge_index, (u, v))
            edge_anomalies.append({
                'edge_index': (u, v),
                'explanation': explanation,                
                'anomaly_score': pos_loss[idx].item()
            })

        node_scores = torch.zeros(data.num_nodes, dtype=torch.float, device=pos_loss.device)
        src_all, dst_all = pos_edge_index
        node_scores.scatter_add_(0, src_all, pos_loss)
        node_scores.scatter_add_(0, dst_all, pos_loss)

        results = {
            'edge_anomalies': edge_anomalies,
            'node_anomaly_scores': node_scores.detach().cpu()
        }
        self.save_explain_csv("system", "gnn_ae", edge_anomalies, "./logs")
        modelrunner_log.info(f"explaination results: {results}")

        return results
    def save_explain_csv(self, system_id, model_name, explanations: List[Dict[str, Any]], path: str):
        """Save explanations to a CSV file."""
        
        rows = []
        for exp in explanations:
            edge = exp.get('edge_index', (-1, -1))
            anomaly_score = exp.get('anomaly_score', 0.0)
            explanation = exp.get('explanation', {})
            row = {
                'source_node': edge[0],
                'target_node': edge[1],
                'anomaly_score': anomaly_score,
                'explanation': str(explanation)
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(path, f"{system_id}_{model_name}_explanations_{ts}.csv"), index=False)

    def save_model(self, path: str=None):
        """Save the trained model to the specified path."""
        if path is None:
            path = os.path.join(self.config.get("export_path"), f"{self.config.get('system_id')}_{self.config.get('model_type')}_{self.config.get('bucket_duration')}s.pt")
        logging.info(f"Saving model to: {path}")
        model_values = {
            "edge_threshold": self.edge_threshold,
            "snapshot_threshold": self.snapshot_threshold,
            "state_dict": self.model.state_dict(),
            "config": self.config,           
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "device": self.device

        }

        torch.save(model_values, path)
    
    @staticmethod
    def load_model(config: dict, path: str=None) -> 'GNNAEModelRunner':
        """Load a trained model from the specified path."""
        if path is None:
            path = f'{config.get("export_path")}/{config.get("system_id")}_{config.get("model_type")}_{config.get("bucket_duration")}s.pt'
        model_values = torch.load(path)
        model_runner = GNNAEModelRunner(
            xai_runner=None, # TODO: Load or pass appropriate XAI runner
            input_dim=model_values.get("input_dim"),
            hidden_dim=model_values.get("hidden_dim"),
            output_dim=model_values.get("output_dim"),
            config=model_values.get("config"),
            device=model_values.get("device"),
            edge_threshold=model_values.get("edge_threshold"),
            snapshot_threshold=model_values.get("snapshot_threshold")
        )
        model_runner.model.load_state_dict(model_values.get("state_dict"))
        model_runner.model.to(model_runner.device)        
        return model_runner
    
    def evaluate(self, data):
        """Evaluate the model's performance on the provided data."""
        self.model.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            z = self.model.encode(x, edge_index)

            pos = data.edge_index.to(self.device)
            neg = self.generate_neg_edge_index(pos, data.num_nodes)
            auc, ap = self.model.test(z, pos, neg)

            loss, pos_loss, neg_loss, neg_edge_index = self.compute_loss_per_edge(z, pos, data.num_nodes)
            return {
                'total_loss': loss.item(),
                'pos_loss': pos_loss.mean().item(),
                'neg_loss': neg_loss.mean().item(),
                'AUC': auc,
                'AP': ap
            }

    def fit_evaluate(self, dataset, epochs, val_fraction=0.1):
        """Fit the model and evaluate on a validation set."""
        transform = RandomLinkSplit(num_val=val_fraction, num_test=0, is_undirected=True, add_negative_train_samples=False)
        train_data, val_data, _ = transform(dataset)[0]

        best_val_auc = 0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            loss = self.train(train_data)
            val_metrics = self.evaluate(val_data)
            val_auc = val_metrics['AUC']

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict()

            modelrunner_log.info(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {
            'best_val_AUC': best_val_auc
        }

    @staticmethod
    def save_test_csv(system_id, model_name, test_results: List[Dict[str, Any]], path: str):
        """Save test results to a CSV file."""
        df = pd.DataFrame(test_results)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(path, f"{system_id}_{model_name}_test_results_{ts}.csv"), index=False)
    
    @staticmethod
    def save_history_csv(system_id, model_name, history: List[Dict[str, Any]], path: str):
        """Save training history to a CSV file."""
        df = pd.DataFrame(history)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(path, f"{system_id}_{model_name}_history_{ts}.csv"), index=False)