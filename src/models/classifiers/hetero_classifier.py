# Captum wrapper should reconstruct the graph

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from repositories.graphs.pyg_builder import y_labels

from xai.captum_explainer import CaptumExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GNNHeteroClassifierModel(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        metadata=None
    ):
        super().__init__()

        if metadata is None:
            raise ValueError("Metadata required.")

        self.metadata = metadata
        self.config = config
        self.bin_thres = config.get(
            "binary_threshold",
            0.35
        )

        hd = config.get(
            "hidden_dim",
            64
        )

        self.encoder = GNNHeteroEncoderModel(
            config,
            metadata
        )

        self.out = nn.Linear(
            hd,
            len(y_labels)
        )

        self.criterion = None
        
        self.explainer = CaptumExplainer(self)

        self.to(DEVICE)

    
    def forward(
        self,
        x_or_data: HeteroData | Dict[str, torch.Tensor],
        edge_index_dict: Optional[dict] = None
    ) -> torch.Tensor:

        if isinstance(x_or_data, HeteroData):
            data = x_or_data
        else:
            data = HeteroData()            # exists solely for Captum, model should accept forward

            for ntype in x_or_data.keys():
                data[ntype].x = x_or_data[ntype]

            data.edge_index_dict = edge_index_dict

        h = self.encoder(data)
        return self.out(h)

    @torch.no_grad()
    def predict(self, data):
        self.eval()

        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)

        logits = self(batch)

        pred = logits.argmax(dim=1)

        return pred.cpu().tolist()

    def save_model(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "metadata": self.metadata,
            },
            path
        )

    def load_model(self, path):
        checkpoint = torch.load(
            path,
            map_location=DEVICE
        )
    
        self.load_state_dict(
            checkpoint["state_dict"]
        )

    def explain(self, data):
        return self.explainer.explain(data)

    def predict_proba(            # helper for predicting probabilities in XAI reports
        self,
        data: HeteroData
    ):
        self.eval()
    
        batch = Batch.from_data_list(
            [data]
        ).to(DEVICE)
    
        logits = self(batch)
    
        probs = torch.softmax(
            logits,
            dim=1
        )
    
        return probs.cpu()

    @property                # target = model.num_classes, instead of target = len(y_labels)
    def num_classes(self):
        return len(y_labels)
