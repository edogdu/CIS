# Captum wrapper should reconstruct the graph

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from repositories.graphs.pyg_builder import y_labels

from xai.captum_explainer import CaptumExplainer        # let this file do all the graph reconstruction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GNNHeteroClassifierModel(nn.Module):
    def __init__(
        self,
        config,
        metadata
    ):
        super().__init__()

        self.metadata = metadata
        self.config = config

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
        data: HeteroData
    ) -> torch.Tensor:
        h = self.encoder(data)
        return self.out(h)

    @torch.no_grad()
    def predict(
        self,
        data: HeteroData
    ):
        self.eval()

        batch = Batch.from_data_list(
            [data]
        ).to(DEVICE)

        logits = self(batch)

        return logits.argmax(
            dim=1
        ).cpu().tolist()

    @torch.no_grad()
    def predict_proba(
        self,
        data: HeteroData
    ):
        self.eval()

        batch = Batch.from_data_list(
            [data]
        ).to(DEVICE)

        logits = self(batch)

        return torch.softmax(
            logits,
            dim=1
        ).cpu()

    def explain(
        self,
        data
    ):
        return self.explainer.explain(data)

    def save_model(
        self,
        path
    ):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "metadata": self.metadata,
            },
            path
        )

    def load_model(
        self,
        path
    ):
        checkpoint = torch.load(
            path,
            map_location=DEVICE
        )

        self.load_state_dict(
            checkpoint["state_dict"]
        )
