from typing import Any, Dict, List
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from models.encoders.hetero_encoder import GNNHeteroEncoderModel
from repositories.graphs.pyg_builder import y_labels

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

        self.to(DEVICE)

    def forward(
        self,
        x_or_data,
        edge_index_dict=None
    ):
        if isinstance(x_or_data, HeteroData):
            data = x_or_data
        else:
            data = HeteroData()

            for ntype in x_or_data.keys():
                data[ntype].x = x_or_data[ntype]

            data.edge_index_dict = edge_index_dict

        h = self.encoder(data)
        return self.out(h)

    @torch.no_grad()
    def predict(self, data):
        self.eval()

        batch = Batch.from_data_list([data]).to(DEVICE)

        logits = self(batch)

        pred = logits.argmax(dim=1)

        return pred.cpu().tolist()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(
            torch.load(
                path,
                map_location=DEVICE
            )
        )
