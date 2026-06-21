from models.encoders.hetero_encoder import (
    GNNHeteroEncoderModel
)

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
        hd = config.get("hidden_dim", 64)
        self.criterion = None
        self.scalers = {

        }
        self.encoder = GNNHeteroEncoderModel(config, metadata)

        self.out = nn.Linear(hd, len(y_labels))   # -> [B,5] (classes: 1..5 shifted to 0..4 in loss)

        self.to(DEVICE)


    def forward(self, x_or_data, edge_index_dict=None) -> torch.Tensor:

        if isinstance(x_or_data, HeteroData):
            data = x_or_data
        else:
            # Assume x_or_data is a dict of node feature tensors
            data = HeteroData()
            for ntype in x_or_data.keys():
                data[ntype].x = x_or_data[ntype]
            data.edge_index_dict = edge_index_dict            

        h = self.encoder(data)
        logits = self.out(h)                   # [B, 5]
        return logits

    @torch.no_grad()
    def predict(self, data: HeteroData) -> List[int]:
        """Predict anomaly classes for the given data HeteroData."""
        self.eval()
        data = data.to(DEVICE)
        batch = Batch.from_data_list([data]).to(DEVICE)
        anom_logits = self(batch)      # [B,5]
        pred = anom_logits.argmax(dim=1)
        return pred.detach().cpu().tolist()
