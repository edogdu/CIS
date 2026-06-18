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
