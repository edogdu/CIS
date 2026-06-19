class GNNHeteroBinEncoderModel(nn.Module):
    """GNN module to produce node embeddings for heterogeneous graphs."""
    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroBinEncoderModel, self).__init__()
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

class GNNHeteroAnomalyDetectionModel(nn.Module):
    """GNN model for anomaly detection.  It is supervised model,
    which classifies each graph as normal, MITM, DoS, scan, physical fault, anomaly
    This will allow for heterogeneous graphs with different node types.
    """

    def __init__(self, config: Dict[str, Any], metadata=None):
        super(GNNHeteroAnomalyDetectionModel, self).__init__()
        if metadata is None:
            raise ValueError("Metadata must be provided for heterogeneous graphs.")
        self.metadata = metadata
        self.config = config
        self.criterion = None
        self.scalers = {
            
        }
        self.bin_threshold = config.get("bin_threshold", 0.5)
        
        hd = config.get("hidden_dim", 32)        

        self.encoder = GNNHeteroBinEncoderModel(config, metadata)
        
        self.out = nn.Linear(hd, 1)   # -> [B,1] then squeeze to [B]

        self.to(DEVICE)


    def forward(self, data: HeteroData) -> torch.Tensor:
        h = self.encoder(data)   # [B, hd]
        logits = self.out(h).squeeze(dim=-1)   # [B]
        return logits
