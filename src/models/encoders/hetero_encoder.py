from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HGTConv,
    Linear,
    global_max_pool,
)


class GNNHeteroEncoderModel(nn.Module):
    """
    Produces graph embeddings from heterogeneous graphs.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        metadata
    ):
        super().__init__()

        if metadata is None:
            raise ValueError(
                "Metadata must be provided for heterogeneous graphs."
            )

        self.metadata = metadata
        self.config = config

        hd = config.get("hidden_dim", 32)
        heads = config.get("num_heads", 4)

        self.pooled_types: List[str] = config.get(
            "pooled_types",
            [
                "Pumps",
                "FlowSensors",
                "Tanks",
                "Valves",
                "Connections",
                "Endpoints",
            ],
        )

        self.lin_dict = nn.ModuleDict()

        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hd)

        self.num_layers = config.get("num_layers", 3)

        self.convs = nn.ModuleList(
            [
                HGTConv(
                    hd,
                    hd,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.dropout = float(
            config.get("dropout", 0.5)
        )

        pooled_width = hd * max(
            1,
            len(self.pooled_types)
        )

        self.lin1 = Linear(
            pooled_width,
            hd
        )

    def forward(
        self,
        data: HeteroData
    ) -> torch.Tensor:

        x_dict = {
            ntype: self.lin_dict[ntype](x).relu()
            for ntype, x in data.x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(
                x_dict,
                data.edge_index_dict
            )
            x_dict = {
                k: F.relu(v)
                for k, v in x_dict.items()
            }

        num_graphs = data.num_graphs
        device = next(iter(x_dict.values())).device

        pools = []

        for ntype in self.pooled_types:
            if (
                ntype in x_dict
                and hasattr(data[ntype], "batch")
            ):
                pools.append(
                    global_max_pool(
                        x_dict[ntype],
                        data[ntype].batch,
                        size=num_graphs,
                    )
                )
            else:
                pools.append(
                    torch.zeros(
                        (
                            num_graphs,
                            self.config.get(
                                "hidden_dim",
                                32,
                            ),
                        ),
                        device=device,
                    )
                )

        h = (
            torch.cat(pools, dim=1)
            if len(pools) > 1
            else pools[0]
        )

        h = F.relu(
            self.lin1(
                F.dropout(
                    h,
                    p=self.dropout,
                    training=self.training,
                )
            )
        )

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        return h
