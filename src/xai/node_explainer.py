# src/xai/node_explainer.py

from typing import List, Dict
import torch
import numpy as np


class NodeExplainer:

    @staticmethod
    def compute_node_importances(
        attr_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Computes a single importance score per node.

        Input:
            [num_nodes, num_features]

        Output:
            [num_nodes]
        """
        return (
            attr_tensor
            .abs()
            .sum(dim=1)
            .cpu()
            .numpy()
        )

    @staticmethod
    def rank_nodes(
        data,
        ntype: str,
        attr_tensor: torch.Tensor,
        snapshot_id=None,
        true_label=None,
        pred_label=None,
        threshold: float = 1e-4,
    ) -> List[Dict]:
        """
        Creates ranked node importance records.
        """

        node_imp = NodeExplainer.compute_node_importances(
            attr_tensor
        )

        num_nodes = len(node_imp)

        #
        # Recover original IDs
        #
        orig_ids = []

        if hasattr(data[ntype], "original_id"):
            raw = data[ntype].original_id
            orig_ids = (
                raw.tolist()
                if torch.is_tensor(raw)
                else raw
            )

        elif hasattr(data[ntype], "n_id"):
            orig_ids = data[ntype].n_id.tolist()

        if len(orig_ids) != num_nodes:
            orig_ids = [
                f"{ntype}_{i}"
                for i in range(num_nodes)
            ]

        #
        # Build records
        #
        rankings = []

        for i, score in enumerate(node_imp):
            if score <= threshold:
                continue

            rankings.append(
                {
                    "snapshot_id": snapshot_id,
                    "node_type": ntype,
                    "pyg_index": i,
                    "original_id": str(orig_ids[i]),
                    "importance_score": float(score),
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
            )

        rankings.sort(
            key=lambda x: x["importance_score"],
            reverse=True
        )

        return rankings
