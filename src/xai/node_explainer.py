# src/xai/node_explainer.py

from typing import List, Dict
import torch
import numpy as np

class NodeExplainer:

    @staticmethod
    def compute_node_importances(attr_tensor):
        """
        attr_tensor: [num_nodes, num_features]
        returns: [num_nodes]
        """
        return (
            attr_tensor.abs()
            .sum(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

    @staticmethod
    def rank_nodes(
        data,
        ntype,
        attr_tensor,
        snapshot_id=None,
        true_label=None,
        pred_label=None,
        threshold=0.0,
    ):
        """
        Rank nodes of a given type by importance.
        """

        node_imp = NodeExplainer.compute_node_importances(attr_tensor)
        num_nodes = len(node_imp)

        # original IDs
        orig_ids = data[ntype].original_id

        # build records
        rankings = []
        for i in range(num_nodes):
            score = float(node_imp[i])
            if score <= threshold:
                continue

            oid = orig_ids[i]

            rankings.append({
                "snapshot_id": snapshot_id,
                "node_type": ntype,
                "pyg_index": i,
                "original_id": oid,
                "importance_score": score,
                "true_label": true_label,
                "pred_label": pred_label,
            })

        rankings.sort(
            key=lambda x: x["importance_score"],
            reverse=True
        )

        return rankings

    # metadata inspector
    def inspect_node(data, node_type, idx, score):
        original_id = data[node_type].original_id[idx]
        feature_names = data[node_type].feature_names
        feature_values = data[node_type].x[idx].tolist()
        context = data.context_lookup.get(node_type, {}).get(original_id, {})

        return {
            "node_type": node_type,
            "original_id": original_id,
            "score": float(score),
            "features": dict(zip(feature_names, feature_values)),
            "context": context,
        }
