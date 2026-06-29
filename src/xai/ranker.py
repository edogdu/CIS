# src/xai/ranker.py

from typing import Dict, List
import torch
import numpy as np


# ------------------------------------------------------------
# Feature Ranking
# ------------------------------------------------------------

class FeatureRanker:
    """
    Rank feature importance for a single node type.
    """

    @staticmethod
    def rank(attr_tensor: torch.Tensor, feature_names: List[str]) -> Dict[str, float]:
        """
        attr_tensor: [num_nodes, num_features]
        returns: {feature_name: importance_score}
        """

        feat_imp = (
            attr_tensor.abs()
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        if len(feature_names) != len(feat_imp):
            feature_names = [f"feat_{i}" for i in range(len(feat_imp))]

        return {name: float(score) for name, score in zip(feature_names, feat_imp)}

    @staticmethod
    def top_k(feature_scores: Dict[str, float], k: int = 10):
        return sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ------------------------------------------------------------
# Node Ranking
# ------------------------------------------------------------

class NodeExplainer:
    """
    Rank nodes by importance for a single node type.
    """

    @staticmethod
    def compute_node_importances(attr_tensor: torch.Tensor) -> np.ndarray:
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
        ntype: str,
        attr_tensor: torch.Tensor,
        snapshot_id=None,
        true_label=None,
        pred_label=None,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Rank nodes of a given type by importance.
        """

        node_imp = NodeExplainer.compute_node_importances(attr_tensor)
        orig_ids = data[ntype].original_id

        rankings = []
        for i, score in enumerate(node_imp):
            if score <= threshold:
                continue

            rankings.append({
                "snapshot_id": snapshot_id,
                "node_type": ntype,
                "pyg_index": i,
                "original_id": orig_ids[i],
                "importance_score": float(score),
                "true_label": true_label,
                "pred_label": pred_label,
            })

        rankings.sort(key=lambda x: x["importance_score"], reverse=True)
        return rankings


# ------------------------------------------------------------
# Metadata Inspector
# ------------------------------------------------------------

def inspect_node(data, node_type, idx, score):
    """
    Produce a full metadata-aware explanation for a single node.
    """

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
