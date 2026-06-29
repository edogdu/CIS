# src/xai/feature_ranker.py

from typing import List, Dict
import numpy as np

class FeatureRanker:
    """
    Utility for ranking feature importance scores produced
    by Captum, SHAP, GNNExplainer, etc.
    """

    @staticmethod
    def rank(attr_tensor, feature_names):
        """
        Convert an attribution tensor into a feature->importance mapping.
        attr_tensor: [num_nodes, num_features]
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

        return {
            feature: float(score)
            for feature, score in zip(feature_names, feat_imp)
        }

    @staticmethod
    def top_k(feature_scores, k=10):
        return sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
