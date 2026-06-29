# src/xai/anomaly_report.py

import json
from typing import Dict, List


def build_anomaly_report(
    data,
    explanation: Dict,
    node_rankings_by_type: Dict[str, List[Dict]],
    feature_rankings_by_type: Dict[str, List[tuple]],
    top_k_nodes: int = 10,
    top_k_features: int = 10,
) -> Dict:
    """
    Build a full anomaly explanation report from:
      - data: HeteroData snapshot
      - explanation: model.explain(batch) output
      - node_rankings_by_type: {node_type: [rankings]}
      - feature_rankings_by_type: {node_type: [(feature, score)]}
    """

    target_class = explanation.get("target_class")
    confidence = explanation.get("confidence")
    logits = explanation.get("logits")

    report = {
        "summary": {
            "target_class": target_class,
            "confidence": float(confidence) if confidence is not None else None,
            "logits": logits.detach().cpu().tolist() if hasattr(logits, "detach") else logits,
        },
        "node_types": {},
    }

    for ntype, rankings in node_rankings_by_type.items():
        # top-k nodes
        top_nodes = rankings[:top_k_nodes]

        # attach metadata for each node
        node_entries = []
        for r in top_nodes:
            idx = r["pyg_index"]
            original_id = r["original_id"]
            score = r["importance_score"]

            feature_names = getattr(data[ntype], "feature_names", [])
            feature_values = data[ntype].x[idx].tolist()
            context = data.context_lookup.get(ntype, {}).get(original_id, {})

            node_entries.append({
                "snapshot_id": r["snapshot_id"],
                "node_type": ntype,
                "pyg_index": idx,
                "original_id": original_id,
                "importance_score": score,
                "true_label": r.get("true_label"),
                "pred_label": r.get("pred_label"),
                "features": dict(zip(feature_names, feature_values)),
                "context": context,
            })

        # top-k features
        feat_scores = feature_rankings_by_type.get(ntype, [])
        top_features = feat_scores[:top_k_features]

        report["node_types"][ntype] = {
            "top_nodes": node_entries,
            "top_features": [
                {"feature": f, "importance": float(s)}
                for f, s in top_features
            ],
        }

    return report


def save_anomaly_report(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
