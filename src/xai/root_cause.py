# src/xai/root_cause.py

from typing import Dict, List


def build_root_cause_chain(
    node_rankings_by_type: Dict[str, List[Dict]],
    feature_rankings_by_type: Dict[str, List[tuple]],
    explanation: Dict,
) -> Dict:
    """
    Build a root-cause chain from ranked nodes + ranked features.
    """

    target_class = explanation.get("target_class")
    confidence = explanation.get("confidence")

    chain = {
        "target_class": target_class,
        "confidence": float(confidence),
        "steps": []
    }

    # 1. Measurement anomalies → strongest root cause
    meas = node_rankings_by_type.get("Measurement", [])
    if meas:
        top_meas = meas[0]
        chain["steps"].append({
            "stage": "Sensor anomaly",
            "node_type": "Measurement",
            "original_id": top_meas["original_id"],
            "importance": top_meas["importance_score"],
            "top_features": feature_rankings_by_type.get("Measurement", [])
        })

    # 2. FlowSensor anomalies → propagation
    flows = node_rankings_by_type.get("FlowSensors", [])
    if flows:
        top_flow = flows[0]
        chain["steps"].append({
            "stage": "Flow instability",
            "node_type": "FlowSensors",
            "original_id": top_flow["original_id"],
            "importance": top_flow["importance_score"],
            "top_features": feature_rankings_by_type.get("FlowSensors", [])
        })

    # 3. Tank anomalies → system-level physical deviation
    tanks = node_rankings_by_type.get("Tanks", [])
    if tanks:
        top_tank = tanks[0]
        chain["steps"].append({
            "stage": "Tank pressure/level deviation",
            "node_type": "Tanks",
            "original_id": top_tank["original_id"],
            "importance": top_tank["importance_score"],
            "top_features": feature_rankings_by_type.get("Tanks", [])
        })

    # 4. Valve anomalies → reaction to upstream instability
    valves = node_rankings_by_type.get("Valves", [])
    if valves:
        top_valve = valves[0]
        chain["steps"].append({
            "stage": "Valve reaction",
            "node_type": "Valves",
            "original_id": top_valve["original_id"],
            "importance": top_valve["importance_score"],
            "top_features": feature_rankings_by_type.get("Valves", [])
        })

    # 5. Connections → confirm no cyber root cause
    conns = node_rankings_by_type.get("Connections", [])
    if conns:
        top_conn = conns[0]
        chain["steps"].append({
            "stage": "Network activity (non-causal)",
            "node_type": "Connections",
            "original_id": top_conn["original_id"],
            "importance": top_conn["importance_score"],
            "top_features": feature_rankings_by_type.get("Connections", [])
        })

    return chain
