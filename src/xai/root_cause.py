# src/xai/root_cause.py

from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt

NODE_COLORS = {
    "Measurement": "#1f77b4",
    "FlowSensors": "#2ca02c",
    "Tanks": "#ff7f0e",
    "Valves": "#d62728",
    "Connections": "#7f7f7f",
    "Asset": "#9467bd",
    "Endpoints": "#8c564b",
    "Pumps": "#17becf",
}

def visualize_causal_paths(
    causal_paths,
    title="Causal Paths Diagram",
    label_font_size=14,
    title_font_size=20,
    legend_font_size=14,
    arrow_size=25,
):
    G = nx.DiGraph()

    # Build graph
    for chain in causal_paths:
        path = chain["path"]

        for node in path:
            label = f"{node['node_type']}:{node['original_id']}"
            G.add_node(
                label,
                node_type=node["node_type"],
                importance=node["importance"] or 0.00001
            )

        for a, b in zip(path[:-1], path[1:]):
            src = f"{a['node_type']}:{a['original_id']}"
            dst = f"{b['node_type']}:{b['original_id']}"
            G.add_edge(src, dst)

    # Build colors and sizes
    node_colors = []
    node_sizes = []

    for node, attrs in G.nodes(data=True):
        ntype = attrs["node_type"]
        imp = attrs["importance"]

        node_colors.append(NODE_COLORS.get(ntype, "#333333"))
        node_sizes.append(max(400, imp * 6000))

    # Layout
    pos = nx.spring_layout(G, k=1.0, seed=42)

    plt.figure(figsize=(18, 14))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        linewidths=1.5,
        edgecolors="black"
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=arrow_size,
        width=2,
        edge_color="#555555"
    )

    # Draw labels (bigger font)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=label_font_size,
        font_color="white",
        font_weight="bold",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.6, pad=2)
    )

    # Title
    plt.title(title, fontsize=title_font_size)

    # Legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=ntype,
            markerfacecolor=color,
            markersize=14
        )
        for ntype, color in NODE_COLORS.items()
    ]
    plt.legend(handles=handles, loc="upper left", fontsize=legend_font_size)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

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
