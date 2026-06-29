# src/xai/graph_paths.py

from typing import Dict, List, Tuple

def build_causal_paths(
    data,
    node_rankings_by_type: Dict[str, List[Dict]],
    max_hops: int = 2,
) -> List[Dict]:
    """
    Follow hetero edges from top-ranked nodes to build causal paths.
    """

    paths = []

    # helper: get neighbors for (ntype, idx)
    def get_neighbors(ntype: str, idx: int) -> List[Tuple[str, int]]:
        neighbors = []
        for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
            if src_type == ntype:
                src, dst = edge_index
                mask = (src == idx)
                for j in dst[mask].tolist():
                    neighbors.append((dst_type, j))
            if dst_type == ntype:
                src, dst = edge_index
                mask = (dst == idx)
                for j in src[mask].tolist():
                    neighbors.append((src_type, j))
        return neighbors

    # start from top Measurement / FlowSensors / Tanks / Valves
    start_types = ["Measurement", "FlowSensors", "Tanks", "Valves"]
    for ntype in start_types:
        rankings = node_rankings_by_type.get(ntype, [])
        if not rankings:
            continue

        root = rankings[0]  # top node
        root_idx = root["pyg_index"]
        root_oid = root["original_id"]

        path = [{
            "node_type": ntype,
            "pyg_index": root_idx,
            "original_id": root_oid,
            "importance": root["importance_score"],
        }]

        frontier = [(ntype, root_idx)]
        visited = {(ntype, root_idx)}

        for _ in range(max_hops):
            new_frontier = []
            for t, i in frontier:
                for nt2, j in get_neighbors(t, i):
                    if (nt2, j) in visited:
                        continue
                    visited.add((nt2, j))
                    new_frontier.append((nt2, j))
                    path.append({
                        "node_type": nt2,
                        "pyg_index": j,
                        "original_id": data[nt2].original_id[j],
                        "importance": None,  # can fill from rankings if desired
                    })
            frontier = new_frontier

        paths.append({"root_type": ntype, "root_id": root_oid, "path": path})

    return paths
