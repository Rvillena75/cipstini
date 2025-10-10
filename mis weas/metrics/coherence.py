import math
from typing import Dict, List, Optional, Tuple

from .dispersion import _to_xy

Coords = Tuple[float, float]
Route = List[int]


def _ensure_nodes_xy(data: Dict) -> List[Coords]:
    nodes_xy = data.get("nodes_xy")
    if nodes_xy is not None:
        return nodes_xy

    nodes = data["nodes"]
    lat0, lon0 = nodes[0]
    nodes_xy = [_to_xy(lat, lon, lat0, lon0) for lat, lon in nodes]
    data["nodes_xy"] = nodes_xy
    return nodes_xy


def compute_coherence(
    routes: List[Route],
    data: Dict,
    nodes_xy: Optional[List[Coords]] = None,
) -> float:
    """Fracción de clientes más cercanos al centroide de su propia ruta."""
    if not routes:
        return 0.0

    if nodes_xy is None:
        nodes_xy = _ensure_nodes_xy(data)

    centroids: List[Optional[Coords]] = []
    for route in routes:
        if not route:
            centroids.append(None)
            continue
        xs = [nodes_xy[i][0] for i in route]
        ys = [nodes_xy[i][1] for i in route]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        centroids.append((cx, cy))

    total_clients = sum(len(r) for r in routes)
    if total_clients == 0:
        return 0.0

    misplaced = 0
    for idx, route in enumerate(routes):
        own_centroid = centroids[idx]
        if not route or own_centroid is None:
            continue
        ox, oy = own_centroid
        for cust in route:
            cx, cy = nodes_xy[cust]
            own_dist = math.hypot(cx - ox, cy - oy)
            better = False
            for j, other_centroid in enumerate(centroids):
                if j == idx or other_centroid is None:
                    continue
                dx, dy = other_centroid
                if math.hypot(cx - dx, cy - dy) + 1e-9 < own_dist:
                    better = True
                    break
            if better:
                misplaced += 1

    return float(misplaced) / float(total_clients)


__all__ = ["compute_coherence"]
