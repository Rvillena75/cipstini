import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _cross(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 1:
        return pts

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """Ray casting. True si pt está dentro o en borde del polígono."""
    x, y = pt
    inside = False
    n = len(poly)
    if n == 0:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        den = (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)
        if abs(den) < 1e-12:
            if min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
                return True
        inter = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1)
        if inter:
            inside = not inside
    return inside


def inter_route_area_overlap_score(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    include_depot: bool = False,
    depot_id: int = 0,
) -> float:
    """Devuelve un score ∈ [0, 1] que mide solapamiento promedio entre rutas."""
    route_points: List[List[Tuple[float, float]]] = []
    for r in routes:
        if not r:
            route_points.append([])
            continue
        ids = r[:] if include_depot else [nid for nid in r if nid != depot_id]
        pts = [nodes[nid] for nid in ids]
        route_points.append(pts)

    hulls: List[List[Tuple[float, float]]] = []
    for pts in route_points:
        if len(set(pts)) >= 3:
            hulls.append(_convex_hull(pts))
        else:
            hulls.append([])

    pair_scores = []
    R = len(routes)
    for i in range(R):
        pts_i = route_points[i]
        hull_i = hulls[i]
        for j in range(i + 1, R):
            pts_j = route_points[j]
            hull_j = hulls[j]

            components = []
            if hull_j and len(pts_i) > 0:
                inside_i_in_j = sum(1 for p in pts_i if _point_in_polygon(p, hull_j))
                components.append(inside_i_in_j / max(1, len(pts_i)))
            if hull_i and len(pts_j) > 0:
                inside_j_in_i = sum(1 for p in pts_j if _point_in_polygon(p, hull_i))
                components.append(inside_j_in_i / max(1, len(pts_j)))

            if components:
                pair_scores.append(sum(components) / len(components))

    if not pair_scores:
        return 0.0
    return sum(pair_scores) / len(pair_scores)


def penalizar_solapamiento_geometrico(routes: List[List[int]], nodes: List[Tuple[float, float]]) -> float:
    """
    Penaliza solapamiento geométrico si el centroide de una ruta queda dentro
    de la envolvente convexa de otra ruta mayor.
    """
    if len(routes) < 2:
        return 0.0

    penalizacion = 0.0
    route_info = []
    for r in routes:
        if not r:
            continue
        route_nodes = np.array([nodes[i] for i in r])
        centroid = np.mean(route_nodes, axis=0)
        hull = _convex_hull(route_nodes)
        route_info.append({"centroid": centroid, "hull": hull, "size": len(r)})

    for i in range(len(route_info)):
        for j in range(i + 1, len(route_info)):
            info_i = route_info[i]
            info_j = route_info[j]

            if info_i["size"] < info_j["size"] and _point_in_polygon(tuple(info_i["centroid"]), info_j["hull"]):
                penalizacion += 1.0
            elif info_j["size"] < info_i["size"] and _point_in_polygon(tuple(info_j["centroid"]), info_i["hull"]):
                penalizacion += 1.0

    return float(penalizacion)


def compute_solapamiento_geometrico(routes: List[List[int]], nodes: List[Tuple[float, float]]) -> float:
    return inter_route_area_overlap_score(routes, nodes, include_depot=False, depot_id=0)


__all__ = [
    "compute_solapamiento_geometrico",
    "inter_route_area_overlap_score",
    "penalizar_solapamiento_geometrico",
]
