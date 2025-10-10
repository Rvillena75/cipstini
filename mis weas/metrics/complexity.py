import math
from typing import List, Optional, Tuple

from .dispersion import _to_xy

Coords = Tuple[float, float]
Route = List[int]


def _segment_heading(p1: Coords, p2: Coords) -> float:
    """Heading angle (rad) of the segment p1→p2 in cartesian coordinates."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def _angle_diff(a: float, b: float) -> float:
    """Absolute smallest rotation between two angles in [-pi, pi]."""
    diff = b - a
    while diff <= -math.pi:
        diff += 2.0 * math.pi
    while diff > math.pi:
        diff -= 2.0 * math.pi
    return abs(diff)


def route_complexity(route: Route, nodes_xy: List[Coords]) -> float:
    """Promedio absoluto de cambio angular entre segmentos consecutivos."""
    if len(route) < 3:
        return 0.0

    headings = []
    for a, b in zip(route, route[1:]):
        headings.append(_segment_heading(nodes_xy[a], nodes_xy[b]))

    if len(headings) < 2:
        return 0.0

    diffs = [_angle_diff(headings[i], headings[i + 1]) for i in range(len(headings) - 1)]
    if not diffs:
        return 0.0

    return float(sum(diffs) / len(diffs))


def compute_complexity(
    routes: List[Route],
    nodes: List[Tuple[float, float]],
    nodes_xy: Optional[List[Coords]] = None,
) -> float:
    """Promedia la complejidad de cada ruta con al menos 3 clientes."""
    if not routes:
        return 0.0

    if nodes_xy is None:
        lat0, lon0 = nodes[0]
        nodes_xy = [_to_xy(lat, lon, lat0, lon0) for lat, lon in nodes]

    scores = [
        route_complexity(route, nodes_xy)
        for route in routes
        if len(route) >= 3
    ]
    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))


__all__ = ["compute_complexity", "route_complexity"]
