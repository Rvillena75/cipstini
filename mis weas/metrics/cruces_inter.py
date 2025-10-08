import math
from typing import List, Tuple

try:  # pragma: no cover
    from line_profiler import profile as line_profile
except ImportError:  # pragma: no cover
    def line_profile(func):
        return func


def _segments_from_route(route: List[int]) -> List[Tuple[int, int]]:
    if not route:
        return []
    path = [0] + route + [0]
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _ccw(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _segments_intersect_points(p1, p2, p3, p4) -> bool:
    (x1, y1), (x2, y2) = p1, p2
    (x3, y3), (x4, y4) = p3, p4
    return (_ccw(x1, y1, x3, y3, x4, y4) != _ccw(x2, y2, x3, y3, x4, y4)) and (
        _ccw(x1, y1, x2, y2, x3, y3) != _ccw(x1, y1, x2, y2, x4, y4)
    )


def _segments_intersect_indices(s1: Tuple[int, int], s2: Tuple[int, int], nodes: List[Tuple[float, float]]) -> bool:
    a, b = s1
    c, d = s2
    if len({a, b, c, d}) < 4:
        return False
    ax, ay = nodes[a]
    bx, by = nodes[b]
    cx, cy = nodes[c]
    dx, dy = nodes[d]
    return (_ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy)) and (
        _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)
    )


def count_inter_route_crossings(route1: List[int], route2: List[int], nodes: List[Tuple[float, float]]) -> int:
    if not route1 or not route2:
        return 0

    segs1 = _segments_from_route(route1)
    segs2 = _segments_from_route(route2)

    count = 0
    for s1 in segs1:
        for s2 in segs2:
            if _segments_intersect_indices(s1, s2, nodes):
                count += 1
    return count


@line_profile
def count_total_inter_route_crossings(routes: List[List[int]], nodes: List[Tuple[float, float]]) -> float:
    if len(routes) < 2:
        return 0.0

    total_crossings = 0
    route_pairs = 0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            total_crossings += count_inter_route_crossings(routes[i], routes[j], nodes)
            route_pairs += 1

    return total_crossings / route_pairs if route_pairs > 0 else 0.0


def count_between_routes_crossings(routes: List[List[int]], nodes: List[Tuple[float, float]]) -> int:
    segs = [_segments_from_route(r) for r in routes if r]
    cnt = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            for s1 in segs[i]:
                for s2 in segs[j]:
                    if _segments_intersect_indices(s1, s2, nodes):
                        cnt += 1
    return cnt


def _seg_angle_penalty(p1, p2, p3, p4) -> float:
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])
    n1 = math.hypot(*v1) + 1e-9
    n2 = math.hypot(*v2) + 1e-9
    cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return 1.0 - abs(cosang)


def _route_segments(route: List[int]) -> List[Tuple[int, int]]:
    path = [0] + route + [0]
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def count_route_cuts(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    distM,
    depot_id: int = 0,
    depot_relief_m: float = 2000.0,
    angle_weight: bool = True,
):
    segs = [_route_segments(r) for r in routes]

    def _near_depot(u: int, v: int) -> bool:
        return (distM[depot_id, u] < depot_relief_m) or (distM[depot_id, v] < depot_relief_m)

    cuts = 0.0
    denom = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            Si = segs[i]
            Sj = segs[j]
            denom += len(Si) * len(Sj)
            for (a, b) in Si:
                if _near_depot(a, b):
                    continue
                p1 = (nodes[a][1], nodes[a][0])
                p2 = (nodes[b][1], nodes[b][0])
                for (c, d) in Sj:
                    if _near_depot(c, d):
                        continue
                    p3 = (nodes[c][1], nodes[c][0])
                    p4 = (nodes[d][1], nodes[d][0])
                    if _segments_intersect_points(p1, p2, p3, p4):
                        cuts += _seg_angle_penalty(p1, p2, p3, p4) if angle_weight else 1.0
    return cuts, max(1, denom)


def route_cuts_norm(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    distM,
    depot_id: int = 0,
    depot_relief_m: float = 2000.0,
    angle_weight: bool = True,
) -> float:
    cuts, denom = count_route_cuts(routes, nodes, distM, depot_id, depot_relief_m, angle_weight)
    return cuts / float(denom)


__all__ = [
    "count_inter_route_crossings",
    "count_total_inter_route_crossings",
    "count_between_routes_crossings",
    "count_route_cuts",
    "route_cuts_norm",
]
