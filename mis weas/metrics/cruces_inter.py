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
__all__ = [
    "count_inter_route_crossings",
    "count_total_inter_route_crossings",
    "count_between_routes_crossings",
]
