from typing import List, Tuple


def _segments_from_path(route: List[int]) -> List[Tuple[int, int]]:
    if not route:
        return []
    path = [0] + route + [0]
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _ccw(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _segments_intersect(p1: Tuple[int, int], p2: Tuple[int, int], nodes: List[Tuple[float, float]]) -> bool:
    a, b = p1
    c, d = p2
    if len({a, b, c, d}) < 4:
        return False
    ax, ay = nodes[a]
    bx, by = nodes[b]
    cx, cy = nodes[c]
    dx, dy = nodes[d]

    return (_ccw(ax, ay, cx, cy, dx, dy) != _ccw(bx, by, cx, cy, dx, dy)) and (
        _ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)
    )


def count_self_crossings(route: List[int], nodes: List[Tuple[float, float]]) -> int:
    segs = _segments_from_path(route)
    count = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if _segments_intersect(segs[i], segs[j], nodes):
                count += 1
    return count


__all__ = ["count_self_crossings"]
