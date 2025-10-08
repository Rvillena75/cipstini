import math
from typing import List, Tuple


def _minimal_angle_intervals(angles: List[float]) -> List[Tuple[float, float]]:
    if not angles:
        return []
    normalized = sorted(a % (2 * math.pi) for a in angles)
    total = len(normalized)
    extended = normalized + [a + 2 * math.pi for a in normalized]

    best_span = 2 * math.pi
    best_start = normalized[0]
    j = 0
    for i in range(total):
        while j < i + total and extended[j] - extended[i] <= 2 * math.pi:
            j += 1
        span = extended[j - 1] - extended[i]
        if span < best_span - 1e-9:
            best_span = span
            best_start = extended[i]

    start = best_start % (2 * math.pi)
    end = (best_start + best_span) % (2 * math.pi)
    if best_span >= 2 * math.pi - 1e-6:
        return [(0.0, 2 * math.pi)]
    if start <= end:
        return [(start, end)]
    return [(0.0, end), (start, 2 * math.pi)]


def compute_sector_overlap(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    depot_id: int = 0,
) -> float:
    if len(routes) < 2:
        return 0.0

    depot_lat, depot_lon = nodes[depot_id]
    intervals_per_route: List[List[Tuple[float, float]]] = []

    for route in routes:
        angs: List[float] = []
        for client in route:
            lat, lon = nodes[client]
            dy = lat - depot_lat
            dx = lon - depot_lon
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            angs.append(math.atan2(dy, dx))
        intervals_per_route.append(_minimal_angle_intervals(angs))

    overlap = 0.0
    for i in range(len(intervals_per_route)):
        a_intervals = intervals_per_route[i]
        if not a_intervals:
            continue
        for j in range(i + 1, len(intervals_per_route)):
            b_intervals = intervals_per_route[j]
            if not b_intervals:
                continue
            for a_start, a_end in a_intervals:
                for b_start, b_end in b_intervals:
                    start = max(a_start, b_start)
                    end = min(a_end, b_end)
                    if end > start:
                        overlap += end - start

    return overlap / math.pi


__all__ = ["compute_sector_overlap"]
