from typing import List


def compute_route_compactness_penalty(route: List[int], distM) -> float:
    if len(route) < 2:
        return 0.0

    total_dist = distM[0, route[0]]
    for a, b in zip(route, route[1:]):
        total_dist += distM[a, b]
    total_dist += distM[route[-1], 0]

    d_max = max(distM[0, i] for i in route)
    base_min = 2.0 * d_max
    if base_min <= 0.0:
        return 0.0

    detour = max(0.0, total_dist - base_min)
    ratio = detour / base_min
    return float(ratio)


__all__ = ["compute_route_compactness_penalty"]
