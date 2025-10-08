from typing import List

import numpy as np


def _route_distance(route: List[int], distM) -> float:
    if not route:
        return 0.0
    total = distM[0, route[0]]
    for a, b in zip(route, route[1:]):
        total += distM[a, b]
    total += distM[route[-1], 0]
    return float(total)


def compute_balance_distance_cv(routes: List[List[int]], distM) -> float:
    if not routes:
        return 0.0
    dists = [_route_distance(r, distM) for r in routes if r]
    if not dists:
        return 0.0
    mean_dist = float(np.mean(dists))
    if mean_dist <= 1e-9:
        return 0.0
    return 100*float(np.std(dists) / mean_dist)


__all__ = ["compute_balance_distance_cv"]
