from typing import List

import numpy as np

from .solapamiento_zonal import _cluster_id


def _count_zone_transitions(route: List[int], cluster_map: np.ndarray) -> float:
    sequence = []
    for client in route:
        cid = _cluster_id(cluster_map, client)
        if cid >= 0:
            sequence.append(cid)

    if len(sequence) <= 1:
        return 0.0

    transitions = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    return transitions / max(1, len(sequence) - 1)


def compute_zone_transitions(routes: List[List[int]], cluster_map: np.ndarray) -> float:
    if cluster_map is None:
        return 0.0
    values = [_count_zone_transitions(r, cluster_map) for r in routes if r]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


__all__ = ["compute_zone_transitions"]
