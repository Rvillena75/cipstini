from typing import List, Set

import numpy as np


def _cluster_id(cluster_map: np.ndarray, client: int) -> int:
    if cluster_map is None or client >= len(cluster_map):
        return -1
    return int(cluster_map[client])


def _zones_for_route(route: List[int], cluster_map: np.ndarray) -> Set[int]:
    zones: Set[int] = set()
    for client in route:
        cid = _cluster_id(cluster_map, client)
        if cid >= 0:
            zones.add(cid)
    return zones


def compute_solapamiento_zonal(routes: List[List[int]], cluster_map: np.ndarray) -> float:
    """
    Penaliza solapamiento de zonas virtuales entre rutas.

    Retorna el número total de rutas que comparten clusters con al menos otra
    ruta. Se normaliza externamente si se requiere.
    """
    if cluster_map is None or len(routes) < 2:
        return 0.0

    cluster_to_routes = {}
    for idx, route in enumerate(routes):
        if not route:
            continue
        for cid in _zones_for_route(route, cluster_map):
            cluster_to_routes.setdefault(cid, set()).add(idx)

    penalty = 0.0
    for rutas in cluster_to_routes.values():
        if len(rutas) > 1:
            penalty += len(rutas) - 1
    return penalty


__all__ = [
    "compute_solapamiento_zonal",
    "_zones_for_route",
]
