from typing import List, Tuple, Optional

_R_EARTH = 6371000.0


def _to_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    import math

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    x = _R_EARTH * (lon_r - lon0_r) * math.cos((lat_r + lat0_r) * 0.5)
    y = _R_EARTH * (lat_r - lat0_r)
    return (x, y)


def _route_dispersion(route: List[int], nodes_xy: List[Tuple[float, float]]) -> float:
    import math

    if len(route) <= 1:
        return 0.0

    pts = [nodes_xy[i] for i in route]
    xs, ys = zip(*pts)
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

    dists = [math.hypot(px - centroid[0], py - centroid[1]) for px, py in pts]
    max_dist = max(dists)
    if max_dist <= 1e-9:
        return 0.0

    return sum(dists) / len(dists) / max_dist


def compute_dispersion(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    nodes_xy: Optional[List[Tuple[float, float]]] = None,
) -> float:
    """
    Calcula cuán extendida está cada ruta respecto a su centroide.
    Para cada ruta:
      - Se proyectan las coordenadas a un plano local.
      - Se calcula el centroide.
      - Se obtiene la razón mean(distancia al centroide) / max(distancia al centroide),
        lo que entrega un valor en [0, 1]. Luego se promedia entre rutas.
    """
    if not routes:
        return 0.0

    if nodes_xy is None:
        lat0, lon0 = nodes[0]
        nodes_xy = [_to_xy(nodes[i][0], nodes[i][1], lat0, lon0) for i in range(len(nodes))]

    penalties = []

    for route in routes:
        penalties.append(_route_dispersion(route, nodes_xy))

    if not penalties:
        return 0.0

    return float(sum(penalties) / len(penalties))
