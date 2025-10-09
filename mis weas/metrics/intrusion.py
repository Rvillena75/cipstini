import math
from typing import Dict, List, Tuple, Optional  # <- asegúrate de incluir Optional

try:  # pragma: no cover
    from line_profiler import profile as line_profile
except ImportError:  # pragma: no cover
    def line_profile(func):
        return func

_R_EARTH = 6371000.0


def _to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    lat_r = math.radians(lat); lon_r = math.radians(lon)
    lat0_r = math.radians(lat0); lon0_r = math.radians(lon0)
    x = _R_EARTH * (lon_r - lon0_r) * math.cos(0.5 * (lat_r + lat0_r))
    y = _R_EARTH * (lat_r - lat0_r)
    return (x, y)


def _bbox(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _clip_segment_convex_poly(p1, p2, poly) -> float:
    """Longitud del segmento p1→p2 que queda dentro del polígono convexo `poly`."""
    if len(poly) < 3:
        return 0.0

    x1, y1 = p1; x2, y2 = p2
    dx = x2 - x1; dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-9:
        return 0.0

    t_enter, t_exit = 0.0, 1.0
    for i in range(len(poly)):
        ax, ay = poly[i]; bx, by = poly[(i + 1) % len(poly)]
        edge_x = bx - ax; edge_y = by - ay
        # Para polígonos CCW, el interior es el lado izquierdo de cada arista.
        num = edge_x * (y1 - ay) - edge_y * (x1 - ax)
        den = edge_x * dy - edge_y * dx

        if abs(den) < 1e-9:
            if num < 0:
                return 0.0
            continue

        t = -num / den
        if den > 0:
            # entra
            if t > t_exit:
                return 0.0
            if t > t_enter:
                t_enter = t
        else:
            # sale
            if t < t_enter:
                return 0.0
            if t < t_exit:
                t_exit = t

    if t_exit <= t_enter:
        return 0.0

    return (t_exit - t_enter) * seg_len


def _ensure_ccw(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(poly) < 3:
        return poly
    area_twice = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        area_twice += x1 * y2 - x2 * y1
    return poly if area_twice >= 0 else list(reversed(poly))


def _route_segments_xy(route: List[int], nodes_xy: List[Tuple[float, float]]):
    path = [0] + route + [0]
    return [(nodes_xy[path[i]], nodes_xy[path[i + 1]]) for i in range(len(path) - 1)]


def _route_hull_xy(route: List[int], nodes_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    from .solapamiento_geometrico import _convex_hull
    pts_xy = [nodes_xy[i] for i in route]
    if len(set(pts_xy)) < 3:
        return []
    return _ensure_ccw(_convex_hull(pts_xy))


def _intrusion_one_way(hull, hull_bbox, segments):
    intr = 0.0
    for p1, p2 in segments:
        seg_bbox = _bbox([p1, p2])
        if not _bbox_intersect(hull_bbox, seg_bbox):
            continue
        intr += _clip_segment_convex_poly(p1, p2, hull)
    return intr


_GLOBAL_INTR_CACHE_KEY = "_intrusion_cache"


def _ensure_nodes_xy(data: Dict) -> List[Tuple[float, float]]:
    nodes = data["nodes"]
    nodes_xy = data.get("nodes_xy")
    if nodes_xy is None:
        lat0, lon0 = nodes[0]
        nodes_xy = [_to_xy_m(lat, lon, lat0, lon0) for lat, lon in nodes]
        data["nodes_xy"] = nodes_xy
    return nodes_xy


@line_profile
def intrusion_length_between_routes_m(routes: List[List[int]], data: Dict) -> float:
    """
    Fracción media por ruta de la longitud que cae dentro del casco convexo de
    alguna otra ruta. Por defecto EXCLUYE los tramos al depósito para no
    sesgar a la baja cuando el depósito queda fuera de los cascos.
    Devuelve un valor en [0,1].
    """
    if not data.get("nodes"):
        return 0.0

    # Cambia a True si quieres incluir los tramos depósito<->ruta en la métrica
    COUNT_DEPOT_LEGS = True

    nodes_xy = _ensure_nodes_xy(data)
    cache = data.setdefault(_GLOBAL_INTR_CACHE_KEY, {"segments": {}, "hulls": {}})

    seg_cache = cache["segments"]; hull_cache = cache["hulls"]

    def _segments_for(route: List[int]):
        key = tuple(route)
        entry = seg_cache.get(key)
        if entry is None:
            segs = _route_segments_xy(route, nodes_xy)
            length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs)
            entry = (segs, length)
            seg_cache[key] = entry
        return entry

    def _hull_for(route: List[int]):
        key = tuple(route)
        entry = hull_cache.get(key)
        if entry is None:
            hull = _route_hull_xy(route, nodes_xy)
            bbox = _bbox(hull) if len(hull) >= 3 else None
            entry = (hull, bbox)
            hull_cache[key] = entry
        return entry

    route_segments_all: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = []
    route_segments_intr: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = []
    route_lengths_intr: List[float] = []
    hulls: List[List[Tuple[float, float]]] = []
    bboxes: List[Optional[Tuple[float, float, float, float]]] = []

    for route in routes:
        segs, _Lall = _segments_for(route)
        hull, bbox = _hull_for(route)
        route_segments_all.append(segs)
        hulls.append(hull); bboxes.append(bbox)

        # segmentos a considerar para intrusión
        segs_intr = segs if COUNT_DEPOT_LEGS else (segs[1:-1] if len(segs) >= 2 else [])
        L_intr = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs_intr)
        route_segments_intr.append(segs_intr)
        route_lengths_intr.append(L_intr)

    fractions = []
    n_routes = len(routes)
    for j in range(n_routes):
        Lj = route_lengths_intr[j]
        segs_j = route_segments_intr[j]
        if Lj <= 1e-9 or not segs_j:
            fractions.append(0.0)
            continue

        # bbox del conjunto de segmentos de j para prefiltrar contra cascos i
        pts_j = [p for seg in segs_j for p in seg]
        bbox_j = _bbox(pts_j)

        inside_len = 0.0
        for i in range(n_routes):
            if i == j:
                continue
            bbox_i = bboxes[i]
            if bbox_i is None or not hulls[i]:
                continue
            if not _bbox_intersect(bbox_i, bbox_j):
                continue
            inside_len += _intrusion_one_way(hulls[i], bbox_i, segs_j)

        fractions.append(min(1.0, inside_len / Lj))

    return float(sum(fractions) / max(1, len(fractions)))


def compute_intrusion_km(routes: List[List[int]], data: Dict) -> float:
    return intrusion_length_between_routes_m(routes, data)


__all__ = ["compute_intrusion_km", "intrusion_length_between_routes_m"]

