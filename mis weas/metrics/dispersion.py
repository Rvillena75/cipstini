from typing import List, Tuple, Optional
import math

_R_EARTH = 6371000.0

def _to_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    lat0_r, lon0_r = math.radians(lat0), math.radians(lon0)
    x = _R_EARTH * (lon_r - lon0_r) * math.cos(0.5 * (lat_r + lat0_r))
    y = _R_EARTH * (lat_r - lat0_r)
    return x, y

def _cov2x2(xs, ys):
    n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)/n
    syy = sum((y-my)**2 for y in ys)/n
    sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/n
    return sxx, sxy, sxy, syy

def _eigvals2x2(a,b,c,d):
    tr = a+d
    det = a*d - b*c
    disc = max(tr*tr - 4*det, 0.0)
    r = math.sqrt(disc)
    l1 = 0.5*(tr + r)
    l2 = 0.5*(tr - r)
    # estabilidad numérica
    if l1 < l2:
        l1, l2 = l2, l1
    return max(l1, 1e-12), max(l2, 1e-12)

def _hull(points: List[Tuple[float,float]]):
    # Monotone chain (O(n log n))
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _poly_area_perimeter(poly: List[Tuple[float,float]]):
    if len(poly) < 3:
        if len(poly) == 2:
            (x1,y1),(x2,y2) = poly
            return 0.0, math.hypot(x2-x1, y2-y1)*2  # segmento "doblado" como perim
        return 0.0, 0.0
    A=0.0; P=0.0
    for (x1,y1),(x2,y2) in zip(poly, poly[1:]+poly[:1]):
        A += x1*y2 - x2*y1
        P += math.hypot(x2-x1, y2-y1)
    return abs(A)*0.5, P

def route_shape_penalty(
    route: List[int],
    nodes_xy: List[Tuple[float, float]],
    weights: Optional[Tuple[float, float, float]] = None,
    e_cap: float = 5.0,
) -> float:
    if len(route) <= 2:
        return 0.0
    pts = [nodes_xy[i] for i in route]
    xs, ys = zip(*pts)

    # 1) Compactación PP del casco convexo
    hull = _hull(pts)
    A, P = _poly_area_perimeter(hull)
    C_pp = 0.0 if P <= 1e-12 else min(4*math.pi*A/(P*P), 1.0)
    pen_pp = 1.0 - C_pp

    # 2) Isotropía (relación de autovalores)
    a,b,c,d = _cov2x2(xs, ys)
    l1,l2 = _eigvals2x2(a,b,c,d)
    E = l1/l2
    pen_iso = min((E - 1.0)/e_cap, 1.0)

    # 3) Variación radial
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    rs = [math.hypot(x-mx, y-my) for x,y in pts]
    r_mean = sum(rs)/len(rs)
    if r_mean <= 1e-12:
        pen_rad = 0.0
    else:
        # std poblacional
        var = sum((r - r_mean)**2 for r in rs)/len(rs)
        cv = math.sqrt(var)/r_mean
        pen_rad = cv/(1.0 + cv)

    if weights is None:
        w1, w2, w3 = 0.45, 0.35, 0.20
    else:
        if len(weights) != 3:
            raise ValueError("weights for route_shape_penalty must have exactly 3 components")
        w1, w2, w3 = weights
    score = w1*pen_pp + w2*pen_iso + w3*pen_rad
    return max(0.0, min(1.0, score))

def compute_dispersion(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    nodes_xy: Optional[List[Tuple[float, float]]] = None,
    weights: Optional[Tuple[float, float, float]] = None,
    e_cap: float = 5.0,
) -> float:
    """Average shape penalty for a set of routes.

    Each route is evaluated with :func:`route_shape_penalty`, combining
    convex-hull compactness, covariance anisotropy y dispersión radial de los
    clientes. ``weights`` permite sobreescribir los pesos internos (por defecto
    ``(0.45, 0.35, 0.20)``) y ``e_cap`` acota la excentricidad antes de
    normalizar.
    """
    if not routes:
        return 0.0
    if nodes_xy is None:
        lat0, lon0 = nodes[0]
        nodes_xy = [_to_xy(lat, lon, lat0, lon0) for lat,lon in nodes]
    scores = [
        route_shape_penalty(r, nodes_xy, weights=weights, e_cap=e_cap)
        for r in routes
        if len(r) > 1
    ]
    return 0.0 if not scores else float(sum(scores)/len(scores))
