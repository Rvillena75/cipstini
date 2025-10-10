from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .balance_dist import compute_balance_distance_cv
from .balance_stops import compute_balance_stops_cv
from .cruces_intra import count_self_crossings
from .cruces_inter import (
    count_between_routes_crossings,
    count_total_inter_route_crossings,
    count_inter_route_crossings,
)
from .dispersion import compute_dispersion, route_shape_penalty
from .complexity import route_complexity, compute_complexity
from .coherence import compute_coherence
from .intrusion import compute_intrusion_km

try:  # pragma: no cover - optional dependency
    from line_profiler import profile as line_profile
except ImportError:  # pragma: no cover
    def line_profile(func):
        return func

_R_EARTH = 6371000.0

if TYPE_CHECKING:
    from ..alns_core import Solution  # pragma: no cover

Routes = List[List[int]]

_METRICS_LOG: Dict[float, Dict[str, List[float]]] = {}
_CURRENT_LAMBDA: Optional[float] = None
_PROFILE_ENABLED: bool = False
_PROFILE_DATA: Dict[str, float] = {}


def _profile_add(name: str, elapsed: float) -> None:
    """Acumula tiempo y número de llamadas para la métrica `name`."""
    _PROFILE_DATA[f"{name}_time"] = _PROFILE_DATA.get(f"{name}_time", 0.0) + elapsed
    _PROFILE_DATA[f"{name}_calls"] = _PROFILE_DATA.get(f"{name}_calls", 0) + 1


class EstheticCache:
    """Cache incremental para métricas estéticas.

    Útil para evitar recalcular geometrías/pares cuando se generan múltiples
    soluciones intermedias que comparten rutas.
    """

    __slots__ = (
        "nodes",
        "distM",
        "lat0",
        "lon0",
        "nodes_xy",
        "route_cache",
        "pair_cache",
        "intrusion_data",
        "disp_weights",
        "disp_e_cap",
    )

    def __init__(self, data: Dict):
        self.nodes = data["nodes"]
        self.distM = data["distM"]
        self.lat0, self.lon0 = self.nodes[0]
        self.nodes_xy = [self._to_xy_point(self.nodes[i][0], self.nodes[i][1]) for i in range(len(self.nodes))]
        disp_w = data.get("dispersion_shape_weights")
        if disp_w is not None and len(disp_w) >= 3:
            self.disp_weights = tuple(float(w) for w in disp_w[:3])
        else:
            self.disp_weights = (0.45, 0.35, 0.20)
        self.disp_e_cap = float(data.get("dispersion_ecc_cap", 5.0))
        self.route_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}
        self.pair_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Dict[str, float]] = {}
        self.intrusion_data = {"nodes": self.nodes, "nodes_xy": self.nodes_xy}

    def _to_xy_point(self, lat: float, lon: float) -> Tuple[float, float]:
        import math

        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        lat0_r = math.radians(self.lat0)
        lon0_r = math.radians(self.lon0)
        x = _R_EARTH * (lon_r - lon0_r) * math.cos((lat_r + lat0_r) * 0.5)
        y = _R_EARTH * (lat_r - lat0_r)
        return (x, y)

    def _route_distance(self, route: List[int]) -> float:
        if not route:
            return 0.0
        dist = self.distM[0, route[0]]
        for a, b in zip(route, route[1:]):
            dist += self.distM[a, b]
        dist += self.distM[route[-1], 0]
        return float(dist)

    def route_metrics(self, key: Tuple[int, ...], route: List[int]) -> Dict[str, float]:
        cached = self.route_cache.get(key)
        if cached is not None:
            return cached

        metrics = {
            'dispersion': 0.0,
            'cruces_intra': 0.0,
            'distance': 0.0,
            'stops': float(len(route)),
            'complexity': 0.0,
        }
        if route:
            metrics['dispersion'] = float(
                route_shape_penalty(
                    route,
                    self.nodes_xy,
                    weights=self.disp_weights,
                    e_cap=self.disp_e_cap,
                )
            )
            metrics['cruces_intra'] = float(count_self_crossings(route, self.nodes))
            metrics['distance'] = self._route_distance(route)
            metrics['complexity'] = float(route_complexity(route, self.nodes_xy))

        self.route_cache[key] = metrics
        return metrics

    def pair_metrics(
        self,
        key_a: Tuple[int, ...],
        route_a: List[int],
        data_a: Dict[str, float],
        key_b: Tuple[int, ...],
        route_b: List[int],
        data_b: Dict[str, float],
    ) -> Dict[str, float]:
        pair_key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
        cached = self.pair_cache.get(pair_key)
        if cached is not None:
            return cached

        if not route_a or not route_b:
            entry = {
                'cross': 0.0,
            }
            self.pair_cache[pair_key] = entry
            return entry

        cross = count_inter_route_crossings(route_a, route_b, self.nodes)

        entry = {
            'cross': float(cross),
        }
        self.pair_cache[pair_key] = entry
        return entry

    def compute_components(self, routes: List[List[int]]) -> Dict[str, object]:
        n = len(routes)
        if n == 0:
            return {
                'dispersion': 0.0,
                'cruces_intra': 0.0,
                'cruces_inter': 0.0,
                'intrusion_ratio': 0.0,
                'complexity': 0.0,
                'coherence': 0.0,
                'distances': [],
                'stops': [],
            }

        route_keys = [tuple(r) for r in routes]
        infos = [self.route_metrics(k, r) for k, r in zip(route_keys, routes)]

        dispersion = sum(info['dispersion'] for info in infos) / n
        cruces_intra = sum(info['cruces_intra'] for info in infos) / n
        complexity = sum(info.get('complexity', 0.0) for info in infos) / n
        distances = [info['distance'] for info in infos]
        stops = [info['stops'] for info in infos]

        cross_sum = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair = self.pair_metrics(route_keys[i], routes[i], infos[i], route_keys[j], routes[j], infos[j])
                cross_sum += pair['cross']
                pairs += 1

        cross_avg = cross_sum / max(1, pairs)
        intrusion_ratio = compute_intrusion_km(routes, self.intrusion_data)
        coherence_ratio = compute_coherence(routes, self.intrusion_data, nodes_xy=self.nodes_xy)

        return {
            'dispersion': dispersion,
            'cruces_intra': cruces_intra,
            'cruces_inter': cross_avg,
            'intrusion_ratio': intrusion_ratio,
            'complexity': complexity,
            'coherence': coherence_ratio,
            'distances': distances,
            'stops': stops,
        }


def activate_logging(lam: float) -> None:
    global _CURRENT_LAMBDA
    _CURRENT_LAMBDA = lam
    _METRICS_LOG.pop(lam, None)


def deactivate_logging() -> None:
    global _CURRENT_LAMBDA
    _CURRENT_LAMBDA = None


def get_metrics_log() -> Dict[float, Dict[str, List[float]]]:
    return _METRICS_LOG


def enable_aesthetic_profiling(reset: bool = True) -> None:
    global _PROFILE_ENABLED, _PROFILE_DATA
    _PROFILE_ENABLED = True
    if reset:
        _PROFILE_DATA = {}


def disable_aesthetic_profiling() -> None:
    global _PROFILE_ENABLED
    _PROFILE_ENABLED = False


def get_aesthetic_profile(reset: bool = False) -> Dict[str, float]:
    global _PROFILE_DATA
    summary: Dict[str, Tuple[float, int]] = {}
    for key, value in _PROFILE_DATA.items():
        if key.endswith("_time"):
            base = key[:-5]
            calls = int(_PROFILE_DATA.get(f"{base}_calls", 0))
            summary[base] = (float(value), calls)
    if reset:
        _PROFILE_DATA = {}
    return summary


def _route_distance(route: List[int], distM) -> float:
    if not route:
        return 0.0
    total = distM[0, route[0]]
    for a, b in zip(route, route[1:]):
        total += distM[a, b]
    total += distM[route[-1], 0]
    return float(total)


def _log_metrics_from_solution(sol, data) -> Dict[str, float]:
    routes = [r for r in sol.routes if r]
    if not routes:
        return {
            "cross_intra_norm": 0.0,
            "cross_inter_norm": 0.0,
            "balance_dist_cv": 0.0,
            "balance_stops_cv": 0.0,
            "dispersion": 0.0,
            "intrusion": 0.0,
            "complexity": 0.0,
            "coherence": 0.0,
        }

    nodes, distM = data["nodes"], data["distM"]
    nodes_xy = data.get("nodes_xy")
    if nodes_xy is None:
        lat0, lon0 = nodes[0]
        lat0_r = math.radians(lat0)
        lon0_r = math.radians(lon0)
        nodes_xy = []
        for lat, lon in nodes:
            lat_r = math.radians(lat)
            lon_r = math.radians(lon)
            x = _R_EARTH * (lon_r - lon0_r) * math.cos(0.5 * (lat_r + lat0_r))
            y = _R_EARTH * (lat_r - lat0_r)
            nodes_xy.append((x, y))
        data["nodes_xy"] = nodes_xy
    raw_disp_w = data.get("dispersion_shape_weights")
    if raw_disp_w is not None and len(raw_disp_w) >= 3:
        disp_weights = tuple(float(w) for w in raw_disp_w[:3])
    else:
        disp_weights = (0.45, 0.35, 0.20)
    disp_e_cap = float(data.get("dispersion_ecc_cap", 5.0))

    dists = [_route_distance(r, distM) for r in routes]
    mean_dist = sum(dists) / max(1, len(dists))

    segs_per_route = [len(r) + 1 for r in routes]

    def _non_adjacent_pairs(s: int) -> int:
        return max(0, (s * (s - 1)) // 2 - (s - 1))

    pairs_intra = sum(_non_adjacent_pairs(s) for s in segs_per_route)
    crosses_intra = sum(count_self_crossings(r, nodes) for r in routes)
    cross_intra_norm = crosses_intra / max(1, pairs_intra)

    crosses_inter = count_between_routes_crossings(routes, nodes)
    denom_inter = 0
    for i in range(len(segs_per_route)):
        for j in range(i + 1, len(segs_per_route)):
            denom_inter += segs_per_route[i] * segs_per_route[j]
    cross_inter_norm = crosses_inter / max(1, denom_inter)

    balance_cv = (float(np.std(dists)) / (mean_dist + 1e-9)) if dists else 0.0
    stops = [len(r) for r in routes]
    mean_stops = sum(stops) / max(1, len(stops))
    stops_cv = (float(np.std(stops)) / (mean_stops + 1e-9)) if stops else 0.0
    dispersion = compute_dispersion(
        routes,
        nodes,
        nodes_xy,
        weights=disp_weights,
        e_cap=disp_e_cap,
    )
    geom_data = {"nodes": nodes, "nodes_xy": nodes_xy}
    intrusion = compute_intrusion_km(routes, geom_data)
    complexity = compute_complexity(routes, nodes, nodes_xy)
    coherence = compute_coherence(routes, geom_data, nodes_xy=nodes_xy)

    return {
        "cross_intra_norm": cross_intra_norm,
        "cross_inter_norm": cross_inter_norm,
        "balance_dist_cv": balance_cv,
        "balance_stops_cv": stops_cv,
        "dispersion": float(dispersion),
        "intrusion": float(intrusion),
        "complexity": float(complexity),
        "coherence": float(coherence),
    }


@line_profile
def aesthetic_penalty(
    sol,
    data: Dict,
    weights: Dict,
    cache: Optional[EstheticCache] = None,
    enable_metrics: bool = True,
) -> float:
    if not enable_metrics:
        return 0.0
    total_start = time.perf_counter() if _PROFILE_ENABLED else None
    routes = [r for r in sol.routes if r]
    if _PROFILE_ENABLED:
        _PROFILE_DATA["calls_total"] = _PROFILE_DATA.get("calls_total", 0) + 1

    if len(routes) < 2:
        value = 0.0
    else:
        if cache is not None:
            comp_start = time.perf_counter() if _PROFILE_ENABLED else None
            comp = cache.compute_components(routes)
            if _PROFILE_ENABLED and comp_start is not None:
                _profile_add("cache_components", time.perf_counter() - comp_start)

            # Métricas de forma
            pen_dispersion = comp["dispersion"]
            pen_complexity = comp["complexity"]
            pen_cruces_intra = comp["cruces_intra"]
            pen_cruces_inter = comp["cruces_inter"] * 3.0
            pen_intrusion_km = comp["intrusion_ratio"]
            pen_coherence = comp["coherence"]

            # Métricas de balance
            distances = np.array(comp["distances"], dtype=float)
            stops = np.array(comp["stops"], dtype=float)
            mean_dist = distances.mean() if distances.size else 0.0
            pen_balance_dist = float(np.std(distances) / (mean_dist + 1e-9)) if distances.size else 0.0
            mean_stops = stops.mean() if stops.size else 0.0
            pen_balance_stops = float(np.std(stops) / (mean_stops + 1e-9)) if stops.size else 0.0

            if _PROFILE_ENABLED:
                for name in (
                    "dispersion",
                    "complexity",
                    "cruces_intra",
                    "cruces_inter",
                    "balance_dist",
                    "balance_stops",
                    "intrusion",
                    "coherence",
                ):
                    _profile_add(name, 0.0)
        else:
            setup_start = time.perf_counter() if _PROFILE_ENABLED else None
            nodes = data["nodes"]
            distM = data["distM"]
            nodes_xy = data.get("nodes_xy")
            if nodes_xy is None:
                lat0, lon0 = nodes[0]
                lat0_r = math.radians(lat0)
                lon0_r = math.radians(lon0)
                nodes_xy = []
                for lat, lon in nodes:
                    lat_r = math.radians(lat)
                    lon_r = math.radians(lon)
                    x = _R_EARTH * (lon_r - lon0_r) * math.cos(0.5 * (lat_r + lat0_r))
                    y = _R_EARTH * (lat_r - lat0_r)
                    nodes_xy.append((x, y))
                data["nodes_xy"] = nodes_xy
            raw_disp_w = data.get("dispersion_shape_weights")
            if raw_disp_w is not None and len(raw_disp_w) >= 3:
                disp_weights = tuple(float(w) for w in raw_disp_w[:3])
            else:
                disp_weights = (0.45, 0.35, 0.20)
            disp_e_cap = float(data.get("dispersion_ecc_cap", 5.0))
            if _PROFILE_ENABLED and setup_start is not None:
                _profile_add("setup", time.perf_counter() - setup_start)

            if _PROFILE_ENABLED:
                start = time.perf_counter()
                pen_dispersion = compute_dispersion(
                    routes,
                    nodes,
                    nodes_xy,
                    weights=disp_weights,
                    e_cap=disp_e_cap,
                )
                _profile_add("dispersion", time.perf_counter() - start)
            else:
                pen_dispersion = compute_dispersion(
                    routes,
                    nodes,
                    nodes_xy,
                    weights=disp_weights,
                    e_cap=disp_e_cap,
                )

            if _PROFILE_ENABLED:
                start = time.perf_counter()
                pen_cruces_intra = sum(count_self_crossings(r, nodes) for r in routes) / len(routes)
                _profile_add("cruces_intra", time.perf_counter() - start)
            else:
                pen_cruces_intra = sum(count_self_crossings(r, nodes) for r in routes) / len(routes)

            if _PROFILE_ENABLED:
                start = time.perf_counter()
                pen_cruces_inter = count_total_inter_route_crossings(routes, nodes)
                _profile_add("cruces_inter", time.perf_counter() - start)
            else:
                pen_cruces_inter = count_total_inter_route_crossings(routes, nodes)
            pen_cruces_inter *= 3.0

            if _PROFILE_ENABLED:
                start = time.perf_counter()
                pen_balance_dist = compute_balance_distance_cv(routes, distM)
                _profile_add("balance_dist", time.perf_counter() - start)
            else:
                pen_balance_dist = compute_balance_distance_cv(routes, distM)

            if _PROFILE_ENABLED:
                start = time.perf_counter()
                pen_balance_stops = compute_balance_stops_cv(routes)
                _profile_add("balance_stops", time.perf_counter() - start)
            else:
                pen_balance_stops = compute_balance_stops_cv(routes)

            shared_geom_data = {"nodes": nodes, "nodes_xy": nodes_xy}

            pen_complexity = compute_complexity(routes, nodes, nodes_xy)
            pen_intrusion_km = compute_intrusion_km(routes, shared_geom_data)
            pen_coherence = compute_coherence(routes, shared_geom_data, nodes_xy=nodes_xy)

            metrics = {
                # Métricas de forma
                "dispersion_rutas": pen_dispersion,
                "complexity_rutas": pen_complexity,

                # Métricas de interferencia
                "cruces_intra_ruta": pen_cruces_intra,
                "cruces_inter_ruta": pen_cruces_inter,
                "intrusion": pen_intrusion_km,

                # Métricas de balance
                "desbalance_dist_cv": pen_balance_dist,
                "desbalance_stops_cv": pen_balance_stops,
                # Coherencia territorial
                "coherence_clientes": pen_coherence,
            }

        # Actualizar mapeo de nombres a claves de peso
        value = 0.0
        for name, metric in metrics.items():
            weight_key = WEIGHT_KEYS.get(name, "")
            if weight_key and weight_key in weights:
                value += weights[weight_key] * metric
        value = float(value)

    if _CURRENT_LAMBDA is not None:
        d = _log_metrics_from_solution(sol, data)
        bucket = _METRICS_LOG.setdefault(_CURRENT_LAMBDA, {})
        for k, v in d.items():
            bucket.setdefault(k, []).append(float(v))
    if _PROFILE_ENABLED and total_start is not None:
        _profile_add("total_penalty", time.perf_counter() - total_start)
    return float(value)


DEFAULT_FAST_WEIGHTS = {
    "w_dispersion": 40.0,
    "w_complexity": 35.0,
    "w_cruces_intra": 30.0,
    "w_cruces_inter": 40.0,
    "w_balance_dist": 60.0,
    "w_balance_stops": 60.0,
    "w_intrusion": 400.0,
    "w_coherence": 50.0,
}


def _ensure_nodes_xy(data: Dict) -> List[Tuple[float, float]]:
    nodes = data["nodes"]
    nodes_xy = data.get("nodes_xy")
    if nodes_xy is None:
        lat0, lon0 = nodes[0]
        lat0_r = math.radians(lat0)
        lon0_r = math.radians(lon0)
        nodes_xy = []
        for lat, lon in nodes:
            lat_r = math.radians(lat)
            lon_r = math.radians(lon)
            x = _R_EARTH * (lon_r - lon0_r) * math.cos(0.5 * (lat_r + lat0_r))
            y = _R_EARTH * (lat_r - lat0_r)
            nodes_xy.append((x, y))
        data["nodes_xy"] = nodes_xy
    return nodes_xy


def aesthetic_penalty_fast(sol, data: Dict) -> float:
    routes = [r for r in sol.routes if r]
    if not routes:
        return 0.0
    nodes = data["nodes"]
    distM = data["distM"]
    nodes_xy = _ensure_nodes_xy(data)

    raw_disp_w = data.get("dispersion_shape_weights")
    if raw_disp_w is not None and len(raw_disp_w) >= 3:
        disp_weights = tuple(float(w) for w in raw_disp_w[:3])
    else:
        disp_weights = (0.45, 0.35, 0.20)
    disp_e_cap = float(data.get("dispersion_ecc_cap", 5.0))

    dists = [_route_distance(r, distM) for r in routes]
    mean_dist = np.mean(dists) if dists else 0.0
    balance_dist = float(np.std(dists) / (mean_dist + 1e-9)) if dists else 0.0

    stops = [len(r) for r in routes]
    mean_stops = np.mean(stops) if stops else 0.0
    balance_stops = float(np.std(stops) / (mean_stops + 1e-9)) if stops else 0.0

    cruces_intra = sum(count_self_crossings(r, nodes) for r in routes) / max(1, len(routes))
    cruces_inter = count_total_inter_route_crossings(routes, nodes) * 3.0
    dispersion = compute_dispersion(
        routes,
        nodes,
        nodes_xy,
        weights=disp_weights,
        e_cap=disp_e_cap,
    )
    complexity = compute_complexity(routes, nodes, nodes_xy)
    geom_data = {"nodes": nodes, "nodes_xy": nodes_xy}
    intrusion = compute_intrusion_km(routes, geom_data)
    coherence = compute_coherence(routes, geom_data, nodes_xy=nodes_xy)

    weights = data.get("fast_metric_weights", DEFAULT_FAST_WEIGHTS)

    return (
        weights.get("w_dispersion", DEFAULT_FAST_WEIGHTS["w_dispersion"]) * float(dispersion)
        + weights.get("w_complexity", DEFAULT_FAST_WEIGHTS["w_complexity"]) * float(complexity)
        + weights.get("w_cruces_intra", DEFAULT_FAST_WEIGHTS["w_cruces_intra"]) * float(cruces_intra)
        + weights.get("w_cruces_inter", DEFAULT_FAST_WEIGHTS["w_cruces_inter"]) * float(cruces_inter)
        + weights.get("w_balance_dist", DEFAULT_FAST_WEIGHTS["w_balance_dist"]) * float(balance_dist)
        + weights.get("w_balance_stops", DEFAULT_FAST_WEIGHTS["w_balance_stops"]) * float(balance_stops)
        + weights.get("w_intrusion", DEFAULT_FAST_WEIGHTS["w_intrusion"]) * float(intrusion)
        + weights.get("w_coherence", DEFAULT_FAST_WEIGHTS["w_coherence"]) * float(coherence)
    )


def esthetics_breakdown_final(sol, data: Dict, weights: Dict, lam: float = 1.0) -> Dict[str, float]:
    key = getattr(sol, "_breakdown_cache_key", None)
    if sol.breakdown_cache is not None and key == (lam, id(data)):
        return sol.breakdown_cache

    routes = [r for r in sol.routes if r]
    if not routes:
        return {
            "dispersion_rutas": 0.0,
            "cruces_intra_ruta": 0.0,
            "cruces_inter_ruta": 0.0,
            "desbalance_dist_cv": 0.0,
            "desbalance_stops_cv": 0.0,
            "complexity_rutas": 0.0,
            "intrusion": 0.0,
            "coherence_clientes": 0.0,
            "penalizacion_cruda": 0.0,
            "detalle_sin_pesos": {},
        }

    if lam <= 0.0:
        zeros = {
            "dispersion_rutas": 0.0,
            "cruces_intra_ruta": 0.0,
            "cruces_inter_ruta": 0.0,
            "desbalance_dist_cv": 0.0,
            "desbalance_stops_cv": 0.0,
            "complexity_rutas": 0.0,
            "intrusion": 0.0,
            "coherence_clientes": 0.0,
        }
        result = dict(zeros)
        result["penalizacion_cruda"] = 0.0
        result["penalizacion_cruda_sin_lambda"] = 0.0
        result["detalle_sin_pesos"] = {k: float(v) for k, v in zeros.items()}
        return result

    cache = EstheticCache(data)
    comp = cache.compute_components(routes)
    pen_cruda = aesthetic_penalty(sol, data, weights, cache=cache, enable_metrics=True)

    distances = np.array(comp["distances"], dtype=float)
    stops = np.array(comp["stops"], dtype=float)
    mean_dist = distances.mean() if distances.size else 0.0
    pen_balance_dist = float(np.std(distances) / (mean_dist + 1e-9)) if distances.size else 0.0
    mean_stops = stops.mean() if stops.size else 0.0
    pen_balance_stops = float(np.std(stops) / (mean_stops + 1e-9)) if stops.size else 0.0

    pen_dispersion = comp["dispersion"]
    pen_complexity = comp["complexity"]
    pen_cruces_intra = comp["cruces_intra"]
    pen_cruces_inter = comp["cruces_inter"] * 3.0
    pen_intrusion = comp["intrusion_ratio"]
    pen_coherence = comp["coherence"]

    metrics = {
        "dispersion_rutas": pen_dispersion,
        "complexity_rutas": pen_complexity,
        "cruces_intra_ruta": pen_cruces_intra,
        "cruces_inter_ruta": pen_cruces_inter,
        "desbalance_dist_cv": pen_balance_dist,
        "desbalance_stops_cv": pen_balance_stops,
        "intrusion": pen_intrusion,
        "coherence_clientes": pen_coherence,
    }

    scale = float(lam)
    weighted_scaled = {}
    for name, metric in metrics.items():
        weighted = weights.get(WEIGHT_KEYS.get(name, ""), 0.0) * metric
        weighted_scaled[name] = float(weighted * scale)

    weighted_scaled["penalizacion_cruda"] = float(pen_cruda * scale)
    weighted_scaled["penalizacion_cruda_sin_lambda"] = float(pen_cruda)
    weighted_scaled["detalle_sin_pesos"] = {name: float(metric) for name, metric in metrics.items()}

    sol.breakdown_cache = weighted_scaled
    sol._breakdown_cache_key = (lam, id(data))
    return weighted_scaled


WEIGHT_KEYS = {
    "dispersion_rutas": "w_dispersion",
    "complexity_rutas": "w_complexity",
    "cruces_intra_ruta": "w_cruces_intra",
    "cruces_inter_ruta": "w_cruces_inter",
    "desbalance_dist_cv": "w_balance_dist",
    "desbalance_stops_cv": "w_balance_stops",
    "intrusion": "w_intrusion",
    "coherence_clientes": "w_coherence",
}

__all__ = [
    "aesthetic_penalty",
    "aesthetic_penalty_fast",
    "esthetics_breakdown_final",
    "activate_logging",
    "deactivate_logging",
    "get_metrics_log",
    "enable_aesthetic_profiling",
    "disable_aesthetic_profiling",
    "get_aesthetic_profile",
    "EstheticCache",
]
