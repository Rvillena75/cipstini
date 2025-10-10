import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import folium
from branca.element import MacroElement
from jinja2 import Template

import os
from pathlib import Path

import time
from dataclasses import dataclass

from tqdm import tqdm

from metrics.aesthetic import (
    EstheticCache,
    activate_logging,
    aesthetic_penalty,
    aesthetic_penalty_fast,
    deactivate_logging,
)

# --- Polilíneas por calle con OSRM (visualización) ---
import json, requests

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATOS_P5_DIR = ROOT_DIR / "Datos P5"

USE_ROAD_GEOMETRY = True         # apágalo si quieres volver a líneas rectas
_ROAD_SHAPE_CACHE_PATH = BASE_DIR / "route_shapes_cache.json"
_ROAD_SHAPE_CACHE = {}

def _load_shape_cache():
    global _ROAD_SHAPE_CACHE
    try:
        with open(_ROAD_SHAPE_CACHE_PATH, "r", encoding="utf-8") as f:
            _ROAD_SHAPE_CACHE = json.load(f)
    except Exception:
        _ROAD_SHAPE_CACHE = {}

def _save_shape_cache():
    try:
        with open(_ROAD_SHAPE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_ROAD_SHAPE_CACHE, f)
    except Exception:
        pass


def _is_valid_coord(lat, lon):
    if any(map(lambda x: x is None or isinstance(x, float) and math.isnan(x), [lat, lon])):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

def _osrm_route_shape(waypoints_latlon):
    """
    waypoints_latlon: [[lat, lon], ...] (incluye depósito al inicio y fin).
    Devuelve [[lat, lon], ...] siguiendo calles, o None si falla.
    """
    # --- 1) Validaciones fuertes
    if not waypoints_latlon or len(waypoints_latlon) < 2:
        print("[OSRM] Necesito ≥2 puntos")
        return None

    for i, (lat, lon) in enumerate(waypoints_latlon):
        if not _is_valid_coord(lat, lon):
            print(f"[OSRM] Coord inválida en idx={i}: lat={lat}, lon={lon}")
            return None

    # Heurística para detectar inversión lat/lon en la entrada
    # (lon en Chile ~ -70, lat ~ -33). Si vemos |lon|<=90 y |lat|>90, probablemente venían invertidas.
    # Ajusta a tu caso si ruteas fuera de Chile.
    sample_lat, sample_lon = waypoints_latlon[0]
    if abs(sample_lon) <= 90 and abs(sample_lat) > 90:
        print("[OSRM] Sospecha de entrada invertida [lon,lat] en lugar de [lat,lon].")
        return None

    # --- 2) Construcción segura del path de coords (lon,lat;...)
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints_latlon)

    base = f"https://router.project-osrm.org/route/v1/driving/{coords}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
        "annotations": "false",  # <- en vez de 'none'
        # "continue_straight": "true",  # opcional
    }

    try:
        r = requests.get(base, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            routes = data.get("routes", [])
            if routes:
                line = routes[0]["geometry"]["coordinates"]  # [[lon,lat], ...]
                shape = [[lat, lon] for lon, lat in line]     # devuelvo [[lat,lon], ...]
                print(f"[OSRM] OK shape pts={len(shape)}")
                return shape
            print("[OSRM] Sin 'routes' en la respuesta")
        else:
            # Log útil: muestra dónde “se rompe” si el server reporta posición
            txt = r.text
            print(f"[OSRM] HTTP {r.status_code}: {txt[:200]}")
    except Exception as e:
        print(f"[OSRM] EXC: {e}")

    return None


def _route_shape_on_roads(route, data):
    """
    Devuelve polilínea [[lat,lon],...] por calles para una ruta (depot + paradas + depot).
    Usa cache y fallback por tramos si falla la ruta completa.
    """
    seq = [0] + route + [0]
    key = ",".join(map(str, seq))
    if key in _ROAD_SHAPE_CACHE:
        return _ROAD_SHAPE_CACHE[key]

    pts = [[data["nodes"][i][0], data["nodes"][i][1]] for i in seq]

    # 1) Intento 1: una sola consulta con todos los waypoints
    shape = _osrm_route_shape(pts)
    if shape is None:
        print(f"[OSRM] Fallback por segmentos (ruta len={len(route)})")
        shape = []
        for a, b in zip(seq, seq[1:]):
            seg_pts = [[data['nodes'][a][0], data['nodes'][a][1]],
                       [data['nodes'][b][0], data['nodes'][b][1]]]
            seg = _osrm_route_shape(seg_pts)
            if seg is None:
                print(f"[OSRM] Fallback recto {a}->{b}")
                seg = seg_pts  # ← aquí se cae a recta si también falla
            if shape and seg:
                shape.extend(seg[1:])
            else:
                shape.extend(seg)

    _ROAD_SHAPE_CACHE[key] = shape
    return shape

_ROUTE_DIST_CACHE = {}
_ROUTE_METRICS_CACHE = {}
_ROUTE_SCHEDULE_CACHE = {}
_EVAL_CACHE = {}  # key: (routes_key, fast, lam), val: (cost, pen, obj)

def _key_from_routes(routes: List[List[int]]) -> tuple:
    """Genera una clave única para una lista de rutas activas."""
    return tuple(tuple(rt) for rt in routes if rt)

def _invalidate_cache_for(routes: List[List[int]]) -> None:
    """Invalida el caché de distancias para las rutas dadas."""
    for r in routes:
        key = tuple(r)
        _ROUTE_DIST_CACHE.pop(key, None)
        _ROUTE_METRICS_CACHE.pop(key, None)
        _ROUTE_SCHEDULE_CACHE.pop(key, None)

def evaluate_cached(
    routes_active: List[List[int]],
    *,
    fast: bool,
    data: dict,
    weights: dict,
    lam: float,
    esthetic_cache: Optional[EstheticCache] = None
) -> tuple:
    """Evalúa una solución con caché de resultados."""
    key = _key_from_routes(routes_active)
    cache_key = (key, bool(fast), float(lam))
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]
    sol_obj = Solution([rt[:] for rt in routes_active])
    cost = solution_cost(sol_obj, data) if sol_obj.cost is None else sol_obj.cost
    if lam > 0.0:
        pen = aesthetic_penalty_fast(sol_obj, data) if fast else aesthetic_penalty(
            sol_obj,
            data,
            weights,
            cache=esthetic_cache,
            enable_metrics=True,
        )
    else:
        pen = 0.0
    obj = cost + lam * pen
    sol_obj.cost = cost
    if lam > 0.0:
        sol_obj.est_penalty = pen
    _EVAL_CACHE[cache_key] = (cost, pen, obj)
    return cost, pen, obj

@dataclass
class Solution:
    routes: List[List[int]]
    cost: Optional[float] = None
    est_penalty: Optional[float] = None
    breakdown_cache: Optional[Dict[str, float]] = None


def _routes_fit_fleet(routes: List[List[int]], idx: int, trial_route: List[int], data: dict) -> bool:
    """Valida si las rutas resultantes pueden asignarse a la flota disponible."""
    candidate = [r[:] for r in routes]
    if idx < len(candidate):
        candidate[idx] = trial_route
    else:
        candidate.append(trial_route)
    feasible, _, _ = pack_routes_to_vehicles(candidate, data)
    return feasible

def preparar_directorio_soluciones(dir_path: str, *, verbose: bool = True) -> None:
    """
    Crea el directorio si no existe y borra todos los archivos .html que contenga.
    Deja logs claros por consola.
    """
    out_dir = Path(dir_path)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(out_dir):
        if filename.endswith(".html"):
            file_path = out_dir / filename
            try:
                os.remove(file_path)
                if verbose:
                    print(f" - Borrado: {filename}")
            except Exception as e:
                if verbose:
                    print(f"Error al borrar {file_path}: {e}")




DEFAULT_I1_DIR = DATOS_P5_DIR / "i1"
DEFAULT_COSTS  = DATOS_P5_DIR / "costs.csv"

def load_data(i1_dir: str = DEFAULT_I1_DIR, costs_path: str = DEFAULT_COSTS) -> Dict:
    """
    Carga overview/demands/vehicles/distances/times/costs, arma matrices NxN y
    preprocesa ventanas de tiempo y servicio. Devuelve un diccionario con todos
    los insumos que el resto del código espera.
    """

    # --- Leer CSVs
    i1_dir = Path(i1_dir)
    costs_path = Path(costs_path)

    df_over = pd.read_csv(i1_dir / "overview.csv")
    df_dem  = pd.read_csv(i1_dir / "demands.csv")
    df_veh  = pd.read_csv(i1_dir / "vehicles.csv")
    df_dist = pd.read_csv(i1_dir / "distances.csv")
    df_time = pd.read_csv(i1_dir / "times.csv")
    df_cost = pd.read_csv(costs_path)

    # --- Checks de tamaños esperados
    exp = int(df_over.loc[0, "expected_matrix_size"])
    dsz = int(df_over.loc[0, "distances_size"])
    tsz = int(df_over.loc[0, "times_size"])
    if dsz != exp or tsz != exp:
        print(f"[WARN] Tamaños esperados no calzan: expected={exp}, dist={dsz}, time={tsz}")

    # --- Depot y jornada
    depot = (float(df_over.loc[0,"depot_latitude"]), float(df_over.loc[0,"depot_longitude"]))
    start_at = pd.to_datetime(df_over.loc[0, "start_at"])
    end_at   = pd.to_datetime(df_over.loc[0, "end_at"])
    horizon_minutes = int((end_at - start_at).total_seconds() / 60)

    # --- ID ↔ índice (útil luego para prints/mapas)
    df_dem["id_str"] = df_dem["id"].astype(str).str.strip()
    id_to_idx = {did: i+1 for i, did in enumerate(df_dem["id_str"].tolist())}
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    # --- Nodos (0 = depot)
    nodes = [(depot[0], depot[1])] + list(zip(df_dem["latitude"].astype(float),
                                              df_dem["longitude"].astype(float)))
    N = len(nodes)

    # --- Construcción de matrices NxN desde formato largo
    def build_matrix(df, value_col: str) -> np.ndarray:
        M = np.zeros((N, N), dtype=float)

        # redondeo a 6 decimales para llaves estables
        def keyfy(lat, lon):
            return (float(f"{lat:.6f}"), float(f"{lon:.6f}"))

        coord_to_idx = {keyfy(lat, lon): i for i, (lat, lon) in enumerate(nodes)}
        hits = 0
        for _, r in df.iterrows():
            o = keyfy(r["origin_latitude"],      r["origin_longitude"])
            d = keyfy(r["destination_latitude"], r["destination_longitude"])
            if o in coord_to_idx and d in coord_to_idx:
                i, j = coord_to_idx[o], coord_to_idx[d]
                M[i, j] = float(r[value_col])
                hits += 1

        if hits < N * N:
            print(f"[WARN] Matriz {value_col}: celdas llenadas={hits} < {N*N}. Revisa redondeos o correspondencias.")

        # Relleno de huecos: copia simétrico si existe; si no, castiga con 1e9
        for i in range(N):
            for j in range(N):
                if i == j:
                    M[i, j] = 0.0
                    continue
                if M[i, j] <= 0.0:
                    if M[j, i] > 0.0:
                        M[i, j] = M[j, i]
                    else:
                        M[i, j] = 1e9  # gran penalización si falta el dato

        return M


    distM = build_matrix(df_dist, "distance")  # metros
    timeM = build_matrix(df_time, "time")      # segundos

    # --- Demandas: tamaño, servicio, ventanas (en minutos desde start_at)
    demand_size  = np.zeros(N, dtype=float)   # idx 0 = depot
    demand_srv_s = np.zeros(N, dtype=float)
    tw_start_min = np.zeros(N, dtype=float)
    tw_end_min   = np.zeros(N, dtype=float)

    for i, row in enumerate(df_dem.itertuples(index=False), start=1):
        demand_size[i]  = float(row.size)
        demand_srv_s[i] = float(row.stop_time)

        tws = getattr(row, "tw_start", None)
        twe = getattr(row, "tw_end", None)

        tw_start_min[i] = 0 if pd.isna(tws) else (pd.to_datetime(tws) - start_at).total_seconds() / 60
        tw_end_min[i]   = horizon_minutes if pd.isna(twe) else (pd.to_datetime(twe) - start_at).total_seconds() / 60

    # --- Vehículos y costos
    vehicle_caps = list(df_veh["capacity"].astype(float).values)
    K = len(vehicle_caps)

    fixed_route_cost = float(df_cost.loc[0, "fixed_route_cost"])
    cost_per_meter   = float(df_cost.loc[0, "cost_per_meter"])
    cost_per_cap     = float(df_cost.loc[0, "cost_per_vehicle_capacity"])

    data = {
        "N": N, "K": K, "nodes": nodes,
        "distM": distM, "timeM": timeM,
        "demand_size": demand_size, "demand_srv_s": demand_srv_s,
        "tw_start_min": tw_start_min, "tw_end_min": tw_end_min,
        "vehicle_caps": vehicle_caps,
        "fixed_route_cost": fixed_route_cost,
        "cost_per_meter": cost_per_meter,
        "cost_per_cap": cost_per_cap,
        "horizon_minutes": horizon_minutes,
        "start_at": start_at,
        # extras útiles:
        "id_to_idx": id_to_idx, "idx_to_id": idx_to_id,
    }

    # --- Precalcular auxiliares para heurísticas y métricas
    data["tw_center_min"] = (tw_start_min + tw_end_min) * 0.5

    if N > 1:
        max_neighbors = min(15, N - 1)
        dist_sub = distM[1:, 1:]
        knn_neighbors: Dict[int, List[int]] = {}
        for idx in range(N - 1):
            order = np.argsort(dist_sub[idx])
            neighs: List[int] = []
            for o in order:
                if o == idx:
                    continue
                neighs.append(o + 1)
                if len(neighs) >= max_neighbors:
                    break
            knn_neighbors[idx + 1] = neighs
        data["knn_neighbors"] = knn_neighbors

        grid_size = 0.01
        customer_cluster: Dict[int, Tuple[int, int]] = {}
        cluster_members: Dict[Tuple[int, int], List[int]] = {}
        for idx in range(1, N):
            lat, lon = nodes[idx]
            key = (int(lat / grid_size), int(lon / grid_size))
            customer_cluster[idx] = key
            cluster_members.setdefault(key, []).append(idx)
        data["customer_cluster"] = customer_cluster
        data["cluster_members"] = cluster_members

    return data


@dataclass
class RouteMetrics:
    """
    Métricas de una ruta individual.
    - load:      suma de tamaños (capacidad consumida) de la ruta
    - time_min:  tiempo total de la ruta (minutos), incluyendo servicio y retorno a depósito
    - dist:      distancia total (metros), incluyendo retorno a depósito
    - feasible:  factible respecto a ventanas de tiempo y jornada
    """
    load: float = 0.0
    time_min: float = 0.0
    dist: float = 0.0
    feasible: bool = True


@dataclass
class RouteSchedule:
    nodes: List[int]
    arrivals: List[float]
    departures: List[float]
    feasible: bool

def calculate_route_metrics(route: List[int], data: Dict) -> RouteMetrics:
    if not route:
        return RouteMetrics(load=0.0, time_min=0.0, dist=0.0, feasible=True)

    key = tuple(route)
    cached = _ROUTE_METRICS_CACHE.get(key)
    if cached is not None:
        return cached

    schedule = compute_route_schedule(route, data)
    load = float(sum(data["demand_size"][i] for i in route))
    dist = _route_distance_direct(route, data)

    if not schedule.feasible or schedule.departures[-1] > data["horizon_minutes"] + 1e-9:
        metrics = RouteMetrics(load=load, time_min=schedule.departures[-1], dist=dist, feasible=False)
        _ROUTE_METRICS_CACHE[key] = metrics
        return metrics

    metrics = RouteMetrics(load=load, time_min=schedule.departures[-1], dist=dist, feasible=True)
    _ROUTE_METRICS_CACHE[key] = metrics
    return metrics


def _route_distance_direct(route: List[int], data: Dict) -> float:
    if not route:
        return 0.0
    distM = data["distM"]
    total = distM[0, route[0]]
    for a, b in zip(route, route[1:]):
        total += distM[a, b]
    total += distM[route[-1], 0]
    return float(total)


def compute_route_schedule(route: List[int], data: Dict) -> RouteSchedule:
    key = tuple(route)
    cached = _ROUTE_SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached

    nodes_path = [0] + route + [0]
    arrivals: List[float] = [float(data["tw_start_min"][0])]
    departures: List[float] = [float(data["tw_start_min"][0])]

    feasible = True
    current_time = departures[0]
    timeM = data["timeM"]
    tw_start = data["tw_start_min"]
    tw_end = data["tw_end_min"]
    service = data["demand_srv_s"]

    for idx in range(1, len(nodes_path)):
        prev = nodes_path[idx - 1]
        node = nodes_path[idx]
        current_time += timeM[prev, node] / 60.0
        if current_time < tw_start[node]:
            current_time = tw_start[node]
        arrivals.append(current_time)
        if idx < len(nodes_path) - 1:
            if current_time > tw_end[node] + 1e-9:
                feasible = False
                arrivals.extend([current_time] * (len(nodes_path) - idx - 1))
                departures.extend([current_time] * (len(nodes_path) - idx))
                break
            current_time += service[node] / 60.0
        departures.append(current_time)

    schedule = RouteSchedule(nodes_path, arrivals, departures, feasible)
    _ROUTE_SCHEDULE_CACHE[key] = schedule
    return schedule


def simulate_insertion_schedule(
    schedule: RouteSchedule,
    route: List[int],
    customer: int,
    pos: int,
    data: Dict,
) -> Optional[RouteSchedule]:
    nodes_old = schedule.nodes
    arrivals_old = schedule.arrivals
    departures_old = schedule.departures

    nodes_new = nodes_old[:pos + 1] + [customer] + nodes_old[pos + 1:]
    arrivals_new = arrivals_old[:pos + 1]
    departures_new = departures_old[:pos + 1]

    current_time = departures_new[-1]
    timeM = data["timeM"]
    tw_start = data["tw_start_min"]
    tw_end = data["tw_end_min"]
    service = data["demand_srv_s"]

    prev = nodes_old[pos]
    current_time += timeM[prev, customer] / 60.0
    if current_time < tw_start[customer]:
        current_time = tw_start[customer]
    if current_time > tw_end[customer] + 1e-9:
        return None
    arrivals_new.append(current_time)
    current_time += service[customer] / 60.0
    departures_new.append(current_time)

    prev = customer
    for idx in range(pos + 1, len(nodes_old)):
        node = nodes_old[idx]
        current_time += timeM[prev, node] / 60.0
        if idx < len(nodes_old) - 1:
            if current_time < tw_start[node]:
                current_time = tw_start[node]
            if current_time > tw_end[node] + 1e-9:
                return None
            arrivals_new.append(current_time)
            current_time += service[node] / 60.0
            departures_new.append(current_time)
        else:
            arrivals_new.append(current_time)
            departures_new.append(current_time)
        prev = node

    if current_time > data["horizon_minutes"] + 1e-9:
        return None

    return RouteSchedule(nodes_new, arrivals_new, departures_new, True)
def route_feasible(route: List[int], cap: float, data: Dict) -> bool:
    """
    Factibilidad de una ruta para un vehículo:
    - Capacidad: suma(demand_size) <= cap
    - Ventanas de tiempo + jornada: usando calculate_route_metrics

    Devuelve True si pasa ambos chequeos.
    """
    if not route:
        return True

    # Capacidad
    load_ok = (sum(data["demand_size"][i] for i in route) <= cap + 1e-9)
    if not load_ok:
        return False

    # Tiempo (ventanas + jornada)
    return calculate_route_metrics(route, data).feasible



def pack_routes_to_vehicles(
    routes: List[List[int]],
    data: dict,
    check_time_feasibility: bool = False
) -> Tuple[bool, List[Optional[int]], List[float]]:
    """
    Asigna cada ruta a un vehículo disponible respetando capacidades (y, opcionalmente,
    factibilidad temporal con esa capacidad). Implementa un "best-fit" sin replicar el
    algoritmo completo de empaquetamiento, lo que reduce drásticamente el costo.

    Devuelve:
         - ok (bool): True si TODAS las rutas pudieron asignarse.
         - assign (List[Optional[int]]): índice de vehículo por ruta (en orden original).
         - remaining_caps (List[float]): capacidades remanentes (solo informativo).
    """

    tol = 1e-9
    vehicle_caps = list(map(float, data["vehicle_caps"]))
    n_routes = len(routes)
    n_veh = len(vehicle_caps)
    if n_routes > n_veh:
        return False, [None] * n_routes, vehicle_caps

    route_loads = [sum(data["demand_size"][i] for i in r) for r in routes]
    order = sorted(range(n_routes), key=lambda idx: route_loads[idx], reverse=True)

    remaining = sorted([(idx, cap) for idx, cap in enumerate(vehicle_caps)], key=lambda x: x[1])
    assign: List[Optional[int]] = [None] * n_routes

    for ridx in order:
        load = route_loads[ridx]
        chosen_pos = None
        for pos, (veh_idx, cap) in enumerate(remaining):
            if cap + tol < load:
                continue
            if check_time_feasibility and not route_feasible(routes[ridx], cap, data):
                continue
            chosen_pos = pos
            break
        if chosen_pos is None:
            return False, assign, vehicle_caps
        veh_idx, cap = remaining.pop(chosen_pos)
        assign[ridx] = veh_idx

    return True, assign, [cap for _, cap in remaining]



# --- Destroy 1: Shaw removal (clientes relacionados) ---
def destroy_shaw(
    sol: Solution,
    p: float,
    data: dict,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2,
) -> Tuple[Solution, List[int]]:
    """Destruye usando Shaw removal con candidatos limitados por vecindario y clústeres."""

    routes = [r[:] for r in sol.routes]
    all_clients = [c for r in routes for c in r]
    if not all_clients:
        return Solution(routes), []

    knn = data.get("knn_neighbors")
    customer_cluster = data.get("customer_cluster")
    cluster_members = data.get("cluster_members")
    tw_center_arr = data.get("tw_center_min")

    if not knn or not customer_cluster or not cluster_members or tw_center_arr is None:
        # Fallback al comportamiento original si faltan caches
        k = max(1, int(len(all_clients) * p))
        seed = random.choice(all_clients)

        def tw_center(i: int) -> float:
            return 0.5 * (data["tw_start_min"][i] + data["tw_end_min"][i])

        def rel(i: int, j: int) -> float:
            dij = data["distM"][i, j]
            tij = abs(tw_center(i) - tw_center(j))
            sij = abs(data["demand_size"][i] - data["demand_size"][j])
            return alpha * dij + beta * tij + gamma * sij

        removed = [seed]
        cand = set(all_clients) - {seed}
        while len(removed) < k and cand:
            j = min(cand, key=lambda x: min(rel(x, r) for r in removed))
            removed.append(j)
            cand.remove(j)

        removed_set = set(removed)
        new_routes = [[i for i in r if i not in removed_set] for r in routes]
        return Solution(new_routes), removed

    nodes = data["nodes"]
    distM = data["distM"]
    demand = data["demand_size"]

    cust_to_route: Dict[int, int] = {}
    route_centroids: Dict[int, Tuple[float, float]] = {}
    for ri, route in enumerate(routes):
        if not route:
            continue
        lat_sum = sum(nodes[c][0] for c in route)
        lon_sum = sum(nodes[c][1] for c in route)
        route_centroids[ri] = (lat_sum / len(route), lon_sum / len(route))
        for cust in route:
            cust_to_route[cust] = ri

    neighbor_cap = data.get("shaw_neighbor_limit", 20)
    removed: List[int] = []
    removed_set: set = set()
    remaining = set(all_clients)

    seed = random.choice(all_clients)
    removed.append(seed)
    removed_set.add(seed)
    remaining.discard(seed)

    def tw_center(idx: int) -> float:
        return float(tw_center_arr[idx])

    def centroid_distance(i: int, j: int) -> float:
        ri = cust_to_route.get(i)
        if ri is None:
            return 0.0
        cx, cy = route_centroids.get(ri, nodes[0])
        lat_j, lon_j = nodes[j]
        return math.hypot(lat_j - cx, lon_j - cy)

    def rel(i: int, j: int) -> float:
        dij = distM[i, j]
        tij = abs(tw_center(i) - tw_center(j))
        sij = abs(demand[i] - demand[j])
        cdist = centroid_distance(i, j)
        return alpha * dij + beta * tij + gamma * sij + delta * cdist

    k_remove = max(1, int(len(all_clients) * p))

    while len(removed) < k_remove and remaining:
        candidate_pool: set = set()
        for r in removed:
            candidate_pool.update(knn.get(r, [])[:neighbor_cap])
            cluster_key = customer_cluster.get(r)
            if cluster_key is not None:
                candidate_pool.update(cluster_members.get(cluster_key, []))
            route_idx = cust_to_route.get(r)
            if route_idx is not None:
                candidate_pool.update(routes[route_idx])

        candidate_pool &= remaining
        if not candidate_pool:
            candidate_pool = set(remaining)

        def score(cust: int) -> float:
            return min(rel(cust, r) for r in removed)

        chosen = min(candidate_pool, key=score)
        removed.append(chosen)
        removed_set.add(chosen)
        remaining.discard(chosen)

    new_routes = [[cust for cust in r if cust not in removed_set] for r in routes]
    return Solution(new_routes), removed

# --- Destroy 2: aleatorio (liviano, estable) ---
def destroy_random(sol: Solution, p: float = 0.15, data: dict = None) -> Tuple[Solution, List[int]]:
    """Elimina ~p de los clientes al azar (al menos 1)."""
    all_clients = [i for r in sol.routes for i in r]
    if not all_clients:
        return Solution([r[:] for r in sol.routes]), []
    k = max(1, int(len(all_clients) * p))
    removed_set = set(random.sample(all_clients, k))
    new_routes = [[i for i in r if i not in removed_set] for r in sol.routes]
    return Solution(new_routes), list(removed_set)

# --- Destroy 3: peores por contribución marginal (inteligente) ---
def destroy_worst(sol: Solution, p: float, data: dict) -> Tuple[Solution, List[int]]:
    """
    Remueve ~p de los clientes con mayor “costo marginal” en su posición actual:
      Δ = d(prev,c) + d(c,next) - d(prev,next)
    """
    all_clients = [i for r in sol.routes for i in r]
    if not all_clients:
        return Solution([r[:] for r in sol.routes]), []

    costs = []
    for route in sol.routes:
        if not route:
            continue
        path = [0] + route + [0]
        for i in range(1, len(path) - 1):
            p_, c, n_ = path[i-1], path[i], path[i+1]
            delta = data["distM"][p_, c] + data["distM"][c, n_] - data["distM"][p_, n_]
            costs.append((delta, c))

    costs.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(len(all_clients) * p))
    removed = [c for _, c in costs[:k]]
    removed_set = set(removed)
    new_routes = [[i for i in r if i not in removed_set] for r in sol.routes]
    return Solution(new_routes), removed

    def tw_center(i):
        return 0.5*(data["tw_start_min"][i] + data["tw_end_min"][i])
    def rel(i, j):
        dij = data["distM"][i, j]
        tij = abs(tw_center(i) - tw_center(j))
        sij = abs(data["demand_size"][i] - data["demand_size"][j])
        return alpha*dij + beta*tij + gamma*sij

    removed = [seed]
    cand = set(all_clients) - {seed}
    while len(removed) < k and cand:
        # elige el más relacionado con cualquiera ya removido
        j = min(cand, key=lambda x: min(rel(x, r) for r in removed))
        removed.append(j); cand.remove(j)

    removed_set = set(removed)
    new_routes = [[i for i in r if i not in removed_set] for r in routes]
    return Solution(new_routes), removed
    for r in routes:
        if acc < target:
            removed += r; acc += len(r)
        else:
            keep.append(r)
    return Solution(keep + [[] for _ in range(len(sol.routes)-len(keep))]), removed




# --- Repair: Regret-k (regret=3 por defecto) ---
def q_insert_regret(
    sol: Solution,
    removed: List[int],
    data: dict,
    weights: dict,
    lam: float = 0.0,
    k: int = 3,
    cache: Optional[EstheticCache] = None,
) -> Solution:
    """Versión Regret-k que evalúa costo y penalización estética de las inserciones."""
    routes = [r[:] for r in sol.routes]
    pen_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}
    cost_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}
    esthetic_cache = None
    if lam > 0.0:
        esthetic_cache = cache if cache is not None else EstheticCache(data)
    feas_cache: Dict[Tuple[Tuple[int, ...], float], bool] = {}
    top_m_insert = int(data.get("top_m_insertion", 4))

    def _route_feasible_cached(rt: List[int], cap: float) -> bool:
        key = (tuple(rt), float(cap))
        res = feas_cache.get(key)
        if res is None:
            res = route_feasible(rt, cap, data)
            feas_cache[key] = res
        return res

    def _evaluate(candidate_routes: List[List[int]]) -> Tuple[float, float, float]:
        active = [rt for rt in candidate_routes if rt]
        key = tuple(tuple(rt) for rt in active)
        cost = cost_cache.get(key)
        pen = 0.0
        need_pen = lam > 0.0

        sol_obj: Optional[Solution] = None
        if cost is None or (need_pen and key not in pen_cache):
            sol_obj = Solution([rt[:] for rt in active])

        if cost is None:
            sol_obj = sol_obj or Solution([rt[:] for rt in active])
            cost = solution_cost(sol_obj, data)
            cost_cache[key] = cost

        if need_pen and esthetic_cache is not None:
            if key in pen_cache:
                pen = pen_cache[key]
            else:
                sol_obj = sol_obj or Solution([rt[:] for rt in active])
                pen = aesthetic_penalty(
                    sol_obj,
                    data,
                    weights,
                    cache=esthetic_cache,
                    enable_metrics=True,
                )
                pen_cache[key] = pen

        return cost, pen, cost + lam * pen

    baseline_cost, baseline_pen, _ = _evaluate([r[:] for r in routes])

    max_cap = max(map(float, data["vehicle_caps"])) if data["vehicle_caps"] else 0.0

    while removed:
        route_schedules = {tuple(r): compute_route_schedule(r, data) for r in routes}
        route_loads = {tuple(r): sum(data["demand_size"][i] for i in r) for r in routes}

        options = []
        for cust in removed:
            candidates: List[Dict[str, object]] = []

            for ri, r in enumerate(routes):
                if top_m_insert <= 0:
                    limit = len(r) + 1
                else:
                    limit = top_m_insert

                pos_info: List[Tuple[float, int, int, int]] = []
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    lb = data["distM"][prev, cust] + data["distM"][cust, nxt] - data["distM"][prev, nxt]
                    pos_info.append((float(lb), pos, prev, nxt))

                pos_info.sort(key=lambda x: x[0])

                for lb, pos, prev, nxt in pos_info[:limit]:
                    new_load = route_loads[tuple(r)] + data["demand_size"][cust]
                    if new_load > max_cap + 1e-9:
                        continue
                    base_sched = route_schedules[tuple(r)]
                    new_sched = simulate_insertion_schedule(base_sched, r, cust, pos, data)
                    if new_sched is None:
                        continue
                    trial = r[:pos] + [cust] + r[pos:]
                    if not _routes_fit_fleet(routes, ri, trial, data):
                        continue

                    candidates.append({
                        'lb': lb,
                        'ri': ri,
                        'pos': pos,
                        'new_route': False,
                        'schedule': new_sched,
                    })

            if len(routes) < data["K"]:
                trial = [cust]
                if data["demand_size"][cust] <= max_cap + 1e-9:
                    sched_new = compute_route_schedule(trial, data)
                    if sched_new.feasible and sched_new.departures[-1] <= data["horizon_minutes"] + 1e-9 and _routes_fit_fleet(routes, len(routes), trial, data):
                        lb_new = data["distM"][0, cust] + data["distM"][cust, 0]
                        candidates.append({
                            'lb': float(lb_new),
                            'ri': len(routes),
                            'pos': 0,
                            'new_route': True,
                            'schedule': sched_new,
                        })

            if not candidates:
                continue

            # Filtrar por límite inferior y tomar top-M candidatos
            if any(x['lb'] > 1e6 for x in candidates):
                continue
            
            candidates.sort(key=lambda x: x['lb'])
            M = min(10, len(candidates))  # Ajustar M según tamaño de instancia
            shortlisted = candidates[:M]
            
            # Primera fase: evaluación rápida con estética fast
            evaluated = []
            for cand in shortlisted:
                trial_routes = routes.copy()
                if cand['new_route']:
                    new_route = [cust]
                    trial_routes.append(new_route)
                    route_idx = len(routes)
                else:
                    route_idx = int(cand['ri'])
                    pos = int(cand['pos'])
                    new_route = routes[route_idx][:pos] + [cust] + routes[route_idx][pos:]
                    trial_routes[route_idx] = new_route

                # Primera evaluación rápida
                trial_routes_active = [r for r in trial_routes if r]
                _, _, fast_obj = evaluate_cached(
                    routes_active=trial_routes_active,
                    fast=True,
                    data=data,
                    weights=weights,
                    lam=lam,
                    esthetic_cache=cache
                )
                evaluated.append({
                    'obj': fast_obj,
                    'trial_routes': trial_routes,
                    'ri': route_idx,
                    'route': new_route,
                    'new_route': cand['new_route']
                })

            if not evaluated:
                continue

            # Segunda fase: evaluación completa solo para top-k
            evaluated.sort(key=lambda x: x['obj'])
            top = evaluated[:k]
            for e in top:
                # Reevaluar los k mejores con estética completa
                _, _, full_obj = evaluate_cached(
                    routes_active=[r for r in e['trial_routes'] if r],
                    fast=False,
                    data=data,
                    weights=weights,
                    lam=lam,
                    esthetic_cache=cache
                )
                e['obj'] = full_obj
            
            # Reordenar por obj completo y tomar el mejor
            top.sort(key=lambda x: x['obj'])
            best = top[0]
            regret = sum(top[i]['obj'] - best['obj'] for i in range(1, len(top)))
            options.append({'cust': cust, 'regret': regret, 'ins': best})

        if not options:
            # No hay inserciones factibles para los clientes restantes
            print(f"[WARN] q_insert_regret: No se encontraron inserciones factibles para los clientes restantes: {removed}")
            break

        options.sort(key=lambda x: x['regret'], reverse=True)
        chosen = options[0]
        cust, ins = chosen['cust'], chosen['ins']
        if ins['new_route']:
            new_route = list(ins['route'])
            routes.append(new_route)
            _invalidate_cache_for([new_route])
        else:
            idx = int(ins['ri'])
            new_route = list(ins['route'])
            if idx < len(routes):
                old_route = routes[idx]
                routes[idx] = new_route
                _invalidate_cache_for([old_route, new_route])
            else:
                routes.append(new_route)
                _invalidate_cache_for([new_route])
        removed.remove(cust)

        baseline_cost = ins.get('cost', baseline_cost)
        baseline_pen = ins.get('pen', baseline_pen)

    return Solution(routes)
"""
# --- Búsqueda local (intra-ruta): 2-Opt con control de TW/jornada ---
def improve_route_with_2opt(route: List[int], data: dict) -> List[int]:
    if len(route) < 4:
        return route

    best = route[:]
    distM = data["distM"]
    improved = True
    min_gain = 1.0   # ignora mejoras menores a 1 metro

    while improved:
        improved = False
        path = [0] + best + [0]
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                A, B, C, D = path[i-1], path[i], path[j], path[j+1]
                gain = (distM[A, B] + distM[C, D]) - (distM[A, C] + distM[B, D])
                if gain > min_gain:   # mejora real y significativa
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    if new_route != best and calculate_route_metrics(new_route, data).feasible:
                        best = new_route
                        improved = True
                        break   # rompe for j
            if improved:
                break           # rompe for i
    return best
"""

def improve_route_with_2opt(route: List[int], data: dict, min_gain: float = 5) -> List[int]:
    """
    Búsqueda local basada en 2-opt dentro de una ruta.
    Usa best-improvement con umbral min_gain para ignorar micro-mejoras.
    """
    if len(route) < 4:
        return route

    distM = data["distM"]
    best = route[:]
    max_iterations = 100  # Límite de iteraciones
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        improved = False
        path = [0] + best + [0]
        best_gain = 0.0
        best_ij = None
        
        # Una sola pasada buscando la mejor mejora
        for i in range(1, len(path) - 2):
            Ai, Bi = path[i-1], path[i]
            for j in range(i + 1, len(path) - 1):
                Cj, Dj = path[j], path[j+1]
                # Ignorar arcos inválidos
                if (distM[Ai, Bi] >= 1e9 or distM[Cj, Dj] >= 1e9 or
                    distM[Ai, Cj] >= 1e9 or distM[Bi, Dj] >= 1e9):
                    continue
                gain = (distM[Ai, Bi] + distM[Cj, Dj]) - (distM[Ai, Cj] + distM[Bi, Dj])
                if gain > max(min_gain, best_gain):
                    candidate = best[:i] + best[i:j][::-1] + best[j:]
                    if calculate_route_metrics(candidate, data).feasible:
                        best_gain = gain
                        best_ij = (i, j)
        
        # Aplicar la mejor mejora encontrada
        if best_ij:
            i, j = best_ij
            best = best[:i] + best[i:j][::-1] + best[j:]
            improved = True
            _invalidate_cache_for([best])

    return best






# --- Búsqueda local (inter-ruta): swap entre rutas con control de capacidad + TW ---
def improve_with_swap(sol: Solution, data: dict) -> Tuple[Solution, bool]:
    """Mejora la solución intercambiando clientes entre rutas diferentes."""
    routes = [r[:] for r in sol.routes]
    best = Solution([r[:] for r in routes])
    best_val = sum(route_distance(r, data["distM"]) for r in routes if r)
    improved = False
    
    for i, r1 in enumerate(routes):
        if not r1:  # Skip empty routes
            continue
        for j, r2 in enumerate(routes):
            if j <= i or not r2:  # Skip same route and empty routes
                continue
            for a_idx, a in enumerate(r1):
                for b_idx, b in enumerate(r2):
                    # Try swapping a and b between routes
                    nr1 = r1[:a_idx] + [b] + r1[a_idx+1:]
                    nr2 = r2[:b_idx] + [a] + r2[b_idx+1:]
                    
                    # Check time windows feasibility
                    if not calculate_route_metrics(nr1, data).feasible or \
                       not calculate_route_metrics(nr2, data).feasible:
                        continue
                        
                    # Check vehicle capacity feasibility
                    ok, _, _ = pack_routes_to_vehicles(
                        [rr for k, rr in enumerate(routes) if k not in (i,j)] + [nr1, nr2],
                        data
                    )
                    if not ok:
                        continue
                        
                    # Calculate new total distance
                    val = sum(
                        route_distance(rr, data["distM"])
                        for k, rr in enumerate(routes)
                        if k not in (i,j)
                    )
                    val += route_distance(nr1, data["distM"]) + route_distance(nr2, data["distM"])
                    
                    # Update if improvement found
                    if val + 1e-6 < best_val:
                        routes[i], routes[j] = nr1, nr2
                        best = Solution([r[:] for r in routes])
                        best_val = val
                        improved = True
                        # Invalidar caché para las rutas modificadas
                        _invalidate_cache_for([nr1, nr2])
                        
    return best, improved

# --- Búsqueda local (inter-ruta): relocate con control de capacidad + TW ---
def improve_with_relocate(sol: Solution, data: dict) -> Tuple[Solution, bool]:
    """
    Mueve un cliente entre rutas si reduce distancia y mantiene factibilidad.
    Devuelve (nueva_sol, True) si aplicó un movimiento; de lo contrario (sol, False).
    """
    routes = [r[:] for r in sol.routes]
    distM = data["distM"]

    # Mapea capacidades (heurística: rutas más demandantes obtienen caps más grandes)
    route_dem = {i: sum(data["demand_size"][c] for c in r) for i, r in enumerate(routes)}
    sorted_ridx = sorted(route_dem, key=route_dem.get, reverse=True)
    sorted_caps = sorted(data["vehicle_caps"], reverse=True)
    route_caps = {ri: (sorted_caps[k] if k < len(sorted_caps) else 0.0) for k, ri in enumerate(sorted_ridx)}

    for r1_idx, r1 in enumerate(routes):
        if not r1:
            continue
        for pos1, cust in enumerate(r1):
            p1 = r1[pos1 - 1] if pos1 > 0 else 0
            n1 = r1[pos1 + 1] if pos1 < len(r1) - 1 else 0
            gain_remove = distM[p1, n1] - distM[p1, cust] - distM[cust, n1]

            for r2_idx, r2 in enumerate(routes):
                if r2_idx == r1_idx:
                    continue
                for pos2 in range(len(r2) + 1):
                    p2 = r2[pos2 - 1] if pos2 > 0 else 0
                    n2 = r2[pos2] if pos2 < len(r2) else 0
                    delta_insert = distM[p2, cust] + distM[cust, n2] - distM[p2, n2]

                    if gain_remove + delta_insert < -1e-9:
                        new_r1 = r1[:pos1] + r1[pos1+1:]
                        new_r2 = r2[:pos2] + [cust] + r2[pos2:]

                        cap1 = route_caps.get(r1_idx, 0.0)
                        cap2 = route_caps.get(r2_idx, 0.0)

                        if route_feasible(new_r1, cap1, data) and route_feasible(new_r2, cap2, data):
                            old_r1, old_r2 = routes[r1_idx], routes[r2_idx]
                            routes[r1_idx], routes[r2_idx] = new_r1, new_r2

                            candidate_active = [r for r in routes if r]
                            ok, _, _ = pack_routes_to_vehicles(candidate_active, data)
                            if ok:
                                # Invalida el caché para las rutas modificadas
                                _invalidate_cache_for([new_r1, new_r2])
                                return Solution(candidate_active), True

                            routes[r1_idx], routes[r2_idx] = old_r1, old_r2
    return sol, False


def route_distance(route: List[int], distM: np.ndarray) -> float:
    if not route:
        return 0.0
    key = tuple(route)
    cached = _ROUTE_DIST_CACHE.get(key)
    if cached is not None:
        return cached
    d = distM[0, route[0]]
    for a, b in zip(route, route[1:]):
        d += distM[a, b]
    d += distM[route[-1], 0]
    d = float(d)
    _ROUTE_DIST_CACHE[key] = d
    return d

def solution_cost(sol, data) -> float:
    active_routes = [r for r in sol.routes if r]
    m = len(active_routes)
    
    # Si requieres estrictamente 1:1, verificar número máximo de rutas
    if m > data["K"]:
        return float("inf")
        
    # Costo fijo por ruta
    base_cost = m * data["fixed_route_cost"]
    
    # Costo variable por distancia
    total_dist = sum(route_distance(r, data["distM"]) for r in active_routes)
    var_cost = total_dist * data["cost_per_meter"]
    
    ok, assign_idx, _ = pack_routes_to_vehicles(active_routes, data)
    if not ok:
        return float("inf")
    veh_caps = list(map(float, data["vehicle_caps"]))
    cap_cost = 0.0
    for ridx, veh_idx in enumerate(assign_idx):
        if veh_idx is None:
            return float("inf")
        cap_cost += veh_caps[veh_idx] * float(data["cost_per_cap"])

    return base_cost + var_cost + cap_cost


def nearest_neighbor_seed(data: Dict) -> Solution:
    """
    Construye una solución inicial factible con heurística de vecino más cercano,
    respetando capacidad y TW/jornada.
    """
    N = data["N"]
    K = data["K"]
    caps = list(sorted(map(float, data["vehicle_caps"]), reverse=True))
    unassigned = set(range(1, N))
    routes: List[List[int]] = []

    for cap in caps:
        if not unassigned:
            break
        route: List[int] = []
        while True:
            best = None
            best_d = float("inf")
            prev = 0 if not route else route[-1]
            for c in list(unassigned):
                # chequear capacidad y tiempo al insertar al final
                trial = route + [c]
                if not route_feasible(trial, cap, data):
                    continue
                d = data["distM"][prev, c]
                if d < best_d:
                    best, best_d = c, d
            if best is None:
                break
            route.append(best)
            unassigned.remove(best)
        if route:
            routes.append(route)

    # Si quedan clientes sin asignar, crea rutas unitarias mientras queden "vehículos lógicos"
    for c in list(unassigned):
        if len(routes) < K and route_feasible([c], caps[min(len(routes), len(caps)-1)], data):
            routes.append([c])
            unassigned.remove(c)

    return Solution(routes=routes)


def solomon_seed_solution(
    data: Dict,
    alpha: float = 0.6,
    lambda_closeness: float = 1.0,
) -> Solution:
    """
    Implementación del heurístico Solomon I1 para generar una solución inicial.
    Retorna una Solution cuyas rutas respetan capacidad y ventanas de tiempo.
    """
    distM = data["distM"]
    demand = data["demand_size"]
    vehicle_caps = list(map(float, data["vehicle_caps"]))
    unassigned = set(range(1, data["N"]))
    routes: List[List[int]] = []

    if not vehicle_caps:
        return Solution(routes=[])

    def route_load(route: List[int]) -> float:
        return float(sum(demand[i] for i in route))

    while unassigned and vehicle_caps:
        cap_lim = vehicle_caps.pop(0)

        feasible_seeds = [
            u
            for u in unassigned
            if demand[u] <= cap_lim + 1e-9 and route_feasible([u], cap_lim, data)
        ]
        if not feasible_seeds:
            continue

        seed = max(feasible_seeds, key=lambda u: distM[0, u])
        route = [seed]
        metrics = calculate_route_metrics(route, data)
        if not metrics.feasible:
            unassigned.remove(seed)
            continue

        cap_used = route_load(route)
        unassigned.remove(seed)

        while True:
            candidates = []
            for cust in list(unassigned):
                if cap_used + demand[cust] > cap_lim + 1e-9:
                    continue

                insertions = []
                for pos in range(len(route) + 1):
                    trial = route[:pos] + [cust] + route[pos:]
                    if not route_feasible(trial, cap_lim, data):
                        continue

                    trial_metrics = calculate_route_metrics(trial, data)
                    prev_node = route[pos - 1] if pos > 0 else 0
                    next_node = route[pos] if pos < len(route) else 0

                    delta_dist = (
                        distM[prev_node, cust]
                        + distM[cust, next_node]
                        - distM[prev_node, next_node]
                    )
                    delta_time_min = max(0.0, trial_metrics.time_min - metrics.time_min)
                    c1 = delta_dist + alpha * (delta_time_min * 60.0)  # penaliza retrasos
                    insertions.append((c1, pos, trial, trial_metrics))

                if not insertions:
                    continue

                best_c1, best_pos, best_trial, best_metrics = min(insertions, key=lambda x: x[0])
                candidates.append((cust, best_pos, best_c1, best_trial, best_metrics))

            if not candidates:
                break

            cust, pos, c1, best_trial, best_metrics = max(
                candidates,
                key=lambda x: lambda_closeness * distM[0, x[0]] - x[2],
            )

            route = best_trial
            metrics = best_metrics
            cap_used += demand[cust]
            unassigned.remove(cust)

        routes.append(route)
    if unassigned:
        print(f"[WARN] Solomon I1 dejó sin asignar {len(unassigned)} clientes.")

    return Solution(routes=[r[:] for r in routes])


def _all_clients_assigned(sol: Solution, data: Dict) -> bool:
    assigned = {i for r in sol.routes for i in r}
    return assigned == set(range(1, data["N"]))

def alns_single_run(
    data: Dict,
    weights,
    lam: float,
    iters: int = 2000,
    seed: int = 0,
    destroy_p: float = 0.15,
    T_start: float = 1.0,
    cooling_rate: float = 0.997,
    T_min = 1e-6,
    reaction: float = 0.2,
    segment_size: int = 32,  # Tamaño del segmento para actualización de scores (reducido de 50 a 32)
    use_fast_esthetics: bool = False
) -> Solution:
    # --- 1. INICIALIZACIÓN ---
    random.seed(seed)
    np.random.seed(seed)

    _EVAL_CACHE.clear()
    use_visual_metrics = lam > 1e-9
    esthetic_cache = EstheticCache(data) if use_visual_metrics else None
    solomon_seed: Optional[Solution] = None
    solomon_eval: Optional[Tuple[float, float, float]] = None
    baseline_solution: Optional[Solution] = None
    baseline_value: Optional[float] = None


# Sólo activa si quieres analizar tiempos o generar reportes detallados
    #activate_logging(lam)
    try:

        def eval_cost(sol: Solution) -> float:
            if sol.cost is None: sol.cost = solution_cost(sol, data)
            return sol.cost

        # VAMOS A MODIFICAR ESTA FUNCIÓN INTERNA
        def eval_est_full(sol: Solution) -> float:
            if not use_visual_metrics:
                sol.est_penalty = 0.0
                return 0.0
            if sol.est_penalty is None:
                sol.est_penalty = aesthetic_penalty(
                    sol,
                    data,
                    weights,
                    cache=esthetic_cache,
                    enable_metrics=True,
                )
            return sol.est_penalty

        def eval_est_fast(sol: Solution) -> float:
            if not use_visual_metrics:
                return 0.0
            return aesthetic_penalty_fast(sol, data)

        def evaluate(sol: Solution, fast: bool) -> float:
            cost = eval_cost(sol)
            if not use_visual_metrics:
                return cost
            pen = eval_est_fast(sol) if fast else eval_est_full(sol)
            return cost + lam * pen

        # Construir solución Solomon I1 como referencia global
        try:
            candidate = solomon_seed_solution(data)
            if _all_clients_assigned(candidate, data):
                sol_cost, sol_pen, sol_obj = evaluate_cached(
                    routes_active=[r for r in candidate.routes if r],
                    fast=False,
                    data=data,
                    weights=weights,
                    lam=lam,
                    esthetic_cache=esthetic_cache,
                )
                candidate.cost = sol_cost
                candidate.est_penalty = sol_pen if use_visual_metrics else 0.0
                solomon_seed = candidate
                solomon_eval = (sol_cost, candidate.est_penalty, sol_obj)
                baseline_solution = Solution(
                    routes=[r[:] for r in candidate.routes],
                    cost=candidate.cost,
                    est_penalty=candidate.est_penalty,
                )
                baseline_value = sol_obj
            else:
                print("[WARN] Solomon I1 generó una solución infactible (quedaron clientes sin asignar).")
        except Exception as exc:
            print(f"[WARN] Solomon I1 no pudo generar semilla inicial: {exc}")

        # Solución inicial y la mejor solución encontrada
        nn_seed = nearest_neighbor_seed(data)
        if not _all_clients_assigned(nn_seed, data):
            raise ValueError("La solución inicial no cubre a todos los clientes.")

        nn_cost, nn_pen, nn_obj = evaluate_cached(
            routes_active=[r for r in nn_seed.routes if r],
            fast=False,
            data=data,
            weights=weights,
            lam=lam,
            esthetic_cache=esthetic_cache
        )
        nn_seed.cost = nn_cost
        nn_seed.est_penalty = nn_pen if use_visual_metrics else 0.0

        curr = Solution(routes=[r[:] for r in nn_seed.routes], cost=nn_seed.cost, est_penalty=nn_seed.est_penalty)
        f_curr = nn_obj

        if solomon_eval is not None and solomon_eval[2] + 1e-9 < f_curr:
            curr = Solution(
                routes=[r[:] for r in solomon_seed.routes],
                cost=solomon_eval[0],
                est_penalty=solomon_eval[1],
            )
            f_curr = solomon_eval[2]

        best = Solution(routes=[r[:] for r in curr.routes], cost=curr.cost, est_penalty=curr.est_penalty)
        f_best = f_curr

        if baseline_solution is None or baseline_value is None or f_curr + 1e-9 < baseline_value:
            baseline_solution = Solution(routes=[r[:] for r in curr.routes], cost=curr.cost, est_penalty=curr.est_penalty)
            baseline_value = f_curr
    
        base_scale = max(abs(f_curr), 1.0)
        T = max(T_min, T_start * base_scale)

        destroy_ops = {
            'random': {'op': destroy_random, 'scores': [1.0] * segment_size, 'uses': [1] * segment_size},
            'worst':  {'op': lambda s,p,data: destroy_worst(s,p,data), 'scores': [1.0] * segment_size, 'uses': [1] * segment_size},
            'shaw':   {'op': lambda s,p,data: destroy_shaw(s,p,data), 'scores': [1.0] * segment_size, 'uses': [1] * segment_size},
        }

        REWARD_BEST, REWARD_BETTER, REWARD_ACCEPTED = 3.0, 2.0, 1.0
        stage_threshold = float(data.get("stage_threshold", 0.0))

        # --- 2. BUCLE PRINCIPAL DEL ALGORITMO ---
        # Inicialización para p adaptativo y estancamiento
        last_improve_it = 0
        p_min, p_max = 0.05, 0.40  # rango de destrucción adaptativa
        
        for it in tqdm(range(iters), desc=f"ALNS (λ={lam})", leave=False):
            # Adaptar p según temperatura y estancamiento
            stall = max(0, it - last_improve_it)
            heat = min(1.0, T / (T_start * base_scale + 1e-9))
            p_now = p_min + (p_max - p_min) * max(heat, min(1.0, stall/200))
            
            # --- a. Selección de Operador Adaptativo basado en Segmentos ---
            # Determinar el segmento actual basado en el progreso
            current_segment = min(segment_size - 1, int(it * segment_size / iters))
            
            # Calcular pesos usando scores del segmento actual
            op_weights = [max(1e-9, d['scores'][current_segment]) / max(1, d['uses'][current_segment]) 
                         for d in destroy_ops.values()]
            
            suma = sum(op_weights)
            if not np.isfinite(suma) or suma <= 0.0:
                op_weights = [1.0] * len(destroy_ops)  # fallback uniforme

            chosen_name = random.choices(list(destroy_ops.keys()), weights=op_weights, k=1)[0]
            chosen_op_data = destroy_ops[chosen_name]
            chosen_op_data['uses'][current_segment] += 1


            # --- b. Destrucción y Reparación ---
            op_args = {'p': p_now, 'data': data}  # Usar p adaptativo
            destroyed, removed = chosen_op_data['op'](curr, **op_args)
        
            # ¡Llama a la nueva función de reparación con los argumentos extra!
            cand = q_insert_regret(destroyed, removed, data, weights, lam=lam, k=3, cache=esthetic_cache)

            # --- c. Búsqueda Local Intensiva (VNS) ---
            rounds = 0
            while rounds < 3:  # Máximo 3 rondas de mejora
                rounds += 1
                improved = False
                
                # 1. Primero 2-opt intra-ruta
                initial_routes = str(cand.routes)
                cand.routes = [improve_route_with_2opt(r, data) for r in cand.routes if r]
                if str(cand.routes) != initial_routes:
                    improved = True
                
                # 2. Luego relocate inter-ruta
                cand, moved = improve_with_relocate(cand, data)
                if moved:
                    improved = True
                
                # 3. Después swap inter-ruta
                cand, moved = improve_with_swap(cand, data)
                if moved:
                    improved = True
                    
                # 4. Finalmente otra ronda de 2-opt
                initial_routes = str(cand.routes)
                cand.routes = [improve_route_with_2opt(r, data) for r in cand.routes if r]
                if str(cand.routes) != initial_routes:
                    improved = True
                    
                # Si no hubo mejoras en esta ronda, terminamos
                if not improved:
                    break
            # --- d. Evaluación y Criterio de Aceptación (SA) ---
            # SOLO EVALUAMOS Y CONSIDERAMOS CANDIDATOS QUE SEAN 100% COMPLETOS
            if _all_clients_assigned(cand, data):
                if lam <= 0.0:
                    cand_cost = solution_cost(cand, data)
                    cand.cost = cand_cost
                    cand.est_penalty = 0.0
                    cand_pen_effective = 0.0
                    cand_obj = cand_cost
                else:
                    fast_cost, fast_pen, fast_obj = evaluate_cached(
                        routes_active=[r for r in cand.routes if r],
                        fast=True,
                        data=data,
                        weights=weights,
                        lam=lam,
                        esthetic_cache=esthetic_cache
                    )
                    cand_cost = fast_cost
                    cand_pen_effective = fast_pen
                    cand_obj = fast_obj

                    need_full = False
                    if stage_threshold <= 0.0:
                        need_full = (fast_obj - f_curr < 0) or random.random() < 0.1
                    else:
                        if fast_obj <= f_best + stage_threshold:
                            need_full = (fast_obj - f_curr < 0) or random.random() < 0.1

                    if need_full:
                        full_cost, full_pen, full_obj = evaluate_cached(
                            routes_active=[r for r in cand.routes if r],
                            fast=False,
                            data=data,
                            weights=weights,
                            lam=lam,
                            esthetic_cache=esthetic_cache
                        )
                        cand_cost = full_cost
                        cand_pen_effective = full_pen
                        cand_obj = full_obj

                    cand.cost = cand_cost
                    cand.est_penalty = cand_pen_effective

                delta = cand_obj - f_curr

                reward = 0.0

                if cand_obj < f_best - 1e-9:
                    best = Solution(
                        routes=[r[:] for r in cand.routes],
                        cost=cand_cost,
                        est_penalty=cand_pen_effective,
                    )
                    f_best = cand_obj
                    curr = Solution(routes=[r[:] for r in cand.routes], cost=cand_cost, est_penalty=cand_pen_effective)
                    f_curr = cand_obj
                    reward = REWARD_BEST
                    last_improve_it = it
                    if baseline_solution is None or baseline_value is None or cand_obj + 1e-9 < baseline_value:
                        baseline_solution = Solution(routes=[r[:] for r in best.routes], cost=best.cost, est_penalty=best.est_penalty)
                        baseline_value = cand_obj
                else:
                    if delta < -1e-9:
                        curr = Solution(routes=[r[:] for r in cand.routes], cost=cand_cost, est_penalty=cand_pen_effective)
                        f_curr = cand_obj
                        reward = REWARD_BETTER
                        last_improve_it = it
                    elif delta <= 1e-9:
                        curr = Solution(routes=[r[:] for r in cand.routes], cost=cand_cost, est_penalty=cand_pen_effective)
                        f_curr = cand_obj
                        reward = REWARD_ACCEPTED * 0.5
                    else:
                        temp = max(1e-9, T)
                        prob = math.exp(-delta / temp)
                        if random.random() < prob:
                            curr = Solution(routes=[r[:] for r in cand.routes], cost=cand_cost, est_penalty=cand_pen_effective)
                            f_curr = cand_obj
                            reward = REWARD_ACCEPTED
            
                current_segment = min(segment_size - 1, int(it * segment_size / iters))
                chosen_op_data['scores'][current_segment] = (1 - reaction) * chosen_op_data['scores'][current_segment] + reaction * reward
        
            # --- e. Enfriamiento (fuera del if para que siempre ocurra) ---
            T = max(T_min, T * cooling_rate)

        if baseline_solution is not None and baseline_value is not None:
            best_cost = best.cost if best.cost is not None else solution_cost(best, data)
            if best.cost is None:
                best.cost = best_cost
            if use_visual_metrics:
                if best.est_penalty is None:
                    best.est_penalty = aesthetic_penalty(
                        best,
                        data,
                        weights,
                        cache=esthetic_cache,
                        enable_metrics=True,
                    )
                best_penalty = best.est_penalty
            else:
                best_penalty = 0.0
            best_obj = best_cost + lam * best_penalty
            if baseline_value + 1e-9 < best_obj:
                best = Solution(
                    routes=[r[:] for r in baseline_solution.routes],
                    cost=baseline_solution.cost,
                    est_penalty=baseline_solution.est_penalty,
                )

        return best

    finally:
        deactivate_logging()

def improve_with_swap(sol: Solution, data: dict):
    routes = [r[:] for r in sol.routes]
    best = Solution([r[:] for r in routes]); best_val = sum(route_distance(r, data["distM"]) for r in routes)
    for i, r1 in enumerate(routes):
        for j, r2 in enumerate(routes):
            if j <= i: continue
            for a_idx, a in enumerate(r1):
                for b_idx, b in enumerate(r2):
                    nr1 = r1[:a_idx] + [b] + r1[a_idx+1:]
                    nr2 = r2[:b_idx] + [a] + r2[b_idx+1:]
                    # validación tiempo (y capacidad vía pack)
                    if not calculate_route_metrics(nr1, data).feasible or not calculate_route_metrics(nr2, data).feasible:
                        continue
                    ok, _, _ = pack_routes_to_vehicles([rr for k, rr in enumerate(routes) if k not in (i,j)] + [nr1, nr2], data)
                    if not ok: continue
                    val = (route_distance(nr1, data["distM"]) + route_distance(nr2, data["distM"]))
                    val += sum(route_distance(routes[k], data["distM"]) for k in range(len(routes)) if k not in (i,j))
                    if val + 1e-6 < best_val:
                        routes[i], routes[j] = nr1, nr2
                        best = Solution([r[:] for r in routes]); best_val = val
    return best, (best_val < sum(route_distance(r, data["distM"]) for r in sol.routes))


# ----------  MULTI-OBJETIVO POR PONDERACIÓN ----------
def run_pareto(
    data,
    weights,
    lambdas=(0.0, 0.5, 1.0, 2.0, 5.0),
    iters=2000,
    seed=42,
    use_fast_esthetics=False,
    verbose: bool = True,
):
    sols = []
    for lam in lambdas:
        s = alns_single_run(data, weights, lam=lam, iters=iters, seed=seed, use_fast_esthetics=use_fast_esthetics)
        sols.append((lam, s.cost, s.est_penalty, s))
        if verbose:
            print(f"Lambda: {lam}, Costo: {s.cost:.2f}, Penalidad: {s.est_penalty:.2f}")
    # quitar dominadas (min costo, min estética)
    sols_sorted = sorted(sols, key=lambda x: (x[1], x[2]))
    pareto = []
    best_est = math.inf
    for lam, c, e, s in sols_sorted:
        if e < best_est - 1e-9:
            pareto.append((lam, c, e, s))
            best_est = e
    return sols, pareto



def _route_polyline(route, nodes):
    """Devuelve listas x (long) e y (lat) para plotear la ruta con depot al inicio/fin."""
    seq = [0] + route + [0]
    xs = [float(nodes[i][1]) for i in seq]  # longitudes
    ys = [float(nodes[i][0]) for i in seq]  # latitudes
    return xs, ys

def plot_solution_on_map(sol, data, title="Rutas"):
    plt.figure(figsize=(7, 7))
    # depot
    depot_lat, depot_lon = data["nodes"][0]
    plt.scatter([depot_lon], [depot_lat], marker="s", s=80, label="Depot")
    # rutas
    for idx, r in enumerate([rr for rr in sol.routes if rr]):
        xs, ys = _route_polyline(r, data["nodes"])
        plt.plot(xs, ys, linewidth=1.5, label=f"Ruta {idx+1}")
        # nodos de la ruta
        plt.scatter(xs[1:-1], ys[1:-1], s=12)
    plt.title(title)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Paleta para rutas (se cicla si hay más rutas que colores)
ROUTE_COLORS = [
    "red","blue","green","purple","orange","darkred","lightblue",
    "lightgreen","pink","cadetblue","darkpurple","gray","black"
]

def _add_route_legend(m, n_routes):
    """Inyecta una leyenda simple (Ruta i → color) dentro del mapa Folium."""
    items = []
    for i in range(n_routes):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        items.append(f'<div><span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;"></span>Ruta {i+1}</div>')
    html = f"""
    <div style="
        position: fixed; 
        bottom: 20px; left: 20px; z-index: 9999; 
        background: white; padding: 10px 12px; 
        border: 1px solid #999; border-radius: 6px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        font-size: 12px;">
      <div style="font-weight:600; margin-bottom:6px;">Leyenda</div>
      {''.join(items)}
    </div>
    """
    class FloatLegend(MacroElement):
        def __init__(self, html):
            super().__init__()
            self._template = Template(f"""
            {{% macro script(this, kwargs) %}}
            var legend = $(`{html}`);
            $(legend).appendTo(maps[Object.keys(maps)[0]]);
            {{% endmacro %}}
            """)
    m.get_root().html.add_child(folium.Element(html))  # simple: incrusta el HTML

def folium_solution_map(sol, data, outfile="mapa.html", tiles="OpenStreetMap", use_roads=None):
    depot_lat, depot_lon = data["nodes"][0]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12, tiles=tiles)

    folium.Marker([depot_lat, depot_lon],
                  icon=folium.Icon(color="black", icon="home"),
                  tooltip="Depot").add_to(m)

    # decide si usar OSRM (por calles) o líneas rectas
    use_roads_flag = USE_ROAD_GEOMETRY if use_roads is None else use_roads

    active_routes = [r for r in sol.routes if r]
    for idx, r in enumerate(active_routes):
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]

        if use_roads_flag:
            shape = _route_shape_on_roads(r, data)
        else:
            shape = [[data["nodes"][i][0], data["nodes"][i][1]] for i in ([0] + r + [0])]

        folium.PolyLine(shape, weight=3, color=color, tooltip=f"Ruta {idx+1}").add_to(m)

        for j in r:
            lat, lon = data["nodes"][j]
            folium.CircleMarker([lat, lon], radius=2, fill=True, color=color).add_to(m)

    _add_route_legend(m, len(active_routes))
    m.save(str(outfile))
    if use_roads_flag:
        _save_shape_cache()  # solo guarda cache si usamos OSRM
    return outfile


def export_pareto_maps(data, pareto, output_dir="soluciones"):
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for (lam, costo, pen, sol) in pareto:
        safe_lam = str(lam).replace('.', '_')
        base_filename = f"mapa_pareto_lambda_{safe_lam}__cost_{int(costo)}__pen_{int(round(pen))}.html"
        out_path = out_dir / base_filename
        folium_solution_map(sol, data, outfile=out_path)
        outputs.append(out_path)
    return outputs

def export_pareto_dual_maps(data, pareto, output_dir="soluciones"):
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for (lam, costo, pen, sol) in pareto:
        safe_lam = str(lam).replace('.', '_')
        base = f"lambda_{safe_lam}__cost_{int(costo)}__pen_{int(round(pen))}"

        out_roads   = out_dir / f"mapa_{base}__roads.html"
        out_straight= out_dir / f"mapa_{base}__straight.html"

        folium_solution_map(sol, data, outfile=out_roads,    use_roads=True)
        folium_solution_map(sol, data, outfile=out_straight, use_roads=False)

        outputs.extend([out_roads, out_straight])
    return outputs




def run_single_experiment(
    base_seed: int,
    data: dict,
    weights: dict,
    lambdas: list,
    iters: int,
    *,
    verbose: bool = True,
) -> list:
    """
    Ejecuta una corrida completa del algoritmo para todos los lambdas con una semilla base.
    """
    if verbose:
        print(f"\n--- INICIANDO CORRIDA CON SEED BASE = {base_seed} ---")

    # Ejecuta run_pareto, que a su vez llama a alns_single_run para cada lambda
    sols, pareto = run_pareto(
        data,
        weights,
        lambdas=lambdas,
        iters=iters,
        seed=base_seed,
        use_fast_esthetics=False,
        verbose=verbose,
    )

    if verbose:
        print(f"--- CORRIDA CON SEED BASE = {base_seed} FINALIZADA ---")
    return pareto


def validar_solucion_final(sol: Solution, data: dict, nombre_solucion: str, *, verbose: bool = True):
    """
    Verifica si una solución final atiende a todos los clientes y reporta si faltan.
    """
    clientes_requeridos = set(range(1, data["N"]))
    clientes_atendidos = {c for r in sol.routes for c in r}

    clientes_faltantes = clientes_requeridos - clientes_atendidos

    if verbose:
        if not clientes_faltantes:
            print(f"Validación OK para '{nombre_solucion}': Todos los {len(clientes_requeridos)} clientes están atendidos.")
        else:
            print(f"¡ERROR DE VALIDACIÓN en '{nombre_solucion}'!")
            print(f"  - Clientes atendidos: {len(clientes_atendidos)} de {len(clientes_requeridos)}")
            print(f"  - Faltan {len(clientes_faltantes)} clientes. IDs: {sorted(list(clientes_faltantes))}")

    return not clientes_faltantes
