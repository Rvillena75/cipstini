
import math
import random
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import folium
from branca.element import MacroElement
from jinja2 import Template

import os

import time

from sklearn.cluster import KMeans
#agregar wea paq parta con varias seeds diferentes

from dataclasses import dataclass

from tqdm import tqdm

# --- Polilíneas por calle con OSRM (visualización) ---
import json, requests

USE_ROAD_GEOMETRY = True         # apágalo si quieres volver a líneas rectas
_ROAD_SHAPE_CACHE_PATH = "route_shapes_cache.json"
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

import math
import requests

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



def precalculate_clusters(data: dict, n_clusters: Optional[int] = None) -> np.ndarray:
    """
    Agrupa los nodos de clientes en clusters geográficos usando K-Means.
    El depot (nodo 0) no se incluye en el clustering.

    Args:
        data: El diccionario de datos cargado.
        n_clusters: El número de clusters a crear. Si es None, se usará el número de vehículos (K).

    Returns:
        Un array de numpy donde el índice `i` contiene el ID del cluster para el nodo `i`.
        El depot (índice 0) tendrá un ID de -1.
    """
    if n_clusters is None:
        n_clusters = data["K"]

    # Extraemos las coordenadas solo de los clientes (ignorando el depot en el índice 0)
    client_nodes = np.array(data["nodes"][1:])
    
    if len(client_nodes) == 0 or len(client_nodes) < n_clusters:
        print("[WARN] No hay suficientes clientes para el clustering. Se omite.")
        cluster_map = np.full(data["N"], -1, dtype=int)
        return cluster_map

    print(f"Generando {n_clusters} zonas virtuales (clusters) para {data['N']-1} clientes...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(client_nodes)
    
    # Creamos el mapa de clusters. El ID del cluster para el cliente `i` (índice 1..N-1)
    # se encontrará en kmeans.labels_[i-1].
    cluster_map = np.zeros(data["N"], dtype=int)
    cluster_map[0] = -1  # Asignamos -1 al depot para identificarlo fácilmente
    cluster_map[1:] = kmeans.labels_
    
    print("Zonas virtuales generadas.")
    return cluster_map



_ROUTE_DIST_CACHE = {}

@dataclass
class Solution:
    routes: List[List[int]]
    cost: Optional[float] = None
    est_penalty: Optional[float] = None

def preparar_directorio_soluciones(dir_path: str) -> None:
    """
    Crea el directorio si no existe y borra todos los archivos .html que contenga.
    Deja logs claros por consola.
    """
    os.makedirs(dir_path, exist_ok=True)

    for filename in os.listdir(dir_path):
        if filename.endswith(".html"):
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
                print(f" - Borrado: {filename}")
            except Exception as e:
                print(f"Error al borrar {file_path}: {e}")




DEFAULT_I1_DIR = os.path.join("Datos P5", "i1")
DEFAULT_COSTS  = os.path.join("Datos P5", "costs.csv")

def load_data(i1_dir: str = DEFAULT_I1_DIR, costs_path: str = DEFAULT_COSTS) -> Dict:
    """
    Carga overview/demands/vehicles/distances/times/costs, arma matrices NxN y
    preprocesa ventanas de tiempo y servicio. Devuelve un diccionario con todos
    los insumos que el resto del código espera.
    """

    # --- Leer CSVs
    df_over = pd.read_csv(os.path.join(i1_dir, "overview.csv"))
    df_dem  = pd.read_csv(os.path.join(i1_dir, "demands.csv"))
    df_veh  = pd.read_csv(os.path.join(i1_dir, "vehicles.csv"))
    df_dist = pd.read_csv(os.path.join(i1_dir, "distances.csv"))
    df_time = pd.read_csv(os.path.join(i1_dir, "times.csv"))
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

    return {
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


def calculate_route_metrics(route: List[int], data: Dict) -> RouteMetrics:
    """
    Calcula métricas completas para una ruta:
    - Aplica tiempos de viaje y agrega tiempos de servicio.
    - Respeta ventanas de tiempo (espera si llega antes; infactible si llega después).
    - Agrega retorno al depósito y valida contra la jornada (horizon_minutes).
    - Devuelve RouteMetrics con load, tiempo total (min), distancia total (m) y factibilidad.

    Convenciones:
    - Nodo 0 es el depósito.
    - 'route' NO incluye el depósito; es una secuencia de clientes (1..N-1).
    - data debe contener: 'demand_size', 'demand_srv_s', 'tw_start_min', 'tw_end_min',
      'timeM', 'distM', 'horizon_minutes'.
    """
    if not route:
        return RouteMetrics(load=0.0, time_min=0.0, dist=0.0, feasible=True)

    # Carga total
    load = float(sum(data["demand_size"][i] for i in route))

    # Acumuladores de tiempo (en minutos) y distancia (en metros)
    time_min = 0.0
    dist = 0.0
    prev = 0  # partimos en depósito

    for node in route:
        # Viaje depósito/cliente o cliente/cliente
        time_min += data["timeM"][prev, node] / 60.0
        dist     += data["distM"][prev, node]

        # Si llegamos antes, esperamos al inicio de la ventana
        if time_min < data["tw_start_min"][node]:
            time_min = data["tw_start_min"][node]

        # Si llegamos después del fin de ventana, la ruta es infactible
        if time_min > data["tw_end_min"][node] + 1e-9:
            return RouteMetrics(load=load, time_min=time_min, dist=dist, feasible=False)

        # Servicio en el nodo
        time_min += data["demand_srv_s"][node] / 60.0
        prev = node

    # Retorno al depósito
    time_min += data["timeM"][prev, 0] / 60.0
    dist     += data["distM"][prev, 0]

    # Jornada
    if time_min > data["horizon_minutes"] + 1e-9:
        return RouteMetrics(load=load, time_min=time_min, dist=dist, feasible=False)

    return RouteMetrics(load=load, time_min=time_min, dist=dist, feasible=True)


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
    factibilidad temporal con esa capacidad). Implementa Best-Fit Decreasing:
      1) Ordena rutas por demanda descendente.
      2) Intenta asignarlas al vehículo con *menor* capacidad restante que aún alcance.
      3) Devuelve:
         - ok (bool): True si TODAS las rutas pudieron asignarse.
         - assign (List[Optional[int]]): índice de vehículo por ruta (en orden original).
         - remaining_caps (List[float]): capacidades remanentes por vehículo (en orden original).
    
    Notas:
    - No modifica el orden de `data["vehicle_caps"]` (se conserva índice real del vehículo).
    - Si `check_time_feasibility=True`, valida cada ruta con route_feasible(route, cap, data).
    - No mezcla rutas en un mismo vehículo más allá de su capacidad restante (modelo 1-ruta-por-vehículo
      o multi-ruta-por-vehículo depende de cómo definas “vehículo” y “ruta” en tus costos; por defecto
      este empaquetado permite varias rutas por vehículo SOLO si hay saldo de capacidad, aunque lo normal
      en VRPTW es 1 ruta = 1 vehículo por jornada. Si quieres forzar 1:1, ver el flag more_strict_1_route_per_vehicle).
    """
    # --- Si quieres forzar 1 ruta = 1 vehículo por jornada, activa este flag:
    more_strict_1_route_per_vehicle = True

    V = list(map(float, data["vehicle_caps"]))  # Copia de capacidades disponibles
    n_routes = len(routes)
    n_veh = len(V)

    # Precalcular demandas por ruta
    route_loads = [sum(data["demand_size"][i] for i in r) for r in routes]

    # Indices de rutas ordenados por demanda (desc)
    order = sorted(range(n_routes), key=lambda idx: route_loads[idx], reverse=True)

    # Resultado en orden original de rutas
    assign: List[Optional[int]] = [None] * n_routes
    remaining_caps = V[:]  # saldo por vehículo (se irá reduciendo si permitimos multi-ruta por vehículo)

    # Para BFD necesitamos, por cada ruta, encontrar el vehículo con capacidad mínima que aún baste.
    for ridx in order:
        load = route_loads[ridx]

        # Elegibles: vehículos cuyo saldo >= load
        candidates = [(vi, remaining_caps[vi]) for vi in range(n_veh) if remaining_caps[vi] + 1e-9 >= load]

        if not candidates:
            # No hay vehículo que alcance
            return False, assign, remaining_caps

        # Best-fit: el que deje MENOS espacio libre (capacidad más ajustada)
        candidates.sort(key=lambda x: x[1])  # menor capacidad restante primero
        chosen_vi = None

        for vi, cap_left in candidates:
            # Si queremos 1 ruta por vehículo, ese vehículo no debe estar ya asignado a otra ruta
            if more_strict_1_route_per_vehicle and any(a == vi for a in assign):
                continue

            # (Opcional) Chequear factibilidad temporal con esa "capacidad" de vehículo
            if check_time_feasibility:
                if not route_feasible(routes[ridx], remaining_caps[vi], data):
                    # con este vehículo no da; probamos siguiente candidato
                    continue

            chosen_vi = vi
            break

        if chosen_vi is None:
            return False, assign, remaining_caps

        # Asignar ruta → vehículo y descontar capacidad si permitimos multi-ruta por vehículo
        assign[ridx] = chosen_vi
        if not more_strict_1_route_per_vehicle:
            remaining_caps[chosen_vi] -= load

    return True, assign, remaining_caps


def penalizar_solapamiento_geometrico(routes: List[List[int]], data: dict) -> float:
    """
    Calcula una penalización geométrica para el solapamiento de rutas.
    Se activa si el centroide de una ruta está contenido dentro de la envolvente convexa de otra.
    """
    if len(routes) < 2:
        return 0.0

    nodes = data["nodes"]
    penalizacion = 0
    
    # Pre-calcula los centroides y las envolventes de cada ruta
    route_info = []
    for r in routes:
        if not r: continue
        route_nodes = np.array([nodes[i] for i in r])
        centroid = np.mean(route_nodes, axis=0)
        hull = _convex_hull(route_nodes)
        route_info.append({'centroid': centroid, 'hull': hull, 'size': len(r)})

    # Compara cada par de rutas
    for i in range(len(route_info)):
        for j in range(i + 1, len(route_info)):
            info_i = route_info[i]
            info_j = route_info[j]

            # Penaliza si el centroide de la ruta más pequeña está en la envolvente de la más grande
            if info_i['size'] < info_j['size'] and _point_in_polygon(info_i['centroid'], info_j['hull']):
                penalizacion += 1
            elif info_j['size'] < info_i['size'] and _point_in_polygon(info_j['centroid'], info_i['hull']):
                penalizacion += 1
                
    return float(penalizacion)





# --- Destroy 1: aleatorio (liviano, estable) ---
def destroy_random(sol: Solution, p: float = 0.15, data: dict = None) -> Tuple[Solution, List[int]]:
    """Elimina ~p de los clientes al azar (al menos 1)."""
    all_clients = [i for r in sol.routes for i in r]
    if not all_clients:
        return Solution([r[:] for r in sol.routes]), []
    k = max(1, int(len(all_clients) * p))
    removed_set = set(random.sample(all_clients, k))
    new_routes = [[i for i in r if i not in removed_set] for r in sol.routes]
    return Solution(new_routes), list(removed_set)

# --- Destroy 2: peores por contribución marginal (inteligente) ---
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

# --- Repair: Regret-k (regret=3 por defecto) ---
def q_insert_regret(sol: Solution, removed: List[int], data: dict, cluster_map: np.ndarray, weights: dict, k: int = 3) -> Solution:
    """
    Versión mejorada de Regret-k que es "consciente" de las zonas.
    Al calcular el costo de una inserción, no solo considera la distancia,
    sino también si la inserción crea dispersión o solapamiento de zonas.
    """
    routes = [r[:] for r in sol.routes]
    
    # Pre-calcula los clusters atendidos por cada ruta para eficiencia
    clusters_por_ruta = [set(cluster_map[c] for c in r) for r in routes]
    todos_los_clusters_atendidos = set.union(*clusters_por_ruta) if clusters_por_ruta else set()

    while removed:
        options = []
        for cust in removed:
            cand_list = []
            cust_cluster = cluster_map[cust]

            # 1. Probar inserciones en rutas existentes
            for ri, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    # --- Cálculo del Costo de Distancia ---
                    prev = r[pos-1] if pos > 0 else 0
                    nxt  = r[pos]   if pos < len(r) else 0
                    delta_dist = data["distM"][prev, cust] + data["distM"][cust, nxt] - data["distM"][prev, nxt]

                    # --- Cálculo del Costo Estético de esta inserción ---
                    # Penalización por dispersión: si se añade un nuevo cluster a la ruta
                    pen_dispersion = 0
                    if cust_cluster not in clusters_por_ruta[ri]:
                        pen_dispersion = 1
                    
                    # Penalización por solapamiento: si este cluster ya es de otra ruta
                    pen_solapamiento = 0
                    if cust_cluster in todos_los_clusters_atendidos and cust_cluster not in clusters_por_ruta[ri]:
                        pen_solapamiento = 1

                    costo_estetico = (weights.get("w_dispersion", 40.0) * pen_dispersion +
                                      weights.get("w_solapamiento", 50.0) * pen_solapamiento)
                    
                    # El costo total de la inserción es la suma del costo de distancia y el estético
                    costo_total = delta_dist + costo_estetico
                    cand_list.append({'cost': costo_total, 'ri': ri, 'pos': pos})

            cand_list.sort(key=lambda x: x['cost'])

            # Chequeo de factibilidad solo para las mejores 'k' opciones para ahorrar tiempo
            feasible_cands = []
            for cand in cand_list[:k*2]: # Revisa un poco más de k por si algunas no son factibles
                ri, pos = cand['ri'], cand['pos']
                trial = routes[ri][:pos] + [cust] + routes[ri][pos:]
                # (Asumimos una capacidad promedio para la validación, puedes refinar esto si es necesario)
                if route_feasible(trial, np.mean(data["vehicle_caps"]), data):
                    feasible_cands.append(cand)
                if len(feasible_cands) >= k:
                    break

            if feasible_cands:
                best = feasible_cands[0]
                regret = sum(feasible_cands[i]['cost'] - best['cost'] for i in range(1, min(k, len(feasible_cands))))
                options.append({'cust': cust, 'regret': regret, 'ins': best})

        if not options:
            # No hay inserciones factibles para los clientes restantes
            print(f"[WARN] q_insert_regret: No se encontraron inserciones factibles para los clientes restantes: {removed}")
            break

        options.sort(key=lambda x: x['regret'], reverse=True)
        chosen = options[0]
        cust, ins = chosen['cust'], chosen['ins']
        ri, pos = ins['ri'], ins['pos']

        # Realizar la inserción y actualizar estructuras
        routes[ri].insert(pos, cust)
        clusters_por_ruta[ri].add(cluster_map[cust])
        todos_los_clusters_atendidos.add(cluster_map[cust])
        removed.remove(cust)

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

def improve_route_with_2opt(route: List[int], data: dict) -> List[int]:
    """
    2-opt con:
    - first-improvement,
    - umbral de mejora (min_gain),
    - detector de ciclos que conserva explícitamente la ruta de menor distancia.
    """
    if len(route) < 4:
        return route

    distM = data["distM"]
    best = route[:]
    best_cost = route_distance(best, distM)

    seen = set()                 # rutas ya visitadas (tuplas)
    prev_best = best[:]          # respaldo de la mejor ruta antes de la última mejora
    prev_best_cost = best_cost

    min_gain = 1              # exige mejoras > 1 metro (ajusta según escala)
    improved = True

    while improved:
        improved = False

        key = tuple(best)
        if key in seen:
            # Ciclo detectado: quedarse explícitamente con la de menor distancia real
            if prev_best_cost < best_cost:
                best = prev_best[:]
                best_cost = prev_best_cost
            # si best ya era mejor, lo dejamos tal cual
            break
        seen.add(key)

        path = [0] + best + [0]
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                A, B, C, D = path[i-1], path[i], path[j], path[j+1]

                # Evita swaps con arcos castigados
                if (distM[A, B] >= 1e9 or distM[C, D] >= 1e9 or
                    distM[A, C] >= 1e9 or distM[B, D] >= 1e9):
                    continue

                # Ganancia (cuánto baja la distancia si aplico el swap)
                gain = (distM[A, B] + distM[C, D]) - (distM[A, C] + distM[B, D])
                if gain > min_gain:
                    candidate = best[:i] + best[i:j][::-1] + best[j:]
                    # Evitar aceptar la misma secuencia
                    if candidate == best:
                        continue
                    # Chequea factibilidad rápida
                    if not calculate_route_metrics(candidate, data).feasible:
                        continue

                    cand_cost = route_distance(candidate, distM)
                    if cand_cost + 1e-9 < best_cost:   # mejora real
                        # Guarda respaldo antes de actualizar (para resolver ciclos)
                        prev_best = best[:]
                        prev_best_cost = best_cost

                        best = candidate
                        best_cost = cand_cost
                        improved = True
                        break  # first-improvement
            if improved:
                break

    return best




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
                            routes[r1_idx], routes[r2_idx] = new_r1, new_r2
                            return Solution([r for r in routes if r]), True
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

def _invalidate_route_cache(routes: List[List[int]]):
    # Si no quieres cache grande/antiguo, invalida cuando cambias rutas
    for r in routes:
        _ROUTE_DIST_CACHE.pop(tuple(r), None)

def solution_cost(sol, data) -> float:
    """
    Costo operativo de una solución:
      - Costo fijo por cada ruta activa
      - Costo variable por distancia recorrida
      - (Opcional) Costo por capacidad utilizada (aproximación)
    Requiere en `data`:
      - fixed_route_cost, cost_per_meter, cost_per_cap
      - distM, demand_size
    """
    active_routes = [r for r in sol.routes if r]
    # fijo por ruta
    base_cost = len(active_routes) * data["fixed_route_cost"]
    # variable por distancia
    total_dist = sum(route_distance(r, data["distM"]) for r in active_routes)
    var_cost = total_dist * data["cost_per_meter"]
    # costo por capacidad usada (aprox. sumatoria de tamaños atendidos)
    cap_cost = sum(
        sum(data["demand_size"][i] for i in r) * data["cost_per_cap"]
        for r in active_routes
    )
    return base_cost + var_cost + cap_cost

def count_self_crossings(route: List[int], nodes: List[Tuple[float, float]]) -> int:
    """
    Cuenta cruces entre segmentos de UNA misma ruta (incluye tramos al/depot).
    - No cuenta cruces entre segmentos adyacentes.
    - No cuenta “cruces” cuando los segmentos comparten un nodo.
    - Complejidad O(m^2), donde m = #segmentos de la ruta (incluyendo depot).
    """
    if not route or len(route) < 3:
        return 0

    # Camino con depósito al inicio y final
    path = [0] + route + [0]
    segments = [(path[i], path[i+1]) for i in range(len(path) - 1)]

    def ccw(A: int, B: int, C: int) -> bool:
        ax, ay = nodes[A]; bx, by = nodes[B]; cx, cy = nodes[C]
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    def intersect(s1: Tuple[int, int], s2: Tuple[int, int]) -> bool:
        a, b = s1; c, d = s2
        # Si comparten algún nodo, no se considera cruce
        if len({a, b, c, d}) < 4:
            return False
        # Test de intersección de segmentos en 2D
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    cnt = 0
    for i in range(len(segments)):
        a1, b1 = segments[i]
        for j in range(i + 1, len(segments)):
            a2, b2 = segments[j]
            # Evitar comparar segmentos adyacentes del mismo camino
            if b1 == a2 or a1 == b2:
                continue
            if intersect((a1, b1), (a2, b2)):
                cnt += 1
    return cnt



from typing import List, Tuple
import math

# ---------- Utilidades geométricas (sin librerías externas) ----------

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _convex_hull(points):
    """
    Calcula la envolvente convexa de un conjunto de puntos 2D.
    Utiliza el algoritmo Monotone Chain de Andrew.
    """
    # --- LÍNEA CORREGIDA ---
    # La versión anterior (sorted(set(points))) fallaba porque los sets de Python no aceptan arrays de numpy.
    # np.unique con axis=0 está diseñado para encontrar filas únicas en un array, que es exactamente lo que necesitamos.
    pts = np.unique(points, axis=0)
    
    if len(pts) <= 2:
        return pts.tolist()

    # Construye la envolvente inferior
    lower = []
    for p in pts:
        while len(lower) >= 2 and np.cross(np.array(lower[-1]) - np.array(lower[-2]), p - np.array(lower[-1])) <= 0:
            lower.pop()
        lower.append(p.tolist())

    # Construye la envolvente superior
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and np.cross(np.array(upper[-1]) - np.array(upper[-2]), p - np.array(upper[-1])) <= 0:
            upper.pop()
        upper.append(p.tolist())

    return lower[:-1] + upper[:-1]

def _point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """Ray casting. True si pt está dentro o en borde del polígono."""
    x, y = pt
    inside = False
    n = len(poly)
    if n == 0:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        # chequeo sobre el borde (colineal y dentro del segmento)
        den = (y2 - y1)*(x - x1) - (x2 - x1)*(y - y1)
        if abs(den) < 1e-12:
            if min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
                return True
        # cruce de rayos
        inter = ((y1 > y) != (y2 > y)) and (x < (x2 - x1)*(y - y1) / (y2 - y1 + 1e-18) + x1)
        if inter:
            inside = not inside
    return inside

# ---------- Métrica de solapamiento entre áreas de rutas ----------

def inter_route_area_overlap_score(
    routes: List[List[int]],
    nodes: List[Tuple[float, float]],
    include_depot: bool = False,
    depot_id: int = 0
) -> float:
    """
    Devuelve un score ∈ [0, 1] que mide solapamiento promedio entre rutas.
    - Construye el convex hull de cada ruta (por sus nodos) y calcula:
        overlap_ij = 0.5 * ( %nodos_i dentro hull_j + %nodos_j dentro hull_i )
      para cada par i<j. El score final es el promedio de los overlap_ij válidos.
    - Si una ruta tiene <3 puntos únicos para el hull, se omite ese lado del cálculo.
    - 'nodes' es lista de (lat, lon) o (y, x) consistente con el resto del código.
    """
    # 1) Extrae las coordenadas por ruta (opcionalmente excluye depósito)
    route_points: List[List[Tuple[float, float]]] = []
    for r in routes:
        if not r:
            route_points.append([])
            continue
        ids = r[:] if include_depot else [nid for nid in r if nid != depot_id]
        pts = [nodes[nid] for nid in ids]
        route_points.append(pts)

    # 2) Convex hull por ruta
    hulls: List[List[Tuple[float, float]]] = []
    for pts in route_points:
        if len(set(pts)) >= 3:
            hulls.append(_convex_hull(pts))
        else:
            hulls.append([])  # sin área

    # 3) Porcentajes de inclusión cruzada
    pair_scores = []
    R = len(routes)
    for i in range(R):
        pts_i = route_points[i]
        hull_i = hulls[i]
        for j in range(i+1, R):
            pts_j = route_points[j]
            hull_j = hulls[j]

            components = []
            # % nodos de i dentro del hull de j
            if hull_j and len(pts_i) > 0:
                inside_i_in_j = sum(1 for p in pts_i if _point_in_polygon(p, hull_j))
                components.append(inside_i_in_j / max(1, len(pts_i)))
            # % nodos de j dentro del hull de i
            if hull_i and len(pts_j) > 0:
                inside_j_in_i = sum(1 for p in pts_j if _point_in_polygon(p, hull_i))
                components.append(inside_j_in_i / max(1, len(pts_j)))

            if components:
                pair_scores.append(sum(components)/len(components))

    if not pair_scores:
        return 0.0
    return sum(pair_scores) / len(pair_scores)


def _segments_from_route(route: List[int]) -> List[Tuple[int, int]]:
    """Incluye depósito al inicio y fin."""
    if not route: return []
    path = [0] + route + [0]
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def _seg_intersect(s1: Tuple[int,int], s2: Tuple[int,int], pts: List[Tuple[float,float]]) -> bool:
    a,b = s1; c,d = s2
    # no contamos si comparten nodo (unión natural o depósito)
    if len({a,b,c,d}) < 4:
        return False
    ax, ay = pts[a]; bx, by = pts[b]; cx, cy = pts[c]; dx, dy = pts[d]
    def ccw(x1,y1, x2,y2, x3,y3): return (y3 - y1)*(x2 - x1) > (y2 - y1)*(x3 - x1)
    return (ccw(ax,ay, cx,cy, dx,dy) != ccw(bx,by, cx,cy, dx,dy)) and \
           (ccw(ax,ay, bx,by, cx,cy) != ccw(ax,ay, bx,by, dx,dy))

def count_between_routes_crossings(routes: List[List[int]], nodes: List[Tuple[float,float]]) -> int:
    segs = [ _segments_from_route(r) for r in routes if r ]
    cnt = 0
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            for s1 in segs[i]:
                for s2 in segs[j]:
                    if _seg_intersect(s1, s2, nodes):
                        cnt += 1
    return cnt


def _calculate_route_compactness_penalty(route: List[int], data: dict) -> float:
    """
    Penaliza rutas “largas y delgadas” (poca compacidad).
    Idea: compara la distancia real de la ruta con el mínimo inevitable
    (ir al cliente más lejano desde el depósito y volver).
    
    Definiciones:
      total_dist = distancia real de la ruta (incluye salida y retorno al depósito)
      d_max      = max_i dist(depot, i) para i en la ruta
      base_min   = 2 * d_max
      detour     = max(0, total_dist - base_min)
      ratio      = detour / base_min     (≥ 0)
    
    Notas:
    - Si la ruta tiene <2 clientes, no penalizamos (retorna 0.0).
    - Si base_min == 0 (coords coinciden o datos degenerados), retorna 0.0.
    """
    if len(route) < 2:
        return 0.0
    distM = data["distM"]

    # distancia real
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
    #return 1.0 / (1.0 + ratio)

# --- Geometría básica ---
def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

def _segments_intersect(p1, p2, p3, p4):
    # p1=(x1,y1), p2=(x2,y2) de ruta A; p3=(x3,y3), p4=(x4,y4) de ruta B
    (x1, y1), (x2, y2) = p1, p2
    (x3, y3), (x4, y4) = p3, p4
    return (_ccw(x1,y1,x3,y3,x4,y4) != _ccw(x2,y2,x3,y3,x4,y4)) and (_ccw(x1,y1,x2,y2,x3,y3) != _ccw(x1,y1,x2,y2,x4,y4))

def _seg_angle_penalty(p1, p2, p3, p4):
    # mayor penal si el cruce es perpendicular; menor si es casi paralelo
    import math
    v1 = (p2[0]-p1[0], p2[1]-p1[1])
    v2 = (p4[0]-p3[0], p4[1]-p3[1])
    n1 = math.hypot(*v1) + 1e-9
    n2 = math.hypot(*v2) + 1e-9
    cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
    # cos=1 (paralelo) -> peso bajo; cos=0 (perp) -> peso alto
    return 1.0 - abs(cosang)  # en [0..1]


def _route_segments(route):
    # con depósito (0) como in/out
    path = [0] + route + [0]
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def count_route_cuts(routes, nodes, distM, depot_id=0, depot_relief_m=2000.0, angle_weight=True):
    """
    Cuenta intersecciones entre segmentos de rutas distintas (excluye las muy cerca del depósito).
    Devuelve (cuts_weighted, denom), donde denom es producto de #segmentos entre pares de rutas.
    """
    # preconstruir segmentos por ruta
    segs = []
    for r in routes:
        segs.append(_route_segments(r))

    def _near_depot(u, v):
        return (distM[depot_id, u] < depot_relief_m) or (distM[depot_id, v] < depot_relief_m)

    cuts = 0.0
    denom = 0
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            Si = segs[i]; Sj = segs[j]
            denom += len(Si) * len(Sj)
            for (a,b) in Si:
                if _near_depot(a,b): 
                    continue
                p1 = (nodes[a][1], nodes[a][0])  # folium usa (lat,lon); aquí tomo (x=lon,y=lat) consistente para geom
                p2 = (nodes[b][1], nodes[b][0])
                for (c,d) in Sj:
                    if _near_depot(c,d): 
                        continue
                    p3 = (nodes[c][1], nodes[c][0])
                    p4 = (nodes[d][1], nodes[d][0])
                    if _segments_intersect(p1, p2, p3, p4):
                        if angle_weight:
                            cuts += _seg_angle_penalty(p1, p2, p3, p4)
                        else:
                            cuts += 1.0
    return cuts, max(1, denom)

def route_cuts_norm(routes, nodes, distM, depot_id=0, depot_relief_m=2000.0, angle_weight=True):
    cuts, denom = count_route_cuts(routes, nodes, distM, depot_id, depot_relief_m, angle_weight)
    return cuts / float(denom)



def aesthetic_penalty(sol: Solution, data: dict, cluster_map: np.ndarray, weights: dict) -> float:
    """
    Calcula una penalización estética basada en clusters geográficos (zonas virtuales).
    Esta versión reemplaza las métricas de solapamiento (convex hull) y compacidad
    por lógica basada en clusters, que es más robusta y alineada con el plan.
    """
    routes = [r for r in sol.routes if r]
    if len(routes) < 2:
        return 0.0

    # --- Métricas basadas en Zonas Virtuales (Clusters) ---

    # --- 1. Penalización por SOLAPAMIENTO (¡NUEVA LÓGICA!) ---
    pen_solapamiento = penalizar_solapamiento_geometrico(routes, data)

    # [cite_start]1. Penalización por SOLAPAMIENTO de zonas entre rutas [cite: 105, 107, 108]
    # Penaliza cada vez que una ruta entra en un cluster que ya fue "reclamado" por otra.
    clusters_reclamados = set()
    pen_solapamiento = 0
    for r in routes:
        clusters_en_ruta = set(cluster_map[c] for c in r)
        clusters_en_ruta.discard(-1) # Ignorar el depot
        
        pen_solapamiento += len(clusters_en_ruta.intersection(clusters_reclamados))
        clusters_reclamados.update(clusters_en_ruta)
        
    # [cite_start]2. Penalización por DISPERSIÓN de ruta (anti-compacidad) [cite: 112, 113, 114]
    # Penaliza a las rutas que se expanden por múltiples zonas.
    pen_dispersion = 0
    UMBRAL_ZONAS_POR_RUTA = 1  # Ideal: cada ruta opera en 1 sola zona
    for r in routes:
        num_clusters = len(set(cluster_map[c] for c in r))
        if num_clusters > UMBRAL_ZONAS_POR_RUTA:
            pen_dispersion += (num_clusters - UMBRAL_ZONAS_POR_RUTA)
            
    # --- Métricas que se conservan (funcionaban bien) ---

    # [cite_start]3. Penalización por CRUCES intra-ruta [cite: 100, 101]
    pen_cruces_intra = sum(count_self_crossings(r, data["nodes"]) for r in routes)

    # ¡NUEVO! Penalización por CRUCES INTER-RUTA
    pen_cruces_inter = count_between_routes_crossings(routes, data["nodes"])

    # [cite_start]4. Penalización por DESBALANCE de distancia y paradas [cite: 117, 118]
    distM = data["distM"]
    dists = [route_distance(r, distM) for r in routes]
    mean_dist = sum(dists) / len(dists) if dists else 0
    pen_balance_dist = np.std(dists) / (mean_dist + 1e-9) # Coef. de Variación

    stops = [len(r) for r in routes]
    mean_stops = sum(stops) / len(stops) if stops else 0
    pen_balance_stops = np.std(stops) / (mean_stops + 1e-9)

    # --- Suma Ponderada Final ---
    total_penalty = (
        weights.get("w_solapamiento", 50.0) * pen_solapamiento +
        weights.get("w_dispersion", 40.0) * pen_dispersion +
        weights.get("w_cruces_intra", 30.0) * pen_cruces_intra + # Renombrado para claridad
        weights.get("w_cruces_inter", 40.0) * pen_cruces_inter + # NUEVA LÍNEA
        weights.get("w_balance_dist", 5.0) * pen_balance_dist +
        weights.get("w_balance_stops", 5.0) * pen_balance_stops
    )
    
    return float(total_penalty)











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


def aesthetic_penalty_fast(sol, data: dict) -> float:
    """
    Penalización estética barata para acelerar ALNS:
    usa solo balance de distancias, balance de paradas y compacidad.
    """
    routes = [r for r in sol.routes if r]
    if not routes:
        return 0.0
    distM = data["distM"]
    dists = [route_distance(r, distM) for r in routes]
    balance = float(np.std(dists)) if dists else 0.0
    stops = [len(r) for r in routes]
    stops_balance = float(np.std(stops)) if stops else 0.0
    compactness = sum(_calculate_route_compactness_penalty(r, data) for r in routes)
    # pesos moderados para mantener correlación con la estética "full"
    return (1.0 * balance) + (10.0 * stops_balance) + (50.0 * compactness)

def _all_clients_assigned(sol: Solution, data: Dict) -> bool:
    assigned = {i for r in sol.routes for i in r}
    return assigned == set(range(1, data["N"]))

def alns_single_run(
    data: Dict,
    weights,
    cluster_map: np.ndarray,  # <-- AÑADE ESTE PARÁMETRO
    lam: float,
    iters: int = 2000,
    seed: int = 0,
    destroy_p: float = 0.15,
    T_start: float = 1.0,
    cooling_rate: float = 0.997,
    T_min = 1e-6,
    reaction: float = 0.9,
    use_fast_esthetics: bool = False
) -> Solution:
    # --- 1. INICIALIZACIÓN ---
    random.seed(seed)
    np.random.seed(seed)

    def eval_cost(sol: Solution) -> float:
        if sol.cost is None: sol.cost = solution_cost(sol, data)
        return sol.cost

    # VAMOS A MODIFICAR ESTA FUNCIÓN INTERNA
    def eval_est_full(sol: Solution) -> float:
        if sol.est_penalty is None:
            # AHORA LLAMA A LA NUEVA FUNCIÓN CON CLUSTERS
            sol.est_penalty = aesthetic_penalty(sol, data, cluster_map, weights)
        return sol.est_penalty

    def eval_est_fast(sol: Solution) -> float:
        return aesthetic_penalty_fast(sol, data)

    def evaluate(sol: Solution, fast: bool) -> float:
        cost = eval_cost(sol)
        if fast:
            pen = eval_est_fast(sol)
        else:
            # AHORA LLAMA A LA VERSIÓN MODIFICADA DE ARRIBA
            pen = eval_est_full(sol)
        return cost + lam * pen

    # Solución inicial y la mejor solución encontrada
    curr = nearest_neighbor_seed(data)
    if not _all_clients_assigned(curr, data):
        raise ValueError("La solución inicial no cubre a todos los clientes.")
    
    curr.cost = eval_cost(curr)
    curr.est_penalty = eval_est_full(curr)
    best = Solution(routes=[r[:] for r in curr.routes], cost=curr.cost, est_penalty=curr.est_penalty)
    
    # Valores iniciales para el ALNS y SA
    f_curr = evaluate(curr, fast=use_fast_esthetics)
    f_best = f_curr
    
    T = T_start  # Temperatura inicial escalada

    destroy_ops = {
        'random': {'op': destroy_random, 'score': 1.0, 'uses': 1},
        'worst':  {'op': destroy_worst, 'score': 1.0, 'uses': 1},
    }
    REWARD_BEST, REWARD_BETTER, REWARD_ACCEPTED = 3.0, 2.0, 1.0

    # --- 2. BUCLE PRINCIPAL DEL ALGORITMO ---
    for it in tqdm(range(iters), desc=f"ALNS (λ={lam})", leave=False):
        
        # --- a. Selección de Operador Adaptativo ---
        op_weights = [max(1e-9, d['score']) / max(1, d['uses']) for d in destroy_ops.values()]
        suma = sum(op_weights)
        if not np.isfinite(suma) or suma <= 0.0:
            op_weights = [1.0] * len(destroy_ops)  # fallback uniforme

        chosen_name = random.choices(list(destroy_ops.keys()), weights=op_weights, k=1)[0]
        chosen_op_data = destroy_ops[chosen_name]


        # --- b. Destrucción y Reparación ---
        op_args = {'p': destroy_p, 'data': data}
        if chosen_name == 'zone':
            op_args['cluster_map'] = cluster_map

        destroyed, removed = chosen_op_data['op'](curr, **op_args)
        
        # ¡Llama a la nueva función de reparación con los argumentos extra!
        cand = q_insert_regret(destroyed, removed, data, cluster_map, weights, k=3)

        # --- c. Búsqueda Local Intensiva (VNS) ---
        keep_searching_vns = True
        max_loops = 100  # límite de salvavidas
        count = 0
        while keep_searching_vns and count < max_loops:            
            cand, moved = improve_with_relocate(cand, data)
            count += 1
            if moved:
                continue            
            initial_routes_str = str(cand.routes)
            cand.routes = [improve_route_with_2opt(r, data) for r in cand.routes if r]
            
            if str(cand.routes) == initial_routes_str:
                keep_searching_vns = False
        # --- d. Evaluación y Criterio de Aceptación (SA) ---
        # SOLO EVALUAMOS Y CONSIDERAMOS CANDIDATOS QUE SEAN 100% COMPLETOS
        if _all_clients_assigned(cand, data):
            cand.cost = None; cand.est_penalty = None # Resetea para forzar recálculo
            f_cand = evaluate(cand, fast=use_fast_esthetics)
            
            accepted = False
            reward = 0
            
            # Compara con la MEJOR solución global (siempre usando evaluación COMPLETA)
            f_cand_full = evaluate(cand, fast=False)
            if f_cand_full < f_best:
                best = Solution(routes=[r[:] for r in cand.routes],
                    cost=eval_cost(cand),
                    est_penalty=eval_est_full(cand))
                f_best = f_cand_full
                curr, f_curr = cand, f_cand_full
                accepted, reward = True, REWARD_BEST
            
            # Si no es una nueva mejor, compara con la solución ACTUAL
            delta = f_cand - f_curr
            if delta < -1e-9:
                curr, f_curr = cand, f_cand
                accepted, reward = True, REWARD_BETTER
            else:
                prob = math.exp(-delta / max(1e-9, T))
                if random.random() < prob:
                    curr, f_curr = cand, f_cand
                    accepted, reward = True, REWARD_ACCEPTED
                else:
                    accepted, reward = False, 0.0
            
            chosen_op_data['score'] = (1 - reaction) * chosen_op_data['score'] + reaction * reward
        
        # --- e. Enfriamiento (fuera del if para que siempre ocurra) ---
        T = max(T_min, T * cooling_rate)

    return best



# ----------  MULTI-OBJETIVO POR PONDERACIÓN ----------
def run_pareto(data, weights, cluster_map, lambdas=(0.0, 0.5, 1.0, 2.0, 5.0), iters=2000, seed=42, use_fast_esthetics=False): # <-- AÑADE cluster_map
    sols = []
    for lam in lambdas:
        # Pasa el cluster_map a la llamada de alns_single_run
        s = alns_single_run(data, weights, cluster_map, lam=lam, iters=iters, seed=seed, use_fast_esthetics=use_fast_esthetics) # <-- AÑADE cluster_map
        sols.append((lam, s.cost, s.est_penalty, s))
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
    m.save(outfile)
    if use_roads_flag:
        _save_shape_cache()  # solo guarda cache si usamos OSRM
    return outfile


# ==== LOG DE MÉTRICAS SIN MODIFICAR FUNCIONES EXISTENTES ====
_METRICS_LOG = {}          # {lam: {metric: [vals,...]}}
_CURRENT_LAMBDA = None     # lambda en ejecución actual

# guardamos referencia a la original
_ORIG_AESTHETIC_PENALTY = aesthetic_penalty

def _log_metrics_from_solution(sol, data):
    routes = [r for r in sol.routes if r]
    if not routes:
        return {
            "cross_intra_norm": 0.0,
            "cross_inter_norm": 0.0,
            "overlap": 0.0,
            "balance_cv": 0.0,
            "stops_cv": 0.0,
            "compact_mean": 0.0,
            "route_cuts_norm": 0.0,
        }

    nodes, distM = data["nodes"], data["distM"]
    dists = [route_distance(r, distM) for r in routes]
    mean_dist = sum(dists)/max(1,len(dists))

    # --- ALINEAR NORMALIZACIONES CON aesthetic_penalty ---
    segs_per_route = [len(r)+1 for r in routes]  # incluye depósito
    def _non_adjacent_pairs(s):
        return max(0, (s*(s-1))//2 - (s-1))

    # cruces intra (pares no adyacentes)
    pairs_intra = sum(_non_adjacent_pairs(s) for s in segs_per_route)
    crosses_intra = sum(count_self_crossings(r, nodes) for r in routes)
    cross_intra_norm = crosses_intra / max(1, pairs_intra)

    # cruces entre rutas (producto de segmentos entre pares de rutas)
    crosses_inter = count_between_routes_crossings(routes, nodes)
    denom_inter = 0
    for i in range(len(segs_per_route)):
        for j in range(i+1, len(segs_per_route)):
            denom_inter += segs_per_route[i] * segs_per_route[j]
    cross_inter_norm = crosses_inter / max(1, denom_inter)

    # overlap (hulls)
    overlap = inter_route_area_overlap_score(routes, nodes, include_depot=False, depot_id=0)

    # balances (CV)
    balance_cv = (float(np.std(dists)) / (mean_dist + 1e-9)) if dists else 0.0
    stops = [len(r) for r in routes]
    mean_stops = sum(stops)/max(1,len(stops))
    stops_cv = (float(np.std(stops)) / (mean_stops + 1e-9)) if stops else 0.0

    # compacidad
    compact_mean = float(np.mean([_calculate_route_compactness_penalty(r, data) for r in routes])) if routes else 0.0

    # NUEVO: cortes entre rutas normalizado
    cuts = route_cuts_norm(routes, nodes, distM, depot_id=0, depot_relief_m=2000.0, angle_weight=True)

    return {
        "cross_intra_norm": cross_intra_norm,
        "cross_inter_norm": cross_inter_norm,
        "overlap": overlap,
        "balance_cv": balance_cv,
        "stops_cv": stops_cv,
        "compact_mean": compact_mean,
        "route_cuts_norm": cuts,
    }


def _aesthetic_penalty_logged(sol, data, *args, **kwargs):
    """Wrapper: llama a la original y además loguea submétricas por iteración."""
    val = _ORIG_AESTHETIC_PENALTY(sol, data, *args, **kwargs)
    if _CURRENT_LAMBDA is not None:
        d = _log_metrics_from_solution(sol, data)
        bucket = _METRICS_LOG.setdefault(_CURRENT_LAMBDA, {})
        for k, v in d.items():
            bucket.setdefault(k, []).append(float(v))
    return float(val)



def esthetics_breakdown_final(sol: Solution, data: dict, weights: dict, cluster_map: np.ndarray) -> dict:
    """
    Devuelve el desglose de las métricas estéticas para una solución final,
    usando la nueva lógica de CLUSTERS.
    """
    routes = [r for r in sol.routes if r]
    if not routes:
        return {k: 0.0 for k in weights.keys()} | {"penalizacion_cruda": 0.0}

    # Calcula cada componente de la penalización individualmente
    # (reutilizando la lógica de la función principal)
    
    # Solapamiento
    clusters_reclamados = set()
    pen_solapamiento = 0
    for r in routes:
        clusters_en_ruta = set(cluster_map[c] for c in r); clusters_en_ruta.discard(-1)
        pen_solapamiento += len(clusters_en_ruta.intersection(clusters_reclamados))
        clusters_reclamados.update(clusters_en_ruta)

    # Dispersión
    pen_dispersion = 0
    for r in routes:
        num_clusters = len(set(cluster_map[c] for c in r))
        if num_clusters > 1: pen_dispersion += (num_clusters - 1)

    # Cruces
    pen_cruces = sum(count_self_crossings(r, data["nodes"]) for r in routes)

    # Desbalance
    dists = [route_distance(r, data["distM"]) for r in routes]
    mean_dist = sum(dists) / len(dists) if dists else 0
    pen_balance_dist = np.std(dists) / (mean_dist + 1e-9)

    stops = [len(r) for r in routes]
    mean_stops = sum(stops) / len(stops) if stops else 0
    pen_balance_stops = np.std(stops) / (mean_stops + 1e-9)

    # Calcula la penalización total cruda (sin ponderar por lambda)
    pen_cruda = aesthetic_penalty(sol, data, cluster_map, weights)

    pen_cruces_intra = sum(count_self_crossings(r, data["nodes"]) for r in routes)
    pen_cruces_inter = count_between_routes_crossings(routes, data["nodes"])
    
    return {
        "solapamiento_zonas": float(pen_solapamiento),
        "dispersion_rutas": float(pen_dispersion),
        "cruces_intra_ruta": float(pen_cruces_intra), # Renombrado
        "cruces_inter_ruta": float(pen_cruces_inter), # NUEVO
        "desbalance_dist_cv": float(pen_balance_dist),
        "desbalance_stops_cv": float(pen_balance_stops),
        "penalizacion_cruda": float(pen_cruda)
    }

# Ya no necesitas `_log_metrics_from_solution` ni `run_with_esthetics_logging`
# Simplificaremos el flujo usando `run_pareto` directamente.
# ==== FIN LOG MÉTRICAS ====



def export_pareto_maps(data, pareto, output_dir="soluciones"):
    os.makedirs(output_dir, exist_ok=True)  # <-- añade esto
    outputs = []
    for (lam, costo, pen, sol) in pareto:
        safe_lam = str(lam).replace('.', '_')
        base_filename = f"mapa_pareto_lambda_{safe_lam}__cost_{int(costo)}__pen_{int(round(pen))}.html"
        out_path = os.path.join(output_dir, base_filename)
        folium_solution_map(sol, data, outfile=out_path)
        outputs.append(out_path)
    return outputs

def export_pareto_dual_maps(data, pareto, output_dir="soluciones"):
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for (lam, costo, pen, sol) in pareto:
        safe_lam = str(lam).replace('.', '_')
        base = f"lambda_{safe_lam}__cost_{int(costo)}__pen_{int(round(pen))}"

        out_roads   = os.path.join(output_dir, f"mapa_{base}__roads.html")
        out_straight= os.path.join(output_dir, f"mapa_{base}__straight.html")

        folium_solution_map(sol, data, outfile=out_roads,    use_roads=True)
        folium_solution_map(sol, data, outfile=out_straight, use_roads=False)

        outputs.extend([out_roads, out_straight])
    return outputs



        
def run_single_experiment(base_seed: int, data: dict, weights: dict, lambdas: list, iters: int, cluster_map: np.ndarray) -> list:
    """
    Ejecuta una corrida completa del algoritmo para todos los lambdas con una semilla base.
    Recibe el cluster_map pre-calculado.
    """
    print(f"\n--- INICIANDO CORRIDA CON SEED BASE = {base_seed} ---")
    
    # Ejecuta run_pareto, que a su vez llama a alns_single_run para cada lambda
    sols, pareto = run_pareto(
        data, 
        weights, 
        cluster_map,
        lambdas=lambdas, 
        iters=iters,
        seed=base_seed,
        use_fast_esthetics=False 
    )
    
    print(f"--- CORRIDA CON SEED BASE = {base_seed} FINALIZADA ---")
    return pareto


def validar_solucion_final(sol: Solution, data: dict, nombre_solucion: str):
    """
    Verifica si una solución final atiende a todos los clientes y reporta si faltan.
    """
    clientes_requeridos = set(range(1, data["N"]))
    clientes_atendidos = {c for r in sol.routes for c in r}
    
    clientes_faltantes = clientes_requeridos - clientes_atendidos
    
    if not clientes_faltantes:
        print(f"Validación OK para '{nombre_solucion}': Todos los {len(clientes_requeridos)} clientes están atendidos.")
    else:
        print(f"¡ERROR DE VALIDACIÓN en '{nombre_solucion}'!")
        print(f"  - Clientes atendidos: {len(clientes_atendidos)} de {len(clientes_requeridos)}")
        print(f"  - Faltan {len(clientes_faltantes)} clientes. IDs: {sorted(list(clientes_faltantes))}")

    return not clientes_faltantes

# === Bloque de Ejecución Principal Corregido para Múltiples Corridas ===
if __name__ == "__main__":
    _load_shape_cache()
    DIRECTORIO_SALIDA = "soluciones"
    preparar_directorio_soluciones(DIRECTORIO_SALIDA)

    # --- 1. Cargar datos ---
    print("Cargando datos...")
    data = load_data()

    # --- 2. Calcular cluster_map UNA SOLA VEZ ---
    print("Generando zonas virtuales (clusters)...")
    n_clientes = data["N"] - 1
    n_zonas_a_crear = min(n_clientes, data["K"])
    if n_zonas_a_crear > 0:
        cluster_map = precalculate_clusters(data, n_clusters=n_zonas_a_crear)
    else:
        cluster_map = np.full(data["N"], -1, dtype=int)
    
    # --- 3. Definir parámetros del experimento ---
    weights = {
        "w_solapamiento": 50.0, "w_dispersion": 40.0, "w_cruces_intra": 30.0,
        "w_cruces_inter": 40.0, "w_balance_dist": 30.0, "w_balance_stops": 30.0,
    }
    #lambdas_a_probar = [0, 50, 100, 250, 500, 1000]
    lambdas_a_probar = [0, 250, 500]
    ITERACIONES_POR_CORRIDA = 1000
    NUMERO_DE_CORRIDAS = 2

    # --- 4. Bucle principal para ejecutar múltiples corridas ---
    start = time.perf_counter()
    all_pareto_solutions = []

    for i in range(NUMERO_DE_CORRIDAS):
        pareto_de_la_corrida = run_single_experiment(
            base_seed=i,
            data=data,
            weights=weights,
            lambdas=lambdas_a_probar,
            iters=ITERACIONES_POR_CORRIDA,
            cluster_map=cluster_map  # Pasa el cluster_map a la función
        )
        all_pareto_solutions.extend(pareto_de_la_corrida)

    # --- 5. Consolidar la frontera de Pareto final ---
    print("\nConsolidando la frontera de Pareto final de todas las corridas...")
    sols_sorted = sorted(all_pareto_solutions, key=lambda x: (x[1], x[2]))
    final_pareto_front = []
    best_est = float('inf')
    for lam, costo, pen, sol in sols_sorted:
        if pen < best_est - 1e-9:
            is_duplicate = any(abs(s[1] - costo) < 1 and abs(s[2] - pen) < 1 for s in final_pareto_front)
            if not is_duplicate:
                final_pareto_front.append((lam, costo, pen, sol))
                best_est = pen

    # --- 6. Generar reporte y mapas (ahora cluster_map sí está definido) ---
    print("\n" + "="*50)
    print("REPORTE DE LA FRONTERA DE PARETO FINAL CONSOLIDADA")
    print("="*50)
    for i, (lam, costo, pen, sol) in enumerate(final_pareto_front):
        # Esta llamada ahora funciona porque cluster_map existe en este ámbito
        nombre_sol = f"Solución {i+1} (λ={lam})"
        validar_solucion_final(sol, data, nombre_sol)
        breakdown = esthetics_breakdown_final(sol, data, weights, cluster_map)
        print(f"\n--- Solución {i+1} (λ original = {lam}) ---")
        print(f"  - Costo Operativo: {costo:,.2f}")
        print(f"  - Penalización Estética Cruda: {pen:,.2f}")
        print( "  - Desglose de Penalizaciones:")
        for key, val in breakdown.items():
            if key != "penalizacion_cruda":
                print(f"    - {key}: {val:.2f}")

    files = export_pareto_dual_maps(data, final_pareto_front, output_dir=DIRECTORIO_SALIDA)

    end = time.perf_counter()
    _save_shape_cache()
    print(f"\nTiempo total del experimento: {end - start:.2f} segundos")
    print(f"\nSe generaron {len(final_pareto_front)} pares de mapas en la carpeta '{DIRECTORIO_SALIDA}'.")