
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from branca.element import MacroElement
from jinja2 import Template

from metrics.aesthetic import EstheticCache, aesthetic_penalty as core_aesthetic_penalty

def debug(msg: str):
    print(f"[DEBUG] {msg}", flush=True)

def preparar_directorio_soluciones(dir_path: str):
    """
    Crea un directorio si no existe y borra todos los archivos .html que contenga.
    """
    os.makedirs(dir_path, exist_ok=True)
    debug(f"Limpiando directorio de soluciones: '{dir_path}'")
    for filename in os.listdir(dir_path):
        if filename.endswith(".html"):
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
                debug(f" - Borrado: {filename}")
            except Exception as e:
                debug(f"Error al borrar {file_path}: {e}")

# ---------- CARGA DE DATOS ----------
def load_data():
    """Carga todos los datos desde los archivos CSV y los pre-procesa."""
    debug("Iniciando carga de datos")
    df_over = pd.read_csv("Datos P5/i1/overview.csv")
    df_dem = pd.read_csv("Datos P5/i1/demands.csv")
    df_veh = pd.read_csv("Datos P5/i1/vehicles.csv")
    df_dist = pd.read_csv("Datos P5/i1/distances.csv")
    df_time = pd.read_csv("Datos P5/i1/times.csv")
    df_cost = pd.read_csv("Datos P5/costs.csv")
    debug("CSV cargados en memoria")

    exp = int(df_over.loc[0, "expected_matrix_size"])
    dsz = int(df_over.loc[0, "distances_size"])
    tsz = int(df_over.loc[0, "times_size"])
    if dsz != exp or tsz != exp:
        print(f"[WARN] Tamaños esperados no calzan: expected={exp}, dist={dsz}, time={tsz}")

    depot = (float(df_over.loc[0,"depot_latitude"]), float(df_over.loc[0,"depot_longitude"]))
    start_at = pd.to_datetime(df_over.loc[0, "start_at"])
    end_at   = pd.to_datetime(df_over.loc[0, "end_at"])
    horizon_minutes = int((end_at - start_at).total_seconds()/60)

    nodes = [(depot[0], depot[1])] + list(zip(df_dem["latitude"].astype(float), df_dem["longitude"].astype(float)))
    N = len(nodes)

    def build_matrix(df, value_col):
        M = np.zeros((N, N), dtype=float)
        def keyfy(lat, lon):
            return (float(f"{lat:.6f}"), float(f"{lon:.6f}"))
        coord_to_idx = {keyfy(lat,lon):i for i,(lat,lon) in enumerate(nodes)}
        for _, r in df.iterrows():
            o = keyfy(r["origin_latitude"], r["origin_longitude"])
            d = keyfy(r["destination_latitude"], r["destination_longitude"])
            if o in coord_to_idx and d in coord_to_idx:
                i, j = coord_to_idx[o], coord_to_idx[d]
                M[i, j] = float(r[value_col])
        return M

    distM = build_matrix(df_dist, "distance")
    timeM = build_matrix(df_time, "time")
    debug("Matrices de distancia/tiempo construidas")

    demand_size = np.zeros(N, dtype=float)
    demand_srv_s = np.zeros(N, dtype=float)
    tw_start_min = np.zeros(N, dtype=float)
    tw_end_min = np.zeros(N, dtype=float)

    for i, row in enumerate(df_dem.itertuples(index=False), start=1):
        demand_size[i]  = float(row.size)
        demand_srv_s[i] = float(row.stop_time)
        tws = getattr(row, "tw_start", None)
        twe = getattr(row, "tw_end", None)
        tw_start_min[i] = 0 if pd.isna(tws) else (pd.to_datetime(tws) - start_at).total_seconds()/60
        tw_end_min[i] = horizon_minutes if pd.isna(twe) else (pd.to_datetime(twe) - start_at).total_seconds()/60

    vehicle_caps = list(df_veh["capacity"].astype(float).values)
    K = len(vehicle_caps)

    fixed_route_cost = float(df_cost.loc[0, "fixed_route_cost"])
    cost_per_meter   = float(df_cost.loc[0, "cost_per_meter"])
    cost_per_cap     = float(df_cost.loc[0, "cost_per_vehicle_capacity"])

    data = {
        "N": N, "K": K, "nodes": nodes, "distM": distM, "timeM": timeM,
        "demand_size": demand_size, "demand_srv_s": demand_srv_s,
        "tw_start_min": tw_start_min, "tw_end_min": tw_end_min,
        "vehicle_caps": vehicle_caps, "fixed_route_cost": fixed_route_cost,
        "cost_per_meter": cost_per_meter, "cost_per_cap": cost_per_cap,
        "horizon_minutes": horizon_minutes, "start_at": start_at,
    }
    debug(f"Datos preparados: N={N}, K={K}")
    return data

# ---------- REPRESENTACIÓN Y FACTIBILIDAD ----------
@dataclass
class Solution:
    routes: List[List[int]]
    cost: Optional[float] = None
    est_penalty: Optional[float] = None

@dataclass
class RouteMetrics:
    load: float = 0.0
    time_min: float = 0.0
    dist: float = 0.0
    feasible: bool = True

def calculate_route_metrics(route: List[int], data: dict) -> RouteMetrics:
    """Calcula y devuelve las métricas completas de una ruta."""
    load = sum(data["demand_size"][i] for i in route)
    time_min = 0.0
    for i, node_idx in enumerate(route):
        prev = route[i-1] if i > 0 else 0
        time_min += data["timeM"][prev, node_idx] / 60.0
        if time_min < data["tw_start_min"][node_idx]:
            time_min = data["tw_start_min"][node_idx]
        if time_min > data["tw_end_min"][node_idx]:
            return RouteMetrics(load=load, feasible=False)
        time_min += data["demand_srv_s"][node_idx] / 60.0
    
    last_node = route[-1] if route else 0
    time_min += data["timeM"][last_node, 0] / 60.0
    
    if time_min > data["horizon_minutes"]:
        return RouteMetrics(load=load, feasible=False)
        
    return RouteMetrics(load=load, time_min=time_min, feasible=True)

def route_feasible(route: List[int], cap: float, data: dict) -> bool:
    """Chequeo de factibilidad: capacidad, ventanas de tiempo y jornada."""
    if not route: return True
    
    load = sum(data["demand_size"][i] for i in route)
    if load > cap + 1e-9:
        return False
        
    return calculate_route_metrics(route, data).feasible

# --- FUNCIONES AUXILIARES (Sin cambios recientes) ---
def route_distance(route: List[int], distM: np.ndarray) -> float:
    if not route: return 0.0
    d = distM[0, route[0]]
    for a, b in zip(route, route[1:]):
        d += distM[a, b]
    d += distM[route[-1], 0]
    return float(d)

def routes_capacity_feasible(routes: List[List[int]], data: dict) -> bool:
    """Best-fit check to ensure routes can be assigned to vehicles."""
    loads = [sum(data["demand_size"][i] for i in r) for r in routes if r]
    caps = sorted(map(float, data["vehicle_caps"]), reverse=True)

    if len(loads) > len(caps):
        return False

    remaining = caps[:]
    for load in sorted(loads, reverse=True):
        assigned = False
        for idx, cap in enumerate(remaining):
            if cap + 1e-9 >= load:
                remaining.pop(idx)
                assigned = True
                break
        if not assigned:
            return False
    return True

def solution_cost(sol: Solution, data) -> float:
    active_routes = [r for r in sol.routes if r]
    base = len(active_routes) * data["fixed_route_cost"]
    dist = sum(route_distance(r, data["distM"]) for r in active_routes)
    var = dist * data["cost_per_meter"]
    cap_cost = sum(sum(data["demand_size"][i] for i in r) * data["cost_per_cap"] for r in active_routes)
    return base + var + cap_cost

DEFAULT_ESTHETIC_WEIGHTS = {
    "w_balance_dist": 60.0,
    "w_balance_stops": 60.0,
    "w_dispersion": 40.0,
    "w_complexity": 35.0,
    "w_cruces_intra": 30.0,
    "w_cruces_inter": 40.0,
    "w_intrusion": 400.0,
    "w_coherence": 50.0,
}


def aesthetic_penalty(sol: Solution, data, weights: Optional[Dict[str, float]] = None) -> float:
    base_weights = {**DEFAULT_ESTHETIC_WEIGHTS}
    if weights:
        for key, value in weights.items():
            if key in base_weights:
                base_weights[key] = float(value)
    cache = EstheticCache(data)
    return core_aesthetic_penalty(sol, data, base_weights, cache=cache)

def fitness(sol: Solution, data, lam: float) -> float:
    return solution_cost(sol, data) + lam * aesthetic_penalty(sol, data)

# ---------- HEURÍSTICAS Y ALGORITMO ----------
def nearest_neighbor_seed(data) -> Solution:
    unserved = set(range(1, data["N"]))
    routes = []
    for cap in sorted(data["vehicle_caps"]):
        if not unserved: break
        route = []
        curr = 0
        while True:
            candidates = [i for i in unserved if data["demand_size"][i] <= cap - sum(data["demand_size"][c] for c in route)]
            if not candidates: break
            
            candidates.sort(key=lambda j: data["distM"][curr, j])
            best_next = next((c for c in candidates if route_feasible(route + [c], cap, data)), None)
            
            if best_next is None: break
            
            route.append(best_next)
            unserved.remove(best_next)
            curr = best_next
        if route: routes.append(route)
    
    if unserved: print(f"[WARN] Clientes no servidos en la solución inicial: {len(unserved)}")
    return Solution(routes=routes)

def destroy_random(sol: Solution, p: float, **kwargs) -> Tuple[Solution, List[int]]:
    all_clients = [i for r in sol.routes for i in r]
    if not all_clients: return sol, []
    k = max(1, int(len(all_clients) * p))
    removed = random.sample(all_clients, k)
    new_routes = [[c for c in r if c not in removed] for r in sol.routes]
    return Solution(routes=new_routes), removed

def destroy_worst(sol: Solution, p: float, data: dict, **kwargs) -> Tuple[Solution, List[int]]:
    all_clients = [i for r in sol.routes for i in r]
    if not all_clients: return sol, []
    
    costs = []
    for route in sol.routes:
        path = [0] + route + [0]
        for i in range(1, len(path) - 1):
            p, c, n = path[i-1], path[i], path[i+1]
            cost = data["distM"][p, c] + data["distM"][c, n] - data["distM"][p, n]
            costs.append((cost, c))
            
    costs.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(len(all_clients) * p))
    removed = [client for cost, client in costs[:k]]
    new_routes = [[c for c in r if c not in removed] for r in sol.routes]
    return Solution(routes=new_routes), removed

def q_insert_regret(sol: Solution, removed: List[int], data: dict, k: int = 3) -> Solution:
    routes = [r[:] for r in sol.routes]
    max_cap = max(map(float, data["vehicle_caps"])) if data["vehicle_caps"] else 0.0

    while removed:
        options = []
        for cust in removed:
            cust_costs = []
            for ri, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [cust] + r[pos:]
                    if not route_feasible(trial, max_cap, data):
                        continue

                    new_routes = routes[:]
                    new_routes[ri] = trial
                    if not routes_capacity_feasible(new_routes, data):
                        continue

                    p = r[pos-1] if pos > 0 else 0
                    n = r[pos] if pos < len(r) else 0
                    delta = data["distM"][p, cust] + data["distM"][cust, n] - data["distM"][p, n]
                    cust_costs.append({'cost': delta, 'route_idx': ri, 'pos': pos})

            if len(routes) < data["K"]:
                if route_feasible([cust], max_cap, data):
                    new_routes = routes + [[cust]]
                    if routes_capacity_feasible(new_routes, data):
                        delta = data["distM"][0, cust] + data["distM"][cust, 0]
                        cust_costs.append({'cost': delta, 'route_idx': len(routes), 'pos': 0})

            if cust_costs:
                cust_costs.sort(key=lambda x: x['cost'])
                best = cust_costs[0]
                regret = sum(cust_costs[i]['cost'] - best['cost'] for i in range(1, min(k, len(cust_costs))))
                options.append({'cust': cust, 'regret': regret, 'insert': best})

        if not options: break

        options.sort(key=lambda x: x['regret'], reverse=True)
        chosen = options[0]
        cust, insert_info = chosen['cust'], chosen['insert']
        ri, pos = insert_info['route_idx'], insert_info['pos']

        if ri == len(routes): routes.append([])
        routes[ri].insert(pos, cust)
        removed.remove(cust)

    return Solution(routes=routes)

def improve_route_with_2opt(route: List[int], data: dict) -> List[int]:
    if len(route) < 4: return route
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        path = [0] + best_route + [0]
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                A, B, C, D = path[i-1], path[i], path[j], path[j+1]
                if data["distM"][A, C] + data["distM"][B, D] < data["distM"][A, B] + data["distM"][C, D]:
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    if calculate_route_metrics(new_route, data).feasible:
                        best_route = new_route
                        path = [0] + best_route + [0]
                        improved = True
    return best_route

def improve_with_relocate(sol: Solution, data: dict) -> Tuple[Solution, bool]:
    routes = [r[:] for r in sol.routes]
    max_cap = max(map(float, data["vehicle_caps"])) if data["vehicle_caps"] else 0.0

    for r1_idx, r1 in enumerate(routes):
        for node_pos, cust in enumerate(r1):
            p1 = r1[node_pos - 1] if node_pos > 0 else 0
            n1 = r1[node_pos + 1] if node_pos < len(r1) - 1 else 0
            cost_removed = data["distM"][p1, n1] - data["distM"][p1, cust] - data["distM"][cust, n1]

            for r2_idx, r2 in enumerate(routes):
                if r1_idx == r2_idx: continue
                for pos2 in range(len(r2) + 1):
                    p2 = r2[pos2 - 1] if pos2 > 0 else 0
                    n2 = r2[pos2] if pos2 < len(r2) else 0
                    cost_inserted = data["distM"][p2, cust] + data["distM"][cust, n2] - data["distM"][p2, n2]

                    if cost_removed + cost_inserted < -1e-9:
                        new_r1 = r1[:node_pos] + r1[node_pos+1:]
                        new_r2 = r2[:pos2] + [cust] + r2[pos2:]
                        if not route_feasible(new_r1, max_cap, data):
                            continue
                        if not route_feasible(new_r2, max_cap, data):
                            continue

                        candidate_routes = routes[:]
                        candidate_routes[r1_idx] = new_r1
                        candidate_routes[r2_idx] = new_r2

                        if routes_capacity_feasible(candidate_routes, data):
                            # limpia rutas vacías para evitar residuos
                            clean = [r for r in candidate_routes if r]
                            return Solution(routes=clean), True
    return sol, False

def alns_single_run(data, lam: float, iters: int = 2000, seed: int = 0) -> Solution:
    random.seed(seed)
    curr = nearest_neighbor_seed(data)
    best = Solution(routes=curr.routes[:], cost=solution_cost(curr, data), est_penalty=aesthetic_penalty(curr, data))
    
    T, cooling_rate = 1.0, 0.999
    destroy_ops = {'random': {'op': destroy_random, 'score': 1, 'uses': 1}, 'worst': {'op': destroy_worst, 'score': 1, 'uses': 1}}
    R_BEST, R_BETTER, R_ACC = 3, 2, 1
    REACT_FACTOR = 0.9
    
    for it in range(iters):
        weights = [d['score'] / d['uses'] for d in destroy_ops.values()]
        op_name = random.choices(list(destroy_ops.keys()), weights=weights, k=1)[0]
        op_data = destroy_ops[op_name]
        
        destroyed, removed = op_data['op'](curr, p=0.15, data=data)
        op_data['uses'] += 1
        
        cand = q_insert_regret(destroyed, removed, data)
        
        improved = True
        while improved:
            cand, relocated = improve_with_relocate(cand, data)
            if relocated: continue
            
            initial_routes = str(cand.routes)
            cand.routes = [improve_route_with_2opt(r, data) for r in cand.routes if r]
            if str(cand.routes) == initial_routes: improved = False
                
        f_curr, f_cand, f_best = fitness(curr, data, lam), fitness(cand, data, lam), fitness(best, data, lam)
        
        accepted, reward = False, 0
        if f_cand < f_best:
            reward, best, curr, accepted = R_BEST, Solution(routes=cand.routes[:], cost=solution_cost(cand, data), est_penalty=aesthetic_penalty(cand, data)), cand, True
        elif f_cand < f_curr:
            reward, curr, accepted = R_BETTER, cand, True
        elif random.random() < math.exp(-(f_cand - f_curr) / max(1e-9, T)):
            reward, curr, accepted = R_ACC, cand, True
        
        op_data['score'] = (1 - REACT_FACTOR) * op_data['score'] + (REACT_FACTOR * reward if accepted else 0)
        T *= cooling_rate
        
    return best

# ---------- EJECUCIÓN Y VISUALIZACIÓN ----------
def run_pareto(data, lambdas=(0.0, 0.5, 1.0, 2.0, 5.0), iters=2000, seed=0):
    sols = []
    for lam in lambdas:
        debug(f"Iniciando ALNS para lambda={lam}")
        s = alns_single_run(data, lam=lam, iters=iters, seed=seed+int(lam*100))
        sols.append((lam, s.cost, s.est_penalty, s))
        print(f"Lambda {lam}: Costo={s.cost:.2f}, Penalidad={s.est_penalty:.2f}")

    sols_sorted = sorted(sols, key=lambda x: (x[1], x[2]))
    pareto = []
    min_penalty = float('inf')
    for item in sols_sorted:
        if item[2] < min_penalty:
            pareto.append(item)
            min_penalty = item[2]
    return sols, pareto

def folium_solution_map(sol, data, outfile="mapa.html"):
    depot_lat, depot_lon = data["nodes"][0]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12, tiles="CartoDB positron")
    folium.Marker([depot_lat, depot_lon], icon=folium.Icon(color="black", icon="home"), tooltip="Depot").add_to(m)
    
    ROUTE_COLORS = ["red","blue","green","purple","orange","darkred","lightblue", "lightgreen","pink","cadetblue","darkpurple","gray","black"]
    
    for idx, r in enumerate(r for r in sol.routes if r):
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        pts = [[data["nodes"][i][0], data["nodes"][i][1]] for i in [0] + r + [0]]
        folium.PolyLine(pts, weight=3, color=color, opacity=0.8, tooltip=f"Ruta {idx+1}").add_to(m)
        for j in r:
            lat, lon = data["nodes"][j]
            folium.CircleMarker([lat, lon], radius=4, fill=True, color=color, fill_opacity=1).add_to(m)
            
    m.save(outfile)
    return outfile

def export_pareto_maps(data, pareto, output_dir="soluciones"):
    outputs = []
    for lam, costo, pen, sol in pareto:
        safe_lam = str(lam).replace('.', '_')
        fname = f"mapa_pareto_L{safe_lam}_C{int(costo)}_P{int(pen)}.html"
        out_path = os.path.join(output_dir, fname)
        folium_solution_map(sol, data, outfile=out_path)
        outputs.append(out_path)
    return outputs

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    DIRECTORIO_SALIDA = "soluciones"
    preparar_directorio_soluciones(DIRECTORIO_SALIDA)

    data = load_data()
    debug("Datos cargados correctamente")

    lambdas_a_probar = [0.0, 10.0, 50.0]
    iteraciones = 2000 # Aumentar para mejores resultados

    debug("Ejecutando run_pareto")
    sols, pareto = run_pareto(data, lambdas=lambdas_a_probar, iters=iteraciones, seed=42)
    debug("run_pareto finalizado")

    files = export_pareto_maps(data, pareto, output_dir=DIRECTORIO_SALIDA)
    
    print("\n--- Frontera de Pareto ---")
    for lam, costo, pen, sol in pareto:
        print(f"Lambda={lam}: Costo={costo:.2f}, Penalidad={pen:.2f}, Rutas={len([r for r in sol.routes if r])}")
    
    print(f"\nMapas de la frontera generados en la carpeta '{DIRECTORIO_SALIDA}':")
    for f in files:
        print(f" - {f}")
