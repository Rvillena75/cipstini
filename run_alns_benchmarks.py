# run_alns_benchmarks_cost_only.py
"""
Benchmark ALNS (Solomon) sin criterios estéticos:
-    timeM = distM.copy()

    return {
        "N": n,
        "K": int(num_vehicles),
        "nodes": nodes,                   # [(lat,lon), ...] para estética
        "coords": coords,                 # (n,2)
        "distM": distM,                   # (n,n) en "unidades"
        "timeM": timeM,                   # (n,n) en minutos
        "demand_size": demand,            # vector
        "demand_srv_s": service,          # minutos de servicio
        "tw_start_min": ready,            # apertura TW (min)
        "tw_end_min": due,               # cierre TW (min)
        "vehicle_caps": [float(vehicle_capacity)] * int(num_vehicles),
        "fixed_route_cost": float(fixed_route_cost),
        "cost_per_meter": float(cost_per_distance),
        "cost_per_cap": float(cost_per_capacity),
        "horizon_minutes": float(due[0]) if len(due) else 0.0,
        "start_at": 0.0,
        # Parámetros estéticos por defecto
        "dispersion_shape_weights": (0.4, 0.3, 0.2, 0.1),
        "dispersion_ecc_cap": 5.0
    }lam = 0.0
- Pone todos los pesos estéticos en 0.0
- Reporta costo, rutas y gap vs. best-known (si está disponible)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import random
import sys
import numpy as np
import pandas as pd

# Importa directamente tu core
# === FIX DE IMPORTACIÓN (carpeta "mis weas" con espacio) ======================
THIS_DIR = Path(__file__).resolve().parent
CORE_DIR = THIS_DIR / "mis weas"            # <— aquí está alns_core.py
if not CORE_DIR.exists():
    raise RuntimeError(f"No existe la carpeta core: {CORE_DIR}")
sys.path.insert(0, str(CORE_DIR))            # agrega "mis weas/" al sys.path

from alns_core import alns_single_run, solution_cost  # mismo directorio

BASE_DIR = Path(__file__).resolve().parent

# === Construcción de datos desde CSV Solomon ===
def build_solomon_data(
    csv_path: Path,
    vehicle_capacity: float = 200.0,
    num_vehicles: int = 25,
    fixed_route_cost: float = 0.0,
    cost_per_distance: float = 1.0,
    cost_per_capacity: float = 0.0,
) -> Dict:
    """
    Lee un CSV estilo Solomon y retorna el diccionario `data` que espera el core.
    Asume:
      - timeM = distM (velocidad 1 => 1 unidad distancia = 1 minuto)
      - service_min en columna 'SERVICE TIME'
      - ventanas [READY TIME, DUE DATE]
      - depósito en fila 0
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró: {csv_path}")

    # Lee el CSV con los nombres de columnas originales de Solomon
    df = pd.read_csv(csv_path)
    
    # Mapeo de nombres de columnas Solomon a nombres esperados
    column_map = {
        'XCOORD.': 'x',
        'YCOORD.': 'y',
        'DEMAND': 'demand',
        'READY TIME': 'ready',
        'DUE DATE': 'due',
        'SERVICE TIME': 'service'
    }
    
    # Verifica que existan las columnas necesarias
    req_cols = set(column_map.keys())
    faltantes = req_cols - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {csv_path.name}: {faltantes}")
    
    # Renombra las columnas al formato esperado
    df = df.rename(columns=column_map)
    
    # Convierte coordenadas x,y a formato lat,lon (aproximación simple para pruebas)
    # Asumimos escala Solomon -> grados (1 unidad ≈ 1 grado)
    nodes = [(float(row['x']), float(row['y'])) for _, row in df.iterrows()]
    coords = df[["x", "y"]].to_numpy(dtype=float)
    demand = df["demand"].to_numpy(dtype=float)
    ready = df["ready"].to_numpy(dtype=float)
    due = df["due"].to_numpy(dtype=float)
    service = df["service"].to_numpy(dtype=float)

    n = len(df)
    # Matriz de distancias euclidianas
    distM = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            distM[i, j] = float((dx*dx + dy*dy) ** 0.5)

    # timeM = distM (1 unidad = 1 minuto)
    timeM = distM.copy()

    return {
        "N": n,
        "K": int(num_vehicles),
        "coords": coords,                 # (n,2)
        "distM": distM,                   # (n,n) en “unidades”
        "timeM": timeM,                   # (n,n) en minutos
        "demand_size": demand,            # vector
        "demand_srv_s": service,          # minutos de servicio
        "tw_start_min": ready,            # apertura TW (min)
        "tw_end_min": due,                # cierre TW (min)
        "vehicle_caps": [float(vehicle_capacity)] * int(num_vehicles),
        "fixed_route_cost": float(fixed_route_cost),
        "cost_per_meter": float(cost_per_distance),
        "cost_per_cap": float(cost_per_capacity),
        "horizon_minutes": float(due[0]) if len(due) else 0.0,
        "start_at": 0.0,
    }

# === Benchmarks (rutas de ejemplo) ===
BENCHMARKS: Dict[str, Dict] = {
    "C101": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C101.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 828.94,
    },

    '''
    "C102": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C102.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 828.94,
    },
    '''
    "C103": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C103.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 828.06,
    },
    '''
    "C104": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C104.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 824.78,
    },
    '''
    "R101": {
        "path": BASE_DIR / "solomon_dataset" / "R1" / "R101.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1650.80,
    },
    '''
    "R102": {
        "path": BASE_DIR / "solomon_dataset" / "R1" / "R102.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1487.68,
    },
    "R201": {
        "path": BASE_DIR / "solomon_dataset" / "R2" / "R201.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1263.27,
    },
    '''
    "R202": {
        "path": BASE_DIR / "solomon_dataset" / "R2" / "R202.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1084.98,
    },
    
    "RC101": {
        "path": BASE_DIR / "solomon_dataset" / "RC1" / "RC101.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1651.02,
    },
    
    "RC102": {
        "path": BASE_DIR / "solomon_dataset" / "RC1" / "RC102.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1508.27,
    },
}

def format_routes(routes: List[List[int]]) -> str:
    return " | ".join("[" + ",".join(map(str, r)) + "]" for r in routes)

def run_instance(
    name: str,
    cfg: Dict,
    iters: int = 1200,
    seed: int = 42,
    max_attempts: int = 10,
) -> Optional[Tuple[float, List[List[int]], float, int]]:
    """
    Ejecuta ALNS en una instancia SOLO COSTO (lam=0) y devuelve:
      (cost, routes, elapsed_sec, K)
    """
    # Pesos estéticos anulados
    weights = {
        "w_intrusion": 0.0,
        "w_dispersion": 0.0,
        "w_cruces_intra": 0.0,
        "w_cruces_inter": 0.0,
        "w_balance_dist": 0.0,
        "w_balance_stops": 0.0,
    }

    attempts = 0
    while attempts < max_attempts:
        attempt_seed = seed + attempts
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)

        data = build_solomon_data(
            csv_path=cfg["path"],
            vehicle_capacity=float(cfg.get("capacity", 200.0)),
            num_vehicles=int(cfg.get("vehicles", 25)),
        )
        
        # Agregar 'nodes' si no está presente, usando 'coords'
        if "nodes" not in data:
            data["nodes"] = [tuple(row) for row in data["coords"].tolist()]
        
        start = time.perf_counter()
        try:
            sol = alns_single_run(
                data=data,
                weights=weights,        # todos 0.0
                lam=0.0,                # SIN estética en FO
                iters=iters,
                seed=attempt_seed,
                use_fast_esthetics=True # acelera evaluaciones estéticas residuales
            )
        except ValueError:
            attempts += 1
            continue

        elapsed = time.perf_counter() - start
        cost = float(solution_cost(sol, data))
        routes = [r for r in sol.routes if r]
        return cost, routes, elapsed, int(data["K"])

    return None

def main():
    print("=== Benchmark ALNS — SOLO COSTO (lam=0) ===")
    total_gap, gap_count = 0.0, 0

    iters = 1200
    seed = 42
    max_attempts = 10
    gaps = [] # Guardará tuplas con los detalles: (nombre, resultado, diferencia, gap)

    for name, cfg in BENCHMARKS.items():
        path = cfg["path"]
        if not path.exists():
            print(f"- {name}: no se encontró el CSV en {path}")
            continue

        result = run_instance(name, cfg, iters=iters, seed=seed, max_attempts=max_attempts)
        if result is None:
            print(f">>> {name}: no se obtuvo solución tras {max_attempts} intentos.")
            continue

        cost, routes, elapsed, K = result
        print(f"\n[{name}] Vehículos={K}  Costo={cost:,.2f}  Tiempo={elapsed:.1f}s")
        print(f"Rutas (primeras 4): {format_routes(routes[:4])}{' ...' if len(routes) > 4 else ''}")

        best = cfg.get("best_cost")
        
        # --- INICIO DEL BLOQUE MEJORADO ---
        if isinstance(best, (int, float)) and best > 0:
            # 1. Calculamos el GAP una sola vez y lo reutilizamos en todo el bloque.
            gap = 100.0 * (cost - best) / best
            total_gap += gap
            gap_count += 1
            
            # 2. Usamos 'gap:+.2f' para mostrar siempre el signo (+/-), lo que es más informativo.
            print(f"Best-known={best:,.2f}  |  GAP={gap:+.2f}%")
            
            if cost == best:
                print("-> Solución best-known alcanzada.")
            elif cost > best:
                diferencia = cost - best
                print(f"-> Mejor el best-known por: {diferencia:,.2f}")
                # 3. Guardamos datos estructurados en lugar de un string pre-formateado.
                gaps.append((name, "Peor", diferencia, gap))
            else: # cost < best
                diferencia = best - cost
                print(f"-> ¡Mejoramos el best-known por: {diferencia:,.2f}!")
                gaps.append((name, "Mejor", diferencia, gap))
    


    if gap_count:
        print(f"\nGAP PROMEDIO ({gap_count} instancias): {total_gap / gap_count:.2f}%")
        print("--- Detalles de las diferencias ---")
        
        # 4. Mejoramos el reporte final para que sea más claro, usando los datos estructurados.
        for name_inst, resultado, diff, gap_val in gaps:
            print(f"  - {name_inst}: {resultado} que el best-known. Dif: {diff:,.2f} ({gap_val:+.2f}%)")
            
    else:
        print("\nNo se pudo calcular GAP promedio (faltan best-known).")

if __name__ == "__main__":
    main()


