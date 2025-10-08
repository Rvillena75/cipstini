"""Ejecuta ALNS sobre distintos datasets (Solomon y P5) y compara con óptimos conocidos."""

from __future__ import annotations

import math
import random
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ALNS_PATH = BASE_DIR / "mis weas" / "alns_core.py"

spec = importlib.util.spec_from_file_location("alns_module", ALNS_PATH)
alns = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(alns)


def _build_solomon_data(
    csv_path: Path,
    vehicle_capacity: float = 200.0,
    max_vehicles: int = 25,
    fixed_route_cost: float = 0.0,
    cost_per_distance: float = 1.0,
    cost_per_capacity: float = 0.0,
) -> Dict:
    df = pd.read_csv(csv_path)

    coords = df[["XCOORD.", "YCOORD."]].astype(float).to_numpy()
    demands = df["DEMAND"].astype(float).to_numpy()
    service_min = df["SERVICE TIME"].astype(float).to_numpy()
    ready = df["READY TIME"].astype(float).to_numpy()
    due = df["DUE DATE"].astype(float).to_numpy()

    # Distancias euclidianas y tiempos (suponiendo velocidad 1 unidad/min → multiplicar por 60 para segundos)
    diff = coords[:, None, :] - coords[None, :, :]
    distM = np.hypot(diff[..., 0], diff[..., 1])
    timeM = distM * 60.0

    total_demand = demands.sum()
    min_vehicles = math.ceil(total_demand / max(vehicle_capacity, 1e-9))
    num_vehicles = max(max_vehicles, min_vehicles)

    return {
        "N": len(df),
        "K": num_vehicles,
        "nodes": [(float(x), float(y)) for x, y in coords],
        "distM": distM,
        "timeM": timeM,
        "demand_size": demands,
        "demand_srv_s": service_min * 60.0,
        "tw_start_min": ready,
        "tw_end_min": due,
        "vehicle_caps": [float(vehicle_capacity)] * num_vehicles,
        "fixed_route_cost": fixed_route_cost,
        "cost_per_meter": cost_per_distance,
        "cost_per_cap": cost_per_capacity,
        "horizon_minutes": float(due[0]) if len(due) else 0.0,
        "start_at": None,
    }


BENCHMARKS: Dict[str, Dict] = {
    "C101": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C101.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 828.94,
    },
    "C102": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C102.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 828.94,
    },
    "C103": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C103.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 828.06,
    },
    "C104": {
        "path": BASE_DIR / "solomon_dataset" / "C1" / "C104.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 824.78,
    },
    "R101": {
        "path": BASE_DIR / "solomon_dataset" / "R1" / "R101.csv",
        "capacity": 200,
        "vehicles": 25,
        "best_cost": 1650.8,
    },
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

# https://sites.google.com/view/vrptwaalihodzic/po%C4%8Detna-stranica?authuser=0#h.s6qpe53o8nkw


def run_alns_on_dataset(
    name: str,
    config: Dict,
    iters: int = 1000,
    seed: int = 42,
    max_attempts: Optional[int] = None,
    iter_boost_every: int = 5,
    iter_boost: int = 500,
) -> Optional[Tuple[float, List[List[int]], float, int]]:
    base_vehicles = config.get("vehicles", 25)

    weights = {
        "w_solapamiento": 50.0,
        "w_sector": 30.0,
        "w_intrusion": 200.0,
        "w_dispersion": 40.0,
        "w_cruces_intra": 30.0,
        "w_cruces_inter": 40.0,
        "w_balance_dist": 30.0,
        "w_balance_stops": 30.0,
        "w_transiciones": 10.0,
    }

    attempts = 0
    current_iters = iters
    while True:
        if max_attempts is not None and attempts >= max_attempts:
            return None
        attempt_seed = seed + attempts
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)

        vehicles = base_vehicles
        data = _build_solomon_data(
            csv_path=config["path"],
            vehicle_capacity=config.get("capacity", 200.0),
            max_vehicles=vehicles,
        )

        cluster_map = alns.precalculate_clusters(data, n_clusters=min(data["K"], data["N"] - 1))

        start = time.perf_counter()
        try:
            solution = alns.alns_single_run(
                data,
                weights=weights,
                cluster_map=cluster_map,
                lam=0.0,
                iters=current_iters,
                seed=attempt_seed,
                use_fast_esthetics=False,
            )
        except ValueError:
            attempts += 1
            if iter_boost_every > 0 and attempts % iter_boost_every == 0:
                current_iters += iter_boost
                print(
                    f"   - {name}: reintento {attempts}, aumentando iteraciones a {current_iters}"
                )
            continue
        elapsed = time.perf_counter() - start
        cost = alns.solution_cost(solution, data)
        routes = [route for route in solution.routes if route]
        return cost, routes, elapsed, data["K"]


def format_routes(routes: List[List[int]]) -> str:
    return " | ".join("[" + ",".join(map(str, r)) + "]" for r in routes)


def main() -> None:
    print("=== ALNS Benchmarks (Solomon datasets) ===")
    total_gap = 0.0
    gap_count = 0
    max_attempts: Optional[int] = None
    iter_boost_every = 5
    iter_boost = 500
    for name, cfg in BENCHMARKS.items():
        path = cfg["path"]
        if not path.exists():
            print(f"- {name}: archivo no encontrado en {path}")
            continue

        result = run_alns_on_dataset(
            name,
            cfg,
            max_attempts=max_attempts,
            iter_boost_every=iter_boost_every,
            iter_boost=iter_boost,
        )
        if result is None:
            print(
                f"\n>>> {name} <<<\n"
                f"No se encontró solución factible con {cfg.get('vehicles', 'N/A')} vehículos tras "
                f"{('múltiples intentos.' if max_attempts is None else f'{max_attempts} semillas distintas.')}"
            )
            continue

        cost, routes, elapsed, used_k = result
        best = cfg.get("best_cost")
        gap = None
        if best and best > 0:
            gap = (cost - best) / best * 100.0

        print(f"\n>>> {name} ({len(routes)} rutas, K usado={used_k}) <<<")
        print(f"Costo ALNS : {cost:.2f}")
        if best:
            print(f"Costo óptimo/best-known : {best:.2f}")
            if gap is not None:
                print(f"Gap vs óptimo : {gap:.2f}%")
                total_gap += gap
                gap_count += 1
        print(f"Tiempo ejecución : {elapsed:.1f}s")
        print(f"Rutas (indices de clientes): {format_routes(routes[:4])}{' ...' if len(routes) > 4 else ''}")

    if gap_count:
        avg_gap = total_gap / gap_count
        print(f"\nGap promedio sobre {gap_count} instancias: {avg_gap:.2f}%")
    else:
        print("\nNo se pudo calcular el gap promedio (faltan costos de referencia).")


if __name__ == "__main__":
    main()
