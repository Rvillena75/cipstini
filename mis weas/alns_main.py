import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from alns_core import (
    BASE_DIR,
    export_pareto_dual_maps,
    load_data,
    precalculate_clusters,
    preparar_directorio_soluciones,
    run_single_experiment,
    validar_solucion_final,
    _load_shape_cache,
    _save_shape_cache,
)
from metrics.aesthetic import esthetics_breakdown_final


DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_solapamiento": 50.0,
    "w_sector": 30.0,
    "w_intrusion": 400.0,
    "w_dispersion": 40.0,
    "w_cruces_intra": 30.0,
    "w_cruces_inter": 40.0,
    "w_balance_dist": 60.0,
    "w_balance_stops": 60.0,
    "w_transiciones": 10.0,
}

DEFAULT_LAMBDAS: Tuple[int, ...] = (0, 250, 500, 100000)
DEFAULT_ITERATIONS: int = 1000
DEFAULT_RUNS: int = 3
DEFAULT_OUTPUT_DIR: Path = BASE_DIR / "soluciones"


def consolidate_pareto(all_solutions: List[Tuple[float, float, float, object]]) -> List[Tuple[float, float, float, object]]:
    sols_sorted = sorted(all_solutions, key=lambda x: (x[1], x[2]))
    final_front = []
    best_est = float("inf")
    for lam, costo, pen, sol in sols_sorted:
        if pen < best_est - 1e-9:
            is_duplicate = any(abs(s[1] - costo) < 1 and abs(s[2] - pen) < 1 for s in final_front)
            if not is_duplicate:
                final_front.append((lam, costo, pen, sol))
                best_est = pen
    return final_front


def main(
    weights: Dict[str, float] = None,
    lambdas: Tuple[float, ...] = DEFAULT_LAMBDAS,
    iterations: int = DEFAULT_ITERATIONS,
    runs: int = DEFAULT_RUNS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    weights = weights or DEFAULT_WEIGHTS

    _load_shape_cache()
    preparar_directorio_soluciones(output_dir)

    print("Cargando datos...")
    data = load_data()

    print("Generando zonas virtuales (clusters)...")
    n_clientes = data["N"] - 1
    n_zonas = min(n_clientes, data["K"])
    if n_zonas > 0:
        cluster_map = precalculate_clusters(data, n_clusters=n_zonas)
    else:
        cluster_map = np.full(data["N"], -1, dtype=int)

    print("Parámetros de ejecución:")
    print(f"  - Lambdas: {lambdas}")
    print(f"  - Iteraciones por corrida: {iterations}")
    print(f"  - Número de corridas: {runs}")

    start = time.perf_counter()
    all_pareto_solutions: List[Tuple[float, float, float, object]] = []

    for seed in range(runs):
        pareto = run_single_experiment(
            base_seed=seed,
            data=data,
            weights=weights,
            lambdas=list(lambdas),
            iters=iterations,
            cluster_map=cluster_map,
        )
        all_pareto_solutions.extend(pareto)

    final_pareto = consolidate_pareto(all_pareto_solutions)
    best_by_lambda: Dict[float, Tuple[float, Tuple[float, float, float, object]]] = {}
    for lam, costo, pen, sol in final_pareto:
        obj = costo + lam * pen
        entry = best_by_lambda.get(lam)
        if entry is None or obj < entry[0] - 1e-9:
            best_by_lambda[lam] = (obj, (lam, costo, pen, sol))

    final_pareto = [
        best_by_lambda[lam][1]
        for lam in sorted(best_by_lambda)
    ]

    print("\n" + "=" * 50)
    print("REPORTE DE LA FRONTERA DE PARETO FINAL CONSOLIDADA")
    print("=" * 50)
    for idx, (lam, costo, pen, sol) in enumerate(final_pareto, start=1):
        nombre_sol = f"Solución {idx} (λ={lam})"
        validar_solucion_final(sol, data, nombre_sol)
        breakdown = esthetics_breakdown_final(sol, data, weights, cluster_map, lam=lam)
        print(f"\n--- {nombre_sol} ---")
        print(f"  - Costo Operativo: {costo:,.2f}")
        print(f"  - Penalización Estética Cruda: {lam * pen:,.2f}")
        print("  - Desglose de Penalizaciones:")
        for key, val in breakdown.items():
            if key == "detalle_sin_pesos" and isinstance(val, dict):
                print("    - detalle_sin_pesos:")
                for sub_key, sub_val in val.items():
                    print(f"        · {sub_key}: {sub_val:.2f}")
            else:
                print(f"    - {key}: {val:.2f}")

    generated_files = export_pareto_dual_maps(data, final_pareto, output_dir=output_dir)

    elapsed = time.perf_counter() - start
    _save_shape_cache()
    print(f"\nTiempo total del experimento: {elapsed:.2f} segundos")
    print(f"Mapas generados: {len(generated_files)} en '{output_dir}'.")


if __name__ == "__main__":
    main()
