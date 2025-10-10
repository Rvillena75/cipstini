import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

from alns_core import (
    BASE_DIR,
    export_pareto_dual_maps,
    load_data,
    preparar_directorio_soluciones,
    run_single_experiment,
    validar_solucion_final,
    _load_shape_cache,
    _save_shape_cache,
)
from metrics.aesthetic import esthetics_breakdown_final

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib opcional
    Parallel = None
    delayed = None


DEFAULT_WEIGHTS: Dict[str, float] = {
    # Métricas de balance
    "w_balance_dist": 60.0,    # Balance de distancias entre rutas
    "w_balance_stops": 60.0,   # Balance de paradas entre rutas
    
    # Métricas de forma
    "w_dispersion": 40.0,      # Dispersión de clientes en cada ruta
    "w_complexity": 35.0,      # Complejidad geométrica (bending energy)
    
    # Métricas de interferencia
    "w_cruces_intra": 30.0,    # Cruces dentro de una misma ruta
    "w_cruces_inter": 40.0,    # Cruces entre rutas diferentes
    "w_intrusion": 400.0,      # Intrusión entre rutas

    # Coherencia territorial
    "w_coherence": 50.0,       # Clientes “fuera” del centroide de su ruta
}

DEFAULT_LAMBDAS: Tuple[int, ...] = (0.0,)
DEFAULT_ITERATIONS: int = 1000
DEFAULT_RUNS: int = 5
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
    *,
    generate_maps: bool = True,
    quiet: bool = False,
    jobs: int = 1,
) -> None:
    weights = weights or DEFAULT_WEIGHTS
    verbose = not quiet

    _load_shape_cache()
    preparar_directorio_soluciones(output_dir, verbose=verbose)

    if verbose:
        print("Cargando datos...")
    data = load_data()

    if verbose:
        print("Parámetros de ejecución:")
        print(f"  - Lambdas: {lambdas}")
        print(f"  - Iteraciones por corrida: {iterations}")
        print(f"  - Número de corridas: {runs}")
        if jobs != 1:
            print(f"  - Paralelismo (jobs): {jobs}")

    start = time.perf_counter()
    all_pareto_solutions: List[Tuple[float, float, float, object]] = []

    seeds = list(range(runs))

    worker_verbose = verbose and jobs == 1

    if jobs != 1 and (Parallel is None or delayed is None):
        raise RuntimeError("joblib no está instalado; instala 'joblib' o ejecuta con --jobs 1.")

    if jobs != 1 and runs > 1:
        parallel = Parallel(n_jobs=jobs, backend="loky")
        results = parallel(
            delayed(run_single_experiment)(
                base_seed=seed,
                data=data,
                weights=weights,
                lambdas=list(lambdas),
                iters=iterations,
                verbose=False,
            )
            for seed in seeds
        )
        for pareto in results:
            all_pareto_solutions.extend(pareto)
    else:
        for seed in seeds:
            pareto = run_single_experiment(
                base_seed=seed,
                data=data,
                weights=weights,
                lambdas=list(lambdas),
                iters=iterations,
                verbose=worker_verbose,
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

    if verbose:
        print("\n" + "=" * 50)
        print("REPORTE DE LA FRONTERA DE PARETO FINAL CONSOLIDADA")
        print("=" * 50)
    for idx, (lam, costo, pen, sol) in enumerate(final_pareto, start=1):
        nombre_sol = f"Solución {idx} (λ={lam})"
        validar_solucion_final(sol, data, nombre_sol, verbose=verbose)
        breakdown = esthetics_breakdown_final(sol, data, weights, lam=lam)
        if verbose:
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

    if generate_maps:
        generated_files = export_pareto_dual_maps(data, final_pareto, output_dir=output_dir)
    else:
        generated_files = []

    elapsed = time.perf_counter() - start
    _save_shape_cache()
    if verbose:
        print(f"\nTiempo total del experimento: {elapsed:.2f} segundos")
        if generate_maps:
            print(f"Mapas generados: {len(generated_files)} en '{output_dir}'.")
        else:
            print("Mapas no generados (flag --skip-maps activado).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta la heurística ALNS con métricas visuales.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None, help="Listado de valores λ (por defecto usa configuración interna).")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Iteraciones por corrida.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Número de corridas (semillas).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directorio de salida.")
    parser.add_argument("--skip-maps", action="store_true", help="No exportar mapas HTML al finalizar.")
    parser.add_argument("--quiet", action="store_true", help="Minimiza la salida por consola.")
    parser.add_argument("--jobs", type=int, default=1, help="Número de procesos paralelos para corridas independientes.")

    args = parser.parse_args()

    lam_tuple = tuple(args.lambdas) if args.lambdas is not None else DEFAULT_LAMBDAS

    main(
        weights=DEFAULT_WEIGHTS,
        lambdas=lam_tuple,
        iterations=args.iterations,
        runs=args.runs,
        output_dir=args.output_dir,
        generate_maps=not args.skip_maps,
        quiet=args.quiet,
        jobs=args.jobs,
    )
