"""
Herramientas rápidas para perfilar el ALNS y la penalización estética.

Uso:

1. Perfil global con cProfile y abrir luego con snakeviz:

    python profile_alns.py --cprofile --iters 50 --lam 0 250
    snakeviz alns.prof

2. Activar el profiler interno de estética (sin dependencias extra):

    python profile_alns.py --internal --iters 10 --lam 0 250

3. Si tienes line_profiler, ejecuta:

    kernprof -l -v profile_alns.py --iters 10 --lam 0 250

   (Se usan los decoradores @profile ya añadidos en los módulos).
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

import numpy as np

import alns_main  # noqa: F401  (asegura rutas relativas)
from alns_main import main as run_alns, DEFAULT_WEIGHTS, DEFAULT_LAMBDAS
from metrics.aesthetic import (
    enable_aesthetic_profiling,
    disable_aesthetic_profiling,
    get_aesthetic_profile,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_FILE = BASE_DIR / "alns.prof"


def _run_once(lambdas, iterations, runs):
    run_alns(
        weights=DEFAULT_WEIGHTS,
        lambdas=tuple(lambdas),
        iterations=iterations,
        runs=runs,
    )


def run_cprofile(args: argparse.Namespace) -> None:
    profile_path = Path(args.profile_file or DEFAULT_PROFILE_FILE)
    print(f"[cProfile] Ejecutando ALNS, volcando resultados a {profile_path} …")
    cProfile.runctx(
        "_run_once(args.lambdas, args.iterations, args.runs)",
        globals(),
        locals(),
        filename=str(profile_path),
    )
    if args.print_stats:
        stats = pstats.Stats(str(profile_path))
        stats.strip_dirs().sort_stats(args.sort).print_stats(args.top)


def run_internal(args: argparse.Namespace) -> None:
    print("[Profiler interno] activado.")
    enable_aesthetic_profiling(reset=True)
    try:
        _run_once(args.lambdas, args.iterations, args.runs)
    finally:
        disable_aesthetic_profiling()
    data = get_aesthetic_profile(reset=True)
    if not data:
        print("No se registraron datos (¿se llamó a aesthetic_penalty?).")
    else:
        total_time = data.get("total_penalty", (0.0, 0))[0]
        print("Tiempos acumulados por componente (segundos, % sobre estética y número de llamadas):")
        for key, (elapsed, calls) in sorted(data.items(), key=lambda kv: kv[1][0], reverse=True):
            if key == "total_penalty":
                continue
            pct = 100.0 * elapsed / max(total_time, 1e-12)
            print(f"  - {key:20s}: {elapsed:7.4f} s ({pct:5.1f}%)  [{calls} llamadas]")

        other = total_time - sum(val[0] for key, val in data.items() if key != "total_penalty")
        if other > 1e-9:
            pct = 100.0 * other / max(total_time, 1e-12)
            print(f"  - otros (setup/overhead) : {other:7.4f} s ({pct:5.1f}%)")

        total_calls = data.get("total_penalty", (0.0, 0))[1]
        print(f"Total estética: {total_time:.4f} s en {total_calls} llamadas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilidades de profiling para el ALNS.")
    parser.add_argument(
        "--iters",
        dest="iterations",
        type=int,
        default=DEFAULT_LAMBDAS and 10,
        help="Iteraciones por corrida (default: 10).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Número de corridas independientes (default: 1).",
    )
    parser.add_argument(
        "--lam",
        dest="lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="Lista de lambdas a evaluar (default: valores por defecto del main).",
    )

    subparsers = parser.add_subparsers(dest="mode")

    p_cprof = subparsers.add_parser("cprofile", help="Ejecuta con cProfile.")
    p_cprof.add_argument("--profile-file", default=None, help="Ruta del .prof a generar.")
    p_cprof.add_argument("--print-stats", action="store_true", help="Imprime resumen tras el run.")
    p_cprof.add_argument(
        "--sort",
        default="cumtime",
        help="Clave de orden para pstats (cumtime, tottime, etc.).",
    )
    p_cprof.add_argument("--top", type=int, default=25, help="Número de líneas a imprimir.")

    subparsers.add_parser("internal", help="Usa el profiler interno de estética.")

    parser.set_defaults(mode="internal")
    args = parser.parse_args()
    if not args.lambdas:
        args.lambdas = list(DEFAULT_LAMBDAS)
    return args


def main():
    args = parse_args()
    if args.mode == "cprofile":
        run_cprofile(args)
    else:
        run_internal(args)


if __name__ == "__main__":
    main()
