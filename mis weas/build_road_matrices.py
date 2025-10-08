# build_road_matrices.py
# Genera distances.csv y times.csv "por calles" usando OSRM /table.
# Requiere: requests, pandas, numpy, tqdm
import os
import time
import math
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# === Ajusta estas rutas si es necesario (coinciden con tu código actual) ===
I1_DIR = os.path.join("Datos P5", "i1")
OVERVIEW_CSV = os.path.join(I1_DIR, "overview.csv")
DEMANDS_CSV  = os.path.join(I1_DIR, "demands.csv")

# Salidas (se pueden sobreescribir los oficiales si quieres)
OUT_DIST_CSV = os.path.join(I1_DIR, "distances1.csv")
OUT_TIME_CSV = os.path.join(I1_DIR, "times1.csv")

# === OSRM ===
# Usa el público o tu instancia local (recomendado para datasets más grandes)
OSRM_URL = "https://router.project-osrm.org"   # ej. "http://localhost:5000"
PROFILE  = "driving"                            # driving / car

# Tamaño máximo por bloque (fuente + destino) para no saturar el /table
# El público suele aceptar ~100 coordenadas; dividimos en bloques conservadores.
BLOCK = 40

# Retries con backoff simple en caso de 429/500/etc.
MAX_RETRIES = 4
SLEEP_BASE  = 1.5

def load_nodes():
    df_over = pd.read_csv(OVERVIEW_CSV)
    df_dem  = pd.read_csv(DEMANDS_CSV)

    depot_lat = float(df_over.loc[0, "depot_latitude"])
    depot_lon = float(df_over.loc[0, "depot_longitude"])

    # Lista de (lat, lon), con el depósito primero
    nodes = [(depot_lat, depot_lon)] + list(
        zip(df_dem["latitude"].astype(float).tolist(),
            df_dem["longitude"].astype(float).tolist())
    )
    return nodes

def osrm_table_block(coords, src_idx, dst_idx):
    """
    Llama OSRM /table para subconjuntos de índices (src_idx, dst_idx).
    Devuelve matrices parciales (durations [s], distances [m]) como np.array
    de shape (len(src_idx), len(dst_idx)).
    """
    # OSRM GET /table usa "lon,lat;lon,lat;..." en la URL y query params sources/destinations
    def fmt_coord(i):
        lat, lon = coords[i]
        return f"{lon:.6f},{lat:.6f}"

    all_ids = sorted(set(src_idx) | set(dst_idx))
    coord_str = ";".join(fmt_coord(i) for i in all_ids)

    # Mapear posiciones locales
    id_to_local = {nid: k for k, nid in enumerate(all_ids)}
    src_local = ";".join(str(id_to_local[i]) for i in src_idx)
    dst_local = ";".join(str(id_to_local[i]) for i in dst_idx)

    url = f"{OSRM_URL}/table/v1/{PROFILE}/{coord_str}"
    params = {
        "sources": src_local,
        "destinations": dst_local,
        "annotations": "duration,distance"
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                j = r.json()
                durations = j.get("durations")
                distances = j.get("distances")
                if durations is None or distances is None:
                    raise RuntimeError("Respuesta OSRM sin 'durations' o 'distances'")
                D = np.array(durations, dtype=float)  # segundos
                M = np.array(distances, dtype=float)  # metros
                return D, M
            elif r.status_code in (429, 502, 503, 504, 500):
                # Backoff exponencial suave
                sleep_s = SLEEP_BASE * (2 ** (attempt - 1))
                time.sleep(sleep_s)
            else:
                raise RuntimeError(f"OSRM /table HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = SLEEP_BASE * (2 ** (attempt - 1))
            time.sleep(sleep_s)

def build_tables(coords):
    """
    Construye matrices completas NxN de duration [s] y distance [m] por OSRM,
    consultando en bloques para evitar límites del servidor.
    """
    N = len(coords)
    Dur = np.zeros((N, N), dtype=float)
    Dis = np.zeros((N, N), dtype=float)

    # Particionamos indices en bloques
    blocks = [list(range(i, min(i+BLOCK, N))) for i in range(0, N, BLOCK)]

    for bi, src in enumerate(tqdm(blocks, desc="Fuentes", leave=False)):
        for bj, dst in enumerate(tqdm(blocks, desc="Destinos", leave=False)):
            D, M = osrm_table_block(coords, src, dst)
            # Escribimos en el slice correspondiente
            Dur[np.ix_(src, dst)] = D
            Dis[np.ix_(src, dst)] = M

    # OSRM pone 0 en diagonal; mantenemos 0
    return Dur, Dis

def save_long_format(coords, Dur, Dis):
    """
    Guarda CSVs en formato largo como los que espera tu load_data():
      origin_latitude, origin_longitude, destination_latitude, destination_longitude, distance
      ... y análogo con 'time'
    """
    rows_dist = []
    rows_time = []
    N = len(coords)

    for i in range(N):
        oi_lat, oi_lon = coords[i]
        for j in range(N):
            dj_lat, dj_lon = coords[j]
            rows_dist.append({
                "origin_latitude":       oi_lat,
                "origin_longitude":      oi_lon,
                "destination_latitude":  dj_lat,
                "destination_longitude": dj_lon,
                "distance":              float(Dis[i, j]) if np.isfinite(Dis[i, j]) else 0.0
            })
            rows_time.append({
                "origin_latitude":       oi_lat,
                "origin_longitude":      oi_lon,
                "destination_latitude":  dj_lat,
                "destination_longitude": dj_lon,
                "time":                  float(Dur[i, j]) if np.isfinite(Dur[i, j]) else 0.0
            })

    df_dist = pd.DataFrame(rows_dist)
    df_time = pd.DataFrame(rows_time)

    # Guardamos
    os.makedirs(I1_DIR, exist_ok=True)
    df_dist.to_csv(OUT_DIST_CSV, index=False)
    df_time.to_csv(OUT_TIME_CSV, index=False)
    print(f"OK → {OUT_DIST_CSV}  ({len(df_dist)} filas)")
    print(f"OK → {OUT_TIME_CSV}  ({len(df_time)} filas)")

def main():
    coords = load_nodes()
    print(f"Nodos (incluye depot): {len(coords)}")
    Dur, Dis = build_tables(coords)
    save_long_format(coords, Dur, Dis)

if __name__ == "__main__":
    main()
