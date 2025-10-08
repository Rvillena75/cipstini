from typing import List, Tuple

def count_inter_route_crossings(route1: List[int], route2: List[int], nodes: List[Tuple[float, float]]) -> int:
    """
    Cuenta cruces entre segmentos de DOS rutas diferentes.
    - No cuenta "cruces" cuando los segmentos comparten un nodo (e.g., depot).
    - Complejidad O(m*n), donde m,n = #segmentos de cada ruta.
    """
    if not route1 or not route2:
        return 0
        
    # Caminos con depósito al inicio y final
    path1 = [0] + route1 + [0]
    path2 = [0] + route2 + [0]
    segments1 = [(path1[i], path1[i+1]) for i in range(len(path1) - 1)]
    segments2 = [(path2[i], path2[i+1]) for i in range(len(path2) - 1)]

    def ccw(A: int, B: int, C: int) -> bool:
        ax, ay = nodes[A]; bx, by = nodes[B]; cx, cy = nodes[C]
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    def intersect(s1: Tuple[int, int], s2: Tuple[int, int]) -> bool:
        a, b = s1; c, d = s2
        # Si comparten algún nodo, no se considera cruce (e.g., depot)
        if len({a, b, c, d}) < 4:
            return False
        # Test de intersección de segmentos en 2D
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    cnt = 0
    for s1 in segments1:
        for s2 in segments2:
            if intersect(s1, s2):
                cnt += 1
    return cnt

def count_total_inter_route_crossings(routes: List[List[int]], nodes: List[Tuple[float, float]]) -> float:
    """
    Calcula el número total de cruces entre todas las parejas de rutas.
    Devuelve el valor normalizado por el número de rutas.
    """
    if len(routes) < 2:
        return 0.0
        
    total_crossings = 0
    route_pairs = 0
    
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            crossings = count_inter_route_crossings(routes[i], routes[j], nodes)
            total_crossings += crossings
            route_pairs += 1
            
    # Normalización: promedio de cruces por par de rutas
    return total_crossings / route_pairs if route_pairs > 0 else 0.0