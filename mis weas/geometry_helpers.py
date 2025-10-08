def _point_in_polygon(p: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """Determina si un punto está dentro de un polígono usando ray casting."""
    if not poly:
        return False
    x, y = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], 
                       p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Determina si dos segmentos se intersectan."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)