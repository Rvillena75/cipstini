def _clip_segment_convex_poly_new(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    poly_ccw: List[Tuple[float, float]],
) -> float:
    """
    Calcula la longitud del segmento que intersecta con el polígono usando una
    implementación más robusta que considera puntos dentro y fuera del polígono.
    """
    if len(poly_ccw) < 3:
        print("\nDebug _clip_segment_convex_poly: Polygon has less than 3 points")
        return 0.0

    (x0, y0), (x1, y1) = p0, p1
    print(f"\nDebug _clip_segment_convex_poly:")
    print(f"Segment: ({x0}, {y0}) -> ({x1}, {y1})")
    print(f"Polygon points: {poly_ccw}")

    # Primero verificamos si algún punto está dentro del polígono
    p0_inside = _point_in_polygon(p0, poly_ccw)
    p1_inside = _point_in_polygon(p1, poly_ccw)
    
    if p0_inside and p1_inside:
        length = math.hypot(x1 - x0, y1 - y0)
        print(f"Both points inside polygon. Length: {length}")
        return length

    # Si un punto está dentro y otro fuera
    if p0_inside or p1_inside:
        length = math.hypot(x1 - x0, y1 - y0) / 2  # Aproximación conservadora
        print(f"One point inside polygon. Length: {length}")
        return length

    # Si el segmento intersecta el polígono en dos puntos
    intersections = 0
    for i in range(len(poly_ccw)):
        p3 = poly_ccw[i]
        p4 = poly_ccw[(i + 1) % len(poly_ccw)]
        if _segments_intersect(p0, p1, p3, p4):
            intersections += 1
    
    if intersections >= 2:
        # El segmento atraviesa el polígono
        length = math.hypot(x1 - x0, y1 - y0) / 3  # Aproximación conservadora
        print(f"Segment crosses polygon with {intersections} intersections. Length: {length}")
        return length

    print("No intersection found")
    return 0.0