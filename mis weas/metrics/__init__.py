"""
Módulo de métricas estéticas para el proyecto ALNS.

Cada archivo dentro de este paquete implementa una métrica de atractivo
visual independiente, facilitando el mantenimiento y la extensión del
modelo estético.
"""

from .solapamiento_zonal import compute_solapamiento_zonal  # noqa: F401
from .solapamiento_geometrico import (
    compute_solapamiento_geometrico,
    penalizar_solapamiento_geometrico,
    inter_route_area_overlap_score,
)  # noqa: F401
from .dispersion import compute_dispersion  # noqa: F401
from .cruces_intra import count_self_crossings  # noqa: F401
from .cruces_inter import (
    count_total_inter_route_crossings,
    route_cuts_norm,
    count_between_routes_crossings,
)  # noqa: F401
from .balance_dist import compute_balance_distance_cv  # noqa: F401
from .balance_stops import compute_balance_stops_cv  # noqa: F401
from .transiciones import compute_zone_transitions  # noqa: F401
from .sector_overlap import compute_sector_overlap  # noqa: F401
from .intrusion import compute_intrusion_km  # noqa: F401
from .compactness import compute_route_compactness_penalty  # noqa: F401
