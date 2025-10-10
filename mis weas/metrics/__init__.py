"""
Módulo de métricas estéticas para el proyecto ALNS.

Cada archivo dentro de este paquete implementa una métrica de atractivo
visual independiente, facilitando el mantenimiento y la extensión del
modelo estético.
"""

# fmt: off
from .dispersion import compute_dispersion                         # noqa: F401
from .cruces_intra import count_self_crossings                     # noqa: F401
from .cruces_inter import (
    count_total_inter_route_crossings,
    count_between_routes_crossings,
)                                                                   # noqa: F401
from .balance_dist import compute_balance_distance_cv              # noqa: F401
from .balance_stops import compute_balance_stops_cv                # noqa: F401
from .intrusion import compute_intrusion_km                        # noqa: F401
from .complexity import compute_complexity, route_complexity        # noqa: F401
from .coherence import compute_coherence                           # noqa: F401
# fmt: on
