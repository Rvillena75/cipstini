from typing import List

import numpy as np


def compute_balance_stops_cv(routes: List[List[int]]) -> float:
    stops = [len(r) for r in routes if r]
    if not stops:
        return 0.0
    mean_stops = float(np.mean(stops))
    if mean_stops <= 1e-9:
        return 0.0
    return 100*float(np.std(stops) / mean_stops)


__all__ = ["compute_balance_stops_cv"]
