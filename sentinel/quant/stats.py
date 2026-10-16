"""Small numeric helpers."""

from __future__ import annotations

import numpy as np


def weighted_quantile(values, weights, q: float) -> float:
    """Weighted quantile with linear interpolation on the weighted CDF midpoints."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(v)
    v, w = v[mask], w[mask]
    if v.size == 0:
        return float("nan")
    if v.size == 1:
        return float(v[0])
    if w.sum() <= 0:
        w = np.ones_like(v)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cdf, v))


def quantiles(values, weights) -> dict[str, float]:
    return {
        "q25": weighted_quantile(values, weights, 0.25),
        "q50": weighted_quantile(values, weights, 0.50),
        "q75": weighted_quantile(values, weights, 0.75),
        "min": float(np.nanmin(values)) if len(values) else float("nan"),
        "max": float(np.nanmax(values)) if len(values) else float("nan"),
    }
