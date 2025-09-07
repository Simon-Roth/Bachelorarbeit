
from __future__ import annotations
import numpy as np

def uniform_volumes(seed: int, M: int, D: int, low=3.0, high=12.0) -> np.ndarray:
    """Return (D x M) demand matrix; i.e., volumes[d,j]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(D, M))

def lognormal_volumes(seed: int, M: int, D: int, mu=2.0, sigma=0.5, min_val=1e-2) -> np.ndarray:
    """Return (D x M) demand matrix from lognormal per dimension."""
    rng = np.random.default_rng(seed)
    V = np.exp(rng.normal(loc=mu, scale=sigma, size=(D, M)))
    return np.maximum(V, min_val)
