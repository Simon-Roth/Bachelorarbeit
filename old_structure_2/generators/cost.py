
from __future__ import annotations
import numpy as np

def cost_utilization(seed: int, volumes: np.ndarray, capacities: np.ndarray, noise=0.25, base=10.0) -> np.ndarray:
    """
    volumes: (D x M)
    capacities: (D x N)
    Returns cost matrix (M x N) based on sum over d of (v[d,j]/(cap[d,i]+eps)), scaled and noised.
    """
    rng = np.random.default_rng(seed)
    D, M = volumes.shape
    Dc, N = capacities.shape
    assert D == Dc, "dimensions mismatch for volumes/capacities"
    eps = 1e-9
    util = np.zeros((M, N), dtype=float)
    for d in range(D):
        util += volumes[d, :].reshape(M, 1) / (capacities[d, :].reshape(1, N) + eps)
    noise_mat = 1.0 + noise * rng.normal(size=(M, N))
    return base * util * noise_mat
