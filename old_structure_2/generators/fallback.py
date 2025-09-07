
from __future__ import annotations
import numpy as np

def constant_fallback(M: int, value: float = 1000.0) -> np.ndarray:
    """Return (M x 1) fallback cost column."""
    return np.full((M, 1), float(value), dtype=float)
