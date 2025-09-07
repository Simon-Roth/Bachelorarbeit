
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

FALLBACK_COL = "fallback"

@dataclass
class OfflineConfig:
    # Problem sizes
    N: int
    M: int
    D: int = 1  # number of capacity dimensions for vector bin packing

    # Optional reserve per dimension (0..1). If scalar given, broadcast to D.
    reserve: Any = 0.0

    # Fallback controls
    use_fallback: bool = True

    # Random seed for reproducibility
    seed: int = 42

    # Scenario/generator parameters (free-form)
    params: Dict[str, Any] = field(default_factory=dict)

    def reserve_vec(self) -> np.ndarray:
        rv = self.reserve
        if isinstance(rv, (int, float)):
            return np.full(self.D, float(rv), dtype=float)
        arr = np.asarray(rv, dtype=float).reshape(-1)
        assert arr.shape[0] == self.D, "reserve must be scalar or length-D vector"
        return arr
