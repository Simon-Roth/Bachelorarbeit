
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

FALLBACK_COL = "fallback"

@dataclass
class OfflineConfig:
    N: int
    M: int
    D: int = 1
    reserve: Any = 0.0
    use_fallback: bool = True
    seed: int = 42
    params: Dict[str, Any] = field(default_factory=dict)

    def reserve_vec(self) -> np.ndarray:
        rv = self.reserve
        if isinstance(rv, (int, float)):
            return np.full(self.D, float(rv), dtype=float)
        arr = np.asarray(rv, dtype=float).reshape(-1)
        assert arr.shape[0] == self.D, "reserve must be scalar or length-D vector"
        return arr
