
import numpy as np

def constant_fallback(M: int, value: float = 1000.0) -> np.ndarray:
    return np.full((M, 1), float(value))

def fallback_dispatcher(M: int, params: dict):
    if not params.get("use", True):
        return None
    value = params.get("cost", 1000.0)
    return constant_fallback(M, value)
