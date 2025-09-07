
import numpy as np

def uniform_volumes(seed: int, M: int, D: int, low=3.0, high=12.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(D, M))

def lognormal_volumes(seed: int, M: int, D: int, mu=2.0, sigma=0.5, min_val=1e-2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    V = np.exp(rng.normal(loc=mu, scale=sigma, size=(D, M)))
    return np.maximum(V, min_val)

def volume_dispatcher(seed: int, M: int, D: int, params: dict) -> np.ndarray:
    dist = params.get("distribution", "uniform")
    if dist == "uniform":
        return uniform_volumes(seed, M, D, params.get("low", 3.0), params.get("high", 12.0))
    elif dist == "lognormal":
        return lognormal_volumes(seed, M, D, params.get("mu", 2.0), params.get("sigma", 0.5))
    else:
        raise ValueError(f"Unknown volume distribution {dist}")
