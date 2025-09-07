
import numpy as np

def capacity_dispatcher(seed: int, N: int, D: int, params: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dist = params.get("distribution", "uniform")

    if dist == "uniform":
        low = params.get("low", 60.0)
        high = params.get("high", 100.0)
        return np.vstack([rng.uniform(low, high, size=N) for _ in range(D)])

    elif dist == "normal":
        mean = params.get("mean", 100.0)
        std = params.get("std", 10.0)
        return np.maximum(1e-6, rng.normal(mean, std, size=(D, N)))

    elif dist == "fixed":
        val = params.get("value", 100.0)
        return np.full((D, N), val)

    else:
        raise ValueError(f"Unknown capacity distribution {dist}")
