
import numpy as np

def cost_utilization(seed: int, volumes: np.ndarray, capacities: np.ndarray, base=10.0, noise=0.25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D, M = volumes.shape
    Dc, N = capacities.shape
    assert D == Dc, "dimension mismatch"
    eps = 1e-9
    util = np.zeros((M, N))
    for d in range(D):
        util += volumes[d, :].reshape(M, 1) / (capacities[d, :].reshape(1, N) + eps)
    noise_mat = 1.0 + noise * rng.normal(size=(M, N))
    return base * util * noise_mat

def cost_dispatcher(seed: int, volumes: np.ndarray, capacities: np.ndarray, params: dict) -> np.ndarray:
    model = params.get("model", "utilization")
    if model == "utilization":
        return cost_utilization(seed, volumes, capacities, params.get("base", 10.0), params.get("noise", 0.25))
    elif model == "constant":
        base = params.get("base", 10.0)
        M, N = volumes.shape[1], capacities.shape[1]
        return np.full((M, N), base)
    elif model == "random":
        base = params.get("base", 10.0)
        noise = params.get("noise", 0.25)
        M, N = volumes.shape[1], capacities.shape[1]
        rng = np.random.default_rng(seed)
        return base * (1.0 + noise * rng.normal(size=(M, N)))
    else:
        raise ValueError(f"Unknown cost model {model}")
