
import numpy as np

def full_graph(seed: int, M: int, N: int) -> np.ndarray:
    return np.ones((M, N), dtype=bool)

def bernoulli_graph(seed: int, M: int, N: int, p: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=(M, N)) < p
    for j in range(M):
        if not mask[j].any():
            mask[j, rng.integers(0, N)] = True
    return mask
