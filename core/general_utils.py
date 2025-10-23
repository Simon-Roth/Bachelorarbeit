# binpacking/core/utils.py
from __future__ import annotations
from typing import Iterable, Sequence, Optional
import random
import numpy as np

def set_global_seed(seed: int) -> None:
    """
    Set seeds across Python's random and NumPy for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Return a modern NumPy RNG (PCG64). If seed=None, it is non-deterministic.
    """
    return np.random.default_rng(seed)

def safe_argmax(arr: np.ndarray) -> int:
    """
    Argmax that breaks ties by the first index deterministically.
    """
    return int(np.argmax(arr))

def validate_capacities(capacities: Sequence[float]) -> None:
    assert all(c > 0 for c in capacities), "All capacities must be positive."

def validate_mask(mask: np.ndarray) -> None:
    assert mask.dtype == np.int_ or mask.dtype == np.bool_, "Feasibility mask must be int/bool."
    assert mask.min() >= 0 and mask.max() <= 1, "Feasibility mask must be 0/1."

def effective_capacity(capacity: float, enforce_slack: bool, slack_fraction: float) -> float:
    """
    Apply slack if enabled. If enforce_slack=True, return (1 - slack_fraction) * capacity.
    Otherwise return capacity.
    """
    if enforce_slack:
        assert 0.0 <= slack_fraction < 1.0, "Slack fraction must be in [0,1)."
        return (1.0 - slack_fraction) * capacity
    return capacity
