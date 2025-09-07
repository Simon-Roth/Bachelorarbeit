
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BinSpec:
    bin_id: int
    capacity: float
    kind: str = "regular"  # e.g., ICU / ward / day

@dataclass
class ItemSpec:
    item_id: int
    volume: float

@dataclass
class Instance:
    bins: List[BinSpec]
    items: List[ItemSpec]
    cost: pd.DataFrame                # shape (M, N+1), last col 'fallback'
    edge_mask: pd.DataFrame           # bool mask, same shape
    fallback_col: str = "fallback"

def _ensure_positive(arr: np.ndarray, min_val: float) -> np.ndarray:
    return np.maximum(arr, min_val)

def _random_costs(M: int, N: int, base: float = 1.0, noise: float = 0.1, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return base * (1.0 + noise * rng.normal(size=(M, N)))

def make_uniform(seed: int, N: int, M: int, capacity_range=(60, 100)) -> Instance:
    rng = np.random.default_rng(seed)
    caps = rng.uniform(*capacity_range, size=N)
    bins = [BinSpec(i, float(c), kind="regular") for i, c in enumerate(caps)]
    vols = rng.uniform(3, 12, size=M)
    items = [ItemSpec(j, float(v)) for j, v in enumerate(vols)]
    cost = _random_costs(M, N, base=10.0, noise=0.25, rng=rng)
    fallback = 1000.0 * np.ones((M, 1))
    full_cost = np.concatenate([cost, fallback], axis=1)
    edge_mask = np.ones_like(full_cost, dtype=bool)
    df_cost = pd.DataFrame(full_cost, columns=[f"bin_{i}" for i in range(N)] + ["fallback"])
    df_mask = pd.DataFrame(edge_mask, columns=df_cost.columns)
    return Instance(bins=bins, items=items, cost=df_cost, edge_mask=df_mask)

def make_lognormal_sparse(seed: int, N: int, M: int, edge_p: float = 0.4) -> Instance:
    rng = np.random.default_rng(seed)
    caps = rng.uniform(80, 140, size=N)
    bins = [BinSpec(i, float(c), kind="regular") for i, c in enumerate(caps)]
    vols = np.exp(rng.normal(loc=2.0, scale=0.5, size=M))
    vols = _ensure_positive(vols, 1e-2)
    items = [ItemSpec(j, float(v)) for j, v in enumerate(vols)]
    mask = rng.uniform(size=(M, N)) < edge_p
    for j in range(M):
        if not mask[j].any():
            mask[j, rng.integers(0, N)] = True
    caps_arr = np.array([b.capacity for b in bins], dtype=float)
    base_cost = 10.0 * (vols.reshape(-1, 1) / (caps_arr.reshape(1, -1) + 1e-9))
    noise = 1.0 + 0.25 * rng.normal(size=base_cost.shape)
    cost = base_cost * noise
    large = 1e6
    cost = np.where(mask, cost, large)
    fallback = 1000.0 * np.ones((M, 1))
    full_cost = np.concatenate([cost, fallback], axis=1)
    edge_mask = np.concatenate([mask, np.ones((M,1), dtype=bool)], axis=1)
    df_cost = pd.DataFrame(full_cost, columns=[f"bin_{i}" for i in range(N)] + ["fallback"])
    df_mask = pd.DataFrame(edge_mask, columns=df_cost.columns)
    return Instance(bins=bins, items=items, cost=df_cost, edge_mask=df_mask)

def make_hospital(seed: int, N: int, M: int):
    rng = np.random.default_rng(seed)
    kinds = []
    for i in range(N):
        r = rng.uniform()
        if r < 0.2:
            kinds.append("ICU")
        elif r < 0.8:
            kinds.append("WARD")
        else:
            kinds.append("DAY")
    caps = []
    for k in kinds:
        if k == "ICU":
            caps.append(rng.uniform(40, 60))
        elif k == "WARD":
            caps.append(rng.uniform(80, 120))
        else:
            caps.append(rng.uniform(60, 80))
    bins = [BinSpec(i, float(c), kind=k) for i, (c, k) in enumerate(zip(caps, kinds))]
    vols_small = np.exp(rng.normal(1.8, 0.3, size=int(M*0.7)))
    vols_med   = np.exp(rng.normal(2.5, 0.25, size=int(M*0.25)))
    vols_large = np.exp(rng.normal(3.0, 0.2, size=M - len(vols_small) - len(vols_med)))
    vols = np.concatenate([vols_small, vols_med, vols_large])
    rng.shuffle(vols)
    vols = _ensure_positive(vols, 0.5)
    items = [ItemSpec(j, float(v)) for j, v in enumerate(vols)]
    pref = rng.choice(["ICU", "WARD", "DAY"], size=M, p=[0.15, 0.7, 0.15])
    caps_arr = np.array([b.capacity for b in bins], dtype=float)
    base_cost = 8.0 * (vols.reshape(-1,1) / (caps_arr.reshape(1,-1) + 1e-9))
    penalty = np.zeros_like(base_cost)
    for i, b in enumerate(bins):
        for j in range(M):
            if pref[j] != b.kind:
                if b.kind == "ICU":
                    penalty[j, i] = 3.0
                elif b.kind == "WARD":
                    penalty[j, i] = 6.0
                elif b.kind == "DAY":
                    penalty[j, i] = 12.0 if vols[j] > 7.5 else 4.0
    noise = 1.0 + 0.15 * rng.normal(size=base_cost.shape)
    cost = (base_cost + penalty) * noise
    mask = np.ones((M, N), dtype=bool)
    for i, b in enumerate(bins):
        if b.kind == "DAY":
            mask[:, i] &= (vols <= 8.0)
    for j in range(M):
        if not mask[j].any():
            i_best = int(np.argmin(cost[j]))
            mask[j, i_best] = True
    large = 1e6
    cost = np.where(mask, cost, large)
    fallback = 5000.0 * np.ones((M,1))
    full_cost = np.concatenate([cost, fallback], axis=1)
    edge_mask = np.concatenate([mask, np.ones((M,1), dtype=bool)], axis=1)
    df_cost = pd.DataFrame(full_cost, columns=[f"bin_{i}" for i in range(N)] + ["fallback"])
    df_mask = pd.DataFrame(edge_mask, columns=df_cost.columns)
    return Instance(bins=bins, items=items, cost=df_cost, edge_mask=df_mask)

def make_instance(seed: int, N: int, M: int, scenario: str = "uniform"):
    scenario = scenario.lower()
    if scenario == "uniform":
        return make_uniform(seed, N, M)
    elif scenario == "lognormal_sparse":
        return make_lognormal_sparse(seed, N, M)
    elif scenario == "hospital":
        return make_hospital(seed, N, M)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

def instance_to_frames(inst) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins_df = pd.DataFrame({"bin_id": [b.bin_id for b in inst.bins],
                            "capacity": [b.capacity for b in inst.bins],
                            "kind": [b.kind for b in inst.bins]})
    items_df = pd.DataFrame({"item_id": [it.item_id for it in inst.items],
                             "volume": [it.volume for it in inst.items]})
    cost_df  = inst.cost.copy()
    return bins_df, items_df, cost_df
