
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from config import FALLBACK_COL, OfflineConfig

@dataclass
class Instance:
    bins_df: pd.DataFrame          # columns: bin_id, kind, capacity_d0, ..., capacity_d{D-1}
    items_df: pd.DataFrame         # columns: item_id, volume_d0, ..., volume_d{D-1}
    cost_df: pd.DataFrame          # columns: bin_0..bin_{N-1}[, fallback]
    mask_df: pd.DataFrame          # same shape as cost_df (bool)
    D: int                         # number of dimensions

def build_instance(cfg: OfflineConfig,
                   graph_gen, volume_gen, cost_gen, fallback_gen,
                   kind: str = "regular") -> Instance:
    rng = np.random.default_rng(cfg.seed)

    # 1) Build capacities per dimension (D x N)
    caps = []
    for d in range(cfg.D):
        low, high = cfg.params.get(f"cap_range_d{d}", (60.0, 100.0))
        caps.append(rng.uniform(low, high, size=cfg.N))
    capacities = np.vstack(caps)  # (D x N)

    # 2) Graph / mask (M x N) over regular bins
    mask = graph_gen(cfg.seed + 11, cfg.M, cfg.N)
    assert mask.shape == (cfg.M, cfg.N)

    # 3) Volumes per item per dimension (D x M)
    volumes = volume_gen(cfg.seed + 17, cfg.M, cfg.D)
    assert volumes.shape == (cfg.D, cfg.M)

    # 4) Costs (M x N) for allowed edges; disallowed edges will be Big-M checked
    costs = cost_gen(cfg.seed + 23, volumes, capacities)
    assert costs.shape == (cfg.M, cfg.N)

    # 5) Apply mask: enforce big-M on forbidden edges
    bigM = cfg.params.get("bigM_forbidden", 1e6)
    costs_masked = np.where(mask, costs, bigM)

    # 6) Fallback (optional)
    if cfg.use_fallback:
        fallback_col = fallback_gen(cfg.M)  # (M x 1)
        cost_full = np.concatenate([costs_masked, fallback_col], axis=1)
        mask_full = np.concatenate([mask, np.ones((cfg.M, 1), dtype=bool)], axis=1)
        columns = [f"bin_{i}" for i in range(cfg.N)] + [FALLBACK_COL]
    else:
        cost_full = costs_masked
        mask_full = mask
        columns = [f"bin_{i}" for i in range(cfg.N)]

    # 7) DataFrames
    cost_df = pd.DataFrame(cost_full, columns=columns)
    mask_df = pd.DataFrame(mask_full, columns=columns)

    # bins_df with vector capacities per dimension
    bins_dict = {"bin_id": list(range(cfg.N)), "kind": [kind]*cfg.N}
    for d in range(cfg.D):
        bins_dict[f"capacity_d{d}"] = capacities[d, :]
    bins_df = pd.DataFrame(bins_dict)

    # items_df with vector volumes per dimension
    items_dict = {"item_id": list(range(cfg.M))}
    for d in range(cfg.D):
        items_dict[f"volume_d{d}"] = volumes[d, :]
    items_df = pd.DataFrame(items_dict)

    # 8) Basic schema checks
    assert cost_df.shape == mask_df.shape, "cost/mask shape mismatch"
    assert not bins_df.isna().any().any(), "NaNs in bins_df"
    assert not items_df.isna().any().any(), "NaNs in items_df"
    assert not cost_df.isna().any().any(), "NaNs in cost_df"
    assert mask_df.dtypes.eq(bool).all(), "mask_df must be boolean"
    return Instance(bins_df=bins_df, items_df=items_df, cost_df=cost_df, mask_df=mask_df, D=cfg.D)
