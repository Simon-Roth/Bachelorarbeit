
from dataclasses import dataclass
import numpy as np
import pandas as pd
from config import FALLBACK_COL, OfflineConfig
from generators.capacity import capacity_dispatcher
from generators.volume import volume_dispatcher
from generators.cost import cost_dispatcher
from generators.fallback import fallback_dispatcher
from generators.graph import full_graph, bernoulli_graph

@dataclass
class Instance:
    bins_df: pd.DataFrame
    items_df: pd.DataFrame
    cost_df: pd.DataFrame
    mask_df: pd.DataFrame
    D: int

def build_instance(cfg: OfflineConfig, graph_type="full") -> Instance:
    capacities = capacity_dispatcher(cfg.seed, cfg.N, cfg.D, cfg.params.get("capacity", {}))
    if graph_type == "full":
        mask = full_graph(cfg.seed+11, cfg.M, cfg.N)
    elif graph_type == "bernoulli":
        p = cfg.params.get("graph_p", 0.4)
        mask = bernoulli_graph(cfg.seed+11, cfg.M, cfg.N, p)
    else:
        raise ValueError(f"Unknown graph type {graph_type}")
    volumes = volume_dispatcher(cfg.seed+17, cfg.M, cfg.D, cfg.params.get("volume", {}))
    costs = cost_dispatcher(cfg.seed+23, volumes, capacities, cfg.params.get("cost", {}))
    bigM = cfg.params.get("bigM_forbidden", 1e6)
    costs_masked = np.where(mask, costs, bigM)
    fb_col = fallback_dispatcher(cfg.M, cfg.params.get("fallback", {}))
    if fb_col is not None:
        cost_full = np.concatenate([costs_masked, fb_col], axis=1)
        mask_full = np.concatenate([mask, np.ones((cfg.M,1), bool)], axis=1)
        columns = [f"bin_{i}" for i in range(cfg.N)] + [FALLBACK_COL]
    else:
        cost_full, mask_full = costs_masked, mask
        columns = [f"bin_{i}" for i in range(cfg.N)]
    cost_df = pd.DataFrame(cost_full, columns=columns)
    mask_df = pd.DataFrame(mask_full, columns=columns)
    bins_dict = {"bin_id": list(range(cfg.N)), "kind": ["regular"]*cfg.N}
    for d in range(cfg.D):
        bins_dict[f"capacity_d{d}"] = capacities[d,:]
    bins_df = pd.DataFrame(bins_dict)
    items_dict = {"item_id": list(range(cfg.M))}
    for d in range(cfg.D):
        items_dict[f"volume_d{d}"] = volumes[d,:]
    items_df = pd.DataFrame(items_dict)

    # Checks
    assert cost_df.shape == mask_df.shape, "cost/mask shape mismatch"
    assert not bins_df.isna().any().any(), "NaNs in bins_df"
    assert not items_df.isna().any().any(), "NaNs in items_df"
    assert not cost_df.isna().any().any(), "NaNs in cost_df"
    assert mask_df.dtypes.eq(bool).all(), "mask_df must be boolean"
    return Instance(bins_df, items_df, cost_df, mask_df, cfg.D)
