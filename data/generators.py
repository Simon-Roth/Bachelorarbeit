# binpacking/data/generators.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from repo_bachelorarbeit.core.config import Config
from repo_bachelorarbeit.core.models import BinSpec, ItemSpec, Instance, Costs, FeasibleGraph
from repo_bachelorarbeit.core.general_utils import validate_capacities, make_rng,validate_mask

def generate_offline_instance(cfg: Config, seed: int) -> Instance:
    """
    Create an offline instance at t=0:
    - N regular bins with capacities
    - M_off offline items with volumes ~ Uniform(a,b)
    - G_off feasibility mask via Erdos-Renyi with p_off
    - Cost matrix with explicit fallback column for offline items
    """
    rng = make_rng(seed)
    N = cfg.problem.N
    M_off = cfg.problem.M_off
    capacities = np.array(cfg.problem.capacities, dtype=float)
    validate_capacities(capacities)

    # Bins (0..N-1). Fallback's "index" will be N (no capacity).
    bins = [BinSpec(id=i, capacity=float(capacities[i])) for i in range(N)]
    fallback_idx = N  # IMPORTANT: 0-based indexing, fallback is the (N)-th column

    # Offline volumes ~ Beta(a, b)
    alpha_off, beta_off = cfg.volumes.offline_beta
    lo, hi = cfg.volumes.offline_bounds
    assert alpha_off > 0 and beta_off > 0, "offline_beta must be positive."
    assert 0.0 < lo < hi, "offline_bounds must satisfy 0 < lower < upper."
    u = rng.beta(alpha_off, beta_off, size=M_off).astype(float)  # in [0,1]
    offline_volumes = (lo + u * (hi - lo)).astype(float)
    offline_items = [ItemSpec(id=j, volume=float(offline_volumes[j])) for j in range(M_off)]

    # Feasibility mask for OFFLINE items: shape (M_off, N+1)
    # Last column (fallback) is always feasible if fallback enabled; else 0.
    p_off = cfg.graphs.p_off
    feas_mask = (rng.uniform(size=(M_off, N)) < p_off).astype(int)
    if cfg.problem.fallback_is_enabled:
        fallback_col = np.ones((M_off, 1), dtype=int)
    else:
        fallback_col = np.zeros((M_off, 1), dtype=int)
    feas_full = np.hstack([feas_mask, fallback_col])  # (M_off, N+1)
    
    validate_mask(feas_full)
    
    # Assignment costs for OFFLINE items to all N+1 bins (fallback very large)
    c_low, c_high = cfg.costs.base_assign_range
    base_costs = rng.uniform(c_low, c_high, size=(M_off, N)).astype(float)
    fallback_costs = np.full((M_off, 1), cfg.costs.huge_fallback, dtype=float)
    assign = np.hstack([base_costs, fallback_costs])

    # Infeasible edges get zeroed in feasibility mask (solver will enforce x[j,i]=0 later).
    costs = Costs(
        assign=assign,
        reassignment_penalty=cfg.costs.reassignment_penalty,
        penalty_mode=cfg.costs.penalty_mode,
        per_volume_scale=cfg.costs.per_volume_scale,
        huge_fallback=cfg.costs.huge_fallback,
    )

    feasible = FeasibleGraph(feasible=feas_full)
    return Instance(
        bins=bins,
        offline_items=offline_items,
        costs=costs,
        feasible=feasible,
        fallback_bin_index=fallback_idx,
    )

def sample_online_item(
    cfg: Config,
    next_id: int,
    rng: np.random.Generator,
    capacities: np.ndarray,
) -> Tuple[int, float, List[int]]:
    """
    Generate one ONLINE item:
    - volume ~ Beta(alpha, beta) scaled by 'online_scale_to_capacity' * min(capacities)
    - feasible bins via Erdos-Renyi with p_onl (NO fallback)
    """
    alpha, beta = cfg.volumes.online_beta
    scale = cfg.volumes.online_scale_to_capacity * float(np.min(capacities))
    vol = float(np.clip(rng.beta(alpha, beta) * scale, 1e-9, None))  # guard against 0

    N = cfg.problem.N
    p_onl = cfg.graphs.p_onl
    feas = (rng.uniform(size=N) < p_onl)
    feasible_bins = [i for i in range(N) if feas[i]]

    # Guarantee at least one feasible bin to avoid degenerate arrivals
    # Note: this is about the FEASIBILITY GRAPH (compatibility), not capacity.
    # Capacity may still be insufficient; the online policy can evict offline items to free space.
    if not feasible_bins:
        feasible_bins = [int(rng.integers(0, N))]

    return next_id, vol, feasible_bins
