# binpacking/experiments/run_one.py
from __future__ import annotations
from pathlib import Path
from core.config import load_config
from core.utils import set_global_seed
from data.generators import generate_offline_instance, sample_online_item

def main():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)

    # Build offline instance (t=0)
    inst = generate_offline_instance(cfg, seed=seed)
    print(f"N={len(inst.bins)}, M_off={len(inst.offline_items)}, fallback_idx={inst.fallback_bin_index}")
    print(f"Assign matrix shape: {inst.costs.assign.shape}, Feas mask shape: {inst.feasible.feasible.shape}")

    # Show an example online item sample (no solver yet)
    import numpy as np
    capacities = np.array([b.capacity for b in inst.bins], dtype=float)
    from core.utils import make_rng
    rng = make_rng(seed+1)
    online_id, vol, feas_bins = sample_online_item(cfg, next_id=len(inst.offline_items), rng=rng, capacities=capacities)
    print(f"Sample online item -> id={online_id}, vol={vol:.3f}, feasible_bins={feas_bins}")

if __name__ == "__main__":
    main()
