import numpy as np
from core.config import load_config
from data.generators import generate_offline_instance

def test_offline_instance_shapes():
    cfg = load_config("configs/default.yaml")
    inst = generate_offline_instance(cfg, seed=123)
    N = cfg.problem.N
    M = cfg.problem.M_off

    # bins
    assert len(inst.bins) == N
    # items
    assert len(inst.offline_items) == M
    # costs matrix: M x (N+1)
    assert inst.costs.assign.shape == (M, N+1)
    # feasibility mask: M x (N+1)
    assert inst.feasible.feasible.shape == (M, N+1)
    # fallback index is N (0-based)
    assert inst.fallback_bin_index == N

    # feasibility mask in {0,1}
    feas = inst.feasible.feasible
    assert feas.min() >= 0 and feas.max() <= 1

    # fallback col is feasible iff fallback enabled
    if cfg.problem.fallback_is_enabled:
        assert (feas[:, N] == 1).all()
    else:
        assert (feas[:, N] == 0).all()

    # assignment costs finite; fallback column very large
    assign = inst.costs.assign
    assert np.isfinite(assign[:, :N]).all()
    assert (assign[:, N] >= cfg.costs.huge_fallback * 0.999).all()
