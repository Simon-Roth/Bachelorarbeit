import numpy as np
from core.config import load_config
from core.utils import make_rng
from data.generators import generate_offline_instance, sample_online_item

def test_online_item_has_structural_feasible_bin():
    cfg = load_config("configs/default.yaml")
    inst = generate_offline_instance(cfg, seed=1)
    capacities = np.array([b.capacity for b in inst.bins], dtype=float)
    rng = make_rng(2)
    next_id = len(inst.offline_items)

    # sample multiple to reduce fluke probability
    for _ in range(50):
        oid, vol, feas_bins = sample_online_item(cfg, next_id, rng, capacities)
        assert len(feas_bins) >= 1, "Must have at least one structurally feasible bin"
        assert all(0 <= i < cfg.problem.N for i in feas_bins)
        assert vol > 0.0
        next_id += 1
