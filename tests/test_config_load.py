from pathlib import Path
from core.config import load_config

def test_load_config_defaults():
    cfg = load_config("configs/default.yaml")
    assert cfg.problem.N > 0
    assert len(cfg.problem.capacities) == cfg.problem.N
    assert 0.0 < cfg.volumes.offline_uniform[0] < cfg.volumes.offline_uniform[1]
    assert cfg.costs.penalty_mode in {"per_item", "per_volume"}
