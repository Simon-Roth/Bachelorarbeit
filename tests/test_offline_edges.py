import numpy as np
from core.config import load_config
from data.generators import generate_offline_instance
from offline.milp_solver import OfflineMILPSolver

def test_tiny_capacities_force_fallback():
    cfg = load_config("configs/default.yaml")
    cfg.slack.enforce_slack = True
    cfg.slack.fraction = 0.9  # 10% capacity left
    inst = generate_offline_instance(cfg, seed=1)
    solver = OfflineMILPSolver(cfg, time_limit=20, mip_gap=0.0)
    _, info = solver.solve(inst)

    fallback_used = int((info.assignments[:, cfg.problem.N] == 1).sum())
    assert fallback_used >= 1  # likely many

def test_generous_capacities_zero_fallback():
    cfg = load_config("configs/default.yaml")
    # Inflate capacities to ensure everything fits easily
    cfg.problem.capacities = [10.0 for _ in cfg.problem.capacities]
    cfg.slack.enforce_slack = False
    inst = generate_offline_instance(cfg, seed=2)
    solver = OfflineMILPSolver(cfg, time_limit=20, mip_gap=0.0)
    _, info = solver.solve(inst)

    fallback_used = int((info.assignments[:, cfg.problem.N] == 1).sum())
    assert fallback_used == 0
