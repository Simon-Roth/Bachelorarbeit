import numpy as np
import pytest

from core.config import load_config, Config
from core.utils import set_global_seed
from data.generators import generate_offline_instance
from offline.milp_solver import OfflineMILPSolver


def recompute_objective(inst, x):
    """Recompute objective from assignment matrix and cost matrix."""
    assign_cost = inst.costs.assign[:x.shape[0], :x.shape[1]]
    return float((assign_cost * x).sum())


def compute_loads_from_x(inst, x):
    """Aggregate loads on regular bins from x."""
    N = len(inst.bins)
    vols = np.array([it.volume for it in inst.offline_items], dtype=float)
    loads = np.zeros(N, dtype=float)
    for i in range(N):
        mask = (x[:, i] == 1)
        if mask.any():
            loads[i] = vols[mask].sum()
    fallback_load = float(vols[x[:, N] == 1].sum())
    return loads, fallback_load


def test_assignment_and_capacity_basic():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)

    inst = generate_offline_instance(cfg, seed=seed)
    solver = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0, log_to_console=False)
    state, info = solver.solve(inst)

    # A) each offline item assigned exactly once
    x = info.assignments
    assert x.shape == (cfg.problem.M_off, cfg.problem.N + 1)
    assert np.all(x.sum(axis=1) == 1)

    # B) capacity respected with slack
    N = cfg.problem.N
    caps = np.array([b.capacity for b in inst.bins], dtype=float)
    eff = caps * (1.0 - (cfg.slack.fraction if cfg.slack.enforce_slack else 0.0))
    loads, _ = compute_loads_from_x(inst, x)
    assert np.all(loads <= eff + 1e-9)


def test_objective_consistency():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)
    inst = generate_offline_instance(cfg, seed=seed)
    solver = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info = solver.solve(inst)

    model_obj = info.obj_value
    recomputed = recompute_objective(inst, info.assignments)
    assert np.isclose(model_obj, recomputed, rtol=1e-8, atol=1e-8)


def test_mask_enforcement():
    # Build an instance, then manually zero out some feasibility edges and ensure they are never used.
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    inst = generate_offline_instance(cfg, seed=seed)

    # Force item 0 to be infeasible on bin 0 (and also infeasible on fallback to keep feasibility via another bin)
    inst.feasible.feasible[0, 0] = 0
    # Still keep at least one feasible bin for item 0 (e.g., bin 1)
    inst.feasible.feasible[0, 1] = 1

    solver = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info = solver.solve(inst)

    # Ensure (item 0 -> bin 0) is not selected
    assert info.assignments[0, 0] == 0


def test_infeasible_item_raises():
    # If fallback disabled and p_off small, we can construct an item with no feasible edges
    cfg = load_config("configs/default.yaml")
    cfg.problem.fallback_is_enabled = False  # disable fallback for this test only

    seed = cfg.eval.seeds[0]
    inst = generate_offline_instance(cfg, seed=seed)

    # Make item 0 infeasible everywhere
    inst.feasible.feasible[0, :] = 0

    solver = OfflineMILPSolver(cfg, time_limit=10, mip_gap=0.0)
    with pytest.raises(ValueError):
        solver.solve(inst)


def test_monotonicity_slack_and_fallback_cost():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]

    # Base run (no slack)
    cfg.slack.enforce_slack = False
    inst = generate_offline_instance(cfg, seed=seed)
    solver = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info_base = solver.solve(inst)

    # More slack (less capacity) should not decrease objective
    cfg.slack.enforce_slack = True
    cfg.slack.fraction = 0.2
    inst2 = generate_offline_instance(cfg, seed=seed)  # same seed ⇒ same instance stats
    solver2 = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info_slack = solver2.solve(inst2)

    assert info_slack.obj_value >= info_base.obj_value - 1e-8

    # If we increase huge_fallback, objective cannot go down
    cfg.slack.enforce_slack = True
    cfg.slack.fraction = 0.4
    inst3 = generate_offline_instance(cfg, seed=seed)
    old_fallback = inst3.costs.huge_fallback
    inst3.costs.assign[:, -1] = old_fallback  # ensure matrix matches
    solver3 = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info3 = solver3.solve(inst3)

    inst4 = generate_offline_instance(cfg, seed=seed)
    new_fallback = old_fallback * 2.0
    inst4.costs.assign[:, -1] = new_fallback
    solver4 = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info4 = solver4.solve(inst4)

    assert info4.obj_value >= info3.obj_value - 1e-8


def test_warm_start_consistency():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)
    inst = generate_offline_instance(cfg, seed=seed)

    cold = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info_cold = cold.solve(inst)

    # Trivial warm start: assign every item to fallback (feasible)
    warm_start = {j: cfg.problem.N for j in range(cfg.problem.M_off)}
    warm = OfflineMILPSolver(cfg, time_limit=30, mip_gap=0.0)
    _, info_warm = warm.solve(inst, warm_start=warm_start)

    # Optimal objective should be the same
    assert np.isclose(info_cold.obj_value, info_warm.obj_value, rtol=1e-9, atol=1e-9)
