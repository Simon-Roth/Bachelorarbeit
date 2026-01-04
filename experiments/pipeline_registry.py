from __future__ import annotations

import copy
from typing import Callable, Dict, List

from core.config import Config
from experiments.pipeline_runner import PipelineSpec

from offline.offline_solver import OfflineMILPSolver
from offline.offline_heuristics.first_fit_decreasing import FirstFitDecreasing
from offline.offline_heuristics.best_fit_decreasing import BestFitDecreasing
from offline.offline_heuristics.cost_best_fit_decreasing import CostAwareBestFitDecreasing
from offline.offline_heuristics.utilization_priced import UtilizationPricedDecreasing

from online.online_heuristics.best_fit import BestFitOnlinePolicy
from online.online_heuristics.next_fit import NextFitOnlinePolicy
from online.online_heuristics.cost_best_fit import CostAwareBestFitOnlinePolicy
from online.online_heuristics.primal_dual import PrimalDualPolicy
from online.online_heuristics.dynamic_learning import DynamicLearningPolicy

def make_milp_solver(
    *,
    time_limit: int = 60,
    mip_gap: float = 0.03,
    threads: int = 0,
    log_to_console: bool = False,
) -> Callable[[Config], OfflineMILPSolver]:
    def factory(cfg: Config) -> OfflineMILPSolver:
        return OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )

    return factory


def make_warm_milp_solver(
    heuristic_ctor: Callable[[Config], object],
    *,
    time_limit: int = 60,
    mip_gap: float = 0.00,
    threads: int = 0,
    log_to_console: bool = False,
) -> Callable[[Config], object]:
    def factory(cfg: Config):
        solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )
        heuristic = heuristic_ctor(cfg)

        class WarmStartWrapper:
            def __init__(self) -> None:
                self._solver = solver
                self._heuristic = heuristic

            def solve(self, inst):
                warm_state, _ = self._heuristic.solve(inst)
                return self._solver.solve(inst, warm_start=warm_state.assigned_bin)

        return WarmStartWrapper()

    return factory


def _cfg_with_offline_slack(cfg: Config) -> Config:
    cfg_copy = copy.deepcopy(cfg)
    # Slack sized to expected online volume relative to total capacity.
    # E[V_online] = lo + (hi - lo) * a/(a+b) for Beta(a,b) on [0,1] scaled to [lo,hi].
    a, b = cfg_copy.volumes.online_beta
    lo, hi = cfg_copy.volumes.online_bounds
    mean_vol = 0.0
    if (a + b) > 0:
        mean_vol = lo + (hi - lo) * (a / float(a + b))

    horizon = max(0, cfg_copy.stoch.horizon)
    expected_online_volume = horizon * mean_vol

    capacities = cfg_copy.problem.capacities
    if capacities:
        total_capacity = sum(capacities)
    else:
        total_capacity = cfg_copy.problem.N * float(cfg_copy.problem.capacity_mean)

    slack_fraction = 0.0
    if total_capacity > 0 and expected_online_volume > 0:
        slack_fraction = max(0.0, min(0.99, expected_online_volume / total_capacity))
    cfg_copy.slack.enforce_slack = slack_fraction > 0.0
    cfg_copy.slack.fraction = slack_fraction
    cfg_copy.slack.apply_to_online = False
    return cfg_copy


def _cfg_with_util_penalty(cfg: Config) -> Config:
    """
    Enable utilization-aware PWL penalty in the offline MILP objective.
    """
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.util_penalty.enabled = True
    return cfg_copy


PIPELINES: List[PipelineSpec] = [
    PipelineSpec(
        name="MILP+CostAwareBestFit",
        offline_label="MILP",
        online_label="CostAwareBestFit",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="MILP",
    ),
    PipelineSpec(
        name="UtilizationPriced+CostAwareBestFit",
        offline_label="UtilizationPriced",
        online_label="CostAwareBestFit",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
    PipelineSpec(
        name="UtilizationPriced(exp)+CostAwareBestFit",
        offline_label="UtilizationPricedExp",
        online_label="CostAwareBestFit",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg, update_rule="exponential"),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="UtilizationPricedExp",
    ),
    PipelineSpec(
        name="UtilizationPriced+PrimalDual",
        offline_label="UtilizationPriced",
        online_label="PrimalDual",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
    PipelineSpec(
        name="UtilizationPriced(exp)+PrimalDual",
        offline_label="UtilizationPricedExp",
        online_label="PrimalDual",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg, update_rule="exponential"),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="UtilizationPricedExp",
    ),
    PipelineSpec(
        name="MILP+PrimalDual",
        offline_label="MILP",
        online_label="PrimalDual",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="MILP",
    ),
    # PipelineSpec(
    #     name="MILP(offSlack)+PrimalDual",
    #     offline_label="MILP_offSlack",
    #     online_label="PrimalDual",
    #     offline_factory=lambda cfg, base_factory=make_milp_solver(): base_factory(
    #         _cfg_with_offline_slack(cfg)
    #     ),
    #     online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
    #     offline_cache_key="MILP_offSlack",
    # ),
    # PipelineSpec(
    #     name="MILP(offSlack)+CostAwareBestFit",
    #     offline_label="MILP_offSlack",
    #     online_label="CostAwareBestFit",
    #     offline_factory=lambda cfg, base_factory=make_milp_solver(): base_factory(
    #         _cfg_with_offline_slack(cfg)
    #     ),
    #     online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
    #     offline_cache_key="MILP_offSlack",
    # ),
    # PipelineSpec(
    #     name="MILP(offSlack)+DynamicLearning",
    #     offline_label="MILP_offSlack",
    #     online_label="DynamicLearning",
    #     offline_factory=lambda cfg, base_factory=make_milp_solver(): base_factory(
    #         _cfg_with_offline_slack(cfg)
    #     ),
    #     online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
    #     offline_cache_key="MILP_offSlack",
    # ),
    # PipelineSpec(
    #     name="MILP+DynamicLearning",
    #     offline_label="MILP",
    #     online_label="DynamicLearning",
    #     offline_factory=make_milp_solver(),
    #     online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
    #     offline_cache_key="MILP",
    # ),
    # PipelineSpec(
    #     name="UtilizationPriced+DynamicLearning",
    #     offline_label="UtilizationPriced",
    #     online_label="DynamicLearning",
    #     offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
    #     online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
    #     offline_cache_key="UtilizationPriced",
    # ),
    
]

PIPELINE_REGISTRY: Dict[str, PipelineSpec] = {spec.name: spec for spec in PIPELINES}


def get_pipeline(spec_name: str) -> PipelineSpec:
    try:
        return PIPELINE_REGISTRY[spec_name]
    except KeyError as exc:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline '{spec_name}'. Known pipelines: {known}") from exc
