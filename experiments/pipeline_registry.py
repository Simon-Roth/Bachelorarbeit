from __future__ import annotations

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
        name="CostAwareBFD+CostAwareBestFit",
        offline_label="CostAwareBFD",
        online_label="CostAwareBestFit",
        offline_factory=lambda cfg: CostAwareBestFitDecreasing(cfg),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="CostAwareBFD",
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
        name="UtilizationPriced+PrimalDual",
        offline_label="UtilizationPriced",
        online_label="PrimalDual",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
    PipelineSpec(
        name="MILP+PrimalDual",
        offline_label="MILP",
        online_label="PrimalDual",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="MILP",
    ),
    PipelineSpec(
        name="MILP+DynamicLearning",
        offline_label="MILP",
        online_label="DynamicLearning",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
        offline_cache_key="MILP",
    ),
    PipelineSpec(
        name="UtilizationPriced+DynamicLearning",
        offline_label="UtilizationPriced",
        online_label="DynamicLearning",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
]

PIPELINE_REGISTRY: Dict[str, PipelineSpec] = {spec.name: spec for spec in PIPELINES}


def get_pipeline(spec_name: str) -> PipelineSpec:
    try:
        return PIPELINE_REGISTRY[spec_name]
    except KeyError as exc:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline '{spec_name}'. Known pipelines: {known}") from exc
