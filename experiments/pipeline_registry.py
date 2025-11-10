from __future__ import annotations

from typing import Callable, Dict, List

from core.config import Config
from experiments.pipeline_runner import PipelineSpec

from offline.offline_solver import OfflineMILPSolver
from offline.offline_heuristics.first_fit_decreasing import FirstFitDecreasing
from offline.offline_heuristics.best_fit_decreasing import BestFitDecreasing
from offline.offline_heuristics.cost_best_fit_decreasing import CostAwareBestFitDecreasing

from online.heuristics.best_fit import BestFitOnlinePolicy
from online.heuristics.next_fit import NextFitOnlinePolicy
from online.heuristics.cost_best_fit import CostAwareBestFitOnlinePolicy


def make_milp_solver(
    *,
    time_limit: int = 60,
    mip_gap: float = 0.00,
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
        name="MILP+BestFit",
        offline_label="MILP",
        online_label="BestFit",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg: BestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="MILP+NextFit",
        offline_label="MILP",
        online_label="NextFit",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg: NextFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="MILP+CostAwareBestFit",
        offline_label="MILP",
        online_label="CostAwareBestFit",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="MILP(FFD-warm)+BestFit",
        offline_label="MILP+FFD-warm",
        online_label="BestFit",
        offline_factory=make_warm_milp_solver(FirstFitDecreasing),
        online_factory=lambda cfg: BestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="MILP(BFD-warm)+BestFit",
        offline_label="MILP+BFD-warm",
        online_label="BestFit",
        offline_factory=make_warm_milp_solver(BestFitDecreasing),
        online_factory=lambda cfg: BestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="FFD+BestFit",
        offline_label="FFD",
        online_label="BestFit",
        offline_factory=lambda cfg: FirstFitDecreasing(cfg),
        online_factory=lambda cfg: BestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="FFD+NextFit",
        offline_label="FFD",
        online_label="NextFit",
        offline_factory=lambda cfg: FirstFitDecreasing(cfg),
        online_factory=lambda cfg: NextFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="BFD+BestFit",
        offline_label="BFD",
        online_label="BestFit",
        offline_factory=lambda cfg: BestFitDecreasing(cfg),
        online_factory=lambda cfg: BestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="BFD+NextFit",
        offline_label="BFD",
        online_label="NextFit",
        offline_factory=lambda cfg: BestFitDecreasing(cfg),
        online_factory=lambda cfg: NextFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="CostAwareBFD+CostAwareBestFit",
        offline_label="CostAwareBFD",
        online_label="CostAwareBestFit",
        offline_factory=lambda cfg: CostAwareBestFitDecreasing(cfg),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
    ),
    PipelineSpec(
        name="MILP(BFD-warm)+CostAwareBestFit",
        offline_label="MILP+BFD-warm",
        online_label="CostAwareBestFit",
        offline_factory=make_warm_milp_solver(BestFitDecreasing),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
    ),
]

PIPELINE_REGISTRY: Dict[str, PipelineSpec] = {spec.name: spec for spec in PIPELINES}


def get_pipeline(spec_name: str) -> PipelineSpec:
    try:
        return PIPELINE_REGISTRY[spec_name]
    except KeyError as exc:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline '{spec_name}'. Known pipelines: {known}") from exc
