from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from core.config import Config, load_config
from core.general_utils import set_global_seed
from experiments.pipeline_runner import PipelineSpec, run_pipeline

from offline.offline_solver import OfflineMILPSolver
from offline.offline_heuristics.first_fit_decreasing import FirstFitDecreasing
from offline.offline_heuristics.best_fit_decreasing import BestFitDecreasing
from offline.offline_heuristics.cost_best_fit_decreasing import CostAwareBestFitDecreasing

from online.heuristics.best_fit import BestFitOnlinePolicy
from online.heuristics.next_fit import NextFitOnlinePolicy
from online.heuristics.cost_best_fit import CostAwareBestFitOnlinePolicy

from data.generators import generate_instance_with_online
from experiments.optimal_benchmark import solve_full_horizon_optimum
from online.state_utils import count_fallback_items
from experiments.utils import save_combined_result
import copy


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


def main() -> None:
    cfg = load_config("configs/default.yaml")
    base_seed = cfg.eval.seeds[0]

    # Compute full-horizon optimum once for reference
    cfg_opt = load_config("configs/default.yaml")
    set_global_seed(base_seed)
    base_instance = generate_instance_with_online(cfg_opt, seed=base_seed)
    optimal_state, optimal_info = solve_full_horizon_optimum(
        cfg_opt,
        copy.deepcopy(base_instance),
        lambda cfg_: OfflineMILPSolver(cfg_, time_limit=300, mip_gap=0.0, threads=0, log_to_console=False),
    )
    optimal_fallback = count_fallback_items(optimal_state, base_instance)
    optimal_summary = {
        "pipeline": "OPTIMAL_FULL_HORIZON",
        "seed": base_seed,
        "problem": {
            "N": cfg_opt.problem.N,
            "M_off": cfg_opt.problem.M_off,
            "M_on": len(base_instance.online_items),
        },
        "offline": {
            "method": "FullHorizonMILP",
            "status": optimal_info.status,
            "runtime": float(optimal_info.runtime),
            "obj_value": float(optimal_info.obj_value),
            "mip_gap": float(optimal_info.mip_gap),
            "items_in_fallback": optimal_fallback,
        },
        "online": {
            "policy": "FullHorizonMILP",
            "status": optimal_info.status,
            "runtime": 0.0,
            "total_cost": 0.0,
            "fallback_items": optimal_fallback,
            "evicted_offline": 0,
            "total_objective": float(optimal_info.obj_value),
        },
        "final_items_in_fallback": optimal_fallback,
        "offline_assignments": {str(k): int(v) for k, v in optimal_state.assigned_bin.items()},
    }
    optimal_prefix = (
        f"pipeline_optimal_full_horizon_N{cfg_opt.problem.N}_"
        f"Moff{cfg_opt.problem.M_off}_Mon{len(base_instance.online_items)}_seed{base_seed}"
    )
    optimal_path = save_combined_result(optimal_prefix, optimal_summary)
    print(f"Optimal full-horizon benchmark saved to {optimal_path}")

    print("Running offline+online pipelines (alternative runner)...")
    for spec in PIPELINES:
        # fresh copy per run to avoid cross-talk
        cfg_run = load_config("configs/default.yaml")
        set_global_seed(base_seed)

        print(f"\n{'=' * 60}")
        print(f"Running {spec.name}")
        print("=" * 60)

        summary, path = run_pipeline(cfg_run, spec, seed=base_seed)
        print(
            f"{summary['pipeline']}: offline {summary['offline']['status']} "
            f"({summary['offline']['runtime']:.3f}s), "
            f"online {summary['online']['status']} "
            f"({summary['online']['runtime']:.3f}s)"
        )
        print(f"Saved result to {path}")

    print("\nAll pipelines completed!")
    print("Results saved to results/ directory")


if __name__ == "__main__":
    main()
