from __future__ import annotations

"""
Run every registered pipeline for a set of manually curated config variants.

Edit `SCENARIO_SWEEP` below to try different horizon/M_off combinations, then run:
    python -m experiments.run_pipeline_config_sweep --output-root results/config_sweep

The script stores each scenario in its own directory so that
`plots/online/create_config_sweep_plots.py` can aggregate runtimes, objectives, and fallback usage.
"""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from core.config import Config, load_config
from core.general_utils import set_global_seed
from data.generators import generate_instance_with_online
from experiments.pipeline_registry import PIPELINE_REGISTRY, PIPELINES
from experiments.pipeline_runner import run_pipeline
from core.models import AssignmentState
from offline.models import OfflineSolutionInfo
from offline.offline_heuristics.core import HeuristicSolutionInfo
from experiments.optimal_benchmark import solve_full_horizon_optimum
from experiments.utils import save_combined_result
from offline.offline_solver import OfflineMILPSolver
from online.state_utils import count_fallback_items


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Lightweight definition for a config variant that should be evaluated across pipelines.
    """

    name: str
    overrides: Dict[str, Any]
    seeds: Sequence[int] | None = None
    description: str | None = None


def apply_config_overrides(base_cfg: Config, overrides: Dict[str, Any]) -> Config:
    """
    Return a deep-copied config with the overrides merged recursively.
    """
    cfg = copy.deepcopy(base_cfg)

    def _apply(target, patch: Dict[str, Any]) -> None:
        for key, value in patch.items():
            if not hasattr(target, key):
                raise KeyError(f"Config object {target} has no attribute '{key}'")
            current = getattr(target, key)
            if isinstance(value, dict) and hasattr(current, "__dict__"):
                _apply(current, value)
            else:
                setattr(target, key, copy.deepcopy(value))

    _apply(cfg, overrides)
    return cfg



SCENARIO_SWEEP: List[ScenarioConfig] = [
    # ========= FAMILY A: SMOOTH I.I.D. DISTRIBUTION =========
    ScenarioConfig(
        name="smoothdist_ratio_off0_on100",
        overrides={
            "problem": {"M_off": 0},
            "stoch": {"horizon": 300},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Smooth i.i.d. distributions, 0% offline / 100% online.",
    ),
    ScenarioConfig(
        name="smoothdist_ratio_off20_on80",
        overrides={
            "problem": {"M_off": 60},
            "stoch": {"horizon": 240},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Smooth i.i.d. distributions, 20% offline / 80% online.",
    ),
    ScenarioConfig(
        name="smoothdist_ratio_off50_on50",
        overrides={
            "problem": {"M_off": 150},
            "stoch": {"horizon": 150},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Smooth i.i.d. distributions, 50% offline / 50% online.",
    ),
    ScenarioConfig(
        name="smoothdist_ratio_off80_on20",
        overrides={
            "problem": {"M_off": 240},
            "stoch": {"horizon": 60},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Smooth i.i.d. distributions, 80% offline / 20% online.",
    ),

    # ========= FAMILY B: HEAVY-TAILED VOLUMES (SAME MEAN, HIGHER VARIANCE) =========
    ScenarioConfig(
        name="heavyvol_ratio_off0_on100",
        overrides={
            "problem": {"M_off": 0},
            "stoch": {"horizon": 300},
            "volumes": {
                "offline_beta":  [1.0, 2.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [1.0, 2.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Heavy-tailed volumes, 0% offline / 100% online.",
    ),
    ScenarioConfig(
        name="heavyvol_ratio_off20_on80",
        overrides={
            "problem": {"M_off": 60},
            "stoch": {"horizon": 240},
            "volumes": {
                "offline_beta":  [1.0, 2.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [1.0, 2.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Heavy-tailed volumes, 20% offline / 80% online.",
    ),
    ScenarioConfig(
        name="heavyvol_ratio_off50_on50",
        overrides={
            "problem": {"M_off": 150},
            "stoch": {"horizon": 150},
            "volumes": {
                "offline_beta":  [1.0, 2.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [1.0, 2.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Heavy-tailed volumes, 50% offline / 50% online.",
    ),
    ScenarioConfig(
        name="heavyvol_ratio_off80_on20",
        overrides={
            "problem": {"M_off": 240},
            "stoch": {"horizon": 60},
            "volumes": {
                "offline_beta":  [1.0, 2.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [1.0, 2.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.5,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Heavy-tailed volumes, 80% offline / 20% online.",
    ),

    # ========= FAMILY C: SPARSE ONLINE GRAPHS =========
    ScenarioConfig(
        name="sparse_ratio_off0_on100",
        overrides={
            "problem": {"M_off": 0},
            "stoch": {"horizon": 300},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.25,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Sparse online graphs, 0% offline / 100% online.",
    ),
    ScenarioConfig(
        name="sparse_ratio_off20_on80",
        overrides={
            "problem": {"M_off": 60},
            "stoch": {"horizon": 240},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.25,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Sparse online graphs, 20% offline / 80% online.",
    ),
    ScenarioConfig(
        name="sparse_ratio_off50_on50",
        overrides={
            "problem": {"M_off": 150},
            "stoch": {"horizon": 150},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.25,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Sparse online graphs, 50% offline / 50% online.",
    ),
    ScenarioConfig(
        name="sparse_ratio_off80_on20",
        overrides={
            "problem": {"M_off": 240},
            "stoch": {"horizon": 60},
            "volumes": {
                "offline_beta":  [2.5, 5.0],
                "offline_bounds": [0.5, 1.7],
                "online_beta":   [2.5, 5.0],
                "online_bounds": [0.5, 1.7],
            },
            "graphs": {
                "p_off": 0.8,
                "p_onl": 0.25,
            },
            "costs": {
                "assign_beta":   [2.0, 5.0],
                "assign_bounds": [1.0, 5.0],
                "huge_fallback": 50.0,
                "reassignment_penalty": 10.0,
                "penalty_mode": "per_item",
                "per_volume_scale": 10.0,
            },
        },
        description="Sparse online graphs, 80% offline / 20% online.",
    ),

    # ========= FAMILY D: STRUCTURAL OVERLOAD (10–20% expected overflow) =========

ScenarioConfig(
    name="overloaddist_ratio_off0_on100",
    overrides={
        "problem": {"M_off": 0},
        "stoch": {"horizon": 300},
        "volumes": {
            "offline_beta":  [3.0, 3.0],
            "offline_bounds": [0.5, 1.7],
            "online_beta":   [3.0, 3.0],
            "online_bounds": [0.5, 1.7],
        },
    },
    description="Structural overload, 0% offline / 100% online.",
),

ScenarioConfig(
    name="overloaddist_ratio_off20_on80",
    overrides={
        "problem": {"M_off": 60},
        "stoch": {"horizon": 240},
        "volumes": {
            "offline_beta":  [3.0, 3.0],
            "offline_bounds": [0.5, 1.7],
            "online_beta":   [3.0, 3.0],
            "online_bounds": [0.5, 1.7],
        },
    },
    description="Structural overload, 20% offline / 80% online.",
),

ScenarioConfig(
    name="overloaddist_ratio_off50_on50",
    overrides={
        "problem": {"M_off": 150},
        "stoch": {"horizon": 150},
        "volumes": {
            "offline_beta":  [3.0, 3.0],
            "offline_bounds": [0.5, 1.7],
            "online_beta":   [3.0, 3.0],
            "online_bounds": [0.5, 1.7],
        },
    },
    description="Structural overload, 50% offline / 50% online.",
),

ScenarioConfig(
    name="overloaddist_ratio_off80_on20",
    overrides={
        "problem": {"M_off": 240},
        "stoch": {"horizon": 60},
        "volumes": {
            "offline_beta":  [3.0, 3.0],
            "offline_bounds": [0.5, 1.7],
            "online_beta":   [3.0, 3.0],
            "online_bounds": [0.5, 1.7],
        },
    },
    description="Structural overload, 80% offline / 20% online.",
),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every registered pipeline for several config variants."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the reference YAML config that overrides are applied to.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/config_sweep"),
        help="Directory where per-scenario subfolders will be created.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.name for scenario in SCENARIO_SWEEP],
        help="Optional subset of scenario names to run.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        help="Optional subset of pipeline names to run. Defaults to all registered pipelines.",
    )
    return parser.parse_args()


def select_scenarios(requested: Sequence[str] | None) -> Iterable[ScenarioConfig]:
    if not requested:
        yield from SCENARIO_SWEEP
        return
    mapping = {scenario.name: scenario for scenario in SCENARIO_SWEEP}
    for name in requested:
        if name not in mapping:
            known = ", ".join(sorted(mapping))
            raise KeyError(f"Unknown scenario '{name}'. Known: {known}")
        yield mapping[name]


def select_pipelines(names: Sequence[str] | None):
    if not names:
        return PIPELINES
    missing = [name for name in names if name not in PIPELINE_REGISTRY]
    if missing:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(
            f"Unknown pipeline(s) {missing}. Known pipelines are: {known}"
        )
    return [PIPELINE_REGISTRY[name] for name in names]


def run_config_sweep(
    base_cfg: Config,
    scenarios: Iterable[ScenarioConfig],
    pipeline_specs,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    OfflineResult = Tuple[AssignmentState, OfflineSolutionInfo | HeuristicSolutionInfo]
    offline_cache: Dict[Tuple[str, int, str], OfflineResult] = {}

    for scenario in scenarios:
        scenario_cfg = apply_config_overrides(base_cfg, scenario.overrides)
        if scenario.seeds:
            scenario_cfg.eval = copy.deepcopy(scenario_cfg.eval)
            scenario_cfg.eval.seeds = tuple(scenario.seeds)

        seeds = list(scenario_cfg.eval.seeds)
        scenario_dir = output_root / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Scenario: {scenario.name} ({scenario.description or 'no description'}) ===")
        print(
            f"Offline items: {scenario_cfg.problem.M_off}, "
            f"online horizon: {scenario_cfg.stoch.horizon}, "
            f"seeds: {seeds}"
        )

        for seed in seeds:
            set_global_seed(seed)
            shared_instance = generate_instance_with_online(scenario_cfg, seed=seed)
            print(f"\nRunning seed={seed}, offline={len(shared_instance.offline_items)}, online={len(shared_instance.online_items)}")
            offline_count = len(shared_instance.offline_items)
            online_count = len(shared_instance.online_items)

            if offline_count + online_count > 0:
                solver_factory = lambda cfg_: OfflineMILPSolver(
                    cfg_,
                    time_limit=300,
                    mip_gap=0.03,
                    log_to_console=False,
                )
                optimal_state, optimal_info = solve_full_horizon_optimum(
                    scenario_cfg,
                    copy.deepcopy(shared_instance),
                    solver_factory,
                )

                fallback_opt = count_fallback_items(optimal_state, shared_instance)
                optimal_summary = {
                    "pipeline": "OPTIMAL_FULL_HORIZON",
                    "seed": seed,
                    "problem": {
                        "N": scenario_cfg.problem.N,
                        "M_off": offline_count,
                        "M_on": online_count,
                    },
                    "offline": {
                        "method": "FullHorizonMILP",
                        "status": optimal_info.status,
                        "runtime": float(optimal_info.runtime),
                        "obj_value": float(optimal_info.obj_value),
                        "mip_gap": float(optimal_info.mip_gap),
                        "items_in_fallback": fallback_opt,
                    },
                    "online": {
                        "policy": "FullHorizonMILP",
                        "status": optimal_info.status,
                        "runtime": 0.0,
                        "total_cost": 0.0,
                        "fallback_items": fallback_opt,
                        "evicted_offline": 0,
                        "total_objective": float(optimal_info.obj_value),
                    },
                    "final_items_in_fallback": fallback_opt,
                    "offline_assignments": {str(k): int(v) for k, v in optimal_state.assigned_bin.items()},
                }
                optimal_prefix = (
                    f"pipeline_optimal_full_horizon_{scenario.name}_"
                    f"N{scenario_cfg.problem.N}_Moff{offline_count}_Mon{online_count}_seed{seed}"
                )
                optimal_path = save_combined_result(
                    optimal_prefix,
                    optimal_summary,
                    output_dir=str(scenario_dir),
                )
                print(
                    f"  OPTIMAL_FULL_HORIZON: offline {optimal_info.runtime:.3f}s, "
                    f"objective={optimal_info.obj_value:.3f} -> {optimal_path}"
                )

            for spec in pipeline_specs:
                cfg_run = copy.deepcopy(scenario_cfg)
                set_global_seed(seed)
                cache_id = spec.offline_cache_key or spec.name
                cache_key = (scenario.name, seed, cache_id)
                cached_solution = offline_cache.get(cache_key)
                if cached_solution is None and shared_instance.offline_items:
                    solver = spec.offline_factory(cfg_run)
                    offline_state, offline_info = solver.solve(copy.deepcopy(shared_instance))
                    cached_solution = (offline_state, offline_info)
                    offline_cache[cache_key] = cached_solution
                summary, path = run_pipeline(
                    cfg_run,
                    spec,
                    seed=seed,
                    instance=shared_instance,
                    offline_solution=cached_solution,
                    output_dir=str(scenario_dir),
                )
                offline_runtime = summary["offline"]["runtime"]
                online_runtime = summary["online"]["runtime"]
                total_obj = summary["online"]["total_objective"]
                fallback_final = summary["final_items_in_fallback"]
                print(
                    f"  {spec.name}: "
                    f"offline {offline_runtime:.3f}s, online {online_runtime:.3f}s, "
                    f"fallback={fallback_final}, objective={total_obj:.3f} -> {path}"
                )


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)
    scenarios = list(select_scenarios(args.scenarios))
    pipelines = select_pipelines(args.pipelines)
    run_config_sweep(base_cfg, scenarios, pipelines, args.output_root)


if __name__ == "__main__":
    main()
