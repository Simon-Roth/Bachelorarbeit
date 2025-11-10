from __future__ import annotations

from core.config import load_config
from core.general_utils import set_global_seed
from experiments.pipeline_runner import run_pipeline
from experiments.pipeline_registry import PIPELINES

from data.generators import generate_instance_with_online
from experiments.optimal_benchmark import solve_full_horizon_optimum
from online.state_utils import count_fallback_items
from experiments.utils import save_combined_result
from offline.offline_solver import OfflineMILPSolver
import copy


def main() -> None:
    cfg = load_config("configs/default.yaml")
    base_seed = cfg.eval.seeds[0]
    
    # Compute full-horizon optimum once for reference
    compute_full_horizon_baseline(base_seed)
    
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

def compute_full_horizon_baseline(base_seed): 
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
    
if __name__ == "__main__":
    main()
