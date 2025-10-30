from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import load_config
from core.general_utils import set_global_seed
from offline.offline_solver import OfflineMILPSolver
from online.heuristics.cost_best_fit import CostAwareBestFitOnlinePolicy
from experiments.utils import (
    run_offline_online_pipeline,
    build_pipeline_summary,
    save_combined_result,
)


def main() -> None:
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)

    offline_solver = OfflineMILPSolver(
        cfg,
        time_limit=60,
        mip_gap=0.01,
        threads=0,
        log_to_console=False,
    )
    online_policy = CostAwareBestFitOnlinePolicy(cfg)

    (
        instance,
        offline_state,
        final_state,
        offline_info,
        online_info,
    ) = run_offline_online_pipeline(cfg, seed, offline_solver, online_policy)

    pipeline_name = "MILP+CostAwareBestFit"
    summary = build_pipeline_summary(
        pipeline_name,
        seed,
        instance,
        offline_state,
        final_state,
        offline_info,
        online_info,
        offline_method="MILP",
        online_method="CostAwareBestFit",
    )

    problem = summary["problem"]
    prefix = (
        f"pipeline_milp_costaware_best_fit_N{problem['N']}_"
        f"Moff{problem['M_off']}_Mon{problem['M_on']}_seed{seed}"
    )
    save_combined_result(prefix, summary)

    print(
        f"\n{pipeline_name}: offline {summary['offline']['status']} "
        f"({summary['offline']['runtime']:.3f}s), "
        f"online {summary['online']['status']} "
        f"({summary['online']['runtime']:.3f}s)"
    )


if __name__ == "__main__":
    main()
