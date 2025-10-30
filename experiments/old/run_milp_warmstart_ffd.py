from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import load_config
from core.general_utils import set_global_seed
from data.generators import generate_offline_instance
from offline.offline_solver import OfflineMILPSolver
from offline.offline_heuristics.first_fit_decreasing import FirstFitDecreasing
from experiments.utils import save_results

def main():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)
    
    inst = generate_offline_instance(cfg, seed=seed)
    
    # Generate warm start with FFD
    heuristic = FirstFitDecreasing(cfg)
    heuristic_state, _ = heuristic.solve(inst)
    warm_start = heuristic_state.assigned_bin
    
    # Solve with warm start
    solver = OfflineMILPSolver(cfg, time_limit=60, mip_gap=0.01,
                              threads=0, log_to_console=True)
    state, info = solver.solve(inst, warm_start=warm_start)
    
    print(f"\nMILP+FFD: {info.status} in {info.runtime:.3f}s, Obj: {info.obj_value:.2f}")
    
    save_results('MILP+FFD', state, info,
                {'N': cfg.problem.N, 'M_off': cfg.problem.M_off}, seed)

if __name__ == "__main__":
    main()
