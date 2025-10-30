from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import load_config
from core.general_utils import set_global_seed
from data.generators import generate_offline_instance
from offline.offline_heuristics.best_fit_decreasing import BestFitDecreasing
from experiments.utils import save_results

def main():
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)
    
    inst = generate_offline_instance(cfg, seed=seed)
    heuristic = BestFitDecreasing(cfg)
    state, info = heuristic.solve(inst)
    
    print(f"\nBFD: {info.runtime:.3f}s, Obj: {info.obj_value:.2f}")
    
    save_results('BFD', state, info,
                {'N': cfg.problem.N, 'M_off': cfg.problem.M_off}, seed)

if __name__ == "__main__":
    main()
