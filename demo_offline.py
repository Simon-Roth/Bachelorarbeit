
import argparse, pandas as pd
from config import OfflineConfig
from instance_builder import build_instance
from offline.solve import build_and_solve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', type=str, default='full', choices=['full','bernoulli'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--items', type=int, default=40)
    ap.add_argument('--dims', type=int, default=1)
    ap.add_argument('--log', action='store_true')
    args = ap.parse_args()

    # All control is here:
    params = {
        "capacity": {"distribution": "uniform", "low": 60, "high": 100},
        "volume":   {"distribution": "uniform", "low": 3, "high": 12},
        "cost":     {"model": "utilization", "base": 10.0, "noise": 0.25},
        "fallback": {"use": True, "cost": 1000.0},
        "graph_p":  0.4,
        "bigM_forbidden": 1e6
    }
    

    cfg = OfflineConfig(N=args.bins, M=args.items, D=args.dims, seed=args.seed, params=params, use_fallback=params["fallback"]["use"])
    inst = build_instance(cfg, graph_type=args.graph)
    result = build_and_solve(cfg, inst.bins_df, inst.items_df, inst.cost_df, inst.mask_df, log=args.log)
    sol, rep = result['solution'], result['report']

    print("Objective:", sol['objective'])
    print("Each item assigned once:", rep['each_item_assigned_once'])
    print("Overfull constraints:", rep['overfull'])
    print("Num fallback:", rep['num_fallback'])
    for col, ratios in rep['fill_ratios'].items():
        print(f"  {col}: {', '.join(f'{r:.3f}' for r in ratios)}")

    if sol['assignments']:
        df_assign = pd.DataFrame(sol['assignments']).sort_values(['bin_col','item_id'])
        print("\nAssignments (head):\n", df_assign.head())
    else:
        print("\nNo assignments found (check feasibility or fallback).")

if __name__ == '__main__':
    main()
