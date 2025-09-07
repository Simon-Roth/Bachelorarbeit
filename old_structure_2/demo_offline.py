
import argparse
import pandas as pd
from config import OfflineConfig
from scenarios import SCENARIO_REGISTRY
from offline.solve import build_and_solve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', type=str, default='uniform_full', choices=list(SCENARIO_REGISTRY.keys()))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--items', type=int, default=40)
    ap.add_argument('--dims', type=int, default=1)
    ap.add_argument('--reserve', type=float, default=0.0)
    ap.add_argument('--p', type=float, default=0.4, help='edge probability for lognormal_sparse_p')
    ap.add_argument('--no-fallback', action='store_true', help='disable fallback bin')
    ap.add_argument('--log', action='store_true', help='show Gurobi log')
    args = ap.parse_args()

    params = {'p': args.p}  # used by scenarios
    cfg = OfflineConfig(N=args.bins, M=args.items, D=args.dims, reserve=args.reserve,
                        use_fallback=(not args.no_fallback), seed=args.seed, params=params)

    builder = SCENARIO_REGISTRY[args.scenario]
    inst = builder(cfg)

    result = build_and_solve(cfg, inst.bins_df, inst.items_df, inst.cost_df, inst.mask_df, log=args.log)
    sol, rep = result['solution'], result['report']

    print("Objective:", sol['objective'])
    print("Each item assigned once:", rep['each_item_assigned_once'])
    print("Overfull constraints:", rep['overfull'])
    print("Num fallback:", rep['num_fallback'])
    print("Fill ratios per bin (list per dimension):")
    for col, ratios in rep['fill_ratios'].items():
        print(f"  {col}: {', '.join(f'{r:.3f}' for r in ratios)}")

    df_assign = pd.DataFrame(sol['assignments']).sort_values(['bin_col','item_id'])
    print("\nAssignments (head):\n", df_assign.head())

if __name__ == '__main__':
    main()
