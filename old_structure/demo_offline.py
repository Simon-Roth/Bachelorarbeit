
import argparse
import pandas as pd
from instance_generator import make_instance, instance_to_frames
from offline_milp import OfflineMILP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--items', type=int, default=40)
    ap.add_argument('--scenario', type=str, default='hospital',
                    choices=['uniform', 'lognormal_sparse', 'hospital'])
    ap.add_argument('--reserve', type=float, default=0.0)
    ap.add_argument('--log', action='store_true', help='show Gurobi log')
    args = ap.parse_args()

    inst = make_instance(args.seed, args.bins, args.items, scenario=args.scenario)
    bins_df, items_df, cost_df = instance_to_frames(inst)
    edge_mask_df = cost_df.copy() < 1e6

    model = OfflineMILP(bins_df, items_df, cost_df, edge_mask_df, reserve=args.reserve)
    try:
        sol = model.solve(log=args.log)
    except RuntimeError as e:
        print("Solve failed:", e)
        print("\nGurobi might not be available in your environment. Install it locally to solve.")
        return

    report = model.validate(sol)
    print("Objective:", sol['objective'])
    print("Assigned to fallback:", report['num_fallback'])
    print("Each item assigned once:", report['each_item_assigned_once'])
    if report['overfull_bins']:
        print("Overfull bins:", report['overfull_bins'])
    print("Fill ratios per bin:")
    for col, r in report['fill_ratios'].items():
        print(f"  {col}: {r:.3f}")

    df_assign = pd.DataFrame(sol['assignments']).sort_values(['bin_col','item_id'])
    print("\nAssignments (head):\n", df_assign.head(10))

if __name__ == '__main__':
    main()
