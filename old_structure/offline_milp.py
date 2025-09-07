
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

class OfflineMILP:
    """Offline assignment model with x[i,j] (bin i, item j)."""
    def __init__(self, bins_df: pd.DataFrame, items_df: pd.DataFrame,
                 cost_df: pd.DataFrame, edge_mask_df: pd.DataFrame,
                 reserve: float = 0.0):
        self.bins_df = bins_df.reset_index(drop=True)
        self.items_df = items_df.reset_index(drop=True)
        self.cost_df = cost_df.reset_index(drop=True)
        self.edge_mask_df = edge_mask_df.reset_index(drop=True).astype(bool)
        self.reserve = float(reserve)
        self.N = len(self.bins_df)
        self.M = len(self.items_df)
        self.col_names = list(self.cost_df.columns)
        assert self.col_names[-1] == 'fallback', "last column must be 'fallback'"
        self.regular_cols = self.col_names[:-1]
        self.fallback_col = 'fallback'
        assert self.cost_df.shape == self.edge_mask_df.shape, "cost and mask shapes mismatch"
        assert set(self.regular_cols) == set([f"bin_{i}" for i in range(self.N)]), "bin columns mismatch"
        self.model = None
        self.x = None  # dict (i,j) -> var

    def build(self, env: Optional['gp.Env']=None):
        if gp is None:
            raise RuntimeError("gurobipy is not available. Install Gurobi and try again.")
        m = gp.Model(env=env) if env is not None else gp.Model()
        m.Params.OutputFlag = 0
        x = {}
        # variables x[i,j] where assignment allowed
        for i, col in enumerate(self.col_names):
            for j in range(self.M):
                if bool(self.edge_mask_df.loc[j, col]):
                    x[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        # each item exactly once
        for j in range(self.M):
            vars_j = [x[(i, j)] for i, col in enumerate(self.col_names) if (i, j) in x]
            m.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")
        # capacity for regular bins
        for i, col in enumerate(self.regular_cols):
            vol_sum = gp.LinExpr()
            for j in range(self.M):
                if (i, j) in x:
                    vol = float(self.items_df.loc[j, 'volume'])
                    vol_sum += vol * x[(i, j)]
            cap = float(self.bins_df.loc[i, 'capacity']) * (1.0 - self.reserve)
            m.addConstr(vol_sum <= cap, name=f"cap_{col}")
        # objective
        obj = gp.LinExpr()
        for i, col in enumerate(self.col_names):
            for j in range(self.M):
                if (i, j) in x:
                    c = float(self.cost_df.loc[j, col])
                    obj += c * x[(i, j)]
        m.setObjective(obj, GRB.MINIMIZE)
        self.model = m
        self.x = x
        return m

    def solve(self, timelimit: Optional[float]=None, mipgap: Optional[float]=None, log: bool=False) -> Dict:
        if self.model is None:
            self.build()
        if log:
            self.model.Params.OutputFlag = 1
        if timelimit is not None:
            self.model.Params.TimeLimit = float(timelimit)
        if mipgap is not None:
            self.model.Params.MIPGap = float(mipgap)
        self.model.optimize()
        if self.model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(f"Model did not solve to optimality or time limit. Status={self.model.status}")
        return self.get_solution()

    def get_solution(self) -> Dict:
        assert self.model is not None and self.x is not None
        sol = {'assignments': [], 'objective': float(self.model.ObjVal), 'status': int(self.model.status)}
        for (i, j), var in self.x.items():
            if var.X > 0.5:
                col = self.col_names[i]
                sol['assignments'].append({'item_id': int(self.items_df.loc[j, 'item_id']), 'bin_col': col})
        return sol

    def validate(self, sol: Dict) -> Dict:
        assigned = pd.DataFrame(False, index=range(self.M), columns=self.col_names)
        for row in sol['assignments']:
            j = int(row['item_id'])
            col = row['bin_col']
            assigned.loc[j, col] = True
        per_item = assigned.sum(axis=1).tolist()
        all_one = all(v == 1 for v in per_item)
        overfull_bins = []
        for i, col in enumerate(self.regular_cols):
            vol = 0.0
            for j in range(self.M):
                if assigned.loc[j, col]:
                    vol += float(self.items_df.loc[j, 'volume'])
            cap = float(self.bins_df.loc[i, 'capacity']) * (1.0 - self.reserve)
            if vol > cap + 1e-6:
                overfull_bins.append((col, vol, cap))
        return {
            'each_item_assigned_once': all_one,
            'overfull_bins': overfull_bins,
            'num_fallback': int(assigned[self.fallback_col].sum()),
            'fill_ratios': {
                col: (sum(float(self.items_df.loc[j, 'volume']) for j in range(self.M) if assigned.loc[j, col])
                      / (float(self.bins_df.loc[i, 'capacity']) * (1.0 - self.reserve) + 1e-9))
                for i, col in enumerate(self.regular_cols)
            }
        }
