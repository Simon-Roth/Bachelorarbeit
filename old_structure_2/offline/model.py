
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

from config import FALLBACK_COL, OfflineConfig

class OfflineMILP:
    """Offline assignment with x[i,j] and vector capacities (D dims)."""
    def __init__(self, cfg: OfflineConfig, bins_df: pd.DataFrame, items_df: pd.DataFrame,
                 cost_df: pd.DataFrame, mask_df: pd.DataFrame):
        self.cfg = cfg
        self.bins_df = bins_df.reset_index(drop=True)
        self.items_df = items_df.reset_index(drop=True)
        self.cost_df = cost_df.reset_index(drop=True)
        self.mask_df = mask_df.reset_index(drop=True).astype(bool)
        self.N = cfg.N
        self.M = cfg.M
        self.D = cfg.D
        self.reserve = cfg.reserve_vec()  # length D
        # column order must be bin_0..bin_{N-1}[, fallback]
        self.col_names = list(self.cost_df.columns)
        if cfg.use_fallback:
            assert self.col_names[-1] == FALLBACK_COL, "last column must be 'fallback' when enabled"
            self.regular_cols = self.col_names[:-1]
        else:
            self.regular_cols = self.col_names

        # sanity checks
        assert self.cost_df.shape == self.mask_df.shape, "cost/mask shapes mismatch"
        assert set(self.regular_cols) == set([f"bin_{i}" for i in range(self.N)]), "bin column mismatch"
        for d in range(self.D):
            assert f"capacity_d{d}" in self.bins_df.columns, f"missing capacity_d{d}"
            assert f"volume_d{d}" in self.items_df.columns, f"missing volume_d{d}"

        self.model = None
        self.x = None  # dict (i,j) -> binary var

    def build(self, env: Optional['gp.Env']=None):
        if gp is None:
            raise RuntimeError("gurobipy is not available. Install Gurobi and try again.")
        m = gp.Model(env=env) if env is not None else gp.Model()
        m.Params.OutputFlag = 0
        x = {}

        # variables only where mask True
        for i, col in enumerate(self.col_names):
            for j in range(self.M):
                if bool(self.mask_df.loc[j, col]):
                    x[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # assignment constraints: each item exactly once
        for j in range(self.M):
            vars_j = [x[(i, j)] for i, col in enumerate(self.col_names) if (i, j) in x]
            m.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")

        # vector capacity constraints for each dimension d and regular bin i
        for i, col in enumerate(self.regular_cols):
            for d in range(self.D):
                vol_sum = gp.LinExpr()
                for j in range(self.M):
                    if (i, j) in x:
                        vol = float(self.items_df.loc[j, f'volume_d{d}'])
                        vol_sum += vol * x[(i, j)]
                cap = float(self.bins_df.loc[i, f'capacity_d{d}']) * (1.0 - self.reserve[d])
                m.addConstr(vol_sum <= cap, name=f"cap_{col}_d{d}")

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

        # 1) each item exactly once
        per_item = assigned.sum(axis=1).tolist()
        each_once = all(v == 1 for v in per_item)

        # 2) capacities per dimension
        overfull = []
        for i, col in enumerate(self.regular_cols):
            for d in range(self.D):
                vol = 0.0
                for j in range(self.M):
                    if assigned.loc[j, col]:
                        vol += float(self.items_df.loc[j, f'volume_d{d}'])
                cap = float(self.bins_df.loc[i, f'capacity_d{d}']) * (1.0 - self.reserve[d])
                if vol > cap + 1e-6:
                    overfull.append((col, d, vol, cap))

        # 3) fallback count
        num_fallback = int(assigned[FALLBACK_COL].sum()) if self.cfg.use_fallback else 0

        # 4) fill ratios per bin per dimension
        fill = {}
        for i, col in enumerate(self.regular_cols):
            dim_ratios = []
            for d in range(self.D):
                numer = sum(float(self.items_df.loc[j, f'volume_d{d}']) for j in range(self.M) if assigned.loc[j, col])
                denom = float(self.bins_df.loc[i, f'capacity_d{d}']) * (1.0 - self.reserve[d]) + 1e-9
                dim_ratios.append(numer / denom)
            fill[col] = dim_ratios

        return {
            'each_item_assigned_once': each_once,
            'overfull': overfull,
            'num_fallback': num_fallback,
            'fill_ratios': fill
        }
