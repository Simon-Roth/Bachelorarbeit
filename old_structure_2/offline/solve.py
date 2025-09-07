
from __future__ import annotations
from typing import Dict, Optional
from config import OfflineConfig
from offline.model import OfflineMILP

def build_and_solve(cfg: OfflineConfig, bins_df, items_df, cost_df, mask_df,
                    timelimit: Optional[float]=None, mipgap: Optional[float]=None, log: bool=False) -> Dict:
    model = OfflineMILP(cfg, bins_df, items_df, cost_df, mask_df)
    sol = model.solve(timelimit=timelimit, mipgap=mipgap, log=log)
    report = model.validate(sol)
    return {'solution': sol, 'report': report}
