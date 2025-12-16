from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from core.models import Instance, AssignmentState
from offline.offline_heuristics.core import HeuristicSolutionInfo

class BestFitDecreasing():
    """Best-Fit Decreasing heuristic"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def solve(self, inst: Instance) -> Tuple[AssignmentState, HeuristicSolutionInfo]:
        """Solve using Best-Fit Decreasing"""
        import time
        start_time = time.perf_counter()
        
        # Sort items by volume (decreasing)
        items_sorted = sorted(enumerate(inst.offline_items), 
                            key=lambda x: x[1].volume, reverse=True)
        
        # Initialize
        loads = np.zeros(len(inst.bins) + 1)
        assigned_bin = {}
        
        from core.general_utils import effective_capacity
        eff_caps = [effective_capacity(bin.capacity, self.cfg.slack.enforce_slack, 
                                     self.cfg.slack.fraction) for bin in inst.bins]
        
        # Assign items
        for item_idx, item in items_sorted:
            best_bin = -1
            best_remaining = float('inf')
            
            # Find best-fitting bin (minimum remaining space)
            for bin_idx in range(len(inst.bins)):
                if (inst.feasible.feasible[item_idx, bin_idx] == 1 and
                    loads[bin_idx] + item.volume <= eff_caps[bin_idx]):
                    
                    remaining = eff_caps[bin_idx] - (loads[bin_idx] + item.volume)
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_bin = bin_idx
            
            # Assign to best bin or fallback
            if best_bin != -1:
                loads[best_bin] += item.volume
                assigned_bin[item_idx] = best_bin
            elif (
                self.cfg.problem.fallback_is_enabled
                and inst.feasible.feasible[item_idx, len(inst.bins)] == 1
            ):
                fallback_idx = len(inst.bins)
                loads[fallback_idx] += item.volume
                assigned_bin[item_idx] = fallback_idx
            else:
                raise ValueError(f"Item {item_idx} cannot be assigned")
        
        runtime = time.perf_counter() - start_time
        obj_value = self._calculate_objective(assigned_bin, inst)
        
        state = AssignmentState(
            load=loads,
            assigned_bin=assigned_bin,
            offline_evicted=set()
        )
        
        info = HeuristicSolutionInfo(
            algorithm="Best-Fit Decreasing",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
            items_in_fallback=sum(1 for bin_id in assigned_bin.values() 
                                if bin_id == len(inst.bins)),
            utilization=np.mean(loads[:-1] / [b.capacity for b in inst.bins])
        )
        
        return state, info
    
    
    def _calculate_objective(self, assigned_bin: Dict[int, int], inst: Instance) -> float:
        """Calculate objective value"""
        total_cost = 0.0
        for item_idx, bin_idx in assigned_bin.items():
            if bin_idx < len(inst.bins):  # Regular bin
                total_cost += inst.costs.assign[item_idx, bin_idx]
            else:  # Fallback bin
                total_cost += inst.costs.huge_fallback
        return total_cost
