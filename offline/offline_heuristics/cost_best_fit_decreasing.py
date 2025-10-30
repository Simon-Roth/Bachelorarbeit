from __future__ import annotations

import time
from typing import Dict, Tuple, List

import numpy as np

from core.general_utils import effective_capacity
from core.models import AssignmentState, Instance
from offline.offline_heuristics.core import HeuristicSolutionInfo


class CostAwareBestFitDecreasing:
    """
    Greedy offline heuristic that sorts items by volume (descending) and assigns
    each to the feasible bin with the lowest assignment cost. Ties on cost are
    broken by the smallest residual capacity to avoid fragmentation.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def solve(self, inst: Instance) -> Tuple[AssignmentState, HeuristicSolutionInfo]:
        start_time = time.perf_counter()

        items_sorted: List[Tuple[int, float]] = sorted(
            ((idx, item.volume) for idx, item in enumerate(inst.offline_items)),
            key=lambda pair: pair[1],
            reverse=True,
        )

        regular_bins = len(inst.bins)
        fallback_idx = regular_bins
        loads = np.zeros(regular_bins + 1)
        assigned_bin: Dict[int, int] = {}

        eff_caps = [
            effective_capacity(
                bin_spec.capacity,
                self.cfg.slack.enforce_slack,
                self.cfg.slack.fraction,
            )
            for bin_spec in inst.bins
        ]

        for item_idx, volume in items_sorted:
            best_bin = None
            best_cost = float("inf")
            best_residual = float("inf")

            for bin_idx in range(regular_bins):
                if inst.feasible.feasible[item_idx, bin_idx] != 1:
                    continue
                projected = loads[bin_idx] + volume
                if projected > eff_caps[bin_idx] + 1e-9:
                    continue
                cost = float(inst.costs.assign[item_idx, bin_idx])
                residual = eff_caps[bin_idx] - projected
                if (
                    cost < best_cost - 1e-9
                    or (abs(cost - best_cost) <= 1e-9 and residual < best_residual)
                ):
                    best_cost = cost
                    best_residual = residual
                    best_bin = bin_idx

            if best_bin is not None:
                loads[best_bin] += volume
                assigned_bin[item_idx] = best_bin
                continue

            if (
                self.cfg.problem.fallback_is_enabled
                and inst.feasible.feasible[item_idx, fallback_idx] == 1
            ):
                loads[fallback_idx] += volume
                assigned_bin[item_idx] = fallback_idx
                continue

            raise ValueError(f"Item {item_idx} cannot be assigned to any feasible bin.")

        runtime = time.perf_counter() - start_time
        obj_value = self._calculate_objective(assigned_bin, inst)

        state = AssignmentState(
            load=loads,
            assigned_bin=assigned_bin,
            offline_evicted=set(),
        )
        capacities = np.array([bin.capacity for bin in inst.bins], dtype=float)
        utilization = float(np.mean(loads[:regular_bins] / capacities))

        info = HeuristicSolutionInfo(
            algorithm="CostAwareBestFitDecreasing",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
            items_in_fallback=sum(
                1 for bin_id in assigned_bin.values() if bin_id == fallback_idx
            ),
            utilization=utilization,
        )
        return state, info

    def _calculate_objective(
        self, assigned_bin: Dict[int, int], inst: Instance
    ) -> float:
        total_cost = 0.0
        fallback_idx = len(inst.bins)
        for item_idx, bin_idx in assigned_bin.items():
            if bin_idx < fallback_idx:
                total_cost += float(inst.costs.assign[item_idx, bin_idx])
            else:
                total_cost += inst.costs.huge_fallback
        return total_cost
