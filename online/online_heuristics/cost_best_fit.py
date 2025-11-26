from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy

from core.config import Config
from core.models import AssignmentState, Decision, Instance, OnlineItem
from online.policies.base import BaseOnlinePolicy, PolicyInfeasibleError
from online.state_utils import (
    PlacementContext,
    build_context,
    execute_placement,
    TOLERANCE,
)


class CostAwareBestFitOnlinePolicy(BaseOnlinePolicy):
    """
    Selects the feasible bin that minimises incremental cost (assignment plus potential eviction
    penalties). Uses residual capacity as a tie-breaker to avoid fragmentation.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional["numpy.ndarray"],
    ) -> Decision:
        candidate_bins = self._candidate_bins(item, instance, feasible_row)
        if not candidate_bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online item {item.id}")

        best_decision: Optional[Decision] = None
        best_cost = float("inf")
        best_residual = float("inf")

        for bin_id in candidate_bins:
            ctx = build_context(self.cfg, instance, state)
            decision = execute_placement(
                bin_id,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
                allow_eviction=True,
            )
            if decision is None:
                continue

            incremental_cost = decision.incremental_cost
            residual = ctx.effective_caps[bin_id] - ctx.loads[bin_id]
            residual_score = residual if residual >= -TOLERANCE else float("inf")

            if (
                incremental_cost < best_cost - 1e-9
                or (
                    abs(incremental_cost - best_cost) <= 1e-9
                    and residual_score < best_residual - 1e-9
                )
            ):
                best_cost = incremental_cost
                best_residual = residual_score
                best_decision = decision

        if best_decision is None:
            raise PolicyInfeasibleError(
                f"CostAwareBestFitOnlinePolicy could not place item {item.id}"
            )
        return best_decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_bins(
        self,
        item: OnlineItem,
        instance: Instance,
        feasible_row: Optional["numpy.ndarray"],
    ) -> List[int]:
        bins: Set[int] = set(item.feasible_bins)
        if feasible_row is not None:
            regular_bins = len(instance.bins)
            for idx, allowed in enumerate(feasible_row[:regular_bins]):
                if allowed:
                    bins.add(int(idx))
        return sorted(b for b in bins if 0 <= b < len(instance.bins))

    def _eviction_order_desc(
        self,
        bin_id: int,
        ctx: PlacementContext,
    ) -> List[int]:
        regular_bins = len(ctx.instance.bins)
        offline_ids = [
            itm_id
            for itm_id, assigned_bin in ctx.assignments.items()
            if assigned_bin == bin_id and itm_id < len(ctx.instance.offline_items)
        ]
        offline_ids.sort(key=lambda oid: ctx.offline_volumes.get(oid, 0.0), reverse=True)
        return offline_ids

    def _select_reassignment_bin(
        self,
        offline_id: int,
        origin_bin: int,
        ctx: PlacementContext,
    ) -> Optional[int]:
        volume = ctx.offline_volumes.get(offline_id, 0.0)
        instance = ctx.instance
        feasible_row = instance.feasible.feasible[offline_id]
        regular_bins = len(instance.bins)
        fallback_idx = instance.fallback_bin_index

        best_candidate: Optional[int] = None
        best_cost = float("inf")
        best_residual = float("inf")

        for candidate in range(regular_bins):
            if candidate == origin_bin or feasible_row[candidate] != 1:
                continue
            residual = ctx.effective_caps[candidate] - (ctx.loads[candidate] + volume)
            if residual + TOLERANCE < 0:
                continue
            cost = instance.costs.assign[offline_id, candidate]
            if cost < best_cost - 1e-9 or (
                abs(cost - best_cost) <= 1e-9 and residual < best_residual
            ):
                best_cost = cost
                best_residual = residual
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

        if (
            self.cfg.problem.fallback_is_enabled
            and fallback_idx < feasible_row.shape[0]
            and feasible_row[fallback_idx] == 1
        ):
            return fallback_idx
        return None
