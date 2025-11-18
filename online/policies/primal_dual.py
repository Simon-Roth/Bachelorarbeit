from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, List, Dict

import numpy as np

from core.config import Config
from core.models import AssignmentState, Decision, Instance, OnlineItem
from online.policies.base import BaseOnlinePolicy, PolicyInfeasibleError
from online.state_utils import (
    PlacementContext,
    build_context,
    execute_placement,
    TOLERANCE,
)

class BalancedPricePolicy(BaseOnlinePolicy):
    """
    Cost-minimizing Lagrangian policy:
    - score(i) = c_{ji} + λ_i * volume_j
    - choose feasible regular bin with minimum score
    - if none chosen, DO NOT place to fallback explicitly
      -> return a Decision that assigns to fallback with incremental_cost=0.0
         so rejections can be counted later without charging costs.
    """

    def __init__(self, cfg: Config, price_path: Path = Path("results/balanced_prices.json")):
        self.cfg = cfg
        with open(price_path) as f:
            data = json.load(f)
        # per-regular-bin price λ_i
        self.lam: Dict[int, float] = {int(k): float(v) for k, v in data["prices"].items()}

    # Stubs (won't be used because we disallow evictions below)
    @staticmethod
    def _no_eviction_order(_: int, __: PlacementContext) -> List[int]:
        return []

    @staticmethod
    def _no_destination(_: int, __: int, ___: PlacementContext) -> Optional[int]:
        return None

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        ctx: PlacementContext = build_context(self.cfg, instance, state)
        regular_bins = len(instance.bins)
        fallback_idx = instance.fallback_bin_index

        candidate_bins: set[int] = set(item.feasible_bins)
        if feasible_row is not None:
            for idx, allowed in enumerate(feasible_row[:regular_bins]):
                if allowed:
                    candidate_bins.add(int(idx))

        best_score = float("inf")
        best_decision: Optional[Decision] = None

        for i in sorted(candidate_bins):
            if not (0 <= i < regular_bins):
                continue

            # capacity check against effective capacity
            residual = ctx.effective_caps[i] - (ctx.loads[i] + item.volume)
            if residual < -TOLERANCE:
                continue

            c_ji = float(instance.costs.assign[item.id, i])
            lam_i = self.lam.get(i, 0.0)
            score = c_ji + lam_i * float(item.volume)

            if score < best_score - 1e-12:
                decision = execute_placement(
                    i,
                    item,
                    ctx,
                    eviction_order_fn=self._no_eviction_order,
                    destination_fn=self._no_destination,
                    allow_eviction=False,
                )
                if decision is not None:
                    best_score = score
                    best_decision = decision

        if best_decision is not None:
            return best_decision  # already a full Decision with proper incremental_cost

        # No regular bin selected -> place into fallback *without cost*.
        return Decision(
            placed_item=(item.id, fallback_idx),
            evicted_offline=[],
            reassignments=[],
            incremental_cost=0.0,   # <-- intermediate fix: do not charge for fallback
        )
