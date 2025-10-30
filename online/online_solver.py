from __future__ import annotations

import time
from typing import List, Optional, Tuple

from core.config import Config
from core.models import AssignmentState, Instance, Decision
from online.models import OnlineSolutionInfo
from online.policies.base import BaseOnlinePolicy, PolicyInfeasibleError
from online import state_utils


class OnlineSolver:
    """
    Orchestrates the online phase given a policy implementation.
    """

    def __init__(
        self,
        cfg: Config,
        policy: BaseOnlinePolicy,
        *,
        log_to_console: bool = False,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.log_to_console = log_to_console

    def run(
        self,
        instance: Instance,
        initial_state: AssignmentState,
    ) -> Tuple[AssignmentState, OnlineSolutionInfo]:
        """
        Execute the online phase starting from 'initial_state'.
        """
        if not instance.online_items:
            state_copy = state_utils.clone_state(initial_state)
            info = OnlineSolutionInfo(
                status="NO_ITEMS",
                runtime=0.0,
                total_cost=0.0,
                fallback_items=state_utils.count_fallback_items(state_copy, instance),
                evicted_offline=len(state_copy.offline_evicted),
                decisions=[],
            )
            return state_copy, info

        feasible_matrix = None
        if instance.online_feasible is not None:
            feasible_matrix = instance.online_feasible.feasible

        state = state_utils.clone_state(initial_state)
        volume_lookup = state_utils.build_volume_lookup(instance)
        decisions: List[Decision] = []
        total_cost = 0.0

        start_time = time.perf_counter()

        for idx, item in enumerate(instance.online_items):
            feasible_row = None
            if feasible_matrix is not None:
                if idx >= feasible_matrix.shape[0]:
                    raise IndexError(
                        f"Feasibility matrix has fewer rows ({feasible_matrix.shape[0]}) "
                        f"than online items ({len(instance.online_items)})."
                    )
                feasible_row = feasible_matrix[idx, :]

            try:
                decision = self.policy.select_bin(item, state, instance, feasible_row)
            except PolicyInfeasibleError:
                info = OnlineSolutionInfo(
                    status="INFEASIBLE",
                    runtime=time.perf_counter() - start_time,
                    total_cost=total_cost,
                    fallback_items=state_utils.count_fallback_items(state, instance),
                    evicted_offline=len(state.offline_evicted),
                    decisions=decisions,
                )
                return state, info
            state_utils.apply_decision(
                decision,
                item,
                state,
                instance,
                volume_lookup,
            )
            decisions.append(decision)
            total_cost += decision.incremental_cost

        runtime = time.perf_counter() - start_time

        info = OnlineSolutionInfo(
            status="COMPLETED",
            runtime=runtime,
            total_cost=total_cost,
            fallback_items=state_utils.count_fallback_items(state, instance),
            evicted_offline=len(state.offline_evicted),
            decisions=decisions,
        )
        return state, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_online_sequence(
        cfg: Config,
        instance: Instance,
        seed: int,
        *,
        horizon: Optional[int] = None,
    ) -> None:
        """
        Populate 'instance' with an online sequence if it is currently empty.
        Useful for workflows that only generated offline data.
        """
        state_utils.ensure_online_sequence(
            cfg,
            instance,
            seed,
            horizon=horizon,
        )
