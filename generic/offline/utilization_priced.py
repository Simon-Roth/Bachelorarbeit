from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from generic.core.models import AssignmentState, Instance, OfflineSolutionInfo
from generic.core.utils import effective_capacity, scalarize_vector
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.offline.solver import OfflineMILPSolver


class UtilizationPricedDecreasing:
    """
    Generic volume-ordered heuristic with congestion-based pricing.

    Items are processed in descending order of their worst-case normalised
    capacity footprint.  Each capacity constraint j maintains a utilisation-
    based price λ[j].  The assignment score for option k is:

        adjusted_cost[k] = raw_cost[k] / avg_cost
                         + dot(λ, cap_matrix[:, k] / eff_caps)

    This is the natural Lagrangian generalisation of the BGAP per-bin pricing:
    instead of one price per bin, each of the m capacity constraints gets its
    own dual variable that grows with utilisation.

    Compatible with any Instance produced by a generic or BGAP generator.
    """

    def __init__(
        self,
        cfg,
        *,
        price_exponent: float | None = None,
        update_rule: str | None = None,
        exp_rate: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.update_rule = (update_rule or cfg.util_pricing.update_rule).lower()
        self.price_exponent = (
            price_exponent if price_exponent is not None else cfg.util_pricing.price_exponent
        )
        self.exp_rate = exp_rate if exp_rate is not None else cfg.util_pricing.exp_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        start_time = time.perf_counter()

        regular_options = inst.n
        fallback_idx = inst.fallback_option_index
        cols = regular_options + (1 if fallback_idx >= 0 else 0)

        # Early exit for empty offline phase
        if len(inst.offline_steps) == 0:
            runtime = time.perf_counter() - start_time
            state = AssignmentState(
                load=np.zeros(inst.m),
                assigned_option={},
                offline_evicted_steps=set(),
            )
            info = OfflineSolutionInfo(
                algorithm="UtilizationPricedDecreasing",
                status="FEASIBLE",
                runtime=runtime,
                obj_value=0.0,
                feasible=True,
            )
            return state, info

        # Cost normalisation: scale raw costs so pricing is dimensionless
        avg_cost = float(
            np.mean(inst.costs.assignment_costs[: len(inst.offline_steps), :regular_options])
        )
        if not np.isfinite(avg_cost) or avg_cost <= 0.0:
            raise ValueError("Negative or infinite mean cost — cannot normalise for UtilPriced.")

        # Effective capacities: shape (m,)
        eff_caps = np.asarray(
            effective_capacity(inst.b, self.cfg.slack.enforce_slack, self.cfg.slack.fraction),
            dtype=float,
        )

        # Sort steps descending by worst-case normalised footprint.
        # For each step: max over options of normalised usage per constraint,
        # then scalarize over the m constraints.
        size_key = self.cfg.heuristics.size_key
        steps_sorted = self._sort_steps(inst, eff_caps, size_key)

        # State
        loads: np.ndarray = np.zeros(inst.m, dtype=float)
        lam: np.ndarray = np.zeros(inst.m, dtype=float)  # per-constraint prices

        solver = OfflineMILPSolver(self.cfg, log_to_console=False)
        assigned_option: Dict[int, int] = {}

        for step_idx, _ in steps_sorted:
            step = inst.offline_steps[step_idx]
            cap_matrix = np.asarray(step.cap_matrix, dtype=float)  # (m, n)
            remaining = np.maximum(0.0, eff_caps - loads)           # (m,)

            # Build adjusted costs: normalised raw cost + congestion penalty
            raw_costs = np.asarray(
                inst.costs.assignment_costs[step_idx, :cols], dtype=float
            ).reshape(1, -1)
            adjusted = raw_costs / avg_cost  # shape (1, cols)

            # Congestion penalty: dot(λ / eff_caps, cap_matrix[:, k]) for each k
            lam_norm = np.where(eff_caps > 0.0, lam / eff_caps, 0.0)  # (m,)
            for k in range(regular_options):
                adjusted[0, k] += float(np.dot(lam_norm, cap_matrix[:, k]))
            # fallback option gets no congestion penalty (it always "fits")

            data = build_offline_milp_data_from_arrays(
                cap_matrices=cap_matrix.reshape(1, inst.m, inst.n),
                costs=adjusted,
                feas_matrices=[np.asarray(step.feas_matrix, dtype=float)],
                feas_rhs=[np.asarray(step.feas_rhs, dtype=float)],
                b=remaining,
                fallback_idx=fallback_idx,
                slack_enforce=False,
                slack_fraction=0.0,
            )
            step_state, step_info = solver.solve_from_data(data)

            if not step_info.feasible:
                raise ValueError(f"Step {step_idx} cannot be assigned to any feasible option.")
            chosen = step_state.assigned_option.get(0)
            if chosen is None:
                raise ValueError(f"Step {step_idx}: solver returned no assignment.")

            assigned_option[step_idx] = chosen

            # Update loads and prices for all constraints
            if chosen < regular_options:
                loads += cap_matrix[:, chosen]
                lam = self._updated_lambda(loads, eff_caps)

        runtime = time.perf_counter() - start_time
        obj_value = self._objective(assigned_option, inst)

        state = AssignmentState(
            load=loads,
            assigned_option=assigned_option,
            offline_evicted_steps=set(),
        )
        info = OfflineSolutionInfo(
            algorithm="UtilizationPricedDecreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        return state, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sort_steps(
        self, inst: Instance, eff_caps: np.ndarray, size_key: str
    ) -> List[Tuple[int, None]]:
        """Sort offline steps descending by worst-case normalised capacity footprint."""
        denom = np.where(eff_caps > 0.0, eff_caps, 1.0)

        def _key(step_idx: int) -> float:
            cap = np.asarray(inst.offline_steps[step_idx].cap_matrix, dtype=float)  # (m, n)
            # For each constraint j: max fractional usage over all options
            max_frac = np.max(cap / denom[:, None], axis=1)  # (m,)
            return scalarize_vector(max_frac, size_key)

        indices = list(range(len(inst.offline_steps)))
        indices.sort(key=_key, reverse=True)
        return [(i, None) for i in indices]

    def _updated_lambda(self, load: np.ndarray, capacity: np.ndarray) -> np.ndarray:
        """Compute per-constraint prices from current utilisation."""
        util = np.clip(load / np.where(capacity > 0.0, capacity, 1.0), 0.0, 1.0)
        if self.update_rule == "exponential":
            return np.exp(self.exp_rate * util) - 1.0
        return util ** self.price_exponent

    @staticmethod
    def _objective(assigned_option: Dict[int, int], inst: Instance) -> float:
        """Total assignment cost, charging huge_fallback for fallback assignments."""
        total = 0.0
        fb = inst.fallback_option_index
        for step_idx, opt in assigned_option.items():
            if fb >= 0 and opt == fb:
                total += inst.costs.huge_fallback
            else:
                total += float(inst.costs.assignment_costs[step_idx, opt])
        return total
