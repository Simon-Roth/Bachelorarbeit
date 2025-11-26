from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import copy

from core.config import Config
from experiments.utils import (
    build_pipeline_summary,
    save_combined_result,
    build_empty_offline_solution,
)
from pathlib import Path
from online.online_solver import OnlineSolver
from online.prices import compute_prices
from data.generators import generate_instance_with_online
from core.models import AssignmentState, Instance
from offline.models import OfflineSolutionInfo


OfflineFactory = Callable[[Config], object]
OnlineFactory = Callable[[Config], object]


@dataclass(frozen=True)
class PipelineSpec:
    """
    Lightweight description of an offline+online experiment.

    Attributes
    ----------
    name:
        Display name for logs / JSON summary.
    offline_label:
        Label stored in the JSON for the offline component.
    online_label:
        Label stored in the JSON for the online component.
    offline_factory:
        Callable that returns a solver/heuristic with a ``solve(instance)`` method.
    online_factory:
        Callable that returns a policy implementing ``select_bin(...)``.
    """

    name: str
    offline_label: str
    online_label: str
    offline_factory: OfflineFactory
    online_factory: OnlineFactory
    offline_cache_key: str | None = None


def run_pipeline(
    cfg: Config,
    spec: PipelineSpec,
    *,
    seed: Optional[int] = None,
    output_dir: str = "results",
    instance: Optional[Instance] = None,
    offline_solution: Optional[Tuple[AssignmentState, OfflineSolutionInfo]] = None,
) -> Tuple[dict, str]:
    """
    Execute a single offline+online pipeline described by `spec`.

    Parameters
    ----------
    cfg:
        Base configuration. (Caller may provide a copy if reuse is required.)
    spec:
        Pipeline specification containing factories and labels.
    seed:
        Seed used for instance generation; defaults to the first entry in cfg.eval.seeds.
    output_dir:
        Directory where the JSON summary will be written.
    instance:
        Optional pre-generated instance (deep-copied internally) to keep runs aligned.
    offline_solution:
        Optional cached offline result (state, info) to skip re-solving the offline part.

    Returns
    -------
    summary, path:
        JSON-friendly summary dictionary and the file path it was written to.
    """
    chosen_seed = cfg.eval.seeds[0] if seed is None else seed
    if instance is None:
        base_instance = generate_instance_with_online(cfg, seed=chosen_seed)
    else:
        base_instance = copy.deepcopy(instance)

    offline_count = len(base_instance.offline_items)
    online_count = len(base_instance.online_items)

    if offline_count == 0:
        offline_state, offline_info = build_empty_offline_solution(base_instance)
    elif offline_solution is not None:
        offline_state = copy.deepcopy(offline_solution[0])
        offline_info = copy.deepcopy(offline_solution[1])
    else:
        offline_solver = spec.offline_factory(cfg)
        offline_state, offline_info = offline_solver.solve(base_instance)

    prices_path = Path("results/primal_dual.json")
    if spec.online_label == "PrimalDual" and online_count > 0:
        prices_path.parent.mkdir(parents=True, exist_ok=True)
        compute_prices(cfg, base_instance, offline_state, prices_path)

    online_policy = spec.online_factory(cfg)
    online_solver = OnlineSolver(cfg, online_policy)
    final_state, online_info = online_solver.run(base_instance, offline_state)

    summary = build_pipeline_summary(
        spec.name,
        chosen_seed,
        base_instance,
        offline_state,
        final_state,
        offline_info,
        online_info,
        offline_method=spec.offline_label,
        online_method=spec.online_label,
        slack_config=cfg.slack,
    )

    problem = summary["problem"]
    prefix = (
        f"pipeline_{spec.name.replace('+', '_').replace(' ', '_')}_"
        f"N{problem['N']}_Moff{problem['M_off']}_Mon{problem['M_on']}_seed{chosen_seed}"
    )
    path = save_combined_result(prefix, summary, output_dir)
    return summary, path
