from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from core.config import Config
from core.models import AssignmentState
from experiments.utils import (
    run_offline_online_pipeline,
    build_pipeline_summary,
    save_combined_result,
)


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


def run_pipeline(
    cfg: Config,
    spec: PipelineSpec,
    *,
    seed: Optional[int] = None,
    output_dir: str = "results",
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

    Returns
    -------
    summary, path:
        JSON-friendly summary dictionary and the file path it was written to.
    """
    chosen_seed = cfg.eval.seeds[0] if seed is None else seed

    offline_solver = spec.offline_factory(cfg)
    online_policy = spec.online_factory(cfg)

    (
        instance,
        offline_state,
        final_state,
        offline_info,
        online_info,
    ) = run_offline_online_pipeline(cfg, chosen_seed, offline_solver, online_policy)

    summary = build_pipeline_summary(
        spec.name,
        chosen_seed,
        instance,
        offline_state,
        final_state,
        offline_info,
        online_info,
        offline_method=spec.offline_label,
        online_method=spec.online_label,
    )

    problem = summary["problem"]
    prefix = (
        f"pipeline_{spec.name.replace('+', '_').replace(' ', '_')}_"
        f"N{problem['N']}_Moff{problem['M_off']}_Mon{problem['M_on']}_seed{chosen_seed}"
    )
    path = save_combined_result(prefix, summary, output_dir)
    return summary, path
