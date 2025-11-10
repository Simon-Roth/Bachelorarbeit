from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from core.config import Config, load_config
from experiments.pipeline_registry import PIPELINE_REGISTRY, PIPELINES
from experiments.pipeline_runner import run_pipeline


@dataclass(frozen=True)
class ScenarioJob:
    family: str
    variant: str
    pipeline_name: str
    seed: int
    config: Config
    output_dir: Path


def load_manifest(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not data or "scenario_families" not in data:
        raise ValueError(f"Manifest at {path} is missing 'scenario_families'.")
    return data


def resolve_field(
    key: str,
    *,
    variant: Dict[str, Any],
    family: Dict[str, Any],
    defaults: Dict[str, Any],
    fallback: Any = None,
):
    if key in variant:
        return variant[key]
    if key in family:
        return family[key]
    return defaults.get(key, fallback)


def apply_config_overrides(base_cfg: Config, overrides: Dict[str, Any]) -> Config:
    cfg = copy.deepcopy(base_cfg)

    def _apply(target, patch: Dict[str, Any]) -> None:
        for key, value in patch.items():
            if not hasattr(target, key):
                raise KeyError(f"Config object {target} has no attribute '{key}'")
            current = getattr(target, key)
            if isinstance(value, dict) and hasattr(current, "__dict__"):
                _apply(current, value)
            else:
                setattr(target, key, copy.deepcopy(value))

    _apply(cfg, overrides)
    return cfg


def expand_manifest(
    manifest: Dict[str, Any],
    *,
    families: Sequence[str] | None,
    pipelines_filter: Sequence[str] | None,
    limit_seeds: int | None,
) -> Iterable[ScenarioJob]:
    defaults = manifest.get("defaults", {})
    families_data = manifest.get("scenario_families", [])

    for family_entry in families_data:
        family_name = family_entry["name"]
        if families and family_name not in families:
            continue

        variants = family_entry.get("variants", [])
        if not variants:
            continue

        for variant in variants:
            variant_suffix = variant.get("id_suffix", variant.get("name", "variant"))
            variant_id = f"{family_name}_{variant_suffix}"

            base_config_path = Path(resolve_field(
                "base_config",
                variant=variant,
                family=family_entry,
                defaults=defaults,
                fallback="configs/default.yaml",
            ))
            base_cfg = load_config(base_config_path)

            overrides = variant.get("overrides", {})
            cfg_with_overrides = apply_config_overrides(base_cfg, overrides)

            pipelines = resolve_field(
                "pipelines",
                variant=variant,
                family=family_entry,
                defaults=defaults,
                fallback=[spec.name for spec in PIPELINES],
            )
            if pipelines_filter:
                pipelines = [name for name in pipelines if name in pipelines_filter]
                if not pipelines:
                    continue

            seeds = resolve_field(
                "seeds",
                variant=variant,
                family=family_entry,
                defaults=defaults,
                fallback=list(cfg_with_overrides.eval.seeds),
            )
            seeds = list(seeds)

            replications = resolve_field(
                "replications",
                variant=variant,
                family=family_entry,
                defaults=defaults,
                fallback=len(seeds),
            )
            if replications is None:
                replications = len(seeds)
            if replications > len(seeds):
                raise ValueError(
                    f"Variant '{variant_id}' requests {replications} replications but only "
                    f"{len(seeds)} seeds are provided."
                )
            selected_seeds = seeds[:replications]
            if limit_seeds:
                selected_seeds = selected_seeds[:limit_seeds]

            output_dir = Path(resolve_field(
                "output_dir",
                variant=variant,
                family=family_entry,
                defaults=defaults,
                fallback="results",
            )) / family_name / variant_suffix

            for pipeline_name in pipelines:
                if pipeline_name not in PIPELINE_REGISTRY:
                    known = ", ".join(sorted(PIPELINE_REGISTRY))
                    raise KeyError(
                        f"Unknown pipeline '{pipeline_name}' in variant '{variant_id}'. "
                        f"Known pipelines: {known}"
                    )
                for seed in selected_seeds:
                    yield ScenarioJob(
                        family=family_name,
                        variant=variant_suffix,
                        pipeline_name=pipeline_name,
                        seed=int(seed),
                        config=cfg_with_overrides,
                        output_dir=output_dir,
                    )


def run_jobs(jobs: Iterable[ScenarioJob], *, dry_run: bool) -> None:
    total = 0
    for job in jobs:
        total += 1
        prefix = f"[{job.family}/{job.variant}] {job.pipeline_name} (seed={job.seed})"
        if dry_run:
            print(f"DRY-RUN {prefix} -> would write to {job.output_dir}")
            continue

        cfg = copy.deepcopy(job.config)
        cfg.eval.seeds = (job.seed,)

        job.output_dir.mkdir(parents=True, exist_ok=True)
        summary, path = run_pipeline(
            cfg,
            PIPELINE_REGISTRY[job.pipeline_name],
            seed=job.seed,
            output_dir=str(job.output_dir),
        )
        print(f"{prefix} -> {path}")

    if dry_run:
        print(f"Dry run complete. {total} jobs listed.")
    else:
        print(f"Completed {total} jobs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario grid of pipelines.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/scenario_matrix.yaml"),
        help="Path to scenario manifest YAML.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        help="Optional list of family names to run.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        help="Optional list of pipeline names to include.",
    )
    parser.add_argument(
        "--limit-seeds",
        type=int,
        help="Limit the number of seeds per variant (after replication cap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list planned jobs without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    jobs = expand_manifest(
        manifest,
        families=args.families,
        pipelines_filter=args.pipelines,
        limit_seeds=args.limit_seeds,
    )
    run_jobs(jobs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
