from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import json

import pandas as pd


@dataclass(frozen=True)
class ResultRecord:
    family: str
    variant: str
    pipeline: str
    seed: int
    path: Path
    payload: Dict[str, Any]


def _iter_result_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return iter(())
    for path in sorted(root.rglob("pipeline_*.json")):
        if path.is_file():
            yield path


def _infer_family_variant(path: Path, root: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ("unknown", "unknown")
    parts = relative.parts
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], "default"
    return ("unknown", "unknown")


def read_result_records(root: Path) -> Iterable[ResultRecord]:
    for path in _iter_result_files(root):
        with path.open("r") as fh:
            payload = json.load(fh)
        analysis_root = root
        family, variant = _infer_family_variant(path.parent, analysis_root)
        pipeline = str(payload.get("pipeline", "UNKNOWN"))
        seed = int(payload.get("seed", -1))
        yield ResultRecord(
            family=family,
            variant=variant,
            pipeline=pipeline,
            seed=seed,
            path=path,
            payload=payload,
        )


def load_results_dataframe(
    root: Path,
    *,
    include_metadata: bool = True,
) -> pd.DataFrame:
    records = []
    for record in read_result_records(root):
        payload = record.payload
        offline = payload.get("offline", {})
        online = payload.get("online", {})
        problem = payload.get("problem", {})

        row: Dict[str, Any] = {
            "family": record.family,
            "variant": record.variant,
            "pipeline": record.pipeline,
            "seed": record.seed,
            "result_path": str(record.path),
            "offline_status": offline.get("status"),
            "offline_obj": offline.get("obj_value"),
            "offline_runtime": offline.get("runtime"),
            "offline_mip_gap": offline.get("mip_gap"),
            "offline_items_in_fallback": offline.get("items_in_fallback"),
            "online_status": online.get("status"),
            "online_total_cost": online.get("total_cost"),
            "online_runtime": online.get("runtime"),
            "online_fallback_items": online.get("fallback_items"),
            "online_evicted_offline": online.get("evicted_offline"),
            "total_objective": online.get("total_objective"),
            "final_items_in_fallback": payload.get("final_items_in_fallback"),
            "N": problem.get("N"),
            "M_off": problem.get("M_off"),
            "M_on": problem.get("M_on"),
        }
        if include_metadata:
            row.update({
                "problem_meta": problem,
                "offline_meta": offline,
                "online_meta": online,
            })
        records.append(row)

    if not records:
        return pd.DataFrame(
            columns=[
                "family",
                "variant",
                "pipeline",
                "seed",
                "offline_status",
                "online_status",
                "total_objective",
            ]
        )

    df = pd.DataFrame(records)
    df["family"] = df["family"].astype("string")
    df["variant"] = df["variant"].astype("string")
    df["pipeline"] = df["pipeline"].astype("string")
    df["seed"] = df["seed"].astype("int64", errors="ignore")
    return df
