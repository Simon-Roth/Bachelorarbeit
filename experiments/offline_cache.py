from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import hashlib
import numpy as np

from core.models import AssignmentState
from offline.models import OfflineSolutionInfo
from offline.offline_heuristics.core import HeuristicSolutionInfo


CACHE_DIR = Path("results/offline_cache")


def compute_config_signature(config_path: Path) -> str:
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _cache_path(config_sig: str, seed: int, offline_label: str) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in offline_label)
    return CACHE_DIR / f"{config_sig}_seed{seed}_{safe_label}.npz"


InfoType = Union[OfflineSolutionInfo, HeuristicSolutionInfo]


def load_cached_offline_solution(
    config_sig: str,
    seed: int,
    offline_label: str,
) -> Optional[Tuple[AssignmentState, InfoType]]:
    path = _cache_path(config_sig, seed, offline_label)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        load = data["load"].astype(float)
        assigned_keys = data["assigned_keys"].astype(int)
        assigned_vals = data["assigned_vals"].astype(int)
        offline_evicted = set(int(x) for x in data["offline_evicted"])
        assigned_bin = {int(k): int(v) for k, v in zip(assigned_keys, assigned_vals)}
        offline_state = AssignmentState(
            load=load,
            assigned_bin=assigned_bin,
            offline_evicted=offline_evicted,
        )
        info_type = str(data["info_type"].item())
        if info_type == "offline":
            info: InfoType = OfflineSolutionInfo(
                status=str(data["info_status"].item()),
                obj_value=float(data["info_obj_value"].item()),
                mip_gap=float(data["info_mip_gap"].item()),
                runtime=float(data["info_runtime"].item()),
                assignments=data["info_assignments"].astype(int),
            )
        elif info_type == "heuristic":
            info = HeuristicSolutionInfo(
                algorithm=str(data["info_algorithm"].item()),
                runtime=float(data["info_runtime"].item()),
                obj_value=float(data["info_obj_value"].item()),
                feasible=bool(data["info_feasible"].item()),
                items_in_fallback=int(data["info_items_in_fallback"].item()),
                utilization=float(data["info_utilization"].item()),
            )
        else:
            raise ValueError(f"Unknown info_type in cache: {info_type}")
    return offline_state, info


def save_cached_offline_solution(
    config_sig: str,
    seed: int,
    offline_label: str,
    state: AssignmentState,
    info: InfoType,
) -> None:
    path = _cache_path(config_sig, seed, offline_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    assigned_keys = np.array(list(state.assigned_bin.keys()), dtype=int)
    assigned_vals = np.array(list(state.assigned_bin.values()), dtype=int)
    offline_evicted = np.array(list(state.offline_evicted), dtype=int)
    common = dict(
        load=state.load.astype(float, copy=True),
        assigned_keys=assigned_keys,
        assigned_vals=assigned_vals,
        offline_evicted=offline_evicted,
        info_obj_value=np.array(info.obj_value),
        info_runtime=np.array(info.runtime),
    )
    if isinstance(info, OfflineSolutionInfo):
        np.savez(
            path,
            info_type=np.array("offline"),
            info_status=np.array(info.status),
            info_mip_gap=np.array(info.mip_gap),
            info_assignments=info.assignments.astype(int, copy=True),
            **common,
        )
    elif isinstance(info, HeuristicSolutionInfo):
        np.savez(
            path,
            info_type=np.array("heuristic"),
            info_algorithm=np.array(info.algorithm),
            info_feasible=np.array(int(info.feasible)),
            info_items_in_fallback=np.array(info.items_in_fallback),
            info_utilization=np.array(info.utilization),
            **common,
        )
    else:
        raise TypeError(f"Unsupported solution info type: {type(info)}")


def load_cached_full_horizon(
    config_sig: str,
    seed: int,
) -> Optional[Tuple[AssignmentState, OfflineSolutionInfo]]:
    path = _full_horizon_path(config_sig, seed)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        load = data["load"].astype(float)
        assigned_keys = data["assigned_keys"].astype(int)
        assigned_vals = data["assigned_vals"].astype(int)
        assigned_bin = {int(k): int(v) for k, v in zip(assigned_keys, assigned_vals)}
        offline_state = AssignmentState(load=load, assigned_bin=assigned_bin, offline_evicted=set())
        info = OfflineSolutionInfo(
            status=str(data["info_status"].item()),
            obj_value=float(data["info_obj_value"].item()),
            mip_gap=float(data["info_mip_gap"].item()),
            runtime=float(data["info_runtime"].item()),
            assignments=data["info_assignments"].astype(int),
        )
        return offline_state, info


def save_cached_full_horizon(
    config_sig: str,
    seed: int,
    state: AssignmentState,
    info: OfflineSolutionInfo,
) -> None:
    path = _full_horizon_path(config_sig, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    assigned_keys = np.array(list(state.assigned_bin.keys()), dtype=int)
    assigned_vals = np.array(list(state.assigned_bin.values()), dtype=int)
    np.savez(
        path,
        load=state.load.astype(float, copy=True),
        assigned_keys=assigned_keys,
        assigned_vals=assigned_vals,
        info_status=np.array(info.status),
        info_obj_value=np.array(info.obj_value),
        info_mip_gap=np.array(info.mip_gap),
        info_runtime=np.array(info.runtime),
        info_assignments=info.assignments.astype(int, copy=True),
    )
def _full_horizon_path(config_sig: str, seed: int) -> Path:
    return CACHE_DIR / f"{config_sig}_seed{seed}_full_horizon.npz"
