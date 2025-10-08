# binpacking/data/io.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import Any, Dict
from core.models import Instance, BinSpec, ItemSpec, Costs, FeasibleGraph

def save_instance(inst: Instance, path: Path) -> None:
    """
    Serialize an Instance as JSON (NumPy arrays converted to lists).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "bins": [{"id": b.id, "capacity": b.capacity} for b in inst.bins],
        "offline_items": [{"id": it.id, "volume": it.volume} for it in inst.offline_items],
        "costs": {
            "assign": inst.costs.assign.tolist(),
            "reassignment_penalty": inst.costs.reassignment_penalty,
            "penalty_mode": inst.costs.penalty_mode,
            "per_volume_scale": inst.costs.per_volume_scale,
            "huge_fallback": inst.costs.huge_fallback,
        },
        "feasible": inst.feasible.feasible.tolist(),
        "fallback_bin_index": inst.fallback_bin_index,
    }
    path.write_text(json.dumps(data))

def load_instance(path: Path) -> Instance:
    """
    Load an Instance from JSON back into strong types.
    """
    data = json.loads(path.read_text())
    bins = [BinSpec(**b) for b in data["bins"]]
    offline = [ItemSpec(**it) for it in data["offline_items"]]
    costs = Costs(
        assign=np.array(data["costs"]["assign"], dtype=float),
        reassignment_penalty=float(data["costs"]["reassignment_penalty"]),
        penalty_mode=str(data["costs"]["penalty_mode"]),
        per_volume_scale=float(data["costs"]["per_volume_scale"]),
        huge_fallback=float(data["costs"]["huge_fallback"]),
    )
    feasible = FeasibleGraph(feasible=np.array(data["feasible"], dtype=int))
    return Instance(
        bins=bins,
        offline_items=offline,
        costs=costs,
        feasible=feasible,
        fallback_bin_index=int(data["fallback_bin_index"]),
    )

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
