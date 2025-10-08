from pathlib import Path
import tempfile
from core.config import load_config
from data.generators import generate_offline_instance
from data.io import save_instance, load_instance

def test_instance_io_roundtrip():
    cfg = load_config("configs/default.yaml")
    inst = generate_offline_instance(cfg, seed=7)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "inst.json"
        save_instance(inst, p)
        inst2 = load_instance(p)

    # basic equivalence checks
    assert len(inst2.bins) == len(inst.bins)
    assert len(inst2.offline_items) == len(inst.offline_items)
    assert inst2.costs.assign.shape == inst.costs.assign.shape
    assert (inst2.feasible.feasible == inst.feasible.feasible).all()
    assert inst2.fallback_bin_index == inst.fallback_bin_index
