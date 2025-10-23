import numpy as np

def check_unique_assignment(x: np.ndarray):
    """Ensure each item is assigned exactly once."""
    row_sums = x.sum(axis=1)
    if not np.all(row_sums == 1):
        bad_rows = np.where(row_sums != 1)[0].tolist()
        raise AssertionError(f"Each offline item must be assigned exactly once. Violations at rows: {bad_rows[:10]}")

def check_capacity_respected(x, vols, eff_caps):
    """Ensure all regular bins respect capacity constraints."""
    N = eff_caps.shape[0]
    loads = np.zeros(N, dtype=float)
    for i in range(N):
        mask = x[:, i] == 1
        if np.any(mask):
            loads[i] = vols[mask].sum()
    if not np.all(loads <= eff_caps + 1e-9):
        viol = np.where(loads > eff_caps + 1e-9)[0].tolist()
        raise AssertionError(f"Capacity violation at bins (0-based): {viol}")
    return loads

def print_offline_summary(info, state, loads, eff_caps, fallback_count):
    """Standardized result print for offline MILP runs."""
    print("\n=== OFFLINE MILP RESULT ===")
    print(f"Status:           {info.status}")
    print(f"Objective value:  {None if not np.isfinite(info.obj_value) else round(info.obj_value, 6)}")
    print(f"MIP gap:          {None if not np.isfinite(info.mip_gap) else round(info.mip_gap, 6)}")
    print(f"Runtime (s):      {round(info.runtime, 3)}")

    print("Loads per bin (regular):", np.round(loads, 4))
    print("Effective capacities:    ", np.round(eff_caps, 4))
    print(f"Fallback load (state):   {round(state.load[-1], 4)}")
    print(f"Fallback items (count):  {fallback_count}")
    print("All checks passed ✅")
