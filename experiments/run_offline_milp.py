# experiments/run_offline_milp.py
from __future__ import annotations
from pathlib import Path
import numpy as np

from core.config import load_config
from core.utils import set_global_seed
from data.generators import generate_offline_instance
from offline.milp_solver import OfflineMILPSolver
from core.logging_setup import setup_logging,CSVWriter,run_stamp

def _effective_capacity(capacity: float, enforce_slack: bool, slack_fraction: float) -> float:
    """Dupliziert die Logik aus dem Solver für eine saubere, externe Prüfung."""
    if enforce_slack:
        assert 0.0 <= slack_fraction < 1.0, "Slack fraction must be in [0,1)."
        return (1.0 - slack_fraction) * capacity
    return capacity


def main():
    logger = setup_logging(Path("logs"), "milp_experiment")
    
    results_file = Path(f"results/milp_results_{run_stamp()}.csv")
    csv_writer = CSVWriter(results_file, [
        "seed", "N", "M_off", "objective", "status", 
        "runtime", "mip_gap", "fallback_items"
    ])
    
    cfg = load_config("configs/default.yaml")
    seed = cfg.eval.seeds[0]
    set_global_seed(seed)
    logger.info(f"Running experiment with seed {seed}")


    # 1) Instanz generieren
    inst = generate_offline_instance(cfg, seed=seed)

    # 2) MILP lösen
    solver = OfflineMILPSolver(cfg, time_limit=60, mip_gap=0.01, threads=0, log_to_console=True)
    state, info = solver.solve(inst)
    
    logger.info(f"Seed {seed}: Status={info.status}, Obj={info.obj_value:.3f}")
    csv_writer.write_row({
                "seed": seed,
                "N": cfg.problem.N,
                "M_off": cfg.problem.M_off,
                "objective": info.obj_value,
                "status": info.status,
                "runtime": info.runtime,
                "mip_gap": info.mip_gap,
                "fallback_items": int((info.assignments[:, cfg.problem.N] == 1).sum())
            })

    # 3) Robust: Status prüfen
    bad_status = {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}
    print("\n=== OFFLINE MILP RESULT ===")
    print("Status:", info.status)
    print("Objective value:", None if not np.isfinite(info.obj_value) else round(info.obj_value, 6))
    print("MIP gap:", None if not np.isfinite(info.mip_gap) else round(info.mip_gap, 6))
    print("Runtime (s):", round(info.runtime, 3))

    if info.status in bad_status:
        raise RuntimeError(f"Model ended with status {info.status}. Check data or constraints.")

    # 4) Qualitäts-Check A: Jede Item-Zeile genau einmal zugewiesen
    x = info.assignments  # shape: (M_off, N+1), 0/1
    row_sums = x.sum(axis=1)
    if not np.all(row_sums == 1):
        bad_rows = np.where(row_sums != 1)[0].tolist()
        raise AssertionError(f"Each offline item must be assigned exactly once. Violations at rows: {bad_rows[:10]}")

    # 5) Qualitäts-Check B: Kapazitäten (inkl. Slack) eingehalten
    N = cfg.problem.N
    M_off = cfg.problem.M_off
    vols = np.array([it.volume for it in inst.offline_items], dtype=float)
    caps = np.array([b.capacity for b in inst.bins], dtype=float)
    eff_caps = np.array(
        [_effective_capacity(c, cfg.slack.enforce_slack, cfg.slack.fraction) for c in caps],
        dtype=float,
    )
    fallback_count = int((x[:, N] == 1).sum())   # Anzahl Items im Fallback

    # Rechne die Last pro regulärem Bin aus der Lösungsmatrix
    loads_from_x = np.zeros(N, dtype=float)
    for i in range(N):  # nur reguläre Bins
        # Summe der Volumina aller Items, die in Spalte i liegen
        mask = x[:, i] == 1
        if np.any(mask):
            loads_from_x[i] = vols[mask].sum()

    # Prüfen
    if not np.all(loads_from_x <= eff_caps + 1e-9):
        viol = np.where(loads_from_x > eff_caps + 1e-9)[0].tolist()
        raise AssertionError(f"Capacity violation at bins (0-based): {viol}")

    # 6) Kurze Übersicht ausgeben
    print("Loads per bin (regular):", np.round(loads_from_x, 4))
    print("Effective capacities:    ", np.round(eff_caps, 4))
    print("Fallback load (state):   ", round(state.load[N], 4))
    print("Fallback items: (count)   ", round(fallback_count, 4))

    print("All checks passed")


if __name__ == "__main__":
    main()
