
# Modular Offline Bin Packing (MILP, Python+Gurobi)

This package implements a **modular**, PDF-aligned offline MILP for bin packing with **optional fallback** and **vector capacities** (multi-dimensional resource constraints). It is designed so you can later add an online phase (recourse, regret) without changing the core data model.

## Highlights
- **x[i,j]** variable convention (bin i, item j), matching math notation.
- **Vector bin packing**: multiple capacity dimensions D (e.g., bed-days, nurse-hours).
- **Optional fallback** bin (on by default, toggle via flag).
- **Pluggable generators** for graph/mask, volumes (can depend on mask), costs, fallback.
- **Scenarios** wired from generators: `uniform_full`, `lognormal_sparse_p`.
- **Strict validation** and schema checks.

## Quick Start
```bash
pip install gurobipy numpy pandas
python demo_offline.py --scenario uniform_full --seed 42 --bins 8 --items 40 --dims 1 --reserve 0.0
python demo_offline.py --scenario lognormal_sparse_p --p 0.4 --seed 1 --bins 12 --items 120 --dims 2
# to disable fallback:
python demo_offline.py --scenario uniform_full --no-fallback
```

## Structure
- `config.py` — data classes for configuration and schema constants.
- `generators/graph.py` — Bernoulli-p graph & full graph.
- `generators/volume.py` — uniform and lognormal volume generators (vector-ready).
- `generators/cost.py` — cost models based on volume/capacity with noise.
- `generators/fallback.py` — fallback cost generators.
- `instance_builder.py` — stitches generators into a consistent Instance (with checks).
- `offline/model.py` — MILP builder with x[i,j], vector capacities, optional fallback.
- `offline/solve.py` — solve + validation + diagnostics.
- `scenarios.py` — convenient presets wiring the above.
- `demo_offline.py` — CLI to run and validate.

## Why “no variable for forbidden edges”?
- We **do not create variables** for edges where the mask is False (cleaner model, smaller MIP).
- We still assert that any cost provided for forbidden edges is “big-M”, to avoid accidental use.

## Vector capacities (D > 1)
For each bin i and dimension d:
[ sum_j v_{d,j} x_{i,j} <= (1 - rho_d) * C_{d,i} ]
where rho_d is an optional reserve per dimension.
