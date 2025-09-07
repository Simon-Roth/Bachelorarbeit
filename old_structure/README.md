
# Offline MILP (Bin Packing with Costs) — Starter Kit (Python + Gurobi)

This is a clean, modular scaffold to build and *validate* the **offline** part of your integrated offline/online bin packing with costs project. 
It's designed to be easily extended later for robust planning and online re-optimization.

## Files

- `instance_generator.py` — deterministic, seedable generators for bins, items, and assignment-cost graphs. Includes hospital‑style distributions.
- `offline_milp.py` — Gurobi model builder for the offline assignment with an optional capacity reserve (for later robustness).
- `demo_offline.py` — Example script that generates an instance, builds & solves the MILP, and validates the solution. Also prints key diagnostics.

## Quick Start

1. Install requirements (you need a working Gurobi + license):
   ```bash
   pip install gurobipy numpy pandas
   ```

2. Run the demo with a small instance:
   ```bash
   python demo_offline.py --seed 42 --bins 8 --items 40 --scenario hospital --reserve 0.0
   ```

3. Try different scenarios / distributions:
   ```bash
   # uniform volumes & costs
   python demo_offline.py --scenario uniform
   # lognormal volumes, sparse eligibility graph
   python demo_offline.py --scenario lognormal_sparse
   # hospital-like wards (heterogeneous capacities, mismatch penalties)
   python demo_offline.py --scenario hospital
   ```

## Model (Offline)

Binary variables `x[j,i]` indicate assignment of item *j* to bin *i*. Every item must go to exactly one bin (including a fallback bin `N+1` with very large cost). 
Regular bins have capacity constraints. The fallback bin has no capacity limit (or a very large one). Objective is the sum of `c[j,i] * x[j,i]`.

> Later, to anticipate online arrivals, you can:
> - Leave a *reserve* (e.g., use `reserve=0.1` to enforce `sum v_j x[j,i] <= (1-0.1)*C_i`).
> - Add chance constraints or sample average approximation (SAA) terms.
> - Penalize utilization near 100% (convex piecewise-linear penalties).

## Data Design

We provide multiple **scenarios** with realistic distributions and graph sparsity patterns:
- `uniform`: uniform capacities & volumes; fully connected graph with mild random costs.
- `lognormal_sparse`: heavier-tailed volumes (lognormal), sparse eligibility graph.
- `hospital`: heterogeneous wards (ICU, normal, day-care) with different capacities and mismatch costs.

All generators are deterministic given `--seed` to ensure reproducibility.

## Validation

The demo verifies:
- Each item assigned exactly once.
- Capacity respected for all regular bins.
- Reports objective, fill ratios, and number of items sent to fallback.

---
