
# Modular Offline Bin Packing (MILP, Python+Gurobi)

A **modular**, PDF-aligned offline MILP for bin packing with **optional fallback** and **vector capacities** (multi-dimensional constraints).
All stochastic knobs live in `OfflineConfig.params` (nested dicts). No scenario code—**config is the single control point**.

## Parameters (`cfg.params`)

| Section   | Key           | Default     | Options / Meaning |
|-----------|---------------|-------------|-------------------|
| capacity  | distribution  | `uniform`   | `uniform`, `normal`, `fixed` |
|           | low/high      | 60 / 100    | bounds if `uniform` |
|           | mean/std      | 100 / 10    | params if `normal` |
|           | value         | 100         | value if `fixed` |
| volume    | distribution  | `uniform`   | `uniform`, `lognormal` |
|           | low/high      | 3 / 12      | bounds if `uniform` |
|           | mu/sigma      | 2.0 / 0.5   | params if `lognormal` |
| cost      | model         | `utilization` | `utilization`, `constant`, `random` |
|           | base          | 10.0        | cost scale |
|           | noise         | 0.25        | multiplicative noise |
| fallback  | use           | True        | enable fallback column |
|           | cost          | 1000.0      | constant fallback cost |

### Example
```python
from config import OfflineConfig
cfg = OfflineConfig(
  N=12, M=80, D=2, seed=1,
  params={
    "capacity": {"distribution":"normal","mean":120,"std":15},
    "volume":   {"distribution":"lognormal","mu":2.0,"sigma":0.6},
    "cost":     {"model":"utilization","base":10.0,"noise":0.25},
    "fallback": {"use": True, "cost": 2000.0},
    "graph_p":  0.4,                 # used if you choose graph_type='bernoulli'
    "bigM_forbidden": 1e6            # cost to place on masked-off edges in the DF (defensive)
  }
)
```
Run:
```bash
pip install gurobipy numpy pandas
python demo_offline.py --graph bernoulli --seed 1 --bins 12 --items 80 --dims 2
```

## Math (vector capacity)
For each bin i and dimension d:
\[ \sum_j v_{d,j} x_{i,j} \le (1-\rho_d) C_{d,i} \]
Objective:
\[ \min \sum_{j} \sum_{i} c_{j,i} x_{i,j} \]
We create variables only on allowed edges (mask=True). Fallback is an optional extra column.

