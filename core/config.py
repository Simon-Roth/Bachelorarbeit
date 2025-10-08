# binpacking/core/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import yaml
from pathlib import Path

# ---- Problem & generation knobs ----

@dataclass
class ProblemConfig:
    """
    Core structural parameters for an instance.
    - N: number of regular bins
    - M_off: number of offline items
    - capacities: list of bin capacities (len == N)
    - fallback_is_enabled: always True for OFFLINE items; ONLINE must NOT use fallback
    """
    N: int
    M_off: int
    capacities: List[float]
    fallback_is_enabled: bool = True

@dataclass
class VolumeGenerationConfig:
    """
    Volume distributions for offline and online items.
    - offline_uniform: (low, high) for Uniform(low, high)
    - online_beta: (alpha, beta); scaled by 'online_scale_to_capacity' * min(capacities)
      so volumes are reasonable fractions of bin capacity.
    """
    offline_uniform: Tuple[float, float] = (0.05, 0.30)
    online_beta: Tuple[float, float] = (2.0, 5.0)
    online_scale_to_capacity: float = 0.9  # scale by 0.9 * min(C_i) after Beta draw

@dataclass
class GraphGenerationConfig:
    """
    Feasibility graph parameters:
    - p_off: edge prob for G_off (item j can be assigned to bin i)
    - p_onl: edge prob for G_onl^(k) per arrival
    """
    p_off: float = 0.7
    p_onl: float = 0.5

@dataclass
class CostConfig:
    """
    Cost model for assignments and evictions.
    - base_assign_range: range to sample per-item-bin assignment costs
    - huge_fallback: large fallback cost to ensure feasibility but discourage use
    - reassignment_penalty: base penalty for evicting an OFFLINE item (per default PER-ITEM)
    - penalty_mode: 'per_item' | 'per_volume'  (we default to per_item but can switch later)
    - per_volume_scale: if penalty_mode == 'per_volume', use penalty = per_volume_scale * volume
    """
    base_assign_range: Tuple[float, float] = (1.0, 5.0)
    huge_fallback: float = 1e6
    reassignment_penalty: float = 10.0
    penalty_mode: str = "per_item"     # or "per_volume"
    per_volume_scale: float = 10.0     # used only if penalty_mode == "per_volume"

@dataclass
class StochasticConfig:
    """
    Horizon + stochastic arrivals for online phase (already planned for later).
    - horizon_dist: 'fixed' (use 'horizon') or name of a distribution you might add later
    - horizon: default number of online items to generate
    """
    horizon_dist: str = "fixed"
    horizon: int = 100

@dataclass
class PredictionConfig:
    """
    Placeholder for learning-augmented algorithms.
    - use_predictions: if True, downstream code can consume 'horizon_hat' or 'freq_hat'
    - trust_lambda: how aggressively to trust predictions (algorithm-specific)
    """
    use_predictions: bool = False
    horizon_hat: Optional[int] = None
    freq_hat: Optional[Dict[str, float]] = None
    trust_lambda: float = 1.0

@dataclass
class SlackConfig:
    """
    Slack control (even if default is 'no slack') so we can switch later without refactors.
    - enforce_slack: if True, enforce a global fraction of capacity to remain unused
    - fraction: fraction in [0,1); effective capacity is (1 - fraction) * C_i
    """
    enforce_slack: bool = False
    fraction: float = 0.0

@dataclass
class EvalConfig:
    """
    Reproducibility and evaluation bookkeeping.
    - seeds: list of RNG seeds for repeated runs
    """
    seeds: Tuple[int, ...] = (1, 2, 3)

@dataclass
class Config:
    problem: ProblemConfig
    volumes: VolumeGenerationConfig
    graphs: GraphGenerationConfig
    costs: CostConfig
    stoch: StochasticConfig
    pred: PredictionConfig
    slack: SlackConfig
    eval: EvalConfig

def load_config(path: str | Path) -> Config:
    """
    Load YAML into strongly-typed dataclasses. Fails early if keys are missing.
    """
    data = yaml.safe_load(Path(path).read_text())
    return Config(
        problem=ProblemConfig(**data["problem"]),
        volumes=VolumeGenerationConfig(**data["volumes"]),
        graphs=GraphGenerationConfig(**data["graphs"]),
        costs=CostConfig(**data["costs"]),
        stoch=StochasticConfig(**data["stoch"]),
        pred=PredictionConfig(**data["pred"]),
        slack=SlackConfig(**data["slack"]),
        eval=EvalConfig(tuple(data["eval"]["seeds"])),
    )
