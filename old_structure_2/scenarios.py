
from __future__ import annotations
from typing import Dict, Any
from config import OfflineConfig
from generators.graph import full_graph, bernoulli_graph
from generators.volume import uniform_volumes, lognormal_volumes
from generators.cost import cost_utilization
from generators.fallback import constant_fallback
from instance_builder import build_instance

def scenario_uniform_full(cfg: OfflineConfig):
    return build_instance(cfg, full_graph, uniform_volumes, cost_utilization, constant_fallback)

def scenario_lognormal_sparse_p(cfg: OfflineConfig):
    p = cfg.params.get('p', 0.4)
    def graph(seed, M, N): return bernoulli_graph(seed, M, N, p)
    return build_instance(cfg, graph, lognormal_volumes, cost_utilization, constant_fallback)

SCENARIO_REGISTRY = {
    'uniform_full': scenario_uniform_full,
    'lognormal_sparse_p': scenario_lognormal_sparse_p,
}
