from __future__ import annotations

"""
Aggregate config sweep results and visualize runtime/objective/fallback per pipeline.

Run after executing `experiments/run_pipeline_config_sweep.py`.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_ROOT = Path("results/config_sweep")
OUTPUT_DIR = Path("plots/online/config_sweep")


def load_records() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    if not RESULTS_ROOT.exists():
        return pd.DataFrame()

    for scenario_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        for result_file in scenario_dir.glob("pipeline_*.json"):
            data = json.loads(result_file.read_text())
            records.append(
                {
                    "scenario": scenario_dir.name,
                    "pipeline": data["pipeline"],
                    "runtime_offline": float(data["offline"]["runtime"]),
                    "runtime_online": float(data["online"]["runtime"]),
                    "runtime_total": float(data["offline"]["runtime"] + data["online"]["runtime"]),
                    "total_objective": float(data["online"]["total_objective"]),
                    "fallback_final": int(data["final_items_in_fallback"]),
                    "fallback_offline": int(data["offline"]["items_in_fallback"]),
                    "M_off": int(data["problem"]["M_off"]),
                    "M_on": int(data["problem"]["M_on"]),
                    "seed": int(data["seed"]),
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["scenario", "pipeline"])
        .agg(
            runtime_offline=("runtime_offline", "mean"),
            runtime_online=("runtime_online", "mean"),
            runtime_total=("runtime_total", "mean"),
            total_objective=("total_objective", "mean"),
            fallback_final=("fallback_final", "mean"),
            fallback_offline=("fallback_offline", "mean"),
            runs=("seed", "nunique"),
            avg_M_off=("M_off", "mean"),
            avg_M_on=("M_on", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "pipeline"])
    )
    return grouped


def plot_metric(agg: pd.DataFrame, metric: str, ylabel: str, filename: str, title: str) -> None:
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=agg,
        x="pipeline",
        y=metric,
        hue="scenario",
    )
    ax.set_xlabel("Pipeline")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300)
    plt.savefig(OUTPUT_DIR / f"{filename}.pdf", dpi=300)
    plt.close()


def main() -> None:
    df = load_records()
    if df.empty:
        print(f"No results found below {RESULTS_ROOT}. Run experiments/run_pipeline_config_sweep.py first.")
        return

    agg = aggregate(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    print(f"Loaded {len(df)} runs across {len(agg)} scenario/pipeline pairs.")

    plot_metric(
        agg,
        metric="runtime_total",
        ylabel="Runtime (s)",
        filename="config_sweep_runtime",
        title="Average Offline+Online Runtime per Pipeline",
    )
    plot_metric(
        agg,
        metric="total_objective",
        ylabel="Total Objective",
        filename="config_sweep_objective",
        title="Total Objective by Pipeline and Scenario",
    )
    plot_metric(
        agg,
        metric="fallback_final",
        ylabel="Fallback Items After Online Phase",
        filename="config_sweep_fallback",
        title="Fallback Usage by Pipeline and Scenario",
    )
    print(f"Config sweep plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
