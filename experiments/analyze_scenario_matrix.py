from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from experiments.scenario_analysis_utils import load_results_dataframe

OFFLINE_FAILURE_STATUSES = {
    "INFEASIBLE",
    "INF_OR_UNBD",
    "UNBOUNDED",
}


def _summary_stats(series: pd.Series) -> Tuple[float, float, float, float]:
    values = series.dropna().astype(float)
    count = len(values)
    if count == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    sem = std / math.sqrt(count) if count > 0 else float("nan")
    ci95 = 1.96 * sem if count > 0 else float("nan")
    return mean, std, sem, ci95


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["family", "variant", "pipeline"]
    rows = []

    for keys, group in df.groupby(group_cols, dropna=False):
        family, variant, pipeline = keys
        count = len(group)

        offline_fail_rate = float(
            group["offline_status"].isin(OFFLINE_FAILURE_STATUSES).mean()
        )
        online_fail_rate = float((group["online_status"] != "COMPLETED").mean())

        total_mean, total_std, total_sem, total_ci95 = _summary_stats(group["total_objective"])
        offline_mean, offline_std, _, offline_ci95 = _summary_stats(group["offline_obj"])
        online_cost_mean, online_cost_std, _, online_cost_ci95 = _summary_stats(
            group["online_total_cost"]
        )
        offline_rt_mean, offline_rt_std, _, _ = _summary_stats(group["offline_runtime"])
        online_rt_mean, online_rt_std, _, _ = _summary_stats(group["online_runtime"])
        fallback_off_mean, _, _, fallback_off_ci95 = _summary_stats(
            group["offline_items_in_fallback"]
        )
        fallback_on_mean, _, _, fallback_on_ci95 = _summary_stats(
            group["online_fallback_items"]
        )
        final_fallback_mean, _, _, final_fallback_ci95 = _summary_stats(
            group["final_items_in_fallback"]
        )

        row = {
            "family": family,
            "variant": variant,
            "pipeline": pipeline,
            "runs": count,
            "offline_failure_rate": offline_fail_rate,
            "online_failure_rate": online_fail_rate,
            "total_objective_mean": total_mean,
            "total_objective_std": total_std,
            "total_objective_sem": total_sem,
            "total_objective_ci95": total_ci95,
            "offline_obj_mean": offline_mean,
            "offline_obj_std": offline_std,
            "offline_obj_ci95": offline_ci95,
            "online_cost_mean": online_cost_mean,
            "online_cost_std": online_cost_std,
            "online_cost_ci95": online_cost_ci95,
            "offline_runtime_mean": offline_rt_mean,
            "offline_runtime_std": offline_rt_std,
            "online_runtime_mean": online_rt_mean,
            "online_runtime_std": online_rt_std,
            "offline_fallback_mean": fallback_off_mean,
            "offline_fallback_ci95": fallback_off_ci95,
            "online_fallback_mean": fallback_on_mean,
            "online_fallback_ci95": fallback_on_ci95,
            "final_fallback_mean": final_fallback_mean,
            "final_fallback_ci95": final_fallback_ci95,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def analyze(
    results_root: Path,
    output_dir: Path,
) -> None:
    df = load_results_dataframe(results_root, include_metadata=False)
    if df.empty:
        print(f"No result files found under {results_root}.")
        return

    summary = _build_summary(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "scenario_matrix_summary.csv"
    summary.to_csv(summary_csv, index=False)
    _export_plot_ready_tables(summary, output_dir)

    with pd.option_context("display.max_columns", None):
        print("Summary preview:")
        print(summary.head())

    latex_path = output_dir / "scenario_matrix_summary.tex"
    try:
        summary.to_latex(latex_path, index=False, float_format="%.3f")
    except Exception as exc:
        print(f"Failed to export LaTeX summary: {exc}")


def _export_plot_ready_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plot_data"
    plot_dir.mkdir(parents=True, exist_ok=True)

    objective_cols = [
        "family",
        "variant",
        "pipeline",
        "total_objective_mean",
        "total_objective_ci95",
    ]
    summary[objective_cols].to_csv(
        plot_dir / "total_objective.csv",
        index=False,
    )

    failure_cols = [
        "family",
        "variant",
        "pipeline",
        "offline_failure_rate",
        "online_failure_rate",
    ]
    summary[failure_cols].to_csv(
        plot_dir / "failure_rates.csv",
        index=False,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate scenario matrix results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/scenario_grid"),
        help="Directory containing scenario grid results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aggregates/scenario_matrix"),
        help="Directory for aggregated outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    analyze(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
