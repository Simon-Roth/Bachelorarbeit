from __future__ import annotations

"""
Generate thesis-ready plots for the scenario grid aggregates produced by
`experiments/analyze_scenario_matrix.py`.

Usage:
    python plots/scenario/create_scenario_plots.py \
        --summary results/aggregates/scenario_matrix/scenario_matrix_summary.csv
"""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


def sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower() or "variant"


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "family",
        "variant",
        "pipeline",
        "total_objective_mean",
        "total_objective_ci95",
        "offline_failure_rate",
        "online_failure_rate",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Summary file {path} missing columns: {sorted(missing)}")
    return df


def plot_total_objective(summary: pd.DataFrame, out_dir: Path) -> None:
    palette = sns.color_palette("viridis", n_colors=8)
    for (family, variant), group in summary.groupby(["family", "variant"], dropna=False):
        ordered = group.sort_values("total_objective_mean", ascending=True)
        height = max(3.5, 0.45 * len(ordered) + 1.5)
        fig, ax = plt.subplots(figsize=(9.5, height))
        bars = ax.barh(
            ordered["pipeline"],
            ordered["total_objective_mean"],
            color=[palette[i % len(palette)] for i in range(len(ordered))],
            alpha=0.9,
        )
        ax.errorbar(
            ordered["total_objective_mean"],
            ordered["pipeline"],
            xerr=ordered["total_objective_ci95"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        ax.set_xlabel("Total objective (mean ± 95% CI)")
        ax.set_ylabel("")
        ax.set_title(f"{family} – {variant}")
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        sns.despine(left=True, bottom=False)

        for idx, (pipeline, fail_flag) in enumerate(
            zip(ordered["pipeline"], ordered["any_failure"])
        ):
            if fail_flag:
                ci = ordered["total_objective_ci95"].iloc[idx]
                base = ordered["total_objective_mean"].iloc[idx]
                offset = max(ci if pd.notna(ci) else 0.0, 0.02 * base)
                ax.text(
                    base + offset,
                    idx,
                    "✖",
                    va="center",
                    ha="left",
                    color="#d62728",
                    fontsize=11,
                )

        fig.tight_layout()
        fname = f"{sanitise(family)}_{sanitise(variant)}_total_objective"
        fig.savefig(out_dir / f"{fname}.png", dpi=300)
        fig.savefig(out_dir / f"{fname}.pdf")
        plt.close(fig)


def plot_failure_heatmaps(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = {
        "offline_failure_rate": "Offline failure rate",
        "online_failure_rate": "Online failure rate",
    }
    for family, group in summary.groupby("family", dropna=False):
        for metric, title in metrics.items():
            pivot = group.pivot(
                index="pipeline",
                columns="variant",
                values=metric,
            ).sort_index()
            pivot = pivot.fillna(0.0)
            fig, ax = plt.subplots(figsize=(1.5 * len(pivot.columns) + 3, 0.6 * len(pivot.index) + 2))
            sns.heatmap(
                pivot,
                ax=ax,
                vmin=0.0,
                vmax=1.0,
                cmap="Reds",
                cbar_kws={"label": "Rate"},
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                linecolor="white",
            )
            ax.set_xlabel("Variant")
            ax.set_ylabel("Pipeline")
            ax.set_title(f"{title} – {family}")
            fig.tight_layout()
            fname = f"{sanitise(family)}_{metric}"
            fig.savefig(out_dir / f"{fname}.png", dpi=300)
            fig.savefig(out_dir / f"{fname}.pdf")
            plt.close(fig)


def plot_family_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_ax = sns.barplot(
        data=summary,
        x="family",
        y="total_objective_mean",
        hue="pipeline",
        ci=None,
        ax=ax,
    )

    for patch, fail in zip(bar_ax.patches, summary["any_failure"]):
        if fail:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height,
                "xxx",
                color="#d62728",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_ylabel("Total objective (mean)")
    ax.set_xlabel("Scenario family")
    ax.set_title("Scenario family comparison")
    ax.legend(title="Pipeline", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    fig.tight_layout()
    fname = "family_overview_total_objective"
    fig.savefig(out_dir / f"{fname}.png", dpi=300)
    fig.savefig(out_dir / f"{fname}.pdf")
    plt.close(fig)


def create_plots(summary_path: Path, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    summary = load_summary(summary_path)
    summary = summary.copy()
    summary["any_failure"] = (
        (summary["offline_failure_rate"] > 0.0)
        | (summary["online_failure_rate"] > 0.0)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_total_objective(summary, output_dir)
    plot_failure_heatmaps(summary, output_dir)
    plot_family_overview(summary, output_dir)
    print(f"Scenario plots saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create thesis-ready plots for scenario grid results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/aggregates/scenario_matrix/scenario_matrix_summary.csv"),
        help="Path to the aggregated summary CSV produced by analyze_scenario_matrix.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/scenario/results"),
        help="Directory where plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_plots(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
