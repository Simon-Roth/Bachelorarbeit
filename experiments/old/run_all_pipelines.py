from __future__ import annotations
from pathlib import Path
import subprocess


def main() -> None:
    pipelines = [
        "run_milp_best_fit.py",
        "run_milp_next_fit.py",
        "run_ffd_best_fit.py",
        "run_ffd_next_fit.py",
        "run_bfd_best_fit.py",
        "run_bfd_next_fit.py",
        "run_costbfd_costbestfit.py",
        'run_milp_costbestfit.py'
    ]

    print("Running offline+online pipelines...")
    for script in pipelines:
        print(f"\n{'=' * 60}")
        print(f"Running {script}")
        print("=" * 60)
        subprocess.run([".BAvenv/bin/python", f"experiments/old/{script}"], check=True)

    print("\nAll pipelines completed!")
    print("Results saved to results/ directory")


if __name__ == "__main__":
    main()
