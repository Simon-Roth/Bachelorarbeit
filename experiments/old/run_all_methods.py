from __future__ import annotations
from pathlib import Path

import subprocess

def main():
    methods = [
        "run_ffd.py",
        "run_bfd.py", 
        "run_milp.py",
        "run_milp_warmstart_ffd.py",
        "run_milp_warmstart_bfd.py",
        "run_costbfd.py",
    ]
    
    print("Running all methods...")
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method}")
        print('='*60)
        # subprocess for cleaner execution -> enforces that everything goes thorugh the projects virtualenv interpreter
        subprocess.run([".BAvenv/bin/python", f"experiments/{method}"],check=True)
    
    print("\nAll methods completed!")
    print("Results saved to results/ directory")

if __name__ == "__main__":
    main()
