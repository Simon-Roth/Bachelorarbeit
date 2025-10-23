from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import List, Dict, Any
import seaborn as sns

from repo_bachelorarbeit.core.config import load_config
from repo_bachelorarbeit.core.general_utils import set_global_seed
from repo_bachelorarbeit.data.generators import generate_offline_instance
from repo_bachelorarbeit.offline.offline_solver import OfflineMILPSolver
from repo_bachelorarbeit.offline.offline_heuristics.first_fit_decreasing import FirstFitDecreasing
from repo_bachelorarbeit.offline.offline_heuristics.best_fit_decreasing import BestFitDecreasing


def create_problem_sizes() -> List[Dict[str, int]]:
    """Define problem sizes for benchmarking"""
    return [
        # Small problems
        {"N": 3, "M_off": 10},
        {"N": 5, "M_off": 15},
        {"N": 5, "M_off": 25},
        
        # Medium problems
        {"N": 8, "M_off": 40},
        {"N": 10, "M_off": 50},
        {"N": 12, "M_off": 60},
        
        # Large problems
        {"N": 15, "M_off": 75},
        # {"N": 20, "M_off": 100},
        # {"N": 25, "M_off": 125},
        
        # # Very large problems (nur für Heuristiken)
        # {"N": 30, "M_off": 200},
        # {"N": 40, "M_off": 300},
    ]

def benchmark_single_method(method_name: str, cfg, inst, threads: int = 1) -> Dict[str, Any]:
    """Benchmark a single method"""
    
    if method_name == "MILP":
        solver = OfflineMILPSolver(
            cfg, 
            time_limit=300,  # 5 minutes max
            mip_gap=0.00,
            threads=threads,
            log_to_console=False
        )
        
        start_time = time.perf_counter()
        state, info = solver.solve(inst)
        wall_time = time.perf_counter() - start_time
        
        return {
            'method': 'MILP',
            'runtime': info.runtime,
            'wall_time': wall_time,
            'obj_value': info.obj_value,
            'status': info.status,
            'feasible': info.status in ["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"],
            'items_in_fallback': sum(1 for bin_id in state.assigned_bin.values() 
                                   if bin_id >= cfg.problem.N)
        }
        
    elif method_name == "FFD":
        heuristic = FirstFitDecreasing(cfg)
        
        start_time = time.perf_counter()
        state, info = heuristic.solve(inst)
        wall_time = time.perf_counter() - start_time
        
        return {
            'method': 'FFD',
            'runtime': info.runtime,
            'wall_time': wall_time,
            'obj_value': info.obj_value,
            'status': 'HEURISTIC',
            'feasible': info.feasible,
            'items_in_fallback': info.items_in_fallback
        }
        
    elif method_name == "BFD":
        heuristic = BestFitDecreasing(cfg)
        
        start_time = time.perf_counter()
        state, info = heuristic.solve(inst)
        wall_time = time.perf_counter() - start_time
        
        return {
            'method': 'BFD',
            'runtime': info.runtime,
            'wall_time': wall_time,
            'obj_value': info.obj_value,
            'status': 'HEURISTIC',
            'feasible': info.feasible,
            'items_in_fallback': info.items_in_fallback
        }

def benchmark_problem_size(cfg, N: int, M_off: int, num_runs: int = 3, 
                          milp_threads: int = 1) -> Dict[str, Any]:
    """Benchmark all methods for a single problem configuration"""
    
    # Modify config
    cfg.problem.N = N
    cfg.problem.M_off = M_off
    cfg.problem.capacities = [1.0] * N
    
    methods_results = {}
    
    # Determine which methods to run 
    methods_to_run = ["FFD", "BFD","MILP"]  
    
    print(f"  Running methods: {methods_to_run}")
    
    for method in methods_to_run:
        method_runs = []
        
        for run in range(num_runs):
            seed = 42 + run
            set_global_seed(seed)
            
            # Generate instance
            inst = generate_offline_instance(cfg, seed=seed)
            
            # Benchmark method
            try:
                result = benchmark_single_method(method, cfg, inst, milp_threads)
                method_runs.append(result)
                
                print(f"    {method} Run {run+1}: {result['status']:12s} in {result['runtime']:6.3f}s")
                
            except Exception as e:
                print(f"    {method} Run {run+1}: ERROR - {e}")
                continue
        
        if method_runs:
            # Aggregate results
            runtimes = [r['runtime'] for r in method_runs]
            obj_values = [r['obj_value'] for r in method_runs if r['obj_value'] != float('inf')]
            
            methods_results[method] = {
                'mean_runtime': np.mean(runtimes),
                'std_runtime': np.std(runtimes),
                'min_runtime': np.min(runtimes),
                'max_runtime': np.max(runtimes),
                'mean_obj_value': np.mean(obj_values) if obj_values else float('inf'),
                'std_obj_value': np.std(obj_values) if len(obj_values) > 1 else 0.0,
                'success_rate': sum(1 for r in method_runs if r['feasible']) / len(method_runs),
                'all_runs': method_runs
            }
    
    # Calculate optimality gaps (if MILP was run and found optimal solution)
    if 'MILP' in methods_results and methods_results['MILP']['success_rate'] > 0:
        optimal_value = methods_results['MILP']['mean_obj_value']
        
        for method in ['FFD', 'BFD']:
            if method in methods_results:
                heuristic_value = methods_results[method]['mean_obj_value']
                gap = (heuristic_value - optimal_value) / optimal_value if optimal_value > 0 else 0
                methods_results[method]['optimality_gap'] = gap
    
    return {
        'N': N,
        'M_off': M_off,
        'problem_size': N * M_off,
        'methods': methods_results
    }

def run_benchmark(milp_threads: int = 4) -> List[Dict[str, Any]]:
    """Run complete benchmark suite"""
    
    cfg = load_config("repo_bachelorarbeit/configs/default.yaml")
    problem_sizes = create_problem_sizes()
    
    print(f"🚀 Starting method comparison benchmark with {len(problem_sizes)} configurations...")
    print(f"MILP Threads: {milp_threads}")
    print("=" * 70)
    
    results = []
    for i, size_config in enumerate(problem_sizes):
        print(f"📊 Configuration {i+1:2d}/{len(problem_sizes):2d}: "
              f"N={size_config['N']:2d}, M_off={size_config['M_off']:3d}")
        
        result = benchmark_problem_size(cfg, **size_config, milp_threads=milp_threads)
        results.append(result)
        
        # Print summary for this configuration
        print(f"    ➤ Results:")
        for method, data in result['methods'].items():
            print(f"      {method:4s}: {data['mean_runtime']:6.3f}±{data['std_runtime']:5.3f}s, "
                  f"Obj: {data['mean_obj_value']:8.2f}")
        print()
    
    return results

def create_comparison_plots(results: List[Dict], save_dir: str = "repo_bachelorarbeit/plots/offline/results"):
    """Create the four requested comparison plots"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    plot_data = []
    for result in results:
        for method, data in result['methods'].items():
            plot_data.append({
                'N': result['N'],
                'M_off': result['M_off'],
                'problem_size': result['problem_size'],
                'method': method,
                'mean_runtime': data['mean_runtime'],
                'std_runtime': data['std_runtime'],
                'mean_obj_value': data['mean_obj_value'],
                'optimality_gap': data.get('optimality_gap', 0) * 100  # Convert to percentage
            })
    
    df = pd.DataFrame(plot_data)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = {'FFD': '#1f77b4', 'BFD': '#ff7f0e', 'MILP': '#2ca02c'}
    markers = {'FFD': 'o', 'BFD': 's', 'MILP': '^'}
    
    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Bins vs Runtime
    ax1.set_title('Runtime vs Number of Bins', fontsize=14, fontweight='bold')
    for method in ['FFD', 'BFD', 'MILP']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            ax1.errorbar(method_data['N'], method_data['mean_runtime'], 
                        yerr=method_data['std_runtime'],
                        marker=markers[method], label=method, color=colors[method],
                        capsize=5, capthick=2, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Bins (N)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Items vs Runtime
    ax2.set_title('Runtime vs Number of Items', fontsize=14, fontweight='bold')
    for method in ['FFD', 'BFD', 'MILP']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            ax2.errorbar(method_data['M_off'], method_data['mean_runtime'], 
                        yerr=method_data['std_runtime'],
                        marker=markers[method], label=method, color=colors[method],
                        capsize=5, capthick=2, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Items (M_off)', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Bins vs Optimality Gap (only for heuristics)
    ax3.set_title('Optimality Gap vs Number of Bins', fontsize=14, fontweight='bold')
    for method in ['FFD', 'BFD']:
        method_data = df[(df['method'] == method) & (df['optimality_gap'] > 0)]
        if not method_data.empty:
            ax3.plot(method_data['N'], method_data['optimality_gap'],
                    marker=markers[method], label=method, color=colors[method],
                    linewidth=2, markersize=8)
    
    ax3.set_xlabel('Number of Bins (N)', fontsize=12)
    ax3.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Items vs Optimality Gap (only for heuristics)
    ax4.set_title('Optimality Gap vs Number of Items', fontsize=14, fontweight='bold')
    for method in ['FFD', 'BFD']:
        method_data = df[(df['method'] == method) & (df['optimality_gap'] > 0)]
        if not method_data.empty:
            ax4.plot(method_data['M_off'], method_data['optimality_gap'],
                    marker=markers[method], label=method, color=colors[method],
                    linewidth=2, markersize=8)
    
    ax4.set_xlabel('Number of Items (M_off)', fontsize=12)
    ax4.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Method Comparison: MILP vs Heuristics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_dir}/method_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/method_comparison.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"📈 Comparison plots saved to {save_dir}")

def save_comparison_data(results: List[Dict], filename: str = "method_comparison_results.csv"):
    """Save comparison results to CSV"""
    
    # Flatten data for CSV
    rows = []
    for result in results:
        for method, method_data in result['methods'].items():
            for i, run_data in enumerate(method_data['all_runs']):
                rows.append({
                    'N': result['N'],
                    'M_off': result['M_off'],
                    'problem_size': result['problem_size'],
                    'method': method,
                    'run': i,
                    'runtime': run_data['runtime'],
                    'wall_time': run_data['wall_time'],
                    'obj_value': run_data['obj_value'],
                    'status': run_data['status'],
                    'feasible': run_data['feasible'],
                    'items_in_fallback': run_data['items_in_fallback'],
                    'mean_runtime': method_data['mean_runtime'],
                    'optimality_gap': method_data.get('optimality_gap', 0)
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Detailed comparison results saved to {filename}")

def print_summary_table(results: List[Dict]):
    """Print a nice summary table"""
    
    print("\nMETHOD COMPARISON SUMMARY:")
    print("=" * 90)
    print(f"{'Problem':>12} {'Method':>6} {'Runtime (s)':>12} {'Obj Value':>12} {'Gap (%)':>10}")
    print("-" * 90)
    
    for result in results:
        problem_str = f"N={result['N']}, M={result['M_off']}"
        
        for method, data in result['methods'].items():
            runtime_str = f"{data['mean_runtime']:.3f}±{data['std_runtime']:.3f}"
            obj_str = f"{data['mean_obj_value']:.2f}"
            gap_str = f"{data.get('optimality_gap', 0)*100:.1f}" if 'optimality_gap' in data else "-"
            
            print(f"{problem_str:>12} {method:>6} {runtime_str:>12} {obj_str:>12} {gap_str:>10}")

def main():
    """Main benchmark execution"""
    print("Method Comparison Benchmark: MILP vs Heuristics")
    print("=" * 60)
    
    # Ask for thread count
    milp_threads = 1  # You can adjust this
    print(f"Using {milp_threads} threads for MILP solver")
    
    # Run benchmark
    results = run_benchmark(milp_threads=milp_threads)
    
    # Create plots
    create_comparison_plots(results)
    
    # Save data
    save_comparison_data(results)
    
    # Print summary
    print_summary_table(results)

if __name__ == "__main__":
    main()