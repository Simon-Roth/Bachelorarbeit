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
        
        #Large problems
        {"N": 15, "M_off": 75},
        # {"N": 20, "M_off": 150},
        # {"N": 25, "M_off": 200},
        
        # # Very large problems
        # {"N": 30, "M_off": 300},
        # {"N": 40, "M_off": 500},
    ]

def benchmark_problem_size(cfg, N: int, M_off: int, num_runs: int = 3) -> Dict[str, Any]:
    """Benchmark a single problem configuration"""
    
    # Modify config
    cfg.problem.N = N
    cfg.problem.M_off = M_off
    cfg.problem.capacities = [1.0] * N
    
    runtimes = []
    statuses = []
    objectives = []
    
    for run in range(num_runs):
        seed = 42 + run
        set_global_seed(seed)
        
        # Generate and solve
        inst = generate_offline_instance(cfg, seed=seed)
        solver = OfflineMILPSolver(
            cfg, 
            time_limit=300,  # 5 minutes max
            mip_gap=0.01,
            threads=1,
            log_to_console=False
        )
        
        start_time = time.perf_counter()
        state, info = solver.solve(inst)
        runtime = time.perf_counter() - start_time
        
        runtimes.append(info.runtime)  # Gurobi internal time
        statuses.append(info.status)
        objectives.append(info.obj_value)
        
        print(f"  N={N:2d}, M={M_off:3d}, Run {run+1}: {info.status:12s} in {info.runtime:6.2f}s")
    
    return {
        'N': N,
        'M_off': M_off,
        'problem_size': N * M_off,
        'density': M_off / N,  # Items per bin
        'mean_runtime': np.mean(runtimes),
        'std_runtime': np.std(runtimes),
        'min_runtime': np.min(runtimes),
        'max_runtime': np.max(runtimes),
        'success_rate': sum(1 for s in statuses if s == "OPTIMAL") / len(statuses),
        'mean_objective': np.mean([obj for obj in objectives if obj != float('inf')]),
        'all_runtimes': runtimes,
        'all_statuses': statuses
    }

def run_benchmark() -> List[Dict[str, Any]]:
    """Run complete benchmark suite"""
    
    cfg = load_config("repo_bachelorarbeit/configs/default.yaml")
    problem_sizes = create_problem_sizes()
    
    print(f"🚀 Starting benchmark with {len(problem_sizes)} problem configurations...")
    print("=" * 70)
    
    results = []
    for i, size_config in enumerate(problem_sizes):
        print(f"📊 Configuration {i+1:2d}/{len(problem_sizes):2d}: "
              f"N={size_config['N']:2d}, M_off={size_config['M_off']:3d}")
        
        result = benchmark_problem_size(cfg, **size_config)
        results.append(result)
        
        print(f"    ➤ Avg: {result['mean_runtime']:6.2f}s, "
              f"Success: {result['success_rate']:5.1%}")
        print()
    
    return results

def create_runtime_plots(results: List[Dict], save_dir: str = "repo_bachelorarbeit/plots/offline"):
    """Create comprehensive runtime analysis plots"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Runtime vs Problem Size (log-log)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(df['problem_size'], df['mean_runtime'], 
                yerr=df['std_runtime'], marker='o', capsize=5, 
                linewidth=2, markersize=8, capthick=2)
    ax1.set_xlabel('Problem Size (N × M_off)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Runtime Scaling (Log-Log)', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    from scipy import stats
    log_x = np.log10(df['problem_size'])
    log_y = np.log10(df['mean_runtime'])
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
    x_trend = np.logspace(np.log10(df['problem_size'].min()), 
                         np.log10(df['problem_size'].max()), 100)
    y_trend = 10**(slope * np.log10(x_trend) + intercept)
    ax1.plot(x_trend, y_trend, '--', alpha=0.8, 
            label=f'Trend: ∝ N^{slope:.2f} (R²={r_value**2:.3f})')
    ax1.legend()
    
    # 2. Runtime vs Number of Items
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(df['M_off'], df['mean_runtime'], 
                yerr=df['std_runtime'], marker='s', capsize=5,
                linewidth=2, markersize=8, capthick=2, color='orange')
    ax2.set_xlabel('Number of Items (M_off)', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Runtime vs Items', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Runtime vs Number of Bins
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(df['N'], df['mean_runtime'], 
                yerr=df['std_runtime'], marker='^', capsize=5,
                linewidth=2, markersize=8, capthick=2, color='green')
    ax3.set_xlabel('Number of Bins (N)', fontsize=12)
    ax3.set_ylabel('Runtime (seconds)', fontsize=12)
    ax3.set_title('Runtime vs Bins', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Success Rate
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ['red' if rate < 1.0 else 'green' for rate in df['success_rate']]
    bars = ax4.bar(range(len(df)), df['success_rate'], 
                   color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Problem Configuration', fontsize=12)
    ax4.set_ylabel('Success Rate (Optimal Found)', fontsize=12)
    ax4.set_title('Solver Success Rate', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add labels
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([f"N={row['N']}\nM={row['M_off']}" 
                        for _, row in df.iterrows()], 
                       rotation=45, ha='right', fontsize=10)
    
    # 5. Heatmap: Runtime by N and M_off
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(values='mean_runtime', index='N', columns='M_off', aggfunc='first')
    
    im = ax5.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest')
    ax5.set_xticks(range(len(pivot_data.columns)))
    ax5.set_yticks(range(len(pivot_data.index)))
    ax5.set_xticklabels(pivot_data.columns)
    ax5.set_yticklabels(pivot_data.index)
    ax5.set_xlabel('Number of Items (M_off)', fontsize=12)
    ax5.set_ylabel('Number of Bins (N)', fontsize=12)
    ax5.set_title('Runtime Heatmap (Seconds)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Runtime (seconds)', fontsize=11)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if not np.isnan(pivot_data.iloc[i, j]):
                ax5.text(j, i, f'{pivot_data.iloc[i, j]:.1f}', 
                        ha='center', va='center', fontsize=9, 
                        color='white' if pivot_data.iloc[i, j] > pivot_data.values.max()/2 else 'black')
    
    plt.suptitle('MILP Solver Runtime Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plt.savefig(f"{save_dir}/runtime_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/runtime_analysis.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"📈 Plots saved to {save_dir}")

def save_benchmark_data(results: List[Dict], filename: str = "benchmark_results.csv"):
    """Save benchmark results to CSV"""
    
    # Flatten data for CSV
    rows = []
    for result in results:
        for i, (runtime, status) in enumerate(zip(result['all_runtimes'], result['all_statuses'])):
            rows.append({
                'N': result['N'],
                'M_off': result['M_off'],
                'problem_size': result['problem_size'],
                'density': result['density'],
                'run': i,
                'runtime': runtime,
                'status': status,
                'mean_runtime': result['mean_runtime'],
                'std_runtime': result['std_runtime'],
                'success_rate': result['success_rate']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"💾 Detailed results saved to {filename}")

def main():
    """Main benchmark execution"""
    print("🔧 MILP Solver Runtime Benchmark")
    print("=" * 50)
    
    # Run benchmark
    results = run_benchmark()
    
    # Create plots
    create_runtime_plots(results)
    
    # Save data
    save_benchmark_data(results)
    
    # Print summary
    df = pd.DataFrame(results)
    print("\n📋 BENCHMARK SUMMARY:")
    print("=" * 50)
    print(f"Total configurations tested: {len(results)}")
    print(f"Problem sizes range: {df['problem_size'].min()} - {df['problem_size'].max()}")
    print(f"Runtime range: {df['mean_runtime'].min():.2f}s - {df['mean_runtime'].max():.2f}s")
    print(f"Average success rate: {df['success_rate'].mean():.1%}")
    print(f"Fastest solve: {df['min_runtime'].min():.3f}s")
    print(f"Slowest solve: {df['max_runtime'].max():.1f}s")

if __name__ == "__main__":
    main()