"""
PINN Validation Script using Implicit Finite Difference as Source of Truth

This script provides comprehensive validation of PINN results against
high-accuracy implicit finite difference solutions.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from implicit_finite_difference import ImplicitFiniteDifference, create_test_initial_condition
from finite_difference_solver import load_solution_arrays


def calculate_comprehensive_errors(pinn_solution, fd_solution):
    """
    Calculate comprehensive error metrics between PINN and FD solutions
    
    Args:
        pinn_solution: PINN solution array
        fd_solution: Finite difference solution array
        
    Returns:
        Dictionary with various error metrics
    """
    if len(pinn_solution) != len(fd_solution):
        # Interpolate FD solution to match PINN grid if needed
        x_fd = np.linspace(0, 2, len(fd_solution))
        x_pinn = np.linspace(0, 2, len(pinn_solution))
        fd_solution = np.interp(x_pinn, x_fd, fd_solution)
    
    diff = pinn_solution - fd_solution
    
    return {
        'mean_absolute_error': np.mean(np.abs(diff)),
        'rmse': np.sqrt(np.mean(diff**2)),
        'max_absolute_error': np.max(np.abs(diff)),
        'mean_squared_error': np.mean(diff**2),
        'relative_l2_error': np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(fd_solution**2)),
        'relative_max_error': np.max(np.abs(diff)) / np.max(np.abs(fd_solution)),
        'l1_norm': np.sum(np.abs(diff)),
        'l2_norm': np.sqrt(np.sum(diff**2)),
        'linf_norm': np.max(np.abs(diff))
    }


def validate_pinn_experiment(experiment_dir, detailed_output=True):
    """
    Validate a PINN experiment against implicit finite difference solution
    
    Args:
        experiment_dir: Path to experiment results directory
        detailed_output: Whether to provide detailed output and plots
        
    Returns:
        Validation results dictionary
    """
    print(f"🔍 Validating PINN experiment: {os.path.basename(experiment_dir)}")
    print("=" * 80)
    
    # Load experiment hyperparameters
    hyperparams_path = os.path.join(experiment_dir, "hyperparameters.json")
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_path}")
    
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
    
    # Extract key parameters
    nu = hyperparams['physics_params']['nu']
    tf = hyperparams['physics_params']['tf']
    nx_pinn = hyperparams['training_config']['nx']
    
    print(f"📋 Experiment Parameters:")
    print(f"   Diffusion coefficient (ν): {nu}")
    print(f"   Final time: {tf}")
    print(f"   PINN spatial points: {nx_pinn}")
    
    # Load PINN solution arrays
    try:
        pinn_data = load_solution_arrays(experiment_dir)
        pinn_solutions = pinn_data['pinn_solutions']
        pinn_x_coords = pinn_data['x_coordinates']
        pinn_times = pinn_data['time_values']
        print(f"✓ PINN solutions loaded: {pinn_solutions.shape} (time × space)")
    except Exception as e:
        raise RuntimeError(f"Failed to load PINN solutions: {e}")
    
    # Create high-resolution implicit FD solver (source of truth)
    print(f"\n🧮 Creating high-resolution implicit FD solver...")
    implicit_solver = ImplicitFiniteDifference(
        nx=2000,              # High resolution for accuracy
        nt_independent=1000,  # Fine time stepping, independent of PINN
        x_range=(0, 2),
        t_range=(0, tf),
        nu=nu
    )
    
    # Set same initial condition as PINN
    implicit_solver.set_initial_condition(create_test_initial_condition)
    
    # Solve implicit system (source of truth)
    print(f"🚀 Solving implicit FD system...")
    implicit_solution = implicit_solver.solve()
    
    # Validate at each PINN time point
    validation_results = {
        'time_points': [],
        'error_metrics': [],
        'summary_stats': {}
    }
    
    print(f"\n📊 Comparing solutions at {len(pinn_times)} time points...")
    
    all_errors = {'mae': [], 'rmse': [], 'max_error': [], 'rel_l2': [], 'rel_max': []}
    
    for i, t_val in enumerate(pinn_times):
        print(f"   Time {i+1}/{len(pinn_times)}: t = {t_val:.4f}")
        
        # Get PINN solution at this time
        pinn_sol_t = pinn_solutions[i, :]
        
        # Get implicit FD solution at this time (interpolated)
        fd_sol_t = implicit_solver.get_solution_at_time(t_val)
        
        # Calculate errors
        errors = calculate_comprehensive_errors(pinn_sol_t, fd_sol_t)
        
        # Store results
        validation_results['time_points'].append(t_val)
        validation_results['error_metrics'].append(errors)
        
        # Collect for summary statistics
        all_errors['mae'].append(errors['mean_absolute_error'])
        all_errors['rmse'].append(errors['rmse'])
        all_errors['max_error'].append(errors['max_absolute_error'])
        all_errors['rel_l2'].append(errors['relative_l2_error'])
        all_errors['rel_max'].append(errors['relative_max_error'])
        
        if detailed_output:
            print(f"      RMSE: {errors['rmse']:.2e}, Max Error: {errors['max_absolute_error']:.2e}")
    
    # Calculate summary statistics
    validation_results['summary_stats'] = {
        'mean_rmse': np.mean(all_errors['rmse']),
        'std_rmse': np.std(all_errors['rmse']),
        'max_rmse': np.max(all_errors['rmse']),
        'min_rmse': np.min(all_errors['rmse']),
        'mean_max_error': np.mean(all_errors['max_error']),
        'worst_max_error': np.max(all_errors['max_error']),
        'mean_relative_l2': np.mean(all_errors['rel_l2']),
        'worst_relative_l2': np.max(all_errors['rel_l2']),
        'mean_relative_max': np.mean(all_errors['rel_max']),
        'worst_relative_max': np.max(all_errors['rel_max'])
    }
    
    # Print summary
    print(f"\n📈 VALIDATION SUMMARY")
    print("-" * 40)
    stats = validation_results['summary_stats']
    print(f"Average RMSE: {stats['mean_rmse']:.2e} ± {stats['std_rmse']:.2e}")
    print(f"Worst RMSE: {stats['max_rmse']:.2e}")
    print(f"Average Max Error: {stats['mean_max_error']:.2e}")
    print(f"Worst Max Error: {stats['worst_max_error']:.2e}")
    print(f"Average Relative L2 Error: {stats['mean_relative_l2']:.2%}")
    print(f"Worst Relative L2 Error: {stats['worst_relative_l2']:.2%}")
    
    if detailed_output:
        # Create comprehensive validation plots
        create_validation_plots(validation_results, pinn_solutions, implicit_solver, 
                               pinn_times, pinn_x_coords, experiment_dir)
        
        # Save validation results
        save_validation_results(validation_results, experiment_dir)
    
    return validation_results


def create_validation_plots(validation_results, pinn_solutions, implicit_solver, 
                           pinn_times, pinn_x_coords, experiment_dir):
    """Create comprehensive validation plots"""
    
    # Extract error metrics for plotting
    rmse_errors = [metrics['rmse'] for metrics in validation_results['error_metrics']]
    max_errors = [metrics['max_absolute_error'] for metrics in validation_results['error_metrics']]
    rel_l2_errors = [metrics['relative_l2_error'] for metrics in validation_results['error_metrics']]
    
    # Create multi-panel validation plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PINN Validation Against Implicit FD (Source of Truth)', fontsize=16)
    
    # Plot 1: Error evolution over time
    axes[0, 0].plot(pinn_times, rmse_errors, 'b-o', label='RMSE', linewidth=2, markersize=4)
    axes[0, 0].plot(pinn_times, max_errors, 'r-s', label='Max Error', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].set_title('Error Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Relative errors
    axes[0, 1].plot(pinn_times, rel_l2_errors, 'g-^', label='Relative L2', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].set_title('Relative L2 Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Error distribution
    axes[0, 2].hist(rmse_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_xlabel('RMSE')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('RMSE Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Solution comparison at final time
    final_pinn = pinn_solutions[-1, :]
    final_implicit = implicit_solver.get_solution_at_time(pinn_times[-1])
    
    axes[1, 0].plot(pinn_x_coords, final_pinn, 'b-', label='PINN', linewidth=2)
    axes[1, 0].plot(pinn_x_coords, final_implicit, 'r--', label='Implicit FD', linewidth=2)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u(x,t)')
    axes[1, 0].set_title(f'Solution Comparison (t={pinn_times[-1]:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Point-wise error at final time
    final_error = np.abs(final_pinn - final_implicit)
    axes[1, 1].plot(pinn_x_coords, final_error, 'k-', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('|Error|')
    axes[1, 1].set_title(f'Point-wise Error (t={pinn_times[-1]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Plot 6: Convergence quality (error vs time)
    axes[1, 2].semilogy(range(len(rmse_errors)), rmse_errors, 'bo-', markersize=4)
    axes[1, 2].set_xlabel('Time Step Index')
    axes[1, 2].set_ylabel('RMSE')
    axes[1, 2].set_title('RMSE vs Time Step')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save validation plot
    validation_plot_path = os.path.join(experiment_dir, "comprehensive_validation.png")
    plt.savefig(validation_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comprehensive validation plot saved: {validation_plot_path}")
    
    plt.show()


def save_validation_results(validation_results, experiment_dir):
    """Save detailed validation results"""
    
    # Save JSON summary
    validation_summary_path = os.path.join(experiment_dir, "validation_summary.json")
    with open(validation_summary_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_results = {
            'time_points': [float(t) for t in validation_results['time_points']],
            'summary_stats': {k: float(v) for k, v in validation_results['summary_stats'].items()},
            'validation_method': 'Implicit Finite Difference (Source of Truth)'
        }
        json.dump(json_results, f, indent=4)
    
    print(f"✓ Validation summary saved: {validation_summary_path}")
    
    # Save detailed CSV
    detailed_data = []
    for i, t_val in enumerate(validation_results['time_points']):
        row = {'time': t_val}
        row.update(validation_results['error_metrics'][i])
        detailed_data.append(row)
    
    df = pd.DataFrame(detailed_data)
    csv_path = os.path.join(experiment_dir, "detailed_validation_errors.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Detailed validation errors saved: {csv_path}")


def validate_all_experiments(results_base_dir="results"):
    """Validate all experiments in the results directory"""
    
    if not os.path.exists(results_base_dir):
        print(f"Results directory not found: {results_base_dir}")
        return
    
    experiment_dirs = [d for d in os.listdir(results_base_dir) 
                      if d.startswith('PINN_execution_') and 
                      os.path.isdir(os.path.join(results_base_dir, d))]
    
    if not experiment_dirs:
        print(f"No PINN experiments found in {results_base_dir}")
        return
    
    print(f"🔍 Found {len(experiment_dirs)} experiments to validate")
    print("=" * 80)
    
    all_results = {}
    
    for exp_dir in sorted(experiment_dirs):
        exp_path = os.path.join(results_base_dir, exp_dir)
        try:
            results = validate_pinn_experiment(exp_path, detailed_output=True)
            all_results[exp_dir] = results
            print(f"Validation completed for {exp_dir}")
        except Exception as e:
            print(f"Validation failed for {exp_dir}: {e}")
        
        print("\n" + "="*80 + "\n")
    
    # Compare experiments
    if len(all_results) > 1:
        print("📊 EXPERIMENT COMPARISON")
        print("-" * 40)
        for exp_name, results in all_results.items():
            stats = results['summary_stats']
            print(f"{exp_name}:")
            print(f"  Avg RMSE: {stats['mean_rmse']:.2e}")
            print(f"  Worst RMSE: {stats['max_rmse']:.2e}")
            print(f"  Avg Relative L2: {stats['mean_relative_l2']:.2%}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Validate specific experiment
        experiment_path = "/Users/dgomezba/Documents/Tesis/results/PINN_execution_20250926_061901"
        if os.path.exists(experiment_path):
            validate_pinn_experiment(experiment_path, detailed_output=True)
        else:
            print(f"Experiment directory not found: {experiment_path}")
    else:
        # Validate all experiments
        validate_all_experiments()

# %%
