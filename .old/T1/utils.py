"""
Utility functions for PINN experiments
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime


def create_experiment_folders():
    """Create organized folder structure for saving results and models"""
    # Create main directories
    results_dir = "results"
    models_dir = "models"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create unique execution folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exec_name = f"PINN_execution_{timestamp}"
    
    exec_results_dir = os.path.join(results_dir, exec_name)
    exec_models_dir = os.path.join(models_dir, exec_name)
    time_models_dir = os.path.join(exec_models_dir, "time_models")
    
    os.makedirs(exec_results_dir, exist_ok=True)
    os.makedirs(exec_models_dir, exist_ok=True)
    os.makedirs(time_models_dir, exist_ok=True)
    
    print(f"✓ Created experiment folders: {exec_name}")
    return exec_results_dir, exec_models_dir, time_models_dir


def save_hyperparameters(results_dir, hyperparams):
    """Save hyperparameters to JSON file"""
    hyperparams_file = os.path.join(results_dir, "hyperparameters.json")
    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    print(f"✓ Saved hyperparameters to {hyperparams_file}")


def load_model_at_time_step(models_dir, time_step, model_class=None, device='cpu'):
    """
    Load a saved model from a specific time step
    
    Args:
        models_dir: Path to the models directory
        time_step: Time step number
        model_class: The model class to use for reconstruction
        device: Device to load the model on
        
    Returns:
        Loaded model and metadata dictionary
    """
    time_models_dir = os.path.join(models_dir, "time_models")
    model_file = f"time_step_{time_step:04d}.pt"
    model_path = os.path.join(time_models_dir, model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Load the saved data
    saved_data = torch.load(model_path, map_location=device)
    
    # If model_class is provided, recreate the model
    if model_class is not None:
        hyperparams = saved_data['hyperparameters']
        model = model_class(
            x_dim=hyperparams['model_config']['x_dim'],
            hid=hyperparams['model_config']['hidden_size'],
            u_dim=hyperparams['model_config']['u_dim'],
            nu=hyperparams['physics_params']['nu'],
            dt=hyperparams['physics_params']['dt'],
            device=torch.device(device)
        )
        
        # Load the trained weights
        model.load_state_dict(saved_data['model_state_dict'])
        model.eval()
        
        return model, saved_data
    
    # Return just the saved data if no model class provided
    return None, saved_data


def save_solution_dataframe(solution_data, exec_results_dir):
    """
    Create and save DataFrame with solution data
    
    Args:
        solution_data: Dictionary with time_values, pinn_solutions, fd_solutions
        exec_results_dir: Directory to save the DataFrame
    """
    if len(solution_data['time_values']) > 0:
        # Create coordinate grid for DataFrame
        x_coords = np.linspace(0, 2, len(solution_data['pinn_solutions'][0]))
        
        # Prepare data for DataFrame
        df_data = []
        for i, time_val in enumerate(solution_data['time_values']):
            for j, x_val in enumerate(x_coords):
                df_data.append({
                    'time': time_val,
                    'x': x_val,
                    'pinn_solution': solution_data['pinn_solutions'][i][j],
                    'fd_solution': solution_data['fd_solutions'][i][j],
                    'error': abs(solution_data['pinn_solutions'][i][j] - solution_data['fd_solutions'][i][j])
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(df_data)
        
        # Save DataFrame
        df_path = os.path.join(exec_results_dir, "solution_results.csv")
        results_df.to_csv(df_path, index=False)
        print(f"✓ Saved solution results DataFrame to {df_path}")
        return results_df
    return None


def save_solution_arrays(pinn_solutions_2d, fd_solutions_2d, x_plot, t_plot_values, exec_results_dir):
    """
    Save 2D numpy arrays with solution data
    
    Args:
        pinn_solutions_2d: 2D array of PINN solutions
        fd_solutions_2d: 2D array of FD solutions
        x_plot: Spatial coordinates
        t_plot_values: Time values
        exec_results_dir: Directory to save arrays
    """
    arrays_path = os.path.join(exec_results_dir, "solution_arrays.npz")
    np.savez(arrays_path, 
             pinn_solutions=pinn_solutions_2d, 
             fd_solutions=fd_solutions_2d,
             x_coordinates=x_plot,
             time_values=np.array(t_plot_values))
    print(f"✓ Saved 2D solution arrays to {arrays_path}")
    return arrays_path


def calculate_and_save_validation_summary(validation_errors, exec_results_dir):
    """
    Calculate validation error statistics and save summary
    
    Args:
        validation_errors: Dictionary with validation error data
        exec_results_dir: Directory to save validation files
        
    Returns:
        Dictionary with validation statistics
    """
    # Save validation errors to separate file
    validation_path = os.path.join(exec_results_dir, "validation_errors.json")
    with open(validation_path, 'w') as f:
        json.dump(validation_errors, f, indent=4)
    print(f"✓ Saved validation errors to {validation_path}")
    
    # Calculate validation error statistics
    validation_stats = {}
    if len(validation_errors['rmse_errors']) > 0:
        validation_stats = {
            'final_rmse': validation_errors['rmse_errors'][-1],
            'avg_rmse': np.mean(validation_errors['rmse_errors']),
            'max_rmse': np.max(validation_errors['rmse_errors']),
            'min_rmse': np.min(validation_errors['rmse_errors']),
            'final_mean_error': validation_errors['mean_errors'][-1],
            'avg_mean_error': np.mean(validation_errors['mean_errors']),
            'final_max_abs_error': validation_errors['max_absolute_errors'][-1],
            'avg_max_abs_error': np.mean(validation_errors['max_absolute_errors'])
        }
        
        print(f"\n📈 Validation Error Summary:")
        print(f"   Final RMSE: {validation_stats['final_rmse']:.6f}")
        print(f"   Average RMSE: {validation_stats['avg_rmse']:.6f}")
        print(f"   Min RMSE: {validation_stats['min_rmse']:.6f}")
        print(f"   Max RMSE: {validation_stats['max_rmse']:.6f}")
        print(f"   Final Mean Error: {validation_stats['final_mean_error']:.6f}")
        print(f"   Final Max Absolute Error: {validation_stats['final_max_abs_error']:.6f}")
    
    return validation_stats


def create_validation_plots(validation_errors, exec_results_dir):
    """
    Create and save validation error plots
    
    Args:
        validation_errors: Dictionary with validation error data
        exec_results_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    
    if len(validation_errors['rmse_errors']) > 0:
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE error evolution
        plt.subplot(2, 2, 1)
        plt.plot(validation_errors['time_values'], validation_errors['rmse_errors'], 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Time')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Time')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot training loss vs validation error
        plt.subplot(2, 2, 2)
        plt.plot(validation_errors['training_losses'], validation_errors['rmse_errors'], 'ro-', linewidth=2, markersize=6)
        plt.xlabel('Training Loss')
        plt.ylabel('RMSE')
        plt.title('Training Loss vs RMSE')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        
        # Plot all error types
        plt.subplot(2, 2, 3)
        plt.plot(validation_errors['time_values'], validation_errors['rmse_errors'], 'b-o', label='RMSE', linewidth=2)
        plt.plot(validation_errors['time_values'], validation_errors['mean_errors'], 'g-s', label='Mean Error', linewidth=2)
        plt.plot(validation_errors['time_values'], validation_errors['max_absolute_errors'], 'r-^', label='Max Absolute Error', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.title('All Validation Error Types')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot error reduction over time
        plt.subplot(2, 2, 4)
        relative_errors = np.array(validation_errors['rmse_errors']) / validation_errors['rmse_errors'][0]
        plt.plot(validation_errors['time_values'], relative_errors, 'm-d', linewidth=2, markersize=6)
        plt.xlabel('Time')
        plt.ylabel('Relative RMSE')
        plt.title('Relative RMSE (normalized to initial)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        validation_plot_path = os.path.join(exec_results_dir, "validation_errors.png")
        plt.savefig(validation_plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved validation error plot to {validation_plot_path}")
        plt.show()


def save_training_summary(TOTAL_TIME, t, time_models_dir, loss_value, tol, epochs, validation_stats, fd_solver, exec_results_dir, train_time=None):
    """
    Save comprehensive training summary
    
    Args:
        TOTAL_TIME: Total training time
        t: Time array
        time_models_dir: Directory with time models
        loss_value: Final loss value
        tol: Tolerance used
        epochs: Max epochs
        validation_stats: Validation statistics dictionary
        fd_solver: Finite difference solver object
        exec_results_dir: Directory to save summary
    """
    training_summary = {
        "total_runtime_seconds": TOTAL_TIME,
        "total_time_steps": len(t),
        "models_saved": len([f for f in os.listdir(time_models_dir) if f.endswith('.pt')]),
        "average_training_time_per_step": np.mean(train_time) if train_time else None,
        "final_training_loss": loss_value if loss_value is not None else None,
        "convergence_info": {
            "tolerance": tol,
            "max_epochs": epochs
        },
        "validation_summary": {
            "final_rmse": validation_stats.get('final_rmse') if validation_stats else None,
            "average_rmse": validation_stats.get('avg_rmse') if validation_stats else None,
            "min_rmse": validation_stats.get('min_rmse') if validation_stats else None,
            "max_rmse": validation_stats.get('max_rmse') if validation_stats else None,
            "final_mean_error": validation_stats.get('final_mean_error') if validation_stats else None,
            "final_max_absolute_error": validation_stats.get('final_max_abs_error') if validation_stats else None,
            "num_validation_points": len(validation_stats) if validation_stats else 0,
            "finite_difference_stability_ratio": getattr(fd_solver, 'r', None)
        }
    }
    
    summary_path = os.path.join(exec_results_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=4)
    print(f"✓ Saved training summary to {summary_path}")
    return training_summary


def print_final_results(exec_results_dir, exec_models_dir, time_models_dir, pinn_solutions_2d, t_plot_values, tf, validation_stats):
    """Print final experiment results"""
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📁 Results saved in: {exec_results_dir}")
    print(f"🧠 Models saved in: {exec_models_dir}")
    print(f"⏱️  Time models saved in: {time_models_dir}")
    print(f"📊 {len([f for f in os.listdir(time_models_dir) if f.endswith('.pt')])} time-step models saved")
    print(f"📈 2D solution arrays saved: {pinn_solutions_2d.shape} (time x space)")
    print(f"📈 Time points: {len(t_plot_values)} from t=0 to t={tf}")
    
    if validation_stats:
        print(f"🎯 Final RMSE: {validation_stats['final_rmse']:.6f}")
        print(f"📉 Average RMSE: {validation_stats['avg_rmse']:.6f}")
        print(f"🎯 Final Mean Error: {validation_stats['final_mean_error']:.6f}")
        print(f"🎯 Final Max Absolute Error: {validation_stats['final_max_abs_error']:.6f}")
