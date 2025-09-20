#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
import itertools
import tempfile
import pprint
from typing import Dict, Any, List

# --- Correctly import the project's config loader ---
from configs import load_config

# Helper to read the original config file format
def _load_geo_cfg(path: str) -> Dict[str, Any]:
    """Exec-loads a Python file that defines CONFIG = {...} and returns the dict."""
    scope: Dict[str, object] = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, scope)
    if "CONFIG" not in scope or not isinstance(scope["CONFIG"], dict):
        raise ValueError(f"Geodesic config '{path}' must define a CONFIG dict.")
    return scope["CONFIG"]

def write_temp_config(config_dict: Dict[str, Any]) -> str:
    """Writes a config dict to a temporary .py file and returns the path."""
    # Use pprint.pformat to get a pretty-printed, valid Python representation
    config_str = pprint.pformat(config_dict, indent=4)
    
    # The tempfile needs to be a .py file that the driver can import
    fd, temp_path = tempfile.mkstemp(suffix=".py", text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(f"CONFIG = {config_str}\n")
    return temp_path

def parse_rmse_from_log(eval_dir: str) -> float:
    """Parses the 'Avg RMSE vs GT' from the output file."""
    log_file = os.path.join(eval_dir, "geodesic_eval.txt")
    if not os.path.exists(log_file):
        return float('inf')  # Return infinity if the run failed to produce a result

    with open(log_file, 'r') as f:
        for line in f:
            if "Avg RMSE vs GT" in line:
                try:
                    # Extracts the float value from a line like:
                    # "Avg RMSE vs GT (B=10, T=26): 0.170160"
                    rmse_str = line.split(":")[1].strip()
                    return float(rmse_str)
                except (IndexError, ValueError):
                    return float('inf')
    return float('inf')

def run_single_experiment(base_args: List[str], params: Dict[str, Any], base_config: Dict[str, Any], run_idx: int, total_runs: int) -> float:
    """
    Conducts a single evaluation run with a given set of parameters.
    """
    print(f"\n--- Running Experiment {run_idx}/{total_runs} ---")
    print(f"Parameters: {params}")

    # Create a new config by updating the base config with the current parameters
    current_config = base_config.copy()
    current_config.update(params)

    temp_config_path = None
    try:
        # Write the temporary config file
        temp_config_path = write_temp_config(current_config)

        # Construct the full command using `python -m` for robust module loading
        command = [
            "python",
            "-m", "scripts.eval_autoencoder",
            *base_args,
            "--geo_config", temp_config_path
        ]

        print(f"Executing: {' '.join(command)}")

        # Run the subprocess
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("--- ERROR: Subprocess failed! ---")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return float('inf')

        # --- CORRECTED CONFIG LOADING ---
        # Find the evaluation directory to parse results
        main_cfg_path = base_args[base_args.index('--config') + 1]
        main_cfg = load_config(main_cfg_path)
        
        # The returned object might be a class, so we convert to dict if needed
        main_cfg_dict = main_cfg if isinstance(main_cfg, dict) else main_cfg.to_dict()
        
        eval_dir = os.path.join(main_cfg_dict['base_log_dir'], main_cfg_dict['experiment'], 'eval')
        
        rmse = parse_rmse_from_log(eval_dir)
        print(f"--- Finished. Parsed RMSE: {rmse:.6f} ---")
        return rmse

    finally:
        # Clean up the temporary config file
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for geodesic computation.")
    parser.add_argument("--config", required=True, help="Path to the main AE config file where checkpoint paths are defined.")
    parser.add_argument("--geo_config", required=True, help="Path to the base geodesic config .py file to modify.")
    
    args = parser.parse_args()

    # --- Base arguments for all subprocess calls ---
    base_subprocess_args = [
        "--config", args.config,
    ]
    
    base_geo_config = _load_geo_cfg(args.geo_config)

    # =========================================================================
    # FOCUSED HYPERPARAMETER SEARCH
    # =========================================================================
    print("\n" + "="*80)
    print("### Running Focused Hyperparameter Search ###")
    print("="*80)

    # Define the grid for the three parameters of interest
    search_params = {
        'lam_smooth': [0.5, 1.0, 2.0, 4.0],
        'adam_lr': [1e-2, 2.5e-2, 5e-2],
        'armijo_rho': [1e-4, 1e-3]
    }
    
    # Set other influential parameters to fixed, stable values
    fixed_params = {
        'lam_metric': 1e-2,
        'lam_mono': 1.0,
        'armijo_beta': 0.7,
    }
    
    # Create the grid of all combinations
    search_grid = list(itertools.product(*search_params.values()))
    results = []

    print(f"Total experiments to run: {len(search_grid)}")
    print("Fixed parameters for all runs:", fixed_params)

    for i, p_values in enumerate(search_grid):
        params_to_run = dict(zip(search_params.keys(), p_values))
        params_to_run.update(fixed_params)
        
        rmse = run_single_experiment(base_subprocess_args, params_to_run, base_geo_config, i + 1, len(search_grid))
        results.append((params_to_run, rmse))

    # Sort and report final results
    results.sort(key=lambda x: x[1])
    
    print("\n" + "="*80)
    print("### FINAL RESULTS (Sorted by best RMSE) ###")
    print("="*80)
    
    if not results or all(r[1] == float('inf') for r in results):
        print("No successful runs completed. Please check for errors in the logs above.")
    else:
        for params, rmse in results:
            # Format parameters for cleaner printing
            p_str = ", ".join([f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k,v in params.items()])
            if rmse == float('inf'):
                print(f"RMSE: FAILED   | Params: {p_str}")
            else:
                print(f"RMSE: {rmse:.6f} | Params: {p_str}")
        
        print("-"*80)
        best_overall_params, best_overall_rmse = results[0]
        if best_overall_rmse != float('inf'):
            print(f"🏆 Best Overall Parameters: {best_overall_params}")
            print(f"🏆 Best Overall RMSE: {best_overall_rmse:.6f}")
    
    print("="*80)

if __name__ == "__main__":
    main()