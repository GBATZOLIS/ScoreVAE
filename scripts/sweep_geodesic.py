import os
import sys
import time
import random
import pandas as pd
import numpy as np
from itertools import product
from argparse import ArgumentParser
import torch
import torch.multiprocessing as mp

# --- Local Imports ---
from configs import load_config
from data.data_utils import get_dataloaders
from models import get_model
from sde import configure_sde
from utils.train_utils import EMA, load_model
# Import the dual-use script as a library
from geodesic_computation import run_geodesic_computation, load_geo_config

def setup_worker_logging(rank: int, out_dir: str):
    """Redirects stdout and stderr of a worker process to a dedicated log file."""
    log_path = os.path.join(out_dir, f"worker_{rank}.log")
    sys.stdout = open(log_path, 'w', buffering=1)
    sys.stderr = open(log_path, 'w', buffering=1)
    print(f"--- Worker {rank} logging to {log_path} ---")

def worker(
    rank: int,
    device_id: int,
    diffusion_config_path: str,
    base_geo_config_path: str,
    configs_chunk: list,
    results_list: list,
    out_dir: str
):
    """The function run by each parallel process."""
    setup_worker_logging(rank, out_dir)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    try:
        # 1. Load shared resources once per worker to save time
        print(f"[Worker {rank}] Loading shared resources...")
        diff_cfg = load_config(diffusion_config_path)
        base_geo_cfg = load_geo_config(base_geo_config_path)

        # Load Dataset
        train_loader, _, _ = get_dataloaders(diff_cfg.data, seed=base_geo_cfg.get("random_seed", 42))
        dataset = train_loader.dataset

        # Load Model
        model = get_model(diff_cfg.model).to(device)
        ema = EMA(model=model, decay=diff_cfg.model.ema_decay)
        
        ckpt_name = diff_cfg.model.checkpoint
        if not ckpt_name.endswith(".pth"):
            ckpt_name += ".pth"
        ckpt_path = os.path.join(diff_cfg.checkpoint_dir, ckpt_name)

        load_model(model, ema, ckpt_path, "Model", device=device, is_ema=True)
        ema.apply_shadow()
        model.eval()

        # Configure SDE
        sde = configure_sde(diff_cfg)
        print(f"[Worker {rank}] Resources loaded on {device}.")

        # 2. Process the assigned chunk of configurations
        times = []
        total = len(configs_chunk)
        for i, sweep_params in enumerate(configs_chunk, 1):
            start_time = time.time()
            
            geo_cfg = {**base_geo_cfg, **sweep_params}
            
            print(f"\n[Worker {rank}] Run {i}/{total} starting.")
            
            run_results = run_geodesic_computation(
                geo_cfg, model, sde, dataset, device, out_dir, visualize=False
            )
            
            # Combine sweep params with run results for logging
            loss_info = run_results.pop("loss_info", {})
            run_results.pop("path", None)
            full_results = {
                **sweep_params, 
                **run_results,
                "geodesic_energy": loss_info.get('geodesic_energy', np.nan),
                "total_loss": loss_info.get('total', np.nan),
                "best_iter": loss_info.get('iter', -1),
            }

            # Unpack betas tuple into separate columns for easier analysis
            if 'betas' in full_results:
                beta1, beta2 = full_results.pop('betas')
                full_results['beta1'] = beta1
                full_results['beta2'] = beta2

            results_list.append(full_results)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            avg_time = sum(times) / len(times)
            eta = avg_time * (total - i)
            m, s = divmod(int(eta), 60)
            print(f"[Worker {rank}] Run {i}/{total} finished in {elapsed:.1f}s. ETA for this worker: {m}m{s:02d}s.")

    except Exception as e:
        print(f"[Worker {rank}] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = ArgumentParser("Geodesic hyperparameter sweep runner")
    parser.add_argument("--config", required=True, help="Path to the main diffusion config (.py)")
    parser.add_argument("--geo-config", required=True, help="Path to the base geodesic config (.py)")
    parser.add_argument('--gpus', type=str, default='0', help="Comma-separated GPU indices (e.g., '0,1,2')")
    args = parser.parse_args()

    # --- 1. Define Parameter Grid ---
    param_grid = []

    # Parameters that are the same across all experiments
    base_params = {
        "lam_smooth": [200.],
        "lam_mono": [2.0],
        "time_schedule": [[0.05, 0.04, 0.03], [0.03]],
        "patience": [50],
        "betas": [(0.8, 0.990)],
    }

    # Parameters specific to each metric type
    metric_configs = {
        #"stein": {"lam_metric": [1.]},
        "jacobian": {"lam_metric": [0.05]}
    }

    # --- MODIFIED: Define separate configs for each line search method ---
    line_search_configs = [
        # Configs for 'fixed' learning rate
        #{
        #    "line_search": ["fixed"],
        #    "adam_lr": [5e-3, 1e-2],
        #},
        # Configs for 'armijo' line search
        {
            "line_search": ["armijo"],
            "adam_lr":      [1e-2],       # Initial step size guess
            "armijo_rho":   [1e-4, 1e-3, 1e-2, 5e-2],     # Sufficient decrease parameter
            "armijo_beta":  [0.3, 0.5, 0.8],            # Backtracking factor
            "armijo_max_iter": [15],
        }
    ]

    # --- Build the grid without redundant configurations ---
    for metric_type, specific_params in metric_configs.items():
        for ls_config in line_search_configs:
            # Combine base, metric-specific, and line-search-specific params
            current_run_params = {**base_params, **specific_params, **ls_config}
            
            # Get the keys and the product of values for the current group
            keys = list(current_run_params.keys())
            vals = list(product(*current_run_params.values()))
            
            for v in vals:
                cfg = {"metric_type": metric_type, **dict(zip(keys, v))}
                param_grid.append(cfg)

    random.shuffle(param_grid)
    print(f"Generated {len(param_grid)} unique parameter configurations for the sweep.")

    # --- 2. Setup Parallel Execution ---
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_workers = len(gpu_ids)
    configs_per_worker = [param_grid[i::num_workers] for i in range(num_workers)]

    diff_cfg = load_config(args.config)
    out_dir = os.path.join(diff_cfg.base_log_dir, diff_cfg.experiment, "geodesic_sweep_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results and worker logs to: {out_dir}")

    with mp.Manager() as manager:
        shared_results_list = manager.list()
        processes = []

        for i in range(num_workers):
            p = mp.Process(
                target=worker,
                args=(i, gpu_ids[i], args.config, args.geo_config, configs_per_worker[i], shared_results_list, out_dir)
            )
            p.start()
            processes.append(p)
            print(f"Started worker {i} on GPU {gpu_ids[i]} with {len(configs_per_worker[i])} configs.")

        for p in processes:
            p.join()

        # --- 3. Collate, Save, and Analyze Results ---
        print("\nAll workers finished. Collating and analyzing results...")
        final_results = list(shared_results_list)
        if not final_results:
            print("No results were generated. Check worker logs for errors.")
            return

        df = pd.DataFrame(final_results)
        
        # Define columns for display, including new ones if they don't exist
        first_cols = [
            'mean_error', 'geodesic_energy', 'metric_type', 'line_search', 'adam_lr', 
            'lam_metric', 'lam_smooth', 'lam_mono', 'time_schedule',
            'beta1', 'beta2', 'armijo_rho', 'armijo_beta', 'armijo_max_iter'
        ]
        
        # Ensure all columns in first_cols exist in the DataFrame, adding them with NaN if not
        for col in first_cols:
            if col not in df.columns:
                df[col] = np.nan

        other_cols = [col for col in df.columns if col not in first_cols]
        df = df[first_cols + other_cols]

        df = df.sort_values('mean_error', ascending=True)
        
        csv_path = os.path.join(out_dir, "sweep_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Sweep results saved to {csv_path}")

        # --- 4. Display Best Results Summary ---
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)

        # Columns to display in the summary tables
        display_cols = [
            'mean_error', 'geodesic_energy', 'metric_type', 'line_search', 'adam_lr',
            'lam_metric', 'lam_smooth', 'beta1', 'armijo_rho'
        ]
        # Filter display_cols to only those present in the dataframe
        display_cols_exist = [col for col in display_cols if col in df.columns]


        print("\n--- Best Configs by Mean Error vs. GT (lower is better) ---")
        print(df.nsmallest(10, 'mean_error')[display_cols_exist])

        print("\n--- Best Configs by Geodesic Energy (lower is better) ---")
        print(df.nsmallest(10, 'geodesic_energy')[display_cols_exist])

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()