#!/usr/bin/env python3
"""
save_latent_pairs.py
Minimal utility that *only* selects B pairs deterministically and saves them to
latent_pair_selection.pt — no model loading, no geodesics, no plots.

Usage:
  python save_latent_pairs.py --config path/to/ae_or_diff_cfg.py \
                              --geo-config path/to/latent_geo_config.py \
                              [--use_test]

Output:
  <eval_dir>/latent_pair_selection.pt
"""

import os
import torch
from typing import Tuple, List, Any, Dict

from configs import load_config
from utils.train_utils import prepare_training_dirs
from data.data_utils_fast import get_dataloaders

# ── unwrap Subset(s) to the base dataset and get indices in the current split
def _unwrap_base_dataset(eval_dataset) -> Tuple[Any, List[int]]:
    subset = eval_dataset
    while hasattr(subset, "dataset"):
        subset = subset.dataset
    base_ds = subset
    if hasattr(eval_dataset, "indices"):
        idx_pool = list(eval_dataset.indices)
    else:
        idx_pool = list(range(len(eval_dataset)))
    return base_ds, idx_pool

def _load_geo_cfg(path: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, scope)
    if "CONFIG" not in scope:
        raise ValueError(f"Geodesic config '{path}' must define a CONFIG dict.")
    return scope["CONFIG"]  # type: ignore[return-value]

def main():
    import argparse
    p = argparse.ArgumentParser("Save latent pair selection only (no models).")
    p.add_argument("--config", required=True, type=str, help="Main config (used for dataloader + eval dir).")
    p.add_argument("--geo-config", required=True, type=str, help="Latent geodesic CONFIG (reads num_pairs & seed).")
    p.add_argument("--use_test", action="store_true", help="Use test split instead of val.")
    args = p.parse_args()

    cfg = load_config(args.config)
    geo = _load_geo_cfg(args.geo_config)

    # Prepare dirs (for eval_dir output location)
    _, _, eval_dir = prepare_training_dirs(cfg)
    os.makedirs(eval_dir, exist_ok=True)

    # Build loaders with the same seed logic you use elsewhere
    seed = int(geo.get("random_seed", getattr(cfg, "random_seed", 42)))
    _, val_loader, test_loader = get_dataloaders(cfg.data, seed=seed)
    loader = test_loader if args.use_test else val_loader

    # Unwrap to base dataset and get the Subset's index mapping
    base_ds, idx_pool = _unwrap_base_dataset(loader.dataset)

    # Sample indices deterministically (exactly like your latent driver)
    B = int(geo.get("num_pairs", 100))
    need = 2 * B
    g_idx = torch.Generator()
    g_idx.manual_seed(seed)
    perm_local = torch.randperm(len(idx_pool), generator=g_idx).tolist()[:need]

    pair_meta = {
        "seed_used": seed,
        "B": B,
        "perm_local": perm_local,                # length 2*B, each elem idx into idx_pool
        "idx_pool": idx_pool,                    # Subset mapping → base dataset indices
        "split": "test" if args.use_test else "val",
        "dataset_name": str(getattr(cfg.data, "dataset", "unknown")),
    }
    out_path = os.path.join(eval_dir, "latent_pair_selection.pt")
    torch.save(pair_meta, out_path)
    print(f"[Pairs] Saved latent pair selection → {out_path}")
    print(f"[Pairs] Example: 11th pair (1-based) has pair_id=10")

if __name__ == "__main__":
    main()
