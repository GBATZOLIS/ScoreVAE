#!/usr/bin/env python
from __future__ import annotations
import os, json, torch
from argparse import ArgumentParser

from configs import load_config
from data.data_utils_fast import get_dataloaders
from models import get_model
from utils.train_utils import prepare_training_dirs, EMA, load_model
from sde import configure_sde
from utils.entropy_profile import compute_entropy_profile, schedule_from_rescaled_entropic_time


def main():
    P = ArgumentParser("Compute entropy profile / schedule")
    P.add_argument("--config", required=True)
    P.add_argument("--batches", type=int, default=100)
    P.add_argument("--num-t", type=int, default=96)
    P.add_argument("--t-min", type=float, default=None)
    P.add_argument("--t-max", type=float, default=1.0)
    P.add_argument("--prefix", default="entropy_profile")
    P.add_argument("--make-schedule", action="store_true")
    P.add_argument("--n-stages", type=int, default=8)
    args = P.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(cfg.training.device)

    _, _, eval_dir = prepare_training_dirs(cfg)
    train_loader, val_loader, _ = get_dataloaders(cfg.data, seed=cfg.get("random_seed", 42))
    loader = val_loader or train_loader
    orig_shape = tuple(cfg.data.shape)

    model = get_model(cfg.model).to(device)
    ema   = EMA(model, cfg.model.ema_decay)
    ckpt  = os.path.join(cfg.checkpoint_dir, cfg.model.checkpoint)
    load_model(model, ema, ckpt, "Model", device=device, is_ema=True)
    ema.apply_shadow()
    model.eval()

    sde = configure_sde(cfg)

    out = compute_entropy_profile(
        model=model,
        sde=sde,
        loader=loader,
        orig_shape=orig_shape,
        device=device,
        t_min=args.t_min,
        t_max=args.t_max,
        num_t=args.num_t,
        max_batches=args.batches,
        save_dir=eval_dir,
        filename_prefix=args.prefix,
        progress=True,
    )

    png = os.path.join(eval_dir, f"{args.prefix}.png")
    npz = os.path.join(eval_dir, f"{args.prefix}.npz")
    print(f"\nSaved: {png}  /  {npz}")

    # quick summary + discrepancy
    Hs, Hm, t = out["Hdot_score"], out["Hdot_mmse"], out["t"]
    valid = ~torch.isnan(torch.tensor(Hs)).numpy() & ~torch.isnan(torch.tensor(Hm)).numpy()
    if valid.any():
        import numpy as np
        diff = np.abs(Hs[valid] - Hm[valid])
        print(f"  max Ḣ at t={t[np.argmax(Hm[valid])]:.5f}")
        print(f"  mean |ΔḢ| (score vs mmse): {diff.mean():.4e} | max |ΔḢ|: {diff.max():.4e}")
        print(f"  rel L2 discrepancy: {out['discrepancy']['rel_L2']:.4e}")

    if args.make_schedule and args.n_stages > 0:
        # by default, build from *MMSE rescaled* entropic time
        schedule = schedule_from_rescaled_entropic_time(
            out["Phi_mmse_t"], out["Phi_rescaled_mmse"], n_stages=args.n_stages, descending=True
        )
        sched_path = os.path.join(eval_dir, f"{args.prefix}_schedule.json")
        with open(sched_path, "w") as f:
            json.dump({"time_schedule": schedule}, f, indent=2)
        print(f"\nSuggested time_schedule: {schedule}")
        print(f"Wrote: {sched_path}")


if __name__ == "__main__":
    main()
