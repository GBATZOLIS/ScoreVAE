# utils/ae_utils.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from .vis_utils import save_latent_scatter_artifacts

# Try to import SummaryWriter only if available
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = object  # type: ignore


# ────────────────────────── DDP helpers ──────────────────────────

def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _get_world_size() -> int:
    return torch.distributed.get_world_size() if _dist_is_initialized() else 1

def _get_rank() -> int:
    return torch.distributed.get_rank() if _dist_is_initialized() else 0

def _is_primary() -> bool:
    return _get_rank() == 0

def _unwrap_module(m):
    """Return m.module if it exists (DDP/EMA wrappers), otherwise m."""
    return getattr(m, "module", m)

@torch.no_grad()
def _all_gather_variable_batch(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    All-gather a tensor along `dim` when local batch sizes may differ.
    Returns the concatenated tensor on ALL ranks (you can use it only on rank 0 if desired).
    """
    if not _dist_is_initialized():
        return x

    ws = _get_world_size()
    local_shape = torch.tensor(x.shape, device=x.device, dtype=torch.long)
    shapes = [torch.zeros_like(local_shape) for _ in range(ws)]
    torch.distributed.all_gather(shapes, local_shape)
    max_shape = torch.stack(shapes, dim=0).max(dim=0).values.tolist()

    # pad to max shape
    pad_sizes = []
    for i, (cur, mx) in enumerate(zip(x.shape, max_shape)):
        if i == dim:
            pad_sizes.append((0, mx - cur))
        else:
            pad_sizes.append((0, 0))
    # torch.pad expects flattened pairs from last to first
    flat_pad = []
    for (l, r) in reversed(pad_sizes):
        flat_pad.extend([l, r])
    x_padded = torch.nn.functional.pad(x, flat_pad)

    # gather
    gathered = [torch.zeros_like(x_padded) for _ in range(ws)]
    torch.distributed.all_gather(gathered, x_padded)

    # unpad and concat
    out_parts = []
    for r, shape in enumerate(shapes):
        true_len = int(shape[dim].item())
        sl = [slice(None)] * x_padded.ndim
        sl[dim] = slice(0, true_len)
        out_parts.append(gathered[r][tuple(sl)])
    return torch.cat(out_parts, dim=dim)

@torch.no_grad()
def _broadcast_(t: torch.Tensor, src: int = 0):
    """In-place broadcast of tensor t from rank `src`."""
    if _dist_is_initialized():
        torch.distributed.broadcast(t, src=src)

def _writer_ok(writer) -> bool:
    """Write to TensorBoard only on rank 0 (if DDP)."""
    return isinstance(writer, object) and _is_primary()


# ────────────────────────── mode helpers ──────────────────────────

@contextmanager
def evaluation_mode(model):
    """Temporarily sets a torch.nn.Module to eval mode and restores the
    previous training state on exit.
    Works if you pass either the raw module or a DDP wrapper."""
    mm = _unwrap_module(model)
    was_train = getattr(mm, "training", False)
    if hasattr(mm, "eval"):
        mm.eval()
    try:
        yield
    finally:
        if was_train and hasattr(mm, "train"):
            mm.train()


# ────────────────────────── logging callbacks ──────────────────────────


def get_update_latent_normalizer_callback(
    *,
    min_count: int = 4000,
    max_batches: int | None = None,
    tag_prefix: str = "AE",
    eps: float = 1e-6,
    use_amp: bool = True,   # autocast during encode
):
    """
    End-of-epoch callback (DDP-aware, streaming, NO EMA):
      • Each rank encodes ~min_count/world_size examples from val_loader.
      • Locally accumulate S=sum(z), Q=sum(z^2), n in float64 (no storing latents).
      • All-reduce S, Q, n; compute global μ, σ on every rank (no all_gather).
      • Calls model.set_latent_normalization(μ, σ).
    """
    def cb(val_loader, writer, model, device, epoch):
        mm = _unwrap_module(model)
        ws = _get_world_size()
        local_min = int(math.ceil(min_count / max(ws, 1)))

        # Accumulators (float64 for stability)
        S = None
        Q = None
        n_local = torch.zeros((), dtype=torch.float64, device=device)

        # AMP context for faster encode
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (use_amp and device.type == "cuda")
            else contextmanager(lambda: (yield))()
        )

        b = 0
        with evaluation_mode(mm), torch.inference_mode(), amp_ctx:
            for batch in val_loader:
                if (max_batches is not None and b >= max_batches) or (n_local.item() >= local_min):
                    break
                x = batch[0].to(device, non_blocking=True)
                z = mm.encode(x)  # (B, d)

                if S is None:
                    d = z.shape[1]
                    S = torch.zeros(d, dtype=torch.float64, device=device)
                    Q = torch.zeros(d, dtype=torch.float64, device=device)

                z64 = z.to(torch.float64)
                S += z64.sum(dim=0)
                Q += (z64 * z64).sum(dim=0)
                n_local += z.shape[0]
                b += 1

        if S is None:
            raise RuntimeError("[LatentNormStreaming] No validation data was processed.")

        # All-reduce partials
        if _dist_is_initialized():
            torch.distributed.all_reduce(S, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(Q, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(n_local, op=torch.distributed.ReduceOp.SUM)

        n = max(float(n_local.item()), 1.0)
        mu_64 = S / n
        var_64 = (Q / n) - mu_64 * mu_64
        var_64.clamp_(min=(eps ** 2))
        sd_64 = torch.sqrt(var_64)

        mu = mu_64.to(dtype=torch.float32)
        sd = sd_64.to(dtype=torch.float32)

        # Set identical μ/σ on every rank
        mm.set_latent_normalization(mu, sd, eps=eps)

        if _writer_ok(writer):
            writer.add_scalar(f"{tag_prefix}/latent_norm/mean_abs_mean", float(mu.abs().mean().item()), epoch)
            writer.add_scalar(f"{tag_prefix}/latent_norm/mean_std",      float(sd.mean().item()), epoch)

        if _is_primary():
            print(f"[LatentNormStreaming] epoch {epoch}: n={int(n)}  |μ|_mean={mu.abs().mean():.3f}  σ_mean={sd.mean():.3f}")

    return cb


def get_reconstruction_callback():
    """Returns a callback that logs original and reconstructed images (rank 0 only in DDP)."""
    def recon_callback(batch, writer, model, device, epoch, tag_prefix="AE"):
        if not _writer_ok(writer):
            return
        mm = _unwrap_module(model)
        x = batch[0].to(device)
        n = min(x.size(0), 36)
        x = x[:n]
        with evaluation_mode(mm), torch.no_grad():
            x_hat, _ = mm(x)

        nrow = int(math.sqrt(n))
        grid_in  = vutils.make_grid(x, nrow=nrow, normalize=True, scale_each=True)
        grid_out = vutils.make_grid(x_hat.clamp(0,1), nrow=nrow, normalize=True, scale_each=True)

        writer.add_image(f"{tag_prefix}/original", grid_in, epoch)
        writer.add_image(f"{tag_prefix}/reconstruction", grid_out, epoch)
    return recon_callback


def get_latent_scatter_callback(
    num_batches: int = 2,
    max_points: int = 2000,
    mode: str = "both",
    views3d=None,           # list of (elev, azim)
    dims3d=(0, 1, 2),
    point_size_2d: float = 5.0,
    point_size_3d: float = 4.0,
    alpha_2d: float = 0.6,
    alpha_3d: float = 0.6,
    # NEW knobs (forwarded to vis_utils.save_latent_mesh_artifacts)
    prefer_mesh: str = "poisson",     # "poisson" | "bpa" | "auto"
    target_faces: int = 20000,        # triangle budget for the saved mesh
):
    """
    Plots latent scatters.
      mode ∈ {"raw", "norm", "both"}:
        - "raw"  : z = model.encode(x)
        - "norm" : ẑ = model.normalize_latent(model.encode(x)) if available else z
        - "both" : logs both; 3D is ONLY for 'norm'

    NEW:
      • If latent_dim ≥ 3 and normalized latents are available, also builds a generic surface mesh
        and saves interactive HTML, PLY, and a static PNG (logged to TB) via vis_utils.save_latent_mesh_artifacts.
      • Pass `save_dir` at call-site (usually your tb_dir) to control where artifacts are written.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mode = str(mode).lower()
    assert mode in {"raw", "norm", "both"}, "mode must be 'raw', 'norm', or 'both'"

    if views3d is None:
        views3d = [(20, 30), (20, 150), (20, 270)]

    def _maybe_norm(model, z):
        fn = getattr(model, "normalize_latent", None)
        return fn(z) if callable(fn) else z

    def _plot2d_and_log(writer, tag_prefix, epoch, z_all, i, j, subtag):
        if not _writer_ok(writer):  # only rank 0 writes
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(z_all[:, i], z_all[:, j], s=point_size_2d, alpha=alpha_2d)
        ax.set_xlabel(f"z[{i}]"); ax.set_ylabel(f"z[{j}]")
        ax.set_title(f"Latent scatter ({subtag}): z[{i}] vs z[{j}]")
        writer.add_figure(f"{tag_prefix}/latent_scatter/{subtag}_z{i}{j}", fig, epoch)
        plt.close(fig)

    def _set_axes_equal_3d(ax, X, Y, Z):
        x_mid, y_mid, z_mid = (X.min()+X.max())/2, (Y.min()+Y.max())/2, (Z.min()+Z.max())/2
        max_range = max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()) / 2
        if max_range == 0: max_range = 1.0
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

    def _plot3d_views_and_log(writer, tag_prefix, epoch, z_all, i, j, k, subtag):
        if not _writer_ok(writer):
            return
        X, Y, Z = z_all[:, i].numpy(), z_all[:, j].numpy(), z_all[:, k].numpy()
        for v_idx, (elev, azim) in enumerate(views3d, start=1):
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(X, Y, Z, s=point_size_3d, alpha=alpha_3d, depthshade=True)
            ax.set_xlabel(f"z[{i}]"); ax.set_ylabel(f"z[{j}]"); ax.set_zlabel(f"z[{k}]")
            ax.view_init(elev=elev, azim=azim)
            _set_axes_equal_3d(ax, np.asarray(X), np.asarray(Y), np.asarray(Z))
            ax.set_title(f"Latent 3D ({subtag}) view {v_idx}: z[{i}], z[{j}], z[{k}]")
            writer.add_figure(f"{tag_prefix}/latent_scatter3d/{subtag}_z{i}{j}{k}_view{v_idx}", fig, epoch)
            plt.close(fig)

    # NOTE: added optional save_dir arg for artifact output path
    def latent_callback(val_loader, writer, model, device, epoch, save_dir: str | None = None, tag_prefix="AE"):
        mm = _unwrap_module(model)
        zs_raw_local, zs_hat_local = [], []
        has_norm = callable(getattr(mm, "normalize_latent", None))

        with evaluation_mode(mm), torch.no_grad():
            for b_idx, (x, *_) in enumerate(val_loader):
                if b_idx >= num_batches: break
                x = x.to(device, non_blocking=True)
                z = mm.encode(x)
                if mode in {"raw", "both"}:
                    zs_raw_local.append(z.detach().cpu())
                if mode in {"norm", "both"}:
                    z_hat = _maybe_norm(mm, z)
                    zs_hat_local.append(z_hat.detach().cpu())

        # Gather across ranks (variable batch per rank)
        if mode in {"raw", "both"} and zs_raw_local:
            Z_local = torch.cat(zs_raw_local, dim=0)
            Z_all = _all_gather_variable_batch(Z_local, dim=0).cpu()
            if _is_primary():
                Z_all = Z_all[:max_points]
            zs_raw_all = Z_all
        else:
            zs_raw_all = None

        if mode in {"norm", "both"} and zs_hat_local:
            Z_local = torch.cat(zs_hat_local, dim=0)
            Z_all = _all_gather_variable_batch(Z_local, dim=0).cpu()
            if _is_primary():
                Z_all = Z_all[:max_points]
            zs_hat_all = Z_all
        else:
            zs_hat_all = None

        # Plotting on rank 0 only
        def _handle_block(z_all, subtag, do_3d: bool):
            if z_all is None or not _is_primary():
                return
            D = z_all.size(1)

            # 2D panels
            if D >= 2: _plot2d_and_log(writer, tag_prefix, epoch, z_all, 0, 1, subtag)
            if D >= 3:
                _plot2d_and_log(writer, tag_prefix, epoch, z_all, 1, 2, subtag)
                _plot2d_and_log(writer, tag_prefix, epoch, z_all, 0, 2, subtag)

            # 3D only when requested + latent_dim>=3
            if do_3d and D >= 3:
                i, j, k = dims3d
                i = min(i, D-1); j = min(j, D-1); k = min(k, D-1)
                if len({i, j, k}) < 3:
                    # fallback to first 3 unique
                    idxs = []
                    for t in range(D):
                        if t not in idxs: idxs.append(t)
                        if len(idxs) == 3: break
                    if len(idxs) == 3:
                        i, j, k = idxs
                _plot3d_views_and_log(writer, tag_prefix, epoch, z_all, i, j, k, subtag)

        if mode in {"raw", "both"}:
            _handle_block(zs_raw_all, "raw", do_3d=False)

        if mode in {"norm", "both"}:
            if not has_norm and _is_primary():
                print("[LatentScatter] normalize_latent() not found; skipping 3D normalized plots.")
            _handle_block(zs_hat_all, "norm", do_3d=has_norm)

            # Save interactive scatter + exact normalized coordinates (rank 0 only)
            if _is_primary() and has_norm and zs_hat_all is not None:
                try:
                    save_root = save_dir if save_dir is not None else getattr(writer, "log_dir", ".")
                    _ = save_latent_scatter_artifacts(
                        z_norm=zs_hat_all.numpy(),            # full D saved to .npz
                        epoch=epoch,
                        save_root=save_root,
                        tag_prefix=f"{tag_prefix}/latent_scatter",
                        dims=(0, 1, 2),                      # first 3 dims for the HTML
                        save_points=True,
                        save_html=True,
                    )
                except Exception as e:
                    print(f"[LatentScatter] scatter/points save failed: {e}")

    return latent_callback


def get_prior_variance_callback():
    """
    Logs a bar plot of learned PRIOR variances (diag Σ_z) sorted high→low.
    Expects `model.prior_logvar` (shape [d]) to exist. Rank 0 only.
    """
    def cb(writer, model, epoch, tag_prefix="AE"):
        if not _writer_ok(writer):
            return
        mm = _unwrap_module(model)
        prior_logvar = getattr(mm, "prior_logvar", None)
        if prior_logvar is None:
            print("[VarViz] Model has no learnable prior_logvar; skipping prior variance plot.")
            return
        with torch.no_grad():
            var = prior_logvar.detach().exp().cpu()   # σ_p^2 (d,)
            vals, _ = torch.sort(var, descending=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(vals)), vals.numpy())
        ax.set_title("Learned prior variances (sorted)")
        ax.set_xlabel("sorted latent dim")
        ax.set_ylabel("σ_p²")
        writer.add_figure(f"{tag_prefix}/prior_variances_sorted", fig, epoch)
        plt.close(fig)

    return cb


def get_prior_vs_posterior_var_callback(num_batches: int = 3, max_points: int = 4096):
    """
    Plots PRIOR σ_p² vs average POSTERIOR σ_q² from encoder logvar,
    sorted by PRIOR variance. Rank 0 only.
    """
    def cb(val_loader, writer, model, device, epoch, tag_prefix="AE"):
        if not _writer_ok(writer):
            return
        mm = _unwrap_module(model)
        prior_logvar = getattr(mm, "prior_logvar", None)
        if prior_logvar is None:
            print("[VarViz] No learnable prior; plotting posterior variance only.")

        with torch.no_grad():
            logs = []
            count = 0
            for b_idx, (x, *_) in enumerate(val_loader):
                if b_idx >= num_batches or count >= max_points:
                    break
                x = x.to(device, non_blocking=True)
                if hasattr(mm, "encode_stats"):
                    _, logvar = mm.encode_stats(x)
                else:
                    logvar = torch.zeros(x.size(0), getattr(mm, "z_dim", 1), device=x.device)
                logs.append(logvar.detach().cpu())
                count += x.size(0)

            if not logs:
                print("[VarViz] No validation batches for posterior variance.")
                return

            post_var = torch.cat(logs, dim=0).exp().mean(dim=0)  # (d,)
            if prior_logvar is not None:
                prior_var = prior_logvar.detach().exp().cpu()
            else:
                prior_var = torch.ones_like(post_var)

            order = torch.argsort(prior_var, descending=True)
            p = prior_var[order].numpy()
            q = post_var[order].numpy()

        x_idx = np.arange(len(p))
        fig, ax = plt.subplots(figsize=(7, 3))
        w = 0.4
        ax.bar(x_idx - w/2, p, width=w, label="prior σ_p²")
        ax.bar(x_idx + w/2, q, width=w, label="avg posterior σ_q²")
        ax.set_title("Latent variances (sorted by prior)")
        ax.set_xlabel("sorted latent dim")
        ax.set_ylabel("variance")
        ax.legend()
        writer.add_figure(f"{tag_prefix}/prior_vs_posterior_variance", fig, epoch)
        plt.close(fig)

    return cb


# ────────────────────────── data gathering ──────────────────────────

@torch.no_grad()
def gather_images(loader, device: torch.device, needed: int) -> torch.Tensor:
    """Collects just enough images from a loader to reach `needed` samples (local rank view).
    Use ONLY on rank 0 (or use a non-distributed loader)."""
    xs = []
    count = 0
    it = iter(loader)
    while count < needed:
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0].to(device, non_blocking=True)
        xs.append(x)
        count += x.size(0)
    if not xs:
        raise RuntimeError("Loader is empty; cannot gather images.")
    X = torch.cat(xs, dim=0)
    if X.size(0) < needed:
        print(f"[GatherImages] Warning: requested {needed} but only found {X.size(0)}.")
    return X[:needed]


@torch.no_grad()
def encode_latents_from_loader(
    model,
    loader,
    device: torch.device,
    min_count: int = 3000,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Encode batches until we reach at least `min_count` latent points (LOCAL shard).
    Returns Z with len(Z) >= min_count if data allows (otherwise logs a warning).
    This function does NOT gather across ranks; callers can use `_all_gather_variable_batch`.
    """
    mm = _unwrap_module(model)
    zs = []
    total = 0
    it = iter(loader)
    b = 0
    with evaluation_mode(mm):
        while total < min_count:
            if max_batches is not None and b >= max_batches:
                break
            try:
                batch = next(it)
            except StopIteration:
                break
            x = batch[0].to(device, non_blocking=True)
            z = mm.encode(x).detach()
            zs.append(z)
            total += z.size(0)
            b += 1

    if not zs:
        raise RuntimeError("[EncodeLatents] No data in loader to encode.")
    Z = torch.cat(zs, dim=0)
    if Z.size(0) < min_count and _is_primary():
        print(f"[EncodeLatents] Warning: requested {min_count} latents but only encoded {Z.size(0)} on rank 0.")
    return Z  # caller can subsample further if desired


# ────────────────────────── latent corruption viz ──────────────────────────

@torch.no_grad()
def _subsample(latents: torch.Tensor, num_points: int, seed: int = 0) -> torch.Tensor:
    """Subsample rows without replacement for consistent viz."""
    n = latents.shape[0]
    if num_points >= n:
        return latents
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:num_points]
    return latents[idx]


@torch.no_grad()
def _collect_perturbed(
    latents: torch.Tensor,
    latent_sde,
    time_schedule: List[float],
) -> List[Tuple[float, np.ndarray]]:
    """
    Returns list of (t, z_t) where z_t are perturbed points at time t.
    Uses latent_sde.perturb(x, t) which internally calls marginal_prob.
    """
    out: List[Tuple[float, np.ndarray]] = []
    for t in time_schedule:
        t_tensor = torch.tensor(float(t), dtype=latents.dtype, device=latents.device)
        z_t = latent_sde.perturb(latents, t_tensor)  # (M, D)
        out.append((float(t), z_t.detach().cpu().numpy()))
    return out


def _global_axis_limits(
    sets: List[np.ndarray],
    pad_ratio: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension min/max across all sets with padding."""
    all_concat = np.concatenate(sets, axis=0)  # (K*M, D)
    mins = all_concat.min(axis=0)
    maxs = all_concat.max(axis=0)
    pad = pad_ratio * (maxs - mins + 1e-12)
    return mins - pad, maxs + pad

def _save_geodesic_comparison_grid(
    pred_stack: torch.Tensor,   # (B, T, C, H, W) in [0,1]
    gt_stack: torch.Tensor,     # (B, T, C, H, W) in [0,1]
    out_path: str,
    *,
    padding: int = 2,
    row_gap: int = 2,
    pair_gap: int = 12,
) -> str:
    """
    Save a comparison grid per pair:
      Top row: estimated geodesic frames (T columns)
      Bottom row: ground-truth frames (T columns)
    Pairs are stacked vertically with a gap between them.
    """
    pred_stack = pred_stack.detach().cpu().clamp(0.0, 1.0)
    gt_stack   = gt_stack.detach().cpu().clamp(0.0, 1.0)

    B, T, C, H, W = pred_stack.shape
    blocks = []
    for i in range(B):
        # make_grid expects (N, C, H, W)
        pred_grid = vutils.make_grid(pred_stack[i], nrow=T, padding=padding)
        gt_grid   = vutils.make_grid(gt_stack[i],   nrow=T, padding=padding)

        Cg, Hr, Wr = pred_grid.shape
        row_gap_img  = torch.zeros(Cg, row_gap, Wr)
        pair_gap_img = torch.zeros(Cg, pair_gap, Wr)

        pair_block = torch.cat([pred_grid, row_gap_img, gt_grid], dim=1)
        blocks.append(pair_block)
        if i < B - 1:
            blocks.append(pair_gap_img)

    big_img = torch.cat(blocks, dim=1)  # (C, H_total, W)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(big_img, out_path)
    return out_path

# Optional public alias without the leading underscore
def save_geodesic_comparison_grid(*args, **kwargs) -> str:
    return _save_geodesic_comparison_grid(*args, **kwargs)


def plot_latent_corruptions(
    latents: torch.Tensor,
    latent_sde,
    time_schedule: List[float],
    *,
    num_points: int = 4000,
    seed: int = 0,
    out_path: Optional[str] = None,
    figsize_unit: float = 3.0,
    dpi: int = 150,
    point_size: float = 4.0,
    alpha: float = 0.8,
) -> Dict[str, str]:
    """
    Visualize perturbed latent distributions for given diffusion times.

    Layout:
      - D=2: 1 row, C columns (smallest t -> largest t)
      - D=3: 3 rows (xy, yz, xz), C columns (smallest t -> largest t)
    """
    assert latents.ndim == 2, f"Expected latents of shape (N, D); got {tuple(latents.shape)}"
    _, D = latents.shape
    assert D in (2, 3), f"Only D=2 or D=3 supported, got D={D}"

    # Ensure plenty of latents; we still subsample for plotting density & speed.
    latents = latents.detach()
    latents = _subsample(latents, num_points=num_points, seed=seed)

    times = sorted(set(float(t) for t in time_schedule))
    if len(times) == 0:
        raise ValueError("time_schedule must contain at least one time.")

    # Perturb via the latent SDE’s corruption kernel
    perturbed = _collect_perturbed(latents, latent_sde, times)

    # Global axis limits for consistent scale across columns
    mins, maxs = _global_axis_limits([z for _, z in perturbed])

    # Figure layout
    if D == 2:
        nrows, ncols = 1, len(times)
    else:
        nrows, ncols = 3, len(times)  # xy / yz / xz

    # Figure size
    fig_w = max(1, ncols) * figsize_unit
    fig_h = max(1, nrows) * figsize_unit

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)

    def draw_scatter(ax, data: np.ndarray, dims: Tuple[int, int], title: str):
        ax.scatter(data[:, dims[0]], data[:, dims[1]], s=point_size, alpha=alpha)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(mins[dims[0]], maxs[dims[0]])
        ax.set_ylim(mins[dims[1]], maxs[dims[1]])
        ax.set_xlabel(f"dim {dims[0]}")
        ax.set_ylabel(f"dim {dims[1]}")
        ax.set_title(title)

    if D == 2:
        for j, (t, zt) in enumerate(perturbed):
            draw_scatter(axes[0, j], zt, (0, 1), title=f"t = {t:.4f}")
    else:
        for j, (t, zt) in enumerate(perturbed):
            draw_scatter(axes[0, j], zt, (0, 1), title=f"t = {t:.4f} — x–y")
            draw_scatter(axes[1, j], zt, (1, 2), title=f"t = {t:.4f} — y–z")
            draw_scatter(axes[2, j], zt, (0, 2), title=f"t = {t:.4f} — x–z")

    plt.tight_layout()

    if out_path is None:
        name = "latent_corruptions_2d.png" if D == 2 else "latent_corruptions_3d.png"
        out_path = os.path.abspath(name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"saved_path": out_path}


# ────────────────────────── decode/visualize utilities ──────────────────────────

@torch.no_grad()
def decode_and_save_grid(
    decoder,
    z_samples: torch.Tensor,
    eval_dir: str,
    writer: Optional[SummaryWriter] = None,
    *,
    grid_nrow: int | None = None,
    tag: str = "Latent2Image/Samples",
    epoch: int = 0,
) -> str:
    """
    Decode a batch of latent samples and save an image grid.
    - Preserves/restores Module training mode if decoder is an nn.Module.
    - Writes to TensorBoard if `writer` is provided.
    DDP: You should call this on rank 0 only to avoid duplicate files.
    """
    import torch.nn as nn

    dec = decoder
    if isinstance(decoder, nn.Module):
        dec = _unwrap_module(decoder)

    was_training = False
    if isinstance(dec, nn.Module):
        was_training = dec.training
        dec.eval()
        x_samples = dec(z_samples)
        if was_training:
            dec.train()
    else:
        x_samples = dec(z_samples)

    if grid_nrow is None:
        grid_nrow = int(math.sqrt(max(1, x_samples.size(0))))

    grid = vutils.make_grid(x_samples, nrow=grid_nrow, normalize=True, scale_each=True)
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "latent2image_grid.png")
    vutils.save_image(grid, out_path)

    if _writer_ok(writer):
        writer.add_image(tag, grid, epoch)

    if _is_primary():
        print(f"[Eval] Saved generated image grid → {out_path} (nrow={grid_nrow})")
    return out_path


def save_interactive_latent_geodesic_html(
    *,
    path_z_list: List[torch.Tensor],   # List[T] of (B, D) tensors
    z_bg_pert: torch.Tensor,           # (N, D) background cloud at some t_final
    out_dir: str,
    filename: str = "latent_geodesics_latent3d.html",
) -> str:
    """
    Render an interactive 3D HTML of latent geodesics (first 3 dims) overlaid on
    a background cloud of perturbed latent points. Returns the saved path.
    Gracefully no-ops (returns "") if Plotly is not available or D < 3.
    Rank 0-only recommended.
    """
    if not _is_primary():
        return ""
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plotly_plot
    except Exception as e:
        print(f"[Eval] Plotly not available ({e}); skipping interactive latent 3D viz.")
        return ""

    # Stack trajectories → (T, B, D)
    traj = torch.stack([pt.detach().cpu() for pt in path_z_list], dim=0)
    if traj.ndim == 2:  # (T, D) → add batch dim
        traj = traj.unsqueeze(1)
    T, B, D = traj.shape
    if D < 3:
        print(f"[Eval] Latent dim D={D} < 3; skipping interactive 3D viz.")
        return ""

    z_bg = z_bg_pert.detach().cpu()
    if z_bg.shape[1] < 3:
        print(f"[Eval] Background latent dim {z_bg.shape[1]} < 3; skipping interactive 3D viz.")
        return ""

    fig = go.Figure()

    # Add each trajectory as a 3D line
    for i in range(B):
        tr = traj[:, i, :3].numpy()
        fig.add_trace(go.Scatter3d(
            x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
            mode="lines",
            name=f"pair_{i+1}",
        ))

    # Overlay perturbed latent cloud
    cloud = z_bg[:, :3].numpy()
    fig.add_trace(go.Scatter3d(
        x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
        mode="markers",
        name="latent @ t_final",
        marker=dict(size=2, opacity=0.35),
    ))

    fig.update_layout(
        title="Latent geodesics (first 3 dims) with background cloud",
        scene=dict(xaxis_title="z1", yaxis_title="z2", zaxis_title="z3"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plotly_plot(fig, filename=out_path, auto_open=False)
    print(f"[Eval] Interactive latent 3D plot → {out_path}")
    return out_path

def get_generation_callback(
    *,
    sample_steps: int = 250,
    sample_count: int = 36,
    grid_nrow: int | None = None,
    tag_prefix: str = "Latent2Image",
):
    """
    Returns a callback that:
      1) samples ẑ from the latent diffusion model in *normalized* space,
      2) denormalizes with AE buffers (μ,σ),
      3) decodes and logs a grid.

    Usage:
        gen_cb = get_generation_callback(sample_steps=250, sample_count=36)
        gen_cb(writer, model, latent_model, latent_sde, device, save_dir, epoch,
               ema_latent=latent_ema)
    DDP-aware: runs only on rank 0 to avoid duplicate grids.
    """
    def cb(
        writer,
        model,                 # AutoEncoder (must expose normalize/denormalize helpers)
        latent_model,          # latent score model
        latent_sde,            # latent SDE object
        device: torch.device,
        save_dir: str,
        epoch: int,
        *,
        ema_latent=None,       # optional EMA wrapper for latent_model (has apply_shadow/restore)
        ema_ae=None            # optional EMA for AE if you want EMA decode during training
    ) -> str | None:
        if (latent_model is None) or (latent_sde is None):
            if _is_primary():
                print("[Gen] No latent diffusion model/SDE; skipping generation.")
            return None

        if not _is_primary():
            return None  # rank>0 skip

        # Lazy import to avoid hard dependency in other flows
        try:
            from utils.sampling_utils import Algorithm1
        except Exception as e:
            print(f"[Gen] Sampling utility not available ({e}); skipping.")
            return None

        # Make sure we use the raw modules if wrapped
        lm = _unwrap_module(latent_model)
        mm = _unwrap_module(model)

        # Put models into eval temporarily; optionally apply EMA weights
        from contextlib import ExitStack
        with ExitStack() as stack, torch.no_grad():
            stack.enter_context(evaluation_mode(lm))
            stack.enter_context(evaluation_mode(mm))
            if ema_latent is not None:
                ema_latent.apply_shadow()
            if ema_ae is not None:
                ema_ae.apply_shadow()

            try:
                # Score fn as in your eval script
                score_fn = lm.get_score_fn(latent_sde)

                z_dim = getattr(mm, "z_dim", None)
                if z_dim is None:
                    # fallback: many tiny MLPs store "state_size"
                    z_dim = int(getattr(lm, "state_size", 0))
                assert z_dim and z_dim > 0, "Could not infer latent dimensionality."

                # Sample in *normalized* latent space
                z_shape = (int(sample_count), int(z_dim))
                z_hat = Algorithm1(latent_sde, int(sample_steps), score_fn, z_shape, device, y=None)

                # Denormalize using AE buffers and decode
                z = mm.denormalize_latent(z_hat)
                out_path = decode_and_save_grid(
                    decoder=mm.decode,
                    z_samples=z,
                    eval_dir=save_dir,
                    writer=writer,
                    grid_nrow=grid_nrow,
                    tag=f"{tag_prefix}/Samples",
                    epoch=epoch,
                )
            finally:
                if ema_latent is not None:
                    ema_latent.restore()
                if ema_ae is not None:
                    ema_ae.restore()

        return out_path
    return cb

def save_latent_geodesic_2d(
    *,
    path_z_list: List[torch.Tensor],   # list[T] of (B, D) tensors
    z_bg_pert: torch.Tensor,           # (N, D) background cloud at some t_final
    out_dir: str,
    filename: str = "latent_geodesics_latent2d.png",
) -> str:
    """Static 2D viz of latent geodesics over a perturbed background cloud."""
    # Rank 0 only to avoid duplicate files
    if not _is_primary():
        return ""
    import matplotlib.pyplot as plt
    traj = torch.stack([pt.detach().cpu() for pt in path_z_list], dim=0)  # (T,B,D)
    if traj.ndim == 2:  # (T,D) -> (T,1,D)
        traj = traj.unsqueeze(1)
    T, B, D = traj.shape
    if D != 2:
        print(f"[Eval] D={D} != 2; skipping 2D latent viz.")
        return ""
    cloud = z_bg_pert.detach().cpu().numpy()  # (N,2)

    plt.figure(figsize=(6, 6))
    plt.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=0.3, label="latent @ t_final")

    for i in range(B):
        tr = traj[:, i, :2].numpy()
        plt.plot(tr[:, 0], tr[:, 1], linewidth=1.5, label=f"pair_{i+1}" if B <= 8 else None)
        # start/end markers
        plt.scatter(tr[0, 0], tr[0, 1], s=30, marker="o")
        plt.scatter(tr[-1, 0], tr[-1, 1], s=30, marker="x")

    plt.xlabel("z[0]"); plt.ylabel("z[1]")
    plt.gca().set_aspect("equal", adjustable="box")
    if B <= 8:
        plt.legend(loc="best", fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Geodesics] Saved 2D latent geodesics → {out_path}")
    return out_path

def _latent_cloud_at_t_with_marginal(z0: torch.Tensor, sde, t: float, seed: int = 123) -> torch.Tensor:
    """
    Produce a corrupted latent cloud at time t using the SDE marginal.
    Deterministic via a local RNG fork; works on both CPU and CUDA and on all
    torch versions (no use of generator= on randn_like).
    """
    device = z0.device
    t_tensor = torch.tensor(float(t), device=device)

    mean, std = sde.marginal_prob(z0, t_tensor)          # mean: (N,D), std: (N,)
    std = std.view(-1, *([1] * (mean.dim() - 1)))        # broadcast like in driver

    # deterministic noise without polluting global RNG
    if seed is not None:
        if z0.is_cuda:
            dev_idxs = [z0.device.index] if z0.device.index is not None else [torch.cuda.current_device()]
        else:
            dev_idxs = []
        with torch.random.fork_rng(devices=dev_idxs):    # local, deterministic
            torch.manual_seed(int(seed))
            eps = torch.randn_like(mean)
    else:
        eps = torch.randn_like(mean)

    return mean + std * eps


def save_latent_geodesics_stages_2d(
    *,
    stage_paths: list,          # [{'t': float, 'path': [γ0,...,γK]}]
    z_bg_per_stage: dict,       # {float t: (N,2) tensor}
    out_dir: str,
    filename: str = "latent_geodesics_stages_2d.png",
    order: str = "asc",         # "asc" -> small t → large t (left→right); "desc" for the opposite
    figsize_unit: float = 3.0,
    dpi: int = 150,
    point_size: float = 4.0,
    alpha: float = 0.8,
):
    # Rank 0 only to avoid duplicate files
    if not _is_primary():
        return ""
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    # Determine panel order once, then use it for both clouds and paths.
    times = [float(d['t']) for d in stage_paths]
    uniq_times = sorted(set(times))
    times = uniq_times if order == "asc" else list(reversed(uniq_times))

    # Sanity: ensure we have a cloud for every t we’ll draw
    for t in times:
        if float(t) not in z_bg_per_stage:
            raise ValueError(f"Missing background cloud for t={t:.6f}")

    # Global limits from all clouds
    all_bg = np.concatenate([z_bg_per_stage[float(t)].detach().cpu().numpy() for t in times], axis=0)
    xlim = (all_bg[:,0].min(), all_bg[:,0].max())
    ylim = (all_bg[:,1].min(), all_bg[:,1].max())
    pad_x = 0.05 * (xlim[1] - xlim[0]); pad_y = 0.05 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - pad_x, xlim[1] + pad_x)
    ylim = (ylim[0] - pad_y, ylim[1] + pad_y)

    # Group raw paths by time (exact float key)
    from collections import defaultdict
    paths_by_t = defaultdict(list)
    for item in stage_paths:
        paths_by_t[float(item['t'])].append(item['path'])

    S = len(times)
    fig, axes = plt.subplots(1, S, figsize=(S * figsize_unit, figsize_unit), dpi=dpi)
    if S == 1:
        axes = [axes]

    for ax, t in zip(axes, times):
        z_bg = z_bg_per_stage[float(t)]
        ax.scatter(z_bg[:,0].detach().cpu().numpy(),
                   z_bg[:,1].detach().cpu().numpy(),
                   s=point_size, alpha=alpha)

        # overlay all B paths for this stage
        for path in paths_by_t[float(t)]:
            Kp1 = len(path)
            B = path[0].shape[0]
            for b in range(B):
                xy = torch.stack([path[k][b] for k in range(Kp1)], dim=0).detach().cpu().numpy()
                ax.plot(xy[:,0], xy[:,1], linewidth=2.0, alpha=0.95)
                ax.scatter([xy[0,0], xy[-1,0]], [xy[0,1], xy[-1,1]], s=12)

        ax.set_title(f"t = {t:.4f}")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")

    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ───────────────── decode geodesic trajectories into grid ─────────────────

@torch.no_grad()
def decode_latent_path_and_save_grid(
    *,
    path_z_list: List[torch.Tensor],   # [T] of (B, D)
    decoder,                           # AE.decode
    eval_dir: str,
    writer: Optional[SummaryWriter] = None,
    tag: str = "LatentGeodesics/DecodedGrid",
    filename: str = "latent_geodesics_grid.png",
) -> str:
    """
    Make a B×T grid: each row is one pair's trajectory, columns are path nodes.
    Saves the PNG and logs to TensorBoard if writer is provided.
    Rank 0 only to avoid duplicate files.
    """
    if not _is_primary():
        return ""
    import torch.nn as nn
    dec = decoder
    if isinstance(decoder, nn.Module):
        dec = _unwrap_module(decoder)

    # decode each time step
    decoded_steps = [dec(z).detach().cpu() for z in path_z_list]  # list[T] of (B, C, H, W)
    T = len(decoded_steps)
    B = decoded_steps[0].size(0)
    tiles = []
    for i in range(B):
        for t in range(T):
            tiles.append(decoded_steps[t][i])

    grid = vutils.make_grid(tiles, nrow=T, normalize=True, scale_each=True)
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, filename)
    vutils.save_image(grid, out_path)

    if _writer_ok(writer):
        writer.add_image(tag, grid, 0)

    print(f"[Eval] Saved latent geodesics grid → {out_path}  (rows={B}, cols={T})")
    return out_path


# ──────────────── convenience: perturb latents at a single time ───────────

@torch.no_grad()
def perturb_latents_at_time(z: torch.Tensor, sde, t_val: float) -> torch.Tensor:
    """Apply SDE marginal noise at diffusion time t_val to latent points z."""
    t = torch.tensor(float(t_val), device=z.device, dtype=z.dtype)
    mean, std = sde.marginal_prob(z, t)
    xi = torch.randn_like(z)
    # Broadcast std across non-batch dims if needed
    std_view = std.view(-1, *([1] * (z.dim() - 1))) if std.ndim == 1 else std
    return mean + std_view * xi

# ──────────────────────────────────────────────────────────────────────────────
# Compact Isometry Evaluation (2 plots total in TensorBoard)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _take_anchor(val_loader, device, total: int, *, cache: Dict[str, torch.Tensor], key: str = "iso_anchor") -> torch.Tensor:
    """
    Deterministically take ~`total` samples from the start of val_loader and cache them.
    Keeps the anchor set fixed across epochs so curves are comparable.
    """
    if key in cache:
        return cache[key]
    xs = []
    for batch in val_loader:
        x = batch[0].to(device, non_blocking=True)
        xs.append(x)
        if sum(t.size(0) for t in xs) >= total:
            break
    X = torch.cat(xs, dim=0)[:total].contiguous()
    cache[key] = X
    return X


@torch.no_grad()
def _decoder_eigs_and_G(model: nn.Module, x: torch.Tensor, lam: float = 0.0):
    """
    Return eigenvalues (B,d) and G (B,d,d) for decoder pullback metric G = J_d^T J_d.
    Uses the same batched Jacobian builder as curvature code.
    """
    from loss.curvature import _build_J_and_G_batched
    z = model.encode(x).to(torch.float32)
    _, G, _ = _build_J_and_G_batched(model.decode, z, lam=float(lam))
    Gs = 0.5 * (G + G.transpose(-1, -2))
    evals = torch.linalg.eigvalsh(Gs.to(torch.float64)).to(torch.float32)  # (B,d)
    return evals, Gs


@torch.no_grad()
def _encoder_eigs_and_G(model: nn.Module, x: torch.Tensor):
    """
    Return eigenvalues (B,d) and G_e (B,d,d) for encoder metric G_e = J_e J_e^T.
    Uses per-sample jacrev (latent dim is small), still chunked by B_curv in caller.
    """
    from torch.func import jacrev
    B, C, H, W = x.shape
    evals_list, Gs_list = [], []

    def f_single(x_flat: torch.Tensor) -> torch.Tensor:
        x_img = x_flat.view(1, C, H, W)
        return model.encode(x_img).squeeze(0)  # (d,)

    for i in range(B):
        xi = x[i].reshape(-1).to(torch.float32)
        J_e = jacrev(f_single)(xi)                 # (d, D)
        G   = J_e @ J_e.transpose(0, 1)            # (d, d)
        Gs  = 0.5 * (G + G.transpose(0, 1))
        ev  = torch.linalg.eigvalsh(Gs.to(torch.float64)).to(torch.float32)
        evals_list.append(ev.unsqueeze(0)); Gs_list.append(Gs.unsqueeze(0))
    return torch.cat(evals_list, 0), torch.cat(Gs_list, 0)


@torch.no_grad()
def _isometry_distances(evals: torch.Tensor, G: torch.Tensor, eps: float = 1e-12):
    """
    evals: (B,d), G: (B,d,d). Return per-sample scalars:
      dR = ||logm(G)||_F = sqrt(sum (log λ_i)^2)
      dF = ||G - I||_F / sqrt(d)
    """
    dR = torch.sqrt(torch.sum(torch.log(evals.clamp_min(eps))**2, dim=-1))  # (B,)
    B, d = G.shape[0], G.shape[-1]
    I = torch.eye(d, device=G.device, dtype=G.dtype).expand(B, d, d)
    dF = torch.linalg.norm(G - I, ord='fro', dim=(1, 2)) / (d ** 0.5)       # (B,)
    return dR, dF


@torch.no_grad()
def _cond_from_evals(evals: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample condition number κ = λ_max / λ_min."""
    lam_min = evals.min(dim=-1).values.clamp_min(eps)
    lam_max = evals.max(dim=-1).values
    return lam_max / lam_min


@torch.no_grad()
def log_isometry_distance_batched(
    val_loader,
    writer,
    model: nn.Module,
    device: torch.device,
    epoch: int,
    cfg,
    *,
    modes: Tuple[str, ...] = ("decoder", "encoder"),
    total_points: int = 200,
    lam_decoder: float = 0.0,
    tag_prefix: str = "AE",
    _cache: Dict[str, torch.Tensor] = {},
):
    """
    Compute isometry deviation on exactly `total_points`, processed in chunks of size
    cfg.loss.curvature.B_curv (same memory as MECAE). Logs only three means per side:
      dR_mean, dF_mean, log10(cond)_mean → fits into TWO multiline plots total.
    """
    # preserve training flag; evaluate for consistency
    was_training = model.training
    model.eval()

    # geometry mini-batch size
    try:
        B_curv = int(getattr(getattr(cfg.loss, "curvature"), "B_curv", 8))
    except Exception:
        B_curv = 8
    B_curv = max(1, B_curv)

    # fixed anchor set across epochs
    X = _take_anchor(val_loader, device, total=total_points, cache=_cache, key="iso_anchor")

    def _summ(label: str, dR_all: torch.Tensor, dF_all: torch.Tensor, evals_all: torch.Tensor):
        # condition number (per sample), then log10 for better scale
        cond = _cond_from_evals(evals_all)
        logcond = torch.log10(cond.clamp_min(1.0))  # κ≥1

        # means only → two compact plots (decoder and encoder)
        writer.add_scalar(f"{tag_prefix}/reg/{label}_dR_mean",       float(dR_all.mean().item()),   epoch)
        writer.add_scalar(f"{tag_prefix}/reg/{label}_dF_mean",       float(dF_all.mean().item()),   epoch)
        writer.add_scalar(f"{tag_prefix}/reg/{label}_logcond_mean",  float(logcond.mean().item()),  epoch)

        # Optional deep-dive histogram (kept commented to avoid extra tiles)
        # writer.add_histogram(f"{tag_prefix}/reg/{label}_log_eigs",
        #                      torch.log(evals_all.clamp_min(1e-12)), epoch)

    if "decoder" in modes:
        dR_list, dF_list, ev_list = [], [], []
        for i in range(0, X.size(0), B_curv):
            x_chunk = X[i:i + B_curv]
            ev, G = _decoder_eigs_and_G(model, x_chunk, lam=lam_decoder)
            dR, dF = _isometry_distances(ev, G)
            dR_list.append(dR); dF_list.append(dF); ev_list.append(ev)
        _summ("dec", torch.cat(dR_list, 0), torch.cat(dF_list, 0), torch.cat(ev_list, 0))

    if "encoder" in modes:
        dR_list, dF_list, ev_list = [], [], []
        for i in range(0, X.size(0), B_curv):
            x_chunk = X[i:i + B_curv]
            ev, G = _encoder_eigs_and_G(model, x_chunk)
            dR, dF = _isometry_distances(ev, G)
            dR_list.append(dR); dF_list.append(dF); ev_list.append(ev)
        _summ("enc", torch.cat(dR_list, 0), torch.cat(dF_list, 0), torch.cat(ev_list, 0))

    # restore training flag
    model.train(was_training)


def add_isometry_compact_layout(writer, ddp_rank_fn=lambda: 0, tag_prefix: str = "AE"):
    """
    Register a 2-plot compact layout once (call after creating SummaryWriter on rank 0).
    """
    if ddp_rank_fn() != 0:
        return
    layout = {
        "Isometry (compact)": {
            "Decoder dR / dF / log10(cond)": [
                "Multiline",
                [f"{tag_prefix}/reg/dec_dR_mean",
                 f"{tag_prefix}/reg/dec_dF_mean",
                 f"{tag_prefix}/reg/dec_logcond_mean"]
            ],
            "Encoder dR / dF / log10(cond)": [
                "Multiline",
                [f"{tag_prefix}/reg/enc_dR_mean",
                 f"{tag_prefix}/reg/enc_dF_mean",
                 f"{tag_prefix}/reg/enc_logcond_mean"]
            ],
        }
    }
    writer.add_custom_scalars(layout)

# --- RAMP HELPER FUNCTIONS ---

def _clip01(x: float) -> float:
    """Clips a value to the [0, 1] range."""
    return max(0.0, min(1.0, x))

def ramp_value(progress: float, kind: str = "cosine") -> float:
    """
    Calculates a scalar in [0, 1] for a given progress in [0, 1].

    Args:
        progress (float): The input progress, typically from 0 to 1.
        kind (str): The shape of the ramp. One of 'cosine', 'smoothstep', 'linear'.

    Returns:
        float: The ramped value in [0, 1].
    """
    p = _clip01(progress)
    if kind == "linear":
        return p
    if kind == "smoothstep":
        return p * p * (3.0 - 2.0 * p)
    # Default to cosine
    return 0.5 * (1.0 - math.cos(math.pi * p))

def make_weight_ramp(
    base_weight: float,
    cur_epoch: int,
    cur_iter_in_epoch: int,
    steps_per_epoch: int,
    ramp_cfg,
    kind: str = "cosine"
) -> float:
    """
    Smoothly ramps a weight from 0 to a base_weight based on a schedule.
    """
    if base_weight <= 0.0:
        return 0.0

    start_epoch = int(getattr(ramp_cfg, "start_epoch", 0))
    end_epoch = int(getattr(ramp_cfg, "end_epoch", 1))

    if cur_epoch < start_epoch:
        return 0.0

    total_ramp_steps = max(1, int((end_epoch - start_epoch) * steps_per_epoch))
    current_step_in_ramp = max(0, int((cur_epoch - start_epoch) * steps_per_epoch + cur_iter_in_epoch))

    progress = current_step_in_ramp / total_ramp_steps
    val01 = ramp_value(progress, kind=kind)

    return base_weight * val01