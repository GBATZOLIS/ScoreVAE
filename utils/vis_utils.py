# utils/vis_utils.py
# Simple, fast meshing for 3D latent clouds + interactive HTML outputs.

import os
import math
import importlib
from typing import Optional, Tuple, Dict

import numpy as np

# Optional Plotly; fall back gracefully if missing.
try:
    import plotly.graph_objects as go
    from plotly.offline import plot as _plotly_plot
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ---- DDP helpers (standalone; no circular deps) ----
import torch
import torch.distributed as dist

def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def _is_primary() -> bool:
    return (not _dist_is_initialized()) or (dist.get_rank() == 0)

def _writer_ok(writer) -> bool:
    try:
        return _is_primary() and hasattr(writer, "add_figure")
    except Exception:
        return False

# ---- Utilities ----
def _estimate_point_scale(z: np.ndarray, k: int = 10) -> float:
    """Median distance to the k-th NN (robust local length scale)."""
    M = min(len(z), 5000)
    if len(z) > M:
        rs = np.random.RandomState(0)
        z = z[rs.choice(len(z), M, replace=False)]
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    kth = np.partition(d, k, axis=1)[:, k]
    return float(np.median(kth))

def _import_o3d_core():
    """Import Open3D core without pulling open3d.ml (sklearn/scipy)."""
    try:
        return importlib.import_module("open3d.cpu.pybind")  # CPU wheel core
    except Exception:
        return importlib.import_module("open3d")  # fallback

# ---- Interactive plots ----
def _save_interactive_scatter_html(z: np.ndarray, out_dir: str, tag: str, epoch: int) -> Optional[str]:
    if not _HAS_PLOTLY or z.shape[1] < 3 or not _is_primary():
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig = go.Figure(data=[go.Scatter3d(
        x=z[:, 0], y=z[:, 1], z=z[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.6)
    )])
    fig.update_layout(
        title=f"{tag} scatter — epoch {epoch}",
        scene=dict(xaxis_title="z[0]", yaxis_title="z[1]", zaxis_title="z[2]"),
        margin=dict(l=0, r=0, b=0, t=36),
    )
    path = os.path.join(out_dir, f"{tag.replace('/','_')}_scatter_epoch_{epoch}.html")
    try:
        _plotly_plot(fig, filename=path, auto_open=False)
        print(f"[Viz] Saved interactive scatter → {path}")
        return path
    except Exception as e:
        print(f"[Viz] Plotly scatter save failed: {e}")
        return None

def _save_mesh_html(vertices: np.ndarray, faces: np.ndarray, out_dir: str, tag: str, epoch: int) -> Optional[str]:
    if not _HAS_PLOTLY or not _is_primary():
        return None
    os.makedirs(out_dir, exist_ok=True)
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    mesh = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=i, j=j, k=k, opacity=1.0, flatshading=True, name="mesh")
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=f"{tag} mesh — epoch {epoch}",
        scene=dict(xaxis_title="z[0]", yaxis_title="z[1]", zaxis_title="z[2]"),
        margin=dict(l=0, r=0, b=0, t=36),
    )
    path = os.path.join(out_dir, f"{tag.replace('/','_')}_mesh_epoch_{epoch}.html")
    try:
        _plotly_plot(fig, filename=path, auto_open=False)
        print(f"[Viz] Saved interactive mesh → {path}")
        return path
    except Exception as e:
        print(f"[Viz] Plotly mesh save failed: {e}")
        return None

def _save_mesh_with_points_html(vertices: np.ndarray, faces: np.ndarray, out_dir: str, tag: str, epoch: int,
                                scatter_xyz: Optional[np.ndarray] = None) -> Optional[str]:
    if not _HAS_PLOTLY or not _is_primary():
        return None
    os.makedirs(out_dir, exist_ok=True)
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    traces = [go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                        i=i, j=j, k=k, opacity=1.0, flatshading=True, name="mesh")]
    if (scatter_xyz is not None) and (scatter_xyz.shape[1] >= 3):
        traces.append(go.Scatter3d(
            x=scatter_xyz[:, 0], y=scatter_xyz[:, 1], z=scatter_xyz[:, 2],
            mode="markers", marker=dict(size=2, opacity=0.35), name="points"
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{tag} mesh+points — epoch {epoch}",
        scene=dict(xaxis_title="z[0]", yaxis_title="z[1]", zaxis_title="z[2]"),
        margin=dict(l=0, r=0, b=0, t=36),
        showlegend=True,
    )
    path = os.path.join(out_dir, f"{tag.replace('/','_')}_mesh_points_epoch_{epoch}.html")
    try:
        _plotly_plot(fig, filename=path, auto_open=False)
        print(f"[Viz] Saved interactive mesh+points → {path}")
        return path
    except Exception as e:
        print(f"[Viz] Plotly mesh+points save failed: {e}")
        return None

# ---- Disk outputs ----
def _write_ply_ascii(vertices: np.ndarray, faces: np.ndarray, path: str):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

def _save_mesh_png_and_tb(vertices: np.ndarray, faces: np.ndarray, writer, tag: str, epoch: int, view=(28, 52)) -> Optional[str]:
    if not _is_primary():
        return None
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces, linewidth=0.1, antialiased=True, alpha=1.0)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]"); ax.set_zlabel("z[2]")
    ax.set_title(f"{tag} — epoch {epoch}")
    if _writer_ok(writer):
        writer.add_figure(f"{tag}/mesh", fig, epoch)
    try:
        base = getattr(writer, "log_dir", ".")
    except Exception:
        base = "."
    out_dir = os.path.join(base, "meshes")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag.replace('/','_')}_mesh_epoch_{epoch}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as _plt  # ensure close even if TB add_figure fails
    _plt.close(fig)
    print(f"[Viz] Saved static mesh snapshot → {out_path}")
    return out_path

# ---- Meshing (fast BPA + MC fallback) ----
def _reconstruct_bpa_fast(z: np.ndarray, target_faces: int, scale: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        o3d = _import_o3d_core()
    except Exception as e:
        print(f"[Mesh] Open3D core not available for BPA: {e}")
        return None

    max_pts = 70000
    if len(z) > max_pts:
        rs = np.random.RandomState(0)
        z = z[rs.choice(len(z), max_pts, replace=False)]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(z))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    try:
        pcd.orient_normals_consistent_tangent_plane(60)
    except Exception:
        pass

    radii = [0.9*scale, 1.3*scale, 1.8*scale]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if target_faces is not None and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    if len(F) == 0:
        return None
    return V, F

def _reconstruct_mc(z: np.ndarray, scale: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        print(f"[Mesh] scikit-image not available for marching cubes: {e}")
        return None

    P = z
    N = 96 if len(P) <= 3000 else 128
    mins = P.min(0) - 3*scale
    maxs = P.max(0) + 3*scale
    xs = np.linspace(mins[0], maxs[0], N)
    ys = np.linspace(mins[1], maxs[1], N)
    zs = np.linspace(mins[2], maxs[2], N)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    h2 = (1.2*scale)**2
    dens = np.zeros(len(grid), dtype=np.float64)
    B = 20000
    for i in range(0, len(grid), B):
        G = grid[i:i+B]
        d2 = np.sum((G[:, None, :] - P[None, :, :])**2, axis=2)
        dens[i:i+B] = np.exp(-d2/(2*h2)).sum(axis=1)
    dens = dens.reshape(N, N, N)

    isoval = np.percentile(dens, 60)
    verts, faces, _, _ = marching_cubes(
        dens, level=isoval, spacing=(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    )
    verts[:, 0] += mins[0]; verts[:, 1] += mins[1]; verts[:, 2] += mins[2]
    faces = faces.astype(np.int32)
    if len(faces) == 0:
        return None
    return verts, faces

def build_fast_surface_mesh(z_all: np.ndarray, target_faces: int = 20000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if z_all.ndim != 2 or z_all.shape[1] < 3 or len(z_all) < 100:
        return None
    z = z_all.astype(np.float64, copy=False)
    scale = _estimate_point_scale(z, k=10)
    mesh = _reconstruct_bpa_fast(z, target_faces, scale)
    if mesh is None:
        mesh = _reconstruct_mc(z, scale)
    return mesh

# ---- Public entry point ----
def save_latent_mesh_artifacts(
    z_norm: np.ndarray,
    *,
    epoch: int,
    writer,
    save_root: str,
    tag_prefix: str = "AE/latent_surface",
    target_faces: int = 20000,
) -> Dict[str, str]:
    """
    Given normalized latent points (N,3+), save:
      • interactive scatter HTML
      • reconstructed mesh PLY + PNG (logged) + interactive mesh HTML
      • interactive mesh+points HTML
    """
    paths: Dict[str, str] = {}
    if not _is_primary() or z_norm.shape[1] < 3:
        return paths

    html_dir = os.path.join(save_root, "interactive_latent")
    p = _save_interactive_scatter_html(z_norm, html_dir, tag=tag_prefix, epoch=epoch)
    if p: paths["scatter_html"] = p

    mesh = build_fast_surface_mesh(z_norm, target_faces=target_faces)
    if mesh is None:
        print("[Mesh] Reconstruction failed or unavailable backends.")
        return paths

    V, F = mesh
    mesh_dir = os.path.join(save_root, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    ply_path = os.path.join(mesh_dir, f"{tag_prefix.replace('/','_')}_epoch_{epoch}.ply")
    try:
        _write_ply_ascii(V, F, ply_path)
        paths["mesh_ply"] = ply_path
        print(f"[Mesh] Saved PLY → {ply_path}")
    except Exception as e:
        print(f"[Mesh] PLY save failed: {e}")

    png_path = _save_mesh_png_and_tb(V, F, writer, tag=tag_prefix, epoch=epoch)
    if png_path: paths["mesh_png"] = png_path

    p2 = _save_mesh_html(V, F, html_dir, tag=tag_prefix, epoch=epoch)
    if p2: paths["mesh_html"] = p2

    p3 = _save_mesh_with_points_html(V, F, html_dir, tag=tag_prefix, epoch=epoch, scatter_xyz=z_norm)
    if p3: paths["mesh_points_html"] = p3

    return paths

def save_latent_scatter_artifacts(
    z_norm: np.ndarray,
    *,
    epoch: int,
    save_root: str,
    tag_prefix: str = "AE/latent_scatter",
    dims=(0, 1, 2),
    save_points: bool = True,
    save_html: bool = True,
) -> dict:
    """
    Save an interactive scatter HTML (first 3 dims by default) and the exact
    normalized latent array to disk for offline work. Returns a dict of paths.

    - z_norm: (N, D) normalized latents (np.ndarray)
    - dims: which dims to show in the interactive figure (len 2 or 3)
    """
    paths = {}
    if not _is_primary() or z_norm.ndim != 2 or z_norm.shape[0] == 0:
        return paths

    # ensure float32 on disk to keep files small
    z_norm = np.asarray(z_norm, dtype=np.float32)

    # 0) save raw points
    if save_points:
        pts_dir = os.path.join(save_root, "latent_points")
        os.makedirs(pts_dir, exist_ok=True)
        base = f"{tag_prefix.replace('/','_')}_epoch_{epoch}"
        npz_path = os.path.join(pts_dir, base + ".npz")
        # store full D for offline meshing
        np.savez_compressed(npz_path, z=z_norm)
        paths["points_npz"] = npz_path
        print(f"[Viz] Saved normalized latent points → {npz_path}")

    # 1) interactive scatter (2D or 3D), only if requested and plotly available
    if save_html:
        dsel = [d for d in dims if 0 <= d < z_norm.shape[1]]
        if len(dsel) >= 2:
            z_plot = z_norm[:, dsel[:3]]  # plot up to 3 dims
            title = f"{tag_prefix} scatter (dims {dsel[:3]})"
            html_dir = os.path.join(save_root, "interactive_latent")
            if z_plot.shape[1] == 3:
                # reuse your internal helper
                p = _save_interactive_scatter_html(z_plot, html_dir, tag=tag_prefix, epoch=epoch)
            else:
                # 2D fallback (no dependency on mesh code)
                if not _HAS_PLOTLY:
                    p = None
                else:
                    os.makedirs(html_dir, exist_ok=True)
                    import plotly.graph_objects as go
                    from plotly.offline import plot as _plotly_plot
                    fig = go.Figure(data=[go.Scatter(
                        x=z_plot[:, 0], y=z_plot[:, 1], mode="markers",
                        marker=dict(size=4, opacity=0.7)
                    )])
                    fig.update_layout(title=title, xaxis_title=f"z[{dsel[0]}]", yaxis_title=f"z[{dsel[1]}]")
                    p = os.path.join(html_dir, f"{tag_prefix.replace('/','_')}_scatter2d_epoch_{epoch}.html")
                    try:
                        _plotly_plot(fig, filename=p, auto_open=False)
                        print(f"[Viz] Saved interactive 2D scatter → {p}")
                    except Exception as e:
                        print(f"[Viz] 2D scatter save failed: {e}")
                        p = None
            if p:
                paths["scatter_html"] = p

    return paths

