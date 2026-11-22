"""Above-ground 3D surface reconstruction from the full-density LiDAR tiles (visual mesh).

The 2.5D terrain (las_terrain.py) stays the collision / planning surface. Here everything more
than `min_h` above the ground is meshed in true 3D so walls, trunks, railings and overhangs
look right from a robot camera:

    voxel-downsample -> normals (PCA) oriented towards the sensor that measured the point
    (gps_time -> INS pose) -> screened Poisson per tile (with margin) -> density trim ->
    crop to tile core -> quadric decimation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def bilinear(grid, res, xy):
    """Sample grid[y, x] (cell centres at (i+0.5)*res) at metric xy."""
    ny, nx = grid.shape
    fx = np.clip(xy[:, 0] / res - 0.5, 0, nx - 1.001); fy = np.clip(xy[:, 1] / res - 0.5, 0, ny - 1.001)
    x0 = fx.astype(int); y0 = fy.astype(int); ax = fx - x0; ay = fy - y0
    return (grid[y0, x0] * (1 - ax) * (1 - ay) + grid[y0, x0 + 1] * ax * (1 - ay)
            + grid[y0 + 1, x0] * (1 - ax) * ay + grid[y0 + 1, x0 + 1] * ax * ay)


def voxel_reduce(xyz, voxel, extra=None):
    """Mean position (and mean of extra columns) per occupied voxel. Returns (xyz, extra, count)."""
    key = np.floor(xyz / voxel).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    u, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    out = np.stack([np.bincount(inv, xyz[:, k], len(u)) for k in range(3)], 1) / cnt[:, None]
    ex = None
    if extra is not None:
        ex = np.stack([np.bincount(inv, extra[:, k], len(u)) for k in range(extra.shape[1])], 1) / cnt[:, None]
    return out, ex, cnt


def tile_structure_points(tiles, session, ground, ground_res, bounds, margin=1.5, min_h=0.25,
                          max_h=40.0, voxel=0.05, min_count=2):
    """Voxel-reduced above-ground points (+ mean sensor position) inside bounds + margin."""
    x0, y0, x1, y1 = bounds
    acc_x, acc_t = [], []
    T = tiles.tile
    for iy in range(max(0, int((y0 - margin) // T)), min(tiles.ny, int((y1 + margin) // T) + 1)):
        for ix in range(max(0, int((x0 - margin) // T)), min(tiles.nx, int((x1 + margin) // T) + 1)):
            d = tiles.load(ix, iy, ("xyz", "gps_time"))
            xyz = np.asarray(d["xyz"])
            if not len(xyz):
                continue
            m = (xyz[:, 0] >= x0 - margin) & (xyz[:, 0] < x1 + margin) & \
                (xyz[:, 1] >= y0 - margin) & (xyz[:, 1] < y1 + margin)
            xyz = xyz[m].astype(np.float64)
            hag = xyz[:, 2] - bilinear(ground, ground_res, xyz[:, :2])
            k = (hag > min_h) & (hag < max_h)
            acc_x.append(xyz[k]); acc_t.append(np.asarray(d["gps_time"])[m][k])
    if not acc_x:
        return None
    xyz = np.concatenate(acc_x); t = np.concatenate(acc_t)
    if len(xyz) < 500:
        return None
    sensor, _ = session.pose(t)
    pts, sen, cnt = voxel_reduce(xyz, voxel, sensor)
    k = cnt >= min_count
    return pts[k], sen[k]


def reconstruct_tile(pts, sensor, bounds, depth=9, trim_q=0.12, max_faces=60000, min_h_keep=None):
    """Poisson surface for one tile. Returns (V, F) cropped to bounds, or None."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    n = np.asarray(pc.normals)
    flip = np.einsum("ij,ij->i", n, sensor - pts) < 0
    n[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(n)
    # isolated specks produce Poisson blobs
    pc, _ = pc.remove_radius_outlier(nb_points=6, radius=0.2)
    if len(pc.points) < 500:
        return None
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=depth, scale=1.05, linear_fit=True)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_q))
    # keep only surface close to actual points (Poisson closes holes with bubbles)
    tree = o3d.geometry.KDTreeFlann(pc)
    V = np.asarray(mesh.vertices)
    far = np.array([tree.search_knn_vector_3d(v, 1)[2][0] > 0.12 ** 2 for v in V])
    mesh.remove_vertices_by_mask(far)
    x0, y0, x1, y1 = bounds
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
    if len(F) == 0:
        return None
    c = V[F].mean(1)
    keep = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) < 20:
        return None
    return np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
