"""Airborne point cloud -> 2.5D terrain mesh + albedo texture + normal maps -> GLB.

Airborne LiDAR is essentially 2.5D (one height per xy), so instead of Poisson / ball-pivoting
we rasterize to a regular height grid. That gives a watertight, manifold-free-of-holes surface
that is cheap and stable for physics, plus pixel-aligned texture / normal maps.

Height modes:
    dsm   : per-cell max (buildings + tree canopy become solid). Works without classification.
    dtm   : ground points only (cls==2), holes filled. Bare-earth.
    hybrid: DTM everywhere, building cells (cls==6) raised to their roof height,
            optional tall vegetation as obstacles. Needs ASPRS classification.

Output frame: local ENU-like metric frame, origin = (bbox xmin, bbox ymin, ground z min),
Z up. The GLB itself is written in glTF Y-up convention (x, z, -y); `load_glb_zup` undoes it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import ndimage

GROUND, BUILDING = 2, 6
VEG = (3, 4, 5)


@dataclass
class TerrainMeta:
    origin: list            # [x, y, z] in source CRS subtracted from points
    crs: str
    mesh_res: float         # m per mesh grid cell
    tex_res: float          # m per texture pixel
    size_xy: list           # [sx, sy] metres
    grid_shape: list        # [ny, nx] mesh vertices
    mode: str
    z_range: list


# --------------------------------------------------------------------------- rasterization
def _bin(xy, res, shape):
    ix = np.clip((xy[:, 0] / res).astype(np.int64), 0, shape[1] - 1)
    iy = np.clip((xy[:, 1] / res).astype(np.int64), 0, shape[0] - 1)
    return iy * shape[1] + ix


def _reduce(lin, val, n, how):
    out = np.full(n, np.nan, np.float64)
    if how == "max":
        o = np.full(n, -np.inf); np.maximum.at(o, lin, val); m = np.isfinite(o)
    elif how == "min":
        o = np.full(n, np.inf); np.minimum.at(o, lin, val); m = np.isfinite(o)
    elif how == "mean":
        s = np.bincount(lin, val, n); c = np.bincount(lin, minlength=n); m = c > 0
        o = np.where(m, s / np.maximum(c, 1), np.nan)
    else:
        raise ValueError(how)
    out[m] = o[m]
    return out


def _fill_nan(a):
    """Nearest-neighbour hole filling followed by a light blur only inside the holes."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    if mask.all():
        return np.zeros_like(a)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    f = a[tuple(idx)]
    sm = ndimage.uniform_filter(f, 5)
    f[mask] = sm[mask]
    return f


def rasterize_height(xyz, cls, res, shape, mode, veg_min_height=2.0, veg_as_obstacle=False):
    n = shape[0] * shape[1]
    lin = _bin(xyz[:, :2], res, shape)
    z = xyz[:, 2]
    has_cls = cls is not None and (cls == GROUND).any()
    if mode == "auto":
        mode = "hybrid" if has_cls else "dsm"
    if mode in ("dtm", "hybrid") and not has_cls:
        raise ValueError(f"mode={mode} needs ground classification (cls==2); use dsm")

    if mode == "dsm":
        h = _reduce(lin, z, n, "max").reshape(shape)
        # despeckle: isolated high returns (birds, wires) -> median
        h = _fill_nan(h)
        h = ndimage.median_filter(h, 3)
        return h, mode

    g = cls == GROUND
    ground = _fill_nan(_reduce(lin[g], z[g], n, "mean").reshape(shape))
    ground = ndimage.median_filter(ground, 3)
    if mode == "dtm":
        return ground, mode

    h = ground.copy()
    b = cls == BUILDING
    if b.any():
        roof = _reduce(lin[b], z[b], n, "max").reshape(shape)
        bm = ~np.isnan(roof)
        # close small gaps in roofs so walls are continuous
        bm_closed = ndimage.binary_closing(bm, iterations=2)
        roof = np.where(bm, roof, np.nan)
        roof_f = _fill_nan(roof)
        h = np.where(bm_closed, np.maximum(roof_f, ground), h)
    if veg_as_obstacle:
        v = np.isin(cls, VEG)
        if v.any():
            top = _reduce(lin[v], z[v], n, "max").reshape(shape)
            tall = (top - ground) > veg_min_height
            h = np.where(tall, top, h)
    return h, mode


def rasterize_color(xyz, rgb, res, shape):
    n = shape[0] * shape[1]
    lin = _bin(xyz[:, :2], res, shape)
    img = np.stack([_fill_nan(_reduce(lin, rgb[:, c].astype(np.float64), n, "mean").reshape(shape))
                    for c in range(3)], -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def height_normals(h, res):
    """World-space (Z-up) unit normals of a heightfield h[y, x]."""
    gy, gx = np.gradient(h, res)
    nrm = np.stack([-gx, -gy, np.ones_like(h)], -1)
    return nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)


def encode_normal(nrm):
    return np.clip((nrm * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- mesh
def grid_mesh(h, res):
    ny, nx = h.shape
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X.ravel(), Y.ravel(), h.ravel()], 1)
    i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
    i = i.ravel()
    a, b, c, d = i, i + 1, i + nx, i + nx + 1
    F = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])  # CCW seen from +Z
    U = np.stack([V[:, 0] / (nx * res), V[:, 1] / (ny * res)], 1)
    return V, F, U


ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float64)


def build_terrain(cloud: dict, out_dir: str | Path, name: str, mesh_res=0.25, tex_res=0.1,
                  mode="auto", crs="NAD83 / UTM zone 18N", veg_as_obstacle=False,
                  bbox_xy=None):
    import trimesh
    from PIL import Image

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xyz, rgb, cls = cloud["xyz"], cloud["rgb"], cloud.get("cls")
    if bbox_xy is not None:
        x0, y0, x1, y1 = bbox_xy
        m = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
        xyz, rgb = xyz[m], rgb[m]
        cls = cls[m] if cls is not None else None
    else:
        x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
        x1, y1 = xyz[:, 0].max(), xyz[:, 1].max()
    if cls is not None and not cls.any():
        cls = None

    # robust z origin: low percentile of ground (or all) points
    zg = xyz[cls == GROUND, 2] if cls is not None and (cls == GROUND).any() else xyz[:, 2]
    origin = np.array([x0, y0, np.percentile(zg, 0.5)])
    p = xyz - origin
    sx, sy = x1 - x0, y1 - y0

    mshape = (max(2, int(np.ceil(sy / mesh_res))), max(2, int(np.ceil(sx / mesh_res))))
    tshape = (int(np.ceil(sy / tex_res)), int(np.ceil(sx / tex_res)))

    h, mode = rasterize_height(p, cls, mesh_res, mshape, mode, veg_as_obstacle=veg_as_obstacle)
    h_tex, _ = rasterize_height(p, cls, tex_res, tshape, mode, veg_as_obstacle=veg_as_obstacle)
    albedo = rasterize_color(p, rgb, tex_res, tshape)

    return write_terrain(out_dir, name, h, h_tex, albedo, origin, mesh_res, tex_res, mode, crs,
                         float(sx), float(sy))


def write_terrain(out_dir, name, h, h_tex, albedo, origin, mesh_res, tex_res, mode, crs, sx, sy,
                  extra_meta=None):
    """Write GLB + albedo / normal maps + height grid + metadata for a height grid h[y, x]."""
    import trimesh
    from PIL import Image

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mshape, tshape = h.shape, h_tex.shape
    # world-space normals of the fine heightfield (for planners / traversability)
    n_world = height_normals(h_tex, tex_res)
    # tangent-space detail normal map: fine relief relative to the coarse mesh surface
    h_coarse_up = ndimage.zoom(h, (tshape[0] / mshape[0], tshape[1] / mshape[1]), order=1)
    h_coarse_up = h_coarse_up[:tshape[0], :tshape[1]]
    pad = [(0, tshape[0] - h_coarse_up.shape[0]), (0, tshape[1] - h_coarse_up.shape[1])]
    h_coarse_up = np.pad(h_coarse_up, pad, mode="edge")
    n_tan = height_normals(h_tex - h_coarse_up, tex_res)

    # images: row 0 = top (max y) so that UV v=0 at bottom matches glTF after flip
    flip = lambda a: a[::-1]
    Image.fromarray(flip(albedo)).save(out_dir / f"{name}_albedo.png")
    Image.fromarray(flip(encode_normal(n_tan))).save(out_dir / f"{name}_normal_tangent.png")
    Image.fromarray(flip(encode_normal(n_world))).save(out_dir / f"{name}_normal_world.png")
    np.save(out_dir / f"{name}_height.npy", h.astype(np.float32))

    V, F, U = grid_mesh(h, mesh_res)
    mat = trimesh.visual.material.PBRMaterial(
        name=f"{name}_mat",
        baseColorTexture=Image.fromarray(flip(albedo)),
        normalTexture=Image.fromarray(flip(encode_normal(n_tan))),
        metallicFactor=0.0, roughnessFactor=0.9, doubleSided=False)
    mesh = trimesh.Trimesh(V, F, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=U, material=mat))
    mesh_y = mesh.copy(); mesh_y.apply_transform(ZUP_TO_YUP)
    glb = out_dir / f"{name}.glb"
    mesh_y.export(glb)

    meta = TerrainMeta(origin=np.asarray(origin).tolist(), crs=crs, mesh_res=mesh_res, tex_res=tex_res,
                       size_xy=[float(sx), float(sy)], grid_shape=list(mshape), mode=mode,
                       z_range=[float(h.min()), float(h.max())])
    d = asdict(meta)
    d.update(extra_meta or {})
    (out_dir / f"{name}.json").write_text(json.dumps(d, indent=2))
    return glb, meta, mesh


def load_glb_zup(path):
    """Load a terrain GLB written by build_terrain back into Z-up (Isaac Gym) frame."""
    import trimesh
    scene = trimesh.load(path, force="scene")
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    mesh.apply_transform(np.linalg.inv(ZUP_TO_YUP))
    return mesh
