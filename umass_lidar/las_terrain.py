"""LAS -> drivable terrain GLB, streaming (works for multi-GB files, no classification needed).

Unlike the airborne DSM path in mesh.py, this targets dense scans that also see *under*
canopies / overhangs (drone + terrestrial). Per xy cell we build a z-occupancy column and:

    ground   = lowest voxel holding >= 8% of the column's densest voxel (rejects below-ground
               reflection ghosts), refined to the mean z of points around it, then
               median-filtered and morphologically opened
    obstacle = cells with enough points inside the robot band [ground+band_lo, ground+band_hi]
               -> raised to the column top (walls, trunks, poles, cars)
    roof     = cells with a high, locally planar top and nothing in the robot band
               (building interiors seen only from above) -> raised to the top
    else     = ground (so the robot can drive under tree canopy)

Two streaming passes over the file:  (1) voxel occupancy,  (2) ground refinement + color.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from .mesh import _fill_nan, write_terrain

CHUNK = 20_000_000


def las_info(path):
    import laspy
    with laspy.open(str(path)) as f:
        h = f.header
        return dict(count=h.point_count, mins=np.array(h.mins), maxs=np.array(h.maxs),
                    has_rgb="red" in h.point_format.dimension_names)


def dense_core_bbox(path, res=2.0, min_density=150.0, pad=3.0, verbose=True):
    """Bounding box of the largest connected region with >= min_density pts/m^2."""
    import laspy
    info = las_info(path)
    x0, y0 = info["mins"][:2]
    nx = int(np.ceil((info["maxs"][0] - x0) / res)); ny = int(np.ceil((info["maxs"][1] - y0) / res))
    cnt = np.zeros(nx * ny, np.int64)
    with laspy.open(str(path)) as f:
        sx, sy = f.header.scales[:2]; ox, oy = f.header.offsets[:2]
        for p in f.chunk_iterator(CHUNK):
            ix = np.clip(((p.X * sx + ox - x0) / res).astype(np.int64), 0, nx - 1)
            iy = np.clip(((p.Y * sy + oy - y0) / res).astype(np.int64), 0, ny - 1)
            cnt += np.bincount(iy * nx + ix, minlength=nx * ny)
    dense = (cnt.reshape(ny, nx) / res ** 2) >= min_density
    dense = ndimage.binary_closing(dense, iterations=2)
    dense = ndimage.binary_opening(dense, iterations=3)   # drop thin radial streaks
    lab, n = ndimage.label(dense)
    if n == 0:
        raise RuntimeError("no dense region found; pass --bbox or --center/--size")
    k = 1 + int(np.argmax(ndimage.sum(dense, lab, np.arange(1, n + 1))))
    ys, xs = np.nonzero(lab == k)
    bbox = (x0 + xs.min() * res - pad, y0 + ys.min() * res - pad,
            x0 + (xs.max() + 1) * res + pad, y0 + (ys.max() + 1) * res + pad)
    if verbose:
        print(f"[las_terrain] dense core bbox: {np.round(bbox, 1)} "
              f"({bbox[2]-bbox[0]:.0f} x {bbox[3]-bbox[1]:.0f} m)")
    return bbox


def _mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 16.0


def _iter_bbox(path, bbox):
    """Yield (x, y, z, points) for points inside bbox, in local xy (origin = bbox min)."""
    import laspy
    xmin, ymin, xmax, ymax = bbox
    with laspy.open(str(path)) as f:
        hd = f.header
        sx, sy, sz = hd.scales; ox, oy, oz = hd.offsets
        ix0, ix1 = int((xmin - ox) / sx), int((xmax - ox) / sx)
        iy0, iy1 = int((ymin - oy) / sy), int((ymax - oy) / sy)
        for p in f.chunk_iterator(CHUNK):
            X, Y = np.asarray(p.X), np.asarray(p.Y)
            m = (X >= ix0) & (X < ix1) & (Y >= iy0) & (Y < iy1)
            if not m.any():
                continue
            yield (X[m] * sx + ox - xmin, Y[m] * sy + oy - ymin, np.asarray(p.Z)[m] * sz + oz, p, m)


def _elevated_mask(g, res, step=0.5, join=0.6, iters=8):
    """Cells of a lowest-return surface that are not ground (roofs, vehicle tops, walls).
    The surface is cut along steep cells; the largest connected piece is ground, and another
    piece joins it only if it sits within `join` metres of the ground interpolated underneath
    (lawns behind a kerb, terraces reached step by step). Roofs never join: they stand metres
    above the ground interpolated across their footprint."""
    steep = (ndimage.maximum_filter(g, 3) - ndimage.minimum_filter(g, 3)) > step
    lab, n = ndimage.label(~steep)
    if n == 0:
        return np.zeros(g.shape, bool)
    idx = np.arange(1, n + 1)
    is_ground = np.zeros(n + 1, bool)
    is_ground[1 + int(np.argmax(ndimage.sum(np.ones_like(g), lab, idx)))] = True
    for _ in range(iters):
        base = _fill_nan(np.where(is_ground[lab], g, np.nan))
        med = np.r_[0.0, ndimage.median(g - base, lab, idx)]
        new = is_ground | (med < join)
        new[0] = False
        if (new == is_ground).all():
            break
        is_ground = new
    base = _fill_nan(np.where(is_ground[lab], g, np.nan))
    return (g - base) > step


def build_terrain_from_las(las, out_dir, name, bbox=None, mesh_res=0.25, tex_res=0.1, z_res=0.25,
                           min_pts=3, band=(0.3, 2.0), band_pts=4, roof_min_h=3.0,
                           obstacle_cap=None, origin_z=None, bare_earth=False,
                           crs="NAD83 / UTM zone 18N", verbose=True):
    """obstacle_cap: if set, obstacle / roof cells are raised only this far above the ground
    (low collision stubs; the real 3D shape comes from structure.py as a visual mesh).
    bare_earth: for clouds seen from above (aerial photogrammetry), where the lowest return of a
    building column is its roof: roofs are detected as plateaus bounded by drops and the ground is
    interpolated underneath; every column with no ground return and something above is an obstacle."""
    t0 = time.time()
    info = las_info(las)
    if bbox is None:
        bbox = dense_core_bbox(las, verbose=verbose)
    xmin, ymin, xmax, ymax = bbox
    sx, sy = xmax - xmin, ymax - ymin
    ny, nx = int(np.ceil(sy / mesh_res)), int(np.ceil(sx / mesh_res))
    zlo, zhi = info["mins"][2], info["maxs"][2]
    nz = int(np.ceil((zhi - zlo) / z_res)) + 1
    need, avail = ny * nx * nz * 2 / 2 ** 30, _mem_available_gb()
    if need > 0.6 * avail:
        raise MemoryError(f"voxel grid {ny}x{nx}x{nz} needs {need:.1f} GB (available {avail:.1f} GB); "
                          "crop smaller or raise --mesh-res")
    if verbose and need > 2:
        print(f"[las_terrain] voxel grid {ny}x{nx}x{nz}: {need:.1f} GB", flush=True)

    # ---- pass 1: voxel occupancy
    occ = np.zeros((ny * nx, nz), np.uint16)
    n_in = 0
    for x, y, z, _, _ in _iter_bbox(las, bbox):
        cell = np.minimum((y / mesh_res).astype(np.int64), ny - 1) * nx + \
            np.minimum((x / mesh_res).astype(np.int64), nx - 1)
        vz = np.clip(((z - zlo) / z_res).astype(np.int64), 0, nz - 1)
        u, c = np.unique(cell * nz + vz, return_counts=True)
        flat = occ.reshape(-1)
        flat[u] = np.minimum(flat[u].astype(np.int64) + c, 65535).astype(np.uint16)
        n_in += len(x)
        if verbose:
            print(f"[las_terrain] pass 1: {n_in/1e6:.0f} M pts in bbox, {time.time()-t0:.0f}s", flush=True)

    # ground = lowest voxel holding a real share of the column. A plain "lowest occupied voxel"
    # is fooled by mirror-reflection ghost points below wet paving (they are few per voxel,
    # while the true surface next to the scanner holds hundreds).
    # Done in column blocks: the boolean / cumsum temporaries of the whole grid would be several
    # times the size of `occ` itself.
    ncol, blk = ny * nx, max(1, int(2e8 // nz))
    has = np.zeros(ncol, bool); g_idx = np.zeros(ncol, np.int64); top_idx = np.zeros(ncol, np.int64)
    for i in range(0, ncol, blk):
        o = occ[i:i + blk]
        solid = o >= min_pts
        h_ = solid.any(1)
        strong = o >= np.maximum(min_pts, 0.08 * o.max(1, keepdims=True))
        has[i:i + blk] = h_
        g_idx[i:i + blk] = np.where(h_, strong.argmax(1), 0)
        top_idx[i:i + blk] = np.where(h_, nz - 1 - solid[:, ::-1].argmax(1), 0)  # highest occupied voxel
    del solid, strong, o
    g0 = np.where(has, zlo + (g_idx + 0.5) * z_res, np.nan).reshape(ny, nx)
    top = np.where(has, zlo + (top_idx + 1.0) * z_res, np.nan).reshape(ny, nx)

    # ground: fill holes, remove pits/spikes, then opening removes blobs whose lowest return
    # is not ground (dense canopy, car roofs) while keeping real slopes
    nodata = ndimage.distance_transform_edt(np.isnan(g0)) * mesh_res > 0.75
    elevated = None
    if bare_earth:
        elevated = _elevated_mask(_fill_nan(g0.copy()), mesh_res)
        g0 = np.where(elevated, np.nan, g0)
    g = ndimage.median_filter(_fill_nan(g0.copy()), 5)
    g = np.minimum(g, ndimage.grey_opening(g, size=int(6 / mesh_res) | 1))
    g = ndimage.uniform_filter(g, 3)
    g_v = np.clip(((g - zlo) / z_res).astype(np.int64), 0, nz - 1).ravel()

    # robot-band occupancy relative to the *smoothed* ground
    lo = np.clip(g_v + int(round(band[0] / z_res)), 0, nz - 1)
    hi = np.clip(g_v + int(round(band[1] / z_res)), 0, nz - 1)
    band_cnt = np.zeros(ncol, np.int64)
    for i in range(0, ncol, blk):
        cs = np.cumsum(occ[i:i + blk], axis=1, dtype=np.int64)
        rows = np.arange(len(cs))
        band_cnt[i:i + blk] = cs[rows, hi[i:i + blk]] - cs[rows, lo[i:i + blk]]
    del cs, occ
    band_cnt = band_cnt.reshape(ny, nx)
    obstacle = band_cnt >= band_pts
    obstacle = ndimage.binary_opening(obstacle, np.ones((2, 2))) | \
        (band_cnt >= band_pts * 5)                                # keep thin but dense (poles)
    if elevated is not None:
        obstacle |= elevated & ((_fill_nan(top.copy()) - g) > band[0])

    top_f = _fill_nan(top.copy())
    above = (top_f - g) > roof_min_h
    rough = ndimage.maximum_filter(top_f, 5) - ndimage.minimum_filter(top_f, 5)
    roof = above & (rough < 0.6) & ~np.isnan(top)
    roof = ndimage.binary_opening(roof, iterations=int(1.0 / mesh_res))   # >= ~2 m wide
    roof = ndimage.binary_closing(roof, iterations=2)

    # ---- pass 2: refine ground z (mean of points near ground voxel), surface colors
    surf0 = np.where(obstacle | roof, np.maximum(top_f, g), g)    # provisional surface
    ty, tx = int(np.ceil(sy / tex_res)), int(np.ceil(sx / tex_res))
    gsum = np.zeros(ny * nx); gcnt = np.zeros(ny * nx)
    csum = np.zeros((ty * tx, 3)); ccnt = np.zeros(ty * tx)
    isum = np.zeros(ty * tx); icnt = np.zeros(ty * tx)
    gflat, sflat = g.ravel(), surf0.ravel()
    for x, y, z, p, m in _iter_bbox(las, bbox):
        cell = np.minimum((y / mesh_res).astype(np.int64), ny - 1) * nx + \
            np.minimum((x / mesh_res).astype(np.int64), nx - 1)
        near_g = np.abs(z - gflat[cell]) < 0.3
        gsum += np.bincount(cell[near_g], z[near_g], ny * nx)
        gcnt += np.bincount(cell[near_g], minlength=ny * nx)
        near_s = np.abs(z - sflat[cell]) < 0.6
        tcell = np.minimum((y / tex_res).astype(np.int64), ty - 1) * tx + \
            np.minimum((x / tex_res).astype(np.int64), tx - 1)
        inten = np.asarray(p.intensity)[m].astype(np.float64)
        isum += np.bincount(tcell[near_s], inten[near_s], ty * tx)
        icnt += np.bincount(tcell[near_s], minlength=ty * tx)
        if info["has_rgb"]:
            r, gg, b = (np.asarray(p[c])[m] for c in ("red", "green", "blue"))
            col = near_s & ((r > 0) | (gg > 0) | (b > 0))
            for k, a in enumerate((r, gg, b)):
                csum[:, k] += np.bincount(tcell[col], a[col].astype(np.float64), ty * tx)
            ccnt += np.bincount(tcell[col], minlength=ty * tx)
        if verbose:
            print(f"[las_terrain] pass 2: {time.time()-t0:.0f}s", flush=True)

    g_ref = np.where(gcnt >= 2, gsum / np.maximum(gcnt, 1), np.nan).reshape(ny, nx)
    g_ref = np.where(np.isnan(g_ref), g, g_ref)
    g_ref = ndimage.gaussian_filter(ndimage.median_filter(g_ref, 5), 1.0)
    # unobserved areas: smooth extrapolation of the ground, never raised, never drivable
    g_far = ndimage.gaussian_filter(g_ref, 4.0 / mesh_res)
    w = np.clip(ndimage.distance_transform_edt(nodata) * mesh_res / 3.0, 0, 1)
    g_ref = (1 - w) * g_ref + w * g_far
    obstacle &= ~nodata
    roof &= ~nodata
    raised = np.maximum(top_f, g_ref)
    if obstacle_cap is not None:
        raised = np.minimum(raised, g_ref + obstacle_cap)
    h_abs = np.where(obstacle | roof, raised, g_ref)

    z0 = float(np.percentile(g_ref, 0.5)) if origin_z is None else float(origin_z)
    h = h_abs - z0

    # albedo: RGB where available, else intensity grey; then nearest fill
    rgb = csum / np.maximum(ccnt, 1)[:, None]
    if rgb.max() > 255:
        rgb /= 256.0
    inten = isum / np.maximum(icnt, 1)
    if (icnt > 0).any():
        lo_i, hi_i = np.percentile(inten[icnt > 0], [2, 98])
        grey = np.clip((inten - lo_i) / max(hi_i - lo_i, 1e-6), 0, 1) * 200 + 30
    else:
        grey = np.full_like(inten, 128.0)
    alb = np.where((ccnt > 0)[:, None], rgb, np.where((icnt > 0)[:, None], grey[:, None], np.nan))
    albedo = np.stack([_fill_nan(alb[:, k].reshape(ty, tx)) for k in range(3)], -1)
    nd_tex = ndimage.zoom(nodata.astype(np.float32), (ty / ny, tx / nx), order=1)[:ty, :tx]
    nd_tex = np.pad(nd_tex, [(0, ty - nd_tex.shape[0]), (0, tx - nd_tex.shape[1])], mode="edge")
    albedo = albedo * (1 - nd_tex[..., None]) + 95.0 * nd_tex[..., None]   # unobserved -> grey
    albedo = np.clip(albedo, 0, 255).astype(np.uint8)

    h_tex = ndimage.zoom(h, (ty / ny, tx / nx), order=1)[:ty, :tx]
    h_tex = np.pad(h_tex, [(0, ty - h_tex.shape[0]), (0, tx - h_tex.shape[1])], mode="edge")

    out_dir = Path(out_dir)
    extra = dict(source=str(las), bbox_utm=[float(v) for v in bbox], points_used=int(n_in),
                 colored_texel_fraction=float((ccnt > 0).mean()),
                 obstacle_fraction=float(obstacle.mean()), roof_fraction=float(roof.mean()))
    glb, meta, mesh = write_terrain(out_dir, name, h, h_tex, albedo, [xmin, ymin, z0], mesh_res,
                                    tex_res, "ground_obstacle", crs, float(sx), float(sy), extra)
    # semantic grid for planners: 0 ground, 1 obstacle, 2 roof, 3 no data
    sem = np.zeros((ny, nx), np.uint8)
    sem[nodata] = 3
    sem[obstacle] = 1
    sem[roof] = 2
    np.save(out_dir / f"{name}_semantic.npy", sem)
    np.save(out_dir / f"{name}_ground.npy", (g_ref - z0).astype(np.float32))   # bare ground
    if verbose:
        print(f"[las_terrain] {name}: {sx:.0f} x {sy:.0f} m, V={len(mesh.vertices)} "
              f"F={len(mesh.faces)}, obstacle {obstacle.mean()*100:.1f}%, roof {roof.mean()*100:.1f}%, "
              f"no-data {(sem == 3).mean()*100:.1f}%, {time.time()-t0:.0f}s -> {glb}")
    return glb
