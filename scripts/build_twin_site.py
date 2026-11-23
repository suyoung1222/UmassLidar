#!/usr/bin/env python
"""Coloured point cloud seen from above (a region cut with scripts/select_region.py) -> digital twin.

No photos and no scan geometry are needed: the only inputs are point positions and colours.
Same output layout as scripts/build_twin.py (assets/twin/<name>/...), so scripts/run_nav.py and
scripts/blender_twin.py work unchanged with --config configs/<name>.json.

    python scripts/build_twin_site.py --las data/sites/library.las

Stages (cached under data/<name>/; --redo <stage>):
    ground    bare-earth 2.5D terrain (collision + planning grids), DSM
    cloud     local-frame point cache sorted by 20 m tile
    bake      ground tiles coloured from near-ground points, structure tiles = Poisson mesh of the
              points above the ground, coloured from their nearest points
    export    material fill, optional "grade" from the config, PNG + GLB per tile, URDF, twin.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import build_twin as BT                                                    # noqa: E402
from umass_lidar import texture                                            # noqa: E402
from umass_lidar.las_terrain import _fill_nan, build_terrain_from_las      # noqa: E402
from umass_lidar.metashape import Cloud                                    # noqa: E402
from umass_lidar.structure import bilinear, reconstruct_tile, voxel_reduce  # noqa: E402

G = BT.G
TILE = 20.0
SRC_NAMES = ["none", "unused", "point_rgb", "material", "no_data", "point_rgb_lowfreq"]


def load_config(a):
    """configs/<name>.json; written on the first run from the LAS (+ the .json select_region.py wrote)."""
    import laspy
    if a.config:
        cf = Path(a.config)
    else:
        las = Path(a.las).expanduser().resolve()
        cf = ROOT / "configs" / f"{las.stem}.json"
        if not cf.exists():
            side = las.with_suffix(".json")
            if side.exists():
                bbox = json.loads(side.read_text())["bbox_utm"]
            else:
                with laspy.open(str(las)) as f:
                    bbox = [float(f.header.mins[0]), float(f.header.mins[1]),
                            float(f.header.maxs[0]), float(f.header.maxs[1])]
            cf.write_text(json.dumps(dict(name=las.stem, source="las", las=str(las), bbox_utm=bbox,
                                          crs="UTM 18N as in the source LAS"), indent=2))
            print(f"[site] wrote {cf}")
    cfg = json.loads(cf.read_text())
    cfg["twin_dir"] = str(ROOT / "assets/twin" / cfg["name"])
    cfg["cache_dir"] = str(ROOT / "data" / cfg["name"])
    return cfg, cf


def build_cache(cfg, origin, mesh_res, shape, cache):
    """LAS -> cloud.npz (local metres, sorted by tile) and dsm.npy (top surface on the mesh grid)."""
    import laspy
    x0, y0, x1, y1 = cfg["bbox_utm"]
    las = laspy.read(cfg["las"])
    P = np.stack([np.asarray(las.x) - origin[0], np.asarray(las.y) - origin[1], np.asarray(las.z) - origin[2]], 1)
    C = np.stack([las.red, las.green, las.blue], 1)
    C = (C >> 8 if C.max() > 255 else C).astype(np.uint8)
    k = (P[:, 0] >= 0) & (P[:, 0] < x1 - x0) & (P[:, 1] >= 0) & (P[:, 1] < y1 - y0)
    P, C = P[k].astype(np.float32), C[k]
    ny, nx = shape
    lin = np.minimum((P[:, 1] / mesh_res).astype(np.int64), ny - 1) * nx + \
        np.minimum((P[:, 0] / mesh_res).astype(np.int64), nx - 1)
    dsm = np.full(ny * nx, -np.inf, np.float32)
    np.maximum.at(dsm, lin, P[:, 2])
    np.save(cache / "dsm.npy", _fill_nan(np.where(np.isfinite(dsm), dsm, np.nan).reshape(ny, nx)).astype(np.float32))
    tx, ty = int(np.ceil((x1 - x0) / TILE)), int(np.ceil((y1 - y0) / TILE))
    tid = np.minimum((P[:, 1] / TILE).astype(np.int64), ty - 1) * tx + np.minimum((P[:, 0] / TILE).astype(np.int64), tx - 1)
    o = np.argsort(tid, kind="stable")
    np.savez(cache / "cloud.npz", xyz=P[o], rgb=C[o], cam=np.zeros(1, np.uint16),
             starts=np.searchsorted(tid[o], np.arange(tx * ty + 1)), grid=np.array([0.0, 0.0, TILE, tx, ty]))
    print(f"[site] cloud: {len(P) / 1e6:.2f} M points", flush=True)


def _splat(xyz, rgb, bounds, tw, th, z, tol, sigma):
    """Mean colour of the points within tol of the surface z, Gaussian-splatted with sigma (texels).
    Returns (rgb (th,tw,3), support = effective number of points under the kernel)."""
    X0, Y0, X1, Y1 = bounds
    u = ((xyz[:, 0] - X0) / (X1 - X0) * tw).astype(np.int64); v = ((xyz[:, 1] - Y0) / (Y1 - Y0) * th).astype(np.int64)
    k = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
    lin = v[k] * tw + u[k]
    k2 = np.abs(xyz[k, 2] - z.ravel()[lin]) < tol
    lin = lin[k2]; c = rgb[k][k2].astype(np.float64)
    cnt = np.bincount(lin, minlength=th * tw).reshape(th, tw).astype(np.float32)
    out = np.stack([texture._norm_conv(np.bincount(lin, c[:, ch], th * tw).reshape(th, tw).astype(np.float32),
                                       cnt, sigma)[0] for ch in range(3)], -1)
    return out, ndimage.gaussian_filter(cnt, sigma) * (2 * np.pi * sigma ** 2)


def bake_ground(job):
    name, cells = job
    P, N, z, coords, bounds, (th, tw) = texture.ground_texels(G["surf"], G["mesh_res"], G["tex_res"], cells)
    cl = G["cloud"]; idx = cl.query(bounds, 2.0)
    rgb = np.zeros((th * tw, 3), np.float32); src = np.full(th * tw, texture.SRC_NONE, np.uint8)
    if len(idx):
        xyz, col = cl.xyz[idx], cl.rgb[idx]
        pc, sup = _splat(xyz, col, bounds, tw, th, z, G["ground_tol"], G["smooth"] / G["tex_res"])
        use = sup.ravel() > 0.5
        rgb[use] = pc.reshape(-1, 3)[use]; src[use] = texture.SRC_LAS
        # holes (under trees / eaves): low-frequency colour of the ground points around them
        lo, sup = _splat(xyz, col, bounds, tw, th, z, G["ground_tol"], G["fill"] / G["tex_res"])
        use = (src == texture.SRC_NONE) & (sup.ravel() > 3.0)
        rgb[use] = lo.reshape(-1, 3)[use]; src[use] = texture.SRC_LAS_LOW
    semt = ndimage.map_coordinates(G["sem"].astype(np.float32), coords, order=0, mode="nearest").astype(np.uint8)
    src[(semt.ravel() == 3) & (src == texture.SRC_NONE)] = texture.SRC_NODATA
    stype = np.where(N[:, 2] > 0.9, 0, 1).astype(np.uint8)
    flat = np.full((th, tw), 128, np.uint8)
    np.savez(G["bake"] / f"{name}.npz", rgb=np.clip(rgb, 0, 255).astype(np.uint8).reshape(th, tw, 3),
             src=src.reshape(th, tw), i_fine=flat, i_smooth=flat, stype=stype.reshape(th, tw), bounds=np.array(bounds))
    info = dict(shape=(th, tw), points=float((src == texture.SRC_LAS).mean()),
                lowfreq=float((src == texture.SRC_LAS_LOW).mean()))
    print(f"[bake] {name}: {info['shape']} points {info['points']*100:.0f}% lowfreq {info['lowfreq']*100:.0f}%", flush=True)
    return name, info


def bake_structure(job):
    from scipy.spatial import cKDTree
    name, bounds = job
    cl = G["cloud"]; v = G["voxel"]
    idx = cl.query(bounds, 2.0)
    if len(idx) < 2000:
        return name, None
    xyz = cl.xyz[idx].astype(np.float64)
    res = G["mesh_res"]
    hag = xyz[:, 2] - bilinear(G["ground"], res, xyz[:, :2])
    k = hag > G["min_h"]
    if k.sum() < 2000:
        return name, None
    pts, ex, _ = voxel_reduce(xyz[k], v, cl.rgb[idx][k].astype(np.float64))
    dsm = G["dsm"]

    def clearance(p):                                   # height above the top surface (nearest cell)
        i = np.clip((p[:, 1] / res).astype(int), 0, dsm.shape[0] - 1)
        j = np.clip((p[:, 0] / res).astype(int), 0, dsm.shape[1] - 1)
        return p[:, 2] - dsm[i, j]
    extent = max(bounds[2] - bounds[0] + 4.0, float(np.ptp(pts[:, 2])))
    depth = int(np.clip(np.ceil(np.log2(extent / v)), 8, 10))
    m = reconstruct_tile(pts, None, bounds, depth=depth, max_faces=G["max_faces"], normal_radius=4 * v,
                         outlier=(4, 3 * v), max_dist=2.0 * v, clearance=clearance)
    if m is None:
        return name, None
    V, F, UV, size = texture.unwrap(m[0], m[1], texel=G["struct_texel"])
    valid, P, N = texture.atlas_texels(V, F, UV, size)
    h, w = valid.shape
    dist, nn = cKDTree(pts).query(P, k=4, distance_upper_bound=4 * v, workers=4)
    ok = np.isfinite(dist); wgt = np.where(ok, 1 / (dist ** 2 + 1e-3), 0.0); ws = wgt.sum(1)
    rgb = np.einsum("nkc,nk->nc", ex[np.where(ok, nn, 0)], wgt) / np.maximum(ws, 1e-9)[:, None]
    src = np.where(ws > 0, texture.SRC_LAS, texture.SRC_NONE).astype(np.uint8)
    stype = np.where(np.abs(N[:, 2]) < 0.5, 2, 3).astype(np.uint8)

    def img(vals, dtype, ch=None):
        out = np.zeros((h, w) if ch is None else (h, w, ch), dtype)
        out[valid] = vals
        return out
    flat = np.full(len(P), 128, np.uint8)
    np.savez(G["bake"] / f"{name}.npz", rgb=img(np.clip(rgb, 0, 255), np.uint8, 3), src=img(src, np.uint8),
             i_fine=img(flat, np.uint8), i_smooth=img(flat, np.uint8), stype=img(stype, np.uint8), valid=valid,
             bounds=np.array(bounds))
    np.savez(G["bake"] / f"{name}_mesh.npz", V=V.astype(np.float32), F=F.astype(np.int32), UV=UV.astype(np.float32))
    info = dict(shape=(h, w), faces=int(len(F)), points=float((src == texture.SRC_LAS).mean()), depth=depth)
    print(f"[bake] {name}: F={len(F)} tex {info['shape']} points {info['points']*100:.0f}% depth {depth}", flush=True)
    return name, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las", help="region LAS (data/sites/<name>.las); writes configs/<name>.json on first use")
    ap.add_argument("--config", help="existing configs/<name>.json instead of --las")
    ap.add_argument("--redo", nargs="*", default=[], choices=["ground", "cloud", "bake"])
    ap.add_argument("--name", default=None, help="variant (default: full, or crop with --region)")
    ap.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="only bake tiles overlapping this local-metre box (quick tests)")
    ap.add_argument("--ground-split", type=int, nargs=2, default=[6, 6])
    ap.add_argument("--tex-res", type=float, default=0.05, help="ground texel (m); points are ~0.1 m apart")
    ap.add_argument("--struct-texel", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.07, help="point colour splat sigma (m)")
    ap.add_argument("--fill", type=float, default=0.6, help="sigma (m) of the colour that fills ground holes")
    ap.add_argument("--ground-tol", type=float, default=0.3, help="points this close to the ground colour it (m)")
    ap.add_argument("--voxel", type=float, default=0.12, help="structure point voxel (m)")
    ap.add_argument("--min-h", type=float, default=0.4, help="height above ground meshed in 3D")
    ap.add_argument("--max-faces", type=int, default=40000, help="faces per 20 m structure tile")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-structure", action="store_true")
    a = ap.parse_args()
    if not (a.las or a.config):
        ap.error("give --las or --config")
    t0 = time.time()
    cfg, cf = load_config(a)
    cache = Path(cfg["cache_dir"]); twin_root = Path(cfg["twin_dir"])
    variant = a.name or ("full" if a.region is None else "crop")
    twin = twin_root / variant; bake = cache / f"bake_{variant}"
    for p in (cache, twin, bake):
        p.mkdir(parents=True, exist_ok=True)

    # ---- ground
    if "ground" in a.redo or not (twin_root / "ground.glb").exists():
        build_terrain_from_las(cfg["las"], twin_root, "ground", bbox=cfg["bbox_utm"], obstacle_cap=0.6,
                               crs=cfg["crs"], bare_earth=True)
        a.redo.append("cloud")
    meta = json.loads((twin_root / "ground.json").read_text())
    if cfg.get("origin") != meta["origin"]:                     # run_nav.py / Blender read it from the config
        raw = json.loads(cf.read_text()); raw["origin"] = meta["origin"]
        cf.write_text(json.dumps(raw, indent=2)); cfg["origin"] = meta["origin"]
    surf = np.load(twin_root / "ground_height.npy").astype(np.float64)
    ground = np.load(twin_root / "ground_ground.npy").astype(np.float64)
    sem = np.load(twin_root / "ground_semantic.npy")
    res = meta["mesh_res"]; ny, nx = surf.shape

    # ---- cloud
    if "cloud" in a.redo or not (cache / "cloud.npz").exists():
        build_cache(cfg, meta["origin"], res, surf.shape, cache)
    cloud = Cloud(cache / "cloud.npz")
    dsm = np.load(cache / "dsm.npy")

    gjobs = [(n, c) for n, c in BT.ground_tiles(nx, ny, a.ground_split)
             if BT._overlaps(((c[0] + .5) * res, (c[2] + .5) * res, (c[1] + .5) * res, (c[3] + .5) * res), a.region)]
    sx, sy = meta["size_xy"]
    sjobs = [] if a.no_structure else [
        (f"s_{ix}_{iy}", (ix * TILE, iy * TILE, (ix + 1) * TILE, (iy + 1) * TILE))
        for iy in range(int(np.ceil(sy / TILE))) for ix in range(int(np.ceil(sx / TILE)))
        if BT._overlaps((ix * TILE, iy * TILE, (ix + 1) * TILE, (iy + 1) * TILE), a.region)]
    G.update(bake=bake, twin=twin, cloud=cloud, dsm=dsm, surf=surf, ground=ground, sem=sem, mesh_res=res,
             tex_res=a.tex_res, struct_texel=a.struct_texel, max_faces=a.max_faces, voxel=a.voxel, min_h=a.min_h,
             smooth=a.smooth, fill=a.fill, ground_tol=a.ground_tol, gcells=dict(gjobs), grade=cfg.get("grade"))

    # ---- bake
    todo_g = [j for j in gjobs if "bake" in a.redo or not (bake / f"{j[0]}.done").exists()]
    todo_s = [j for j in sjobs if "bake" in a.redo or not (bake / f"{j[0]}.done").exists()]
    if todo_g or todo_s:
        print(f"[site] bake: {len(todo_g)} ground + {len(todo_s)} structure tiles, {time.time()-t0:.0f}s", flush=True)
        for n, _ in todo_s:                               # stale meshes must not survive a failed rebake
            (bake / f"{n}_mesh.npz").unlink(missing_ok=True)
        with Pool(a.workers, maxtasksperchild=1) as pool:
            for fn, jobs in ((bake_ground, todo_g), (bake_structure, todo_s)):
                res_ = [(j[0], pool.apply_async(fn, (j,))) for j in jobs]
                for n, r in res_:
                    try:                                   # a crashed worker must not hang the build
                        _, info = r.get(timeout=1800)
                    except Exception as e:                 # noqa: BLE001
                        print(f"[bake] {n}: FAILED ({type(e).__name__}: {e})", flush=True)
                        info = dict(failed=True)
                    (bake / f"{n}.done").write_text(json.dumps(info))
    table = texture.MaterialTable()
    for name, _ in gjobs + sjobs:
        f = bake / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            k = z["src"][::3, ::3] == texture.SRC_LAS
            table.add(z["stype"][::3, ::3][k], z["i_smooth"][::3, ::3][k].astype(np.float32) / 255,
                      z["rgb"][::3, ::3][k].astype(np.float64))
    G["table_rgb"] = table.table()

    # ---- export
    ejobs = [(n, "ground") for n, _ in gjobs] + [(n, "structure") for n, _ in sjobs if (bake / f"{n}_mesh.npz").exists()]
    with Pool(a.workers) as pool:
        done = pool.map(BT.export_tile, ejobs)
    from umass_lidar.sim_assets import write_visual_urdf
    texture.write_sky(twin / "sky.glb", [sx / 2, sy / 2, 0.0])
    write_visual_urdf(twin, [d[0] for d in done if d[1] == "ground"], [d[0] for d in done if d[1] == "structure"] + ["sky"])
    faces = sum(d[2] for d in done)
    w = np.array([d[2] if d[1] == "structure" else 1e6 for d in done], float)
    frac = (np.array([d[3] for d in done]) * w[:, None]).sum(0) / w.sum()
    failed = [n for n, _ in gjobs + sjobs if "failed" in (bake / f"{n}.done").read_text()]
    summary = dict(config=cfg["name"], source="coloured point cloud (top-down)", las=cfg["las"], variant=variant,
                   region=a.region, tiles_ground=len(gjobs), tiles_structure=len(ejobs) - len(gjobs),
                   failed_tiles=failed, faces_visual=int(faces),
                   texel_source_fraction=dict(zip(SRC_NAMES, np.round(frac, 4).tolist())), ground_tex_res=a.tex_res,
                   struct_texel=a.struct_texel, voxel=a.voxel, smooth=a.smooth, grade=cfg.get("grade"),
                   cloud_points=int(len(cloud.xyz)), collision="../ground.glb", origin=meta["origin"], crs=cfg["crs"])
    (twin / "twin.json").write_text(json.dumps(summary, indent=2))
    if a.region is None:
        BT.write_maps(twin_root, twin, gjobs, res, meta["size_xy"])
    print(f"[site] {variant}: {len(gjobs)} ground + {len(ejobs)-len(gjobs)} structure tiles ({len(failed)} failed), "
          f"{faces/1e6:.2f} M faces, sources {summary['texel_source_fraction']}, {time.time()-t0:.0f}s -> {twin}\n"
          f"[site] waypoints: read local metres off {twin_root / 'map.png'}; config: {cf}")


if __name__ == "__main__":
    main()
