#!/usr/bin/env python
"""Aerial photogrammetry (Agisoft Metashape project + drone photos) -> textured digital twin.

Same output layout as scripts/build_twin.py (assets/twin/<name>/...), so scripts/run_nav.py works
unchanged:  python scripts/run_nav.py --config configs/blue_wall_aerial.json ...

Stages (cached under data/<name>/; --redo <stage>):
    cloud     back-project Metashape depth maps with the project's camera poses -> point cloud
    align     constant shift aerial -> LiDAR frame (only if the config has "align_to")
    ground    bare-earth 2.5D terrain (collision + planning grids)
    bake      ground / structure tiles textured from full-resolution photo crops
    export    PNG + GLB per tile, visual URDF, twin.json
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
from umass_lidar import metashape as MS, texture                           # noqa: E402
from umass_lidar.las_terrain import build_terrain_from_las                 # noqa: E402
from umass_lidar.structure import bilinear, reconstruct_tile, voxel_reduce  # noqa: E402

G = BT.G


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg["twin_dir"] = str(ROOT / "assets/twin" / cfg["name"])
    cfg["cache_dir"] = str(ROOT / "data" / cfg["name"])
    return cfg


def estimate_align(cloud, ref_npz, init=(-3.0, -6.0, -28.2), verbose=True):
    """Constant translation taking the aerial cloud onto the LiDAR cloud: trimmed nearest-neighbour
    grid search followed by translation-only ICP."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    a, _, _ = voxel_reduce(cloud.xyz[::3].astype(np.float64), 0.25)
    tree = cKDTree(a)
    L = np.load(ref_npz)["xyz"]
    L = L[rng.choice(len(L), 120000, replace=False)].astype(np.float64)
    best = (1e9, np.array(init, float))
    for step, span in ((1.0, 4.0), (0.25, 1.0)):
        c0 = best[1].copy()
        for dx in np.arange(-span, span + 1e-6, step):
            for dy in np.arange(-span, span + 1e-6, step):
                for dz in np.arange(-span / 2, span / 2 + 1e-6, step):
                    s = c0 + [dx, dy, dz]
                    d = tree.query(L - s, distance_upper_bound=1.0, workers=-1)[0]
                    e = float(np.minimum(d, 1.0).mean())
                    if e < best[0]:
                        best = (e, s)
        if verbose:
            print(f"[align] grid step {step}: shift {np.round(best[1], 2)} cost {best[0]:.3f}", flush=True)
    s = best[1].copy()
    for _ in range(30):
        d, nn = tree.query(L - s, distance_upper_bound=0.5, workers=-1)
        k = np.isfinite(d)
        s += np.median((L[k] - s) - a[nn[k]], axis=0) * 0.7
    d = tree.query(L - s, distance_upper_bound=1.0, workers=-1)[0]
    info = dict(shift=s.tolist(), inlier_frac_0p5=float((d < 0.5).mean()),
                nn_median=float(np.median(d[np.isfinite(d)])), ref=str(ref_npz))
    if verbose:
        print(f"[align] final shift {np.round(s, 3)}  LiDAR pts within 0.5 m of aerial cloud: "
              f"{info['inlier_frac_0p5']*100:.0f}%, median NN {info['nn_median']:.3f} m", flush=True)
    return info


def tile_bank(bounds, zlo, zhi):
    return MS.AerialBank(G["chunk"], bounds, zlo, zhi)


def _point_color_grid(idx, X0, Y0, X1, Y1, tw, th, z, tol, sigma):
    cl = G["cloud"]
    xyz = cl.xyz[idx]
    u = ((xyz[:, 0] - X0) / (X1 - X0) * tw).astype(np.int64); v = ((xyz[:, 1] - Y0) / (Y1 - Y0) * th).astype(np.int64)
    k = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
    lin = v[k] * tw + u[k]
    k2 = np.abs(xyz[k, 2] - z.ravel()[lin]) < tol
    lin = lin[k2]; c = cl.rgb[idx][k][k2].astype(np.float64)
    cnt = np.bincount(lin, minlength=th * tw).reshape(th, tw).astype(np.float32)
    out = np.stack([texture._norm_conv(np.bincount(lin, c[:, ch], th * tw).reshape(th, tw).astype(np.float32),
                                       cnt, sigma)[0] for ch in range(3)], -1)
    return out, ndimage.gaussian_filter(cnt, sigma) * (2 * np.pi * sigma ** 2)


def bake_ground(job):
    name, cells = job
    P, N, z, coords, bounds, (th, tw) = texture.ground_texels(G["surf"], G["mesh_res"], G["tex_res"], cells)
    bank = tile_bank(bounds, float(z.min()) - 0.5, float(z.max()) + 0.5)
    rgb, wsum = texture.project_photos(bank, P, N, bounds)
    src = np.where(wsum > 0, texture.SRC_PHOTO, texture.SRC_NONE).astype(np.uint8)
    # fallback: colour of nearby cloud points (occluded in every selected photo)
    idx = G["cloud"].query(bounds)
    if len(idx):
        pc, support = _point_color_grid(idx, *bounds, tw, th, z, 0.4, 0.12 / G["tex_res"])
        use = (src == texture.SRC_NONE) & (support.ravel() > 2.0)
        rgb[use] = pc.reshape(-1, 3)[use]; src[use] = texture.SRC_LAS
    semt = ndimage.map_coordinates(G["sem"].astype(np.float32), coords, order=0, mode="nearest").astype(np.uint8)
    src[(semt.ravel() == 3) & (src == texture.SRC_NONE)] = texture.SRC_NODATA
    stype = np.where(N[:, 2] > 0.9, 0, 1).astype(np.uint8)
    flat = np.full((th, tw), 128, np.uint8)
    np.savez(G["bake"] / f"{name}.npz", rgb=np.clip(rgb, 0, 255).astype(np.uint8).reshape(th, tw, 3),
             src=src.reshape(th, tw), i_fine=flat, i_smooth=flat, stype=stype.reshape(th, tw), bounds=np.array(bounds))
    info = dict(shape=(th, tw), photo=float((src == texture.SRC_PHOTO).mean()),
                las=float((src == texture.SRC_LAS).mean()), photos_used=len(bank.k))
    print(f"[bake] {name}: {info['shape']} photo {info['photo']*100:.0f}% points {info['las']*100:.0f}% "
          f"({len(bank.k)} photos)", flush=True)
    return name, info


def bake_structure(job):
    from scipy.spatial import cKDTree
    name, bounds = job
    cl = G["cloud"]; margin = 2.0
    idx = cl.query(bounds, margin)
    if len(idx) < 2000:
        return name, None
    xyz = cl.xyz[idx].astype(np.float64)
    hag = xyz[:, 2] - bilinear(G["ground"], G["mesh_res"], np.clip(xyz[:, :2], 0, None))
    k = hag > G["min_h"]
    if k.sum() < 2000:
        return name, None
    sensor = G["chunk"].C[cl.cam[idx][k]]
    pts, ex, cnt = voxel_reduce(xyz[k], G["voxel"], np.c_[sensor, cl.rgb[idx][k].astype(np.float64)])
    keep = cnt >= 2
    pts, ex = pts[keep], ex[keep]
    if len(pts) < 500:
        return name, None
    v = G["voxel"]
    m = reconstruct_tile(pts, ex[:, :3], bounds, depth=8, max_faces=G["max_faces"], normal_radius=5 * v,
                         outlier=(5, 3.5 * v), max_dist=2.5 * v)
    if m is None:
        return name, None
    V, F, UV, size = texture.unwrap(m[0], m[1], texel=G["struct_texel"])
    valid, P, N = texture.atlas_texels(V, F, UV, size)
    h, w = valid.shape
    bank = tile_bank(bounds, float(V[:, 2].min()), float(V[:, 2].max()))
    rgb, wsum = texture.project_photos(bank, P, N, (bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1))
    src = np.where(wsum > 0, texture.SRC_PHOTO, texture.SRC_NONE).astype(np.uint8)
    dist, nn = cKDTree(pts).query(P, k=4, distance_upper_bound=4 * v, workers=4)
    ok = np.isfinite(dist); wgt = np.where(ok, 1 / (dist ** 2 + 1e-3), 0.0); ws = wgt.sum(1)
    pc = np.einsum("nkc,nk->nc", ex[np.where(ok, nn, 0), 3:6], wgt) / np.maximum(ws, 1e-9)[:, None]
    use = (src == texture.SRC_NONE) & (ws > 0)
    rgb[use] = pc[use]; src[use] = texture.SRC_LAS
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
    info = dict(shape=(h, w), faces=int(len(F)), photo=float((src == texture.SRC_PHOTO).mean()),
                las=float((src == texture.SRC_LAS).mean()))
    print(f"[bake] {name}: F={len(F)} tex {info['shape']} photo {info['photo']*100:.0f}% points {info['las']*100:.0f}%",
          flush=True)
    return name, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/blue_wall_aerial.json"))
    ap.add_argument("--redo", nargs="*", default=[], choices=["cloud", "align", "ground", "bake"])
    ap.add_argument("--name", default="full")
    ap.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--ground-split", type=int, nargs=2, default=[6, 6])
    ap.add_argument("--tex-res", type=float, default=0.025, help="ground texel (m); photo GSD is ~0.03 m")
    ap.add_argument("--struct-texel", type=float, default=0.05)
    ap.add_argument("--voxel", type=float, default=0.25, help="structure point voxel (m); depth-map pixels are ~0.25 m")
    ap.add_argument("--min-h", type=float, default=0.4, help="height above ground meshed in 3D")
    ap.add_argument("--max-faces", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stop-after", choices=["cloud", "align", "ground"])
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    cache = Path(cfg["cache_dir"]); twin_root = Path(cfg["twin_dir"])
    twin = twin_root / a.name; bake = cache / f"bake_{a.name}"
    for p in (cache, twin, bake):
        p.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = cfg["bbox_utm"]; origin = cfg["origin"]
    bbox_local = (0.0, 0.0, x1 - x0, y1 - y0)

    # ---- align (needs an unaligned cloud first)
    af = cache / "align_shift.json"
    if cfg.get("align_to") and ("align" in a.redo or not af.exists()):
        raw = cache / "cloud_raw.npz"
        chunk = MS.Chunk(dict(cfg, align_shift=[-3.0, -6.0, -28.2]))
        MS.build_cloud(chunk, bbox_local, raw)
        info = estimate_align(MS.Cloud(raw), ROOT / "data" / cfg["align_to"] / "points_sub20.npz", init=(0, 0, 0))
        info["shift"] = (np.array(info["shift"]) + [-3.0, -6.0, -28.2]).tolist()
        af.write_text(json.dumps(info, indent=2)); raw.unlink()
        a.redo.append("cloud")
    if af.exists():
        cfg["align_shift"] = json.loads(af.read_text())["shift"]
    if a.stop_after == "align":
        return
    chunk = MS.Chunk(cfg)

    # ---- cloud
    cf = cache / "cloud.npz"
    if "cloud" in a.redo or not cf.exists():
        MS.build_cloud(chunk, bbox_local, cf)
        a.redo.append("ground")
    cloud = MS.Cloud(cf)
    if a.stop_after == "cloud":
        return

    # ---- ground
    if "ground" in a.redo or not (twin_root / "ground.glb").exists():
        las = MS.write_las(cloud, origin, cache / "aerial.las")
        build_terrain_from_las(las, twin_root, "ground", bbox=cfg["bbox_utm"], obstacle_cap=0.6,
                               origin_z=origin[2], crs=cfg["crs"], bare_earth=True)
        las.unlink()
    meta = json.loads((twin_root / "ground.json").read_text())
    surf = np.load(twin_root / "ground_height.npy").astype(np.float64)
    ground = np.load(twin_root / "ground_ground.npy").astype(np.float64)
    sem = np.load(twin_root / "ground_semantic.npy")
    res = meta["mesh_res"]; ny, nx = surf.shape
    if a.stop_after == "ground":
        from umass_lidar.viz import write_terrain_map
        write_terrain_map(twin_root / "ground", out=twin_root / "map.png")
        return

    gjobs = [(n, c) for n, c in BT.ground_tiles(nx, ny, a.ground_split)
             if BT._overlaps(((c[0] + .5) * res, (c[2] + .5) * res, (c[1] + .5) * res, (c[3] + .5) * res), a.region)]
    T = 20.0
    sjobs = [(f"s_{ix}_{iy}", (ix * T, iy * T, (ix + 1) * T, (iy + 1) * T))
             for iy in range(int(np.ceil(bbox_local[3] / T))) for ix in range(int(np.ceil(bbox_local[2] / T)))
             if BT._overlaps((ix * T, iy * T, (ix + 1) * T, (iy + 1) * T), a.region)]
    G.update(bake=bake, twin=twin, chunk=chunk, cloud=cloud, surf=surf, ground=ground, sem=sem, mesh_res=res,
             tex_res=a.tex_res, struct_texel=a.struct_texel, max_faces=a.max_faces, voxel=a.voxel, min_h=a.min_h,
             gcells=dict(gjobs), grade=cfg.get("grade"))
    chunk._build_depth_index()

    # ---- bake
    todo_g = [j for j in gjobs if "bake" in a.redo or not (bake / f"{j[0]}.done").exists()]
    todo_s = [j for j in sjobs if "bake" in a.redo or not (bake / f"{j[0]}.done").exists()]
    if todo_g or todo_s:
        print(f"[twin] bake: {len(todo_g)} ground + {len(todo_s)} structure tiles, {time.time()-t0:.0f}s", flush=True)
        with Pool(a.workers, maxtasksperchild=1) as pool:
            for fn, jobs in ((bake_ground, todo_g), (bake_structure, todo_s)):
                for name, info in pool.imap_unordered(fn, jobs):
                    (bake / f"{name}.done").write_text(json.dumps(info))
    table = texture.MaterialTable()
    for name, _ in gjobs + sjobs:
        f = bake / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            k = z["src"][::3, ::3] == texture.SRC_PHOTO
            table.add(z["stype"][::3, ::3][k], z["i_smooth"][::3, ::3][k].astype(np.float32) / 255,
                      z["rgb"][::3, ::3][k].astype(np.float64))
    G["table_rgb"] = table.table()

    # ---- export
    ejobs = [(n, "ground") for n, _ in gjobs] + [(n, "structure") for n, _ in sjobs if (bake / f"{n}_mesh.npz").exists()]
    with Pool(a.workers) as pool:
        done = pool.map(BT.export_tile, ejobs)
    from umass_lidar.sim_assets import write_visual_urdf
    texture.write_sky(twin / "sky.glb", [meta["size_xy"][0] / 2, meta["size_xy"][1] / 2, 0.0])
    write_visual_urdf(twin, [d[0] for d in done if d[1] == "ground"], [d[0] for d in done if d[1] == "structure"] + ["sky"])
    faces = sum(d[2] for d in done)
    w = np.array([d[2] if d[1] == "structure" else 1e6 for d in done], float)
    frac = (np.array([d[3] for d in done]) * w[:, None]).sum(0) / w.sum()
    names = ["none", "photo", "point_rgb", "material", "no_data", "unused"]
    summary = dict(config=cfg["name"], source="metashape aerial photogrammetry", variant=a.name, region=a.region,
                   tiles_ground=len(gjobs), tiles_structure=len(ejobs) - len(gjobs), faces_visual=int(faces),
                   texel_source_fraction=dict(zip(names, np.round(frac, 4).tolist())), ground_tex_res=a.tex_res,
                   struct_texel=a.struct_texel, cameras=int(len(chunk.ids)), cloud_points=int(len(cloud.xyz)),
                   align_shift=cfg.get("align_shift"), collision="../ground.glb", origin=origin, crs=cfg["crs"])
    (twin / "twin.json").write_text(json.dumps(summary, indent=2))
    if a.region is None:
        BT.write_maps(twin_root, twin, gjobs, res, meta["size_xy"])
    print(f"[twin] {a.name}: {len(gjobs)} ground + {len(ejobs)-len(gjobs)} structure tiles, {faces/1e6:.2f} M faces, "
          f"sources {summary['texel_source_fraction']}, {time.time()-t0:.0f}s -> {twin}")


if __name__ == "__main__":
    main()
