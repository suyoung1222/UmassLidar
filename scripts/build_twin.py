#!/usr/bin/env python
"""RESEPI session (LiDAR LAS + photos + INS trajectory) -> textured digital twin for Isaac Gym.

Stages (each cached; --redo <stage> to recompute):
    prep      LAS<->trajectory shift, point caches (subsample, full-density tiles, voxels)
    calib     camera intrinsics / boresight / per-photo pose refinement  (umass_lidar/camera.py)
    ground    2.5D drivable terrain: collision mesh + planning grids      (umass_lidar/las_terrain.py)
    bake      per-tile meshes + photo / LAS / intensity texels            (umass_lidar/texture.py)
    export    material fill, PNG + GLB per tile, visual URDF, twin.json

Run in the processing env (miniforge `umasslidar`):
    python scripts/build_twin.py                       # configs/blue_wall.json, whole area
    python scripts/build_twin.py --region 80 40 120 80 --name crop   # quick local test (local m)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umass_lidar import resepi, texture                                    # noqa: E402
from umass_lidar.camera import CameraModel, calibrate                      # noqa: E402
from umass_lidar.las_terrain import build_terrain_from_las                 # noqa: E402
from umass_lidar.structure import reconstruct_tile, tile_structure_points  # noqa: E402

G = {}   # globals shared with forked workers


def ground_tiles(nx, ny, n_split):
    xs = np.linspace(0, nx - 1, n_split[0] + 1).round().astype(int)
    ys = np.linspace(0, ny - 1, n_split[1] + 1).round().astype(int)
    return [(f"g_{a}_{b}", (xs[a], xs[a + 1], ys[b], ys[b + 1]))
            for b in range(n_split[1]) for a in range(n_split[0])]


def _overlaps(b, region):
    return region is None or (b[0] < region[2] and b[2] > region[0] and b[1] < region[3] and b[3] > region[1])


def bake_ground(job):
    name, cells = job
    out = G["bake"] / f"{name}.npz"
    table = texture.MaterialTable()
    info = texture.ground_tile_stage_a(G["bank"], G["tiles"], G["imodel"], G["surf"], G["ground"], G["sem"],
                                       G["mesh_res"], G["tex_res"], cells, out, table)
    print(f"[bake] {name}: {info['shape']} photo {info['photo']*100:.0f}% las {info['las']*100:.0f}%", flush=True)
    return name, table.sum, table.cnt, info


def bake_structure(job):
    name, bounds = job
    out = G["bake"] / f"{name}.npz"
    table = texture.MaterialTable()
    r = tile_structure_points(G["tiles"], G["session"], G["ground"], G["mesh_res"], bounds)
    if r is None:
        return name, table.sum, table.cnt, None
    m = reconstruct_tile(r[0], r[1], bounds, max_faces=G["max_faces"])
    if m is None:
        return name, table.sum, table.cnt, None
    V, F, UV, size = texture.unwrap(m[0], m[1], texel=G["struct_texel"])
    info = texture.structure_tile_stage_a(G["bank"], G["tiles"], G["imodel"], V, F, UV, size, bounds, out, table)
    np.savez(G["bake"] / f"{name}_mesh.npz", V=V.astype(np.float32), F=F.astype(np.int32), UV=UV.astype(np.float32))
    info["faces"] = int(len(F))
    print(f"[bake] {name}: F={len(F)} tex {info['shape']} photo {info['photo']*100:.0f}% "
          f"las {info['las']*100:.0f}%", flush=True)
    return name, table.sum, table.cnt, info


def export_tile(job):
    name, kind = job
    from PIL import Image
    rgb, src = texture.finish_texture(G["bake"] / f"{name}.npz", G["table_rgb"])
    rgb = texture.apply_grade(rgb, G.get("grade"))
    out = G["twin"]
    if kind == "ground":
        j0, j1, i0, i1 = G["gcells"][name]
        res = G["mesh_res"]
        h = G["surf"][i0:i1 + 1, j0:j1 + 1]
        ny, nx = h.shape
        X, Y = np.meshgrid((np.arange(j0, j1 + 1) + 0.5) * res, (np.arange(i0, i1 + 1) + 0.5) * res)
        V = np.stack([X.ravel(), Y.ravel(), h.ravel()], 1)
        i = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
        F = np.concatenate([np.stack([i, i + 1, i + nx + 1], 1), np.stack([i, i + nx + 1, i + nx], 1)])
        UV = np.stack([(X.ravel() - X.min()) / (X.max() - X.min()), (Y.ravel() - Y.min()) / (Y.max() - Y.min())], 1)
    else:
        z = np.load(G["bake"] / f"{name}_mesh.npz")
        V, F, UV = z["V"].astype(np.float64), z["F"].astype(np.int64), z["UV"].astype(np.float64)
    Image.fromarray(rgb[::-1]).save(out / f"{name}.png")
    Image.fromarray(texture.SRC_COLORS[src][::-1]).save(out / f"{name}_source.png")
    texture.write_glb(out / f"{name}.glb", V, F, UV, rgb[::-1], name, double_sided=kind == "structure")
    frac = [float((src == k).mean()) for k in range(6)]
    return name, kind, int(len(F)), frac


def write_maps(twin_root, twin, gjobs, res, size_xy, ppm=5.0):
    """map_base.png (ground texture mosaic, north up) and map.png (drivable overlay + metre grid)
    for picking waypoints."""
    import cv2
    from umass_lidar.viz import write_terrain_map
    W, H = int(size_xy[0] * ppm), int(size_xy[1] * ppm)
    base = np.zeros((H, W, 3), np.uint8)
    for name, (j0, j1, i0, i1) in gjobs:
        x0, x1 = int(round((j0 + .5) * res * ppm)), int(round((j1 + .5) * res * ppm))
        y0, y1 = int(round((i0 + .5) * res * ppm)), int(round((i1 + .5) * res * ppm))
        x1, y1 = min(x1, W), min(y1, H)                      # size_xy need not be a multiple of 1/ppm
        im = cv2.imread(str(twin / f"{name}.png"))
        if im is None or x1 <= x0 or y1 <= y0:
            continue
        base[H - y1:H - y0, x0:x1] = cv2.resize(im, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(twin_root / "map_base.png"), base)
    write_terrain_map(twin_root / "ground", image=twin_root / "map_base.png", out=twin_root / "map.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="configs/*.json (default: configs/blue_wall.json)")
    ap.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="only bake tiles overlapping this local-metre box (quick tests)")
    ap.add_argument("--name", default=None, help="output sub-name (default: full)")
    ap.add_argument("--redo", nargs="*", default=[], choices=["prep", "calib", "ground", "bake"])
    ap.add_argument("--ground-split", type=int, nargs=2, default=[6, 6])
    ap.add_argument("--tex-res", type=float, default=0.01, help="ground texel size (m)")
    ap.add_argument("--struct-texel", type=float, default=0.03)
    ap.add_argument("--max-faces", type=int, default=40000, help="faces per 20 m structure tile")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-structure", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = resepi.load_config(a.config)
    cache = Path(cfg["cache_dir"]); twin_root = Path(cfg["twin_dir"])
    variant = a.name or ("full" if a.region is None else "crop")
    twin = twin_root / variant; bake = cache / f"bake_{variant}"
    twin.mkdir(parents=True, exist_ok=True); bake.mkdir(parents=True, exist_ok=True)

    # ---- prep
    sub = resepi.build_subsample(cfg)
    if "prep" in a.redo or not (cache / "las_shift.json").exists():
        resepi.estimate_las_shift(cfg)
    resepi.build_tiles(cfg); resepi.build_vox(cfg)
    session = resepi.load_session(cfg); tiles = resepi.Tiles(cfg)

    # ---- calib
    if "calib" in a.redo or not (cache / "camera.json").exists():
        model = calibrate(cfg, session, tiles)
    else:
        model = CameraModel.from_json(cache / "camera.json")

    # ---- ground
    if "ground" in a.redo or not (twin_root / "ground.glb").exists():
        build_terrain_from_las(cfg["las"], twin_root, "ground", bbox=cfg["bbox_utm"], obstacle_cap=0.6,
                               origin_z=cfg["origin"][2], crs=cfg["crs"])
    meta = json.loads((twin_root / "ground.json").read_text())
    surf = np.load(twin_root / "ground_height.npy").astype(np.float64)
    ground = np.load(twin_root / "ground_ground.npy").astype(np.float64)
    sem = np.load(twin_root / "ground_semantic.npy")
    res = meta["mesh_res"]
    ny, nx = surf.shape

    # ---- jobs
    gjobs = [(n, c) for n, c in ground_tiles(nx, ny, a.ground_split)
             if _overlaps(((c[0] + .5) * res, (c[2] + .5) * res, (c[1] + .5) * res, (c[3] + .5) * res), a.region)]
    sjobs = []
    if not a.no_structure:
        T = tiles.tile
        for iy in range(tiles.ny):
            for ix in range(tiles.nx):
                b = (ix * T, iy * T, (ix + 1) * T, (iy + 1) * T)
                if tiles.counts[iy, ix] > 20000 and _overlaps(b, a.region):
                    sjobs.append((f"s_{ix}_{iy}", b))
    imodel = texture.IntensityModel.fit(sub)
    G.update(bake=bake, twin=twin, tiles=tiles, session=session, imodel=imodel, surf=surf, ground=ground,
             sem=sem, mesh_res=res, tex_res=a.tex_res, struct_texel=a.struct_texel, max_faces=a.max_faces,
             gcells=dict(gjobs), grade=cfg.get("grade"))

    # ---- bake
    todo_g = [j for j in gjobs if "bake" in a.redo or not (bake / f"{j[0]}.done").exists()]
    todo_s = [j for j in sjobs if "bake" in a.redo or not (bake / f"{j[0]}.done").exists()]
    if todo_g or todo_s:
        print(f"[twin] loading photo bank ({len(session.photos)} photos)...", flush=True)
        G["bank"] = texture.PhotoBank(cfg, session, model, tiles)
        print(f"[twin] bake: {len(todo_g)} ground + {len(todo_s)} structure tiles, {time.time()-t0:.0f}s", flush=True)
        # maxtasksperchild=1: a tile's memory is returned to the OS before the next one starts
        with Pool(a.workers, maxtasksperchild=1) as pool:
            for fn, jobs in ((bake_ground, todo_g), (bake_structure, todo_s)):
                for name, _, _, info in pool.imap_unordered(fn, jobs):
                    (bake / f"{name}.done").write_text(json.dumps(info))
    # material table from every baked tile (photo texels only), so it does not depend on run order
    table = texture.MaterialTable()
    for name, _ in gjobs + sjobs:
        f = bake / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            src = z["src"][::3, ::3]
            k = src == texture.SRC_PHOTO
            table.add(z["stype"][::3, ::3][k], z["i_smooth"][::3, ::3][k].astype(np.float32) / 255,
                      z["rgb"][::3, ::3][k].astype(np.float64))
    table.save(bake / "material_table.npz")
    G["table_rgb"] = table.table()

    # ---- export
    ejobs = [(n, "ground") for n, _ in gjobs] + \
            [(n, "structure") for n, _ in sjobs if (bake / f"{n}_mesh.npz").exists()]
    with Pool(a.workers) as pool:
        done = pool.map(export_tile, ejobs)
    from umass_lidar.sim_assets import write_visual_urdf
    texture.write_sky(twin / "sky.glb", [meta["size_xy"][0] / 2, meta["size_xy"][1] / 2, 0.0])
    write_visual_urdf(twin, [d[0] for d in done if d[1] == "ground"],
                      [d[0] for d in done if d[1] == "structure"] + ["sky"])
    faces = sum(d[2] for d in done)
    w = np.array([d[2] if d[1] == "structure" else 1e6 for d in done], float)   # rough area weights
    frac = (np.array([d[3] for d in done]) * w[:, None]).sum(0) / w.sum()
    summary = dict(config=cfg["name"], variant=variant, region=a.region, tiles_ground=len(gjobs),
                   tiles_structure=len(ejobs) - len(gjobs), faces_visual=int(faces),
                   texel_source_fraction=dict(zip(texture.SRC_NAMES, np.round(frac, 4).tolist())),
                   ground_tex_res=a.tex_res, struct_texel=a.struct_texel,
                   intensity_model=imodel.to_dict(), material_table_rgb=np.round(G["table_rgb"]).tolist(),
                   collision="../ground.glb", origin=cfg["origin"], crs=cfg["crs"])
    (twin / "twin.json").write_text(json.dumps(summary, indent=2))
    if a.region is None:
        write_maps(twin_root, twin, gjobs, res, meta["size_xy"])
    print(f"[twin] {variant}: {len(gjobs)} ground + {len(ejobs)-len(gjobs)} structure tiles, "
          f"{faces/1e6:.2f} M faces, sources {summary['texel_source_fraction']}, {time.time()-t0:.0f}s -> {twin}")


if __name__ == "__main__":
    main()
