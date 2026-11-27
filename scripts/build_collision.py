"""Static collision mesh for the twin's 3D structures (walls, buildings, cars, trees) -- processing env.

The structure tiles (s_*.glb) are visual-only in Isaac Gym; this script turns all of them into
one simplified triangle mesh that sim.add_twin() adds as static collision next to ground.glb:

    per tile: merge UV-seam duplicates -> drop small connected pieces (Poisson specks / blobs)
    -> quadric decimation to --ratio of the faces -> push vertices --inset metres against the
    surface normal (Isaac Gym renders collision meshes flat grey; the inset keeps them behind
    the textured visual surface)

    conda activate umasslidar
    python scripts/build_collision.py                               # configs/blue_wall.json, variant full
    python scripts/build_collision.py --config configs/library.json

    --mode voxel: independent of the visual mesh (which has holes where the points are sparse).
    above-ground points -> occupancy on a global voxel lattice -> morphological closing (fills
    gaps up to ~2 x --close) -> drop small blobs -> boundary faces -> quadric decimation.
    The result is a closed shell that is up to one voxel fatter than the real surface.

    python scripts/build_collision.py --config configs/library.json --mode voxel

Output: assets/twin/<name>/<variant>/structure_collision.npz        (mesh mode)
        assets/twin/<name>/<variant>/structure_collision_voxel.npz  (voxel mode)
        (V float32 (N,3) local z-up, F uint32) + .json with the same stem (stats)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umass_lidar.mesh import ZUP_TO_YUP  # noqa: E402


def load_tile(path):
    """Structure GLBs are written double-sided (texture.write_glb: F then F reversed); keep the
    first half = the side facing the sensor (outside)."""
    import trimesh
    m = trimesh.load(path, force="mesh", skip_materials=True, process=False)
    V = np.asarray(m.vertices, np.float64) @ np.linalg.inv(ZUP_TO_YUP)[:3, :3].T
    F = np.asarray(m.faces, np.int64)
    h = len(F) // 2
    if len(F) % 2 == 0 and np.array_equal(F[h:], F[:h, ::-1]):
        F = F[:h]
    return V, F


def simplify_tile(V, F, ratio, min_faces, min_extent, inset):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    mesh.merge_close_vertices(1e-4)
    mesh.remove_duplicated_triangles(); mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    n0 = len(mesh.triangles)
    # connected pieces: drop tiny ones (few faces OR small extent)
    cl, cnt, _ = mesh.cluster_connected_triangles()
    cl, cnt = np.asarray(cl), np.asarray(cnt)
    Vt, Ft = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    ext = np.full(len(cnt), np.inf)
    for k in np.nonzero(cnt < 5 * min_faces)[0]:          # only small clusters can fail
        p = Vt[Ft[cl == k].ravel()]
        ext[k] = np.linalg.norm(p.max(0) - p.min(0))
    small = (cnt < min_faces) | (ext < min_extent)
    mesh.remove_triangles_by_mask(small[cl])
    mesh.remove_unreferenced_vertices()
    dropped = n0 - len(mesh.triangles)
    if len(mesh.triangles) > 200:
        # open3d's quadric decimation stalls on these Poisson meshes; fast_simplification does not
        import fast_simplification
        v, f = fast_simplification.simplify(np.asarray(mesh.vertices), np.asarray(mesh.triangles, np.int64),
                                            target_reduction=1.0 - ratio)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        return None, n0, dropped
    mesh.compute_vertex_normals()
    V2 = np.asarray(mesh.vertices) - inset * np.asarray(mesh.vertex_normals)
    return (V2, np.asarray(mesh.triangles)), n0, dropped


# ----------------------------------------------------------------------------------------------
# voxel mode: collision straight from the point cloud (no dependency on the visual mesh)
# ----------------------------------------------------------------------------------------------
G = {}
CHUNK = 160          # voxels per chunk side (xy); chunks share one global lattice so faces line up


def point_source(cfg):
    """query(bounds) -> (N,3) float64 local-metre points inside the xy box, for either cache kind:
    data/<name>/cloud.npz (aerial / site twins) or data/<name>/tiles/*.vox (LiDAR twin, 3 cm voxels)."""
    cache = ROOT / "data" / cfg["name"]
    if (cache / "cloud.npz").exists():
        from umass_lidar.metashape import Cloud
        cl = Cloud(cache / "cloud.npz")
        return lambda b: cl.xyz[cl.query(b)].astype(np.float64), "cloud.npz"
    idx = cache / "tiles" / "index.json"
    if idx.exists():
        t = json.loads(idx.read_text()); T, nx, ny = t["tile"], t["nx"], t["ny"]

        def query(b):
            acc = []
            for iy in range(max(0, int(b[1] // T)), min(ny, int(b[3] // T) + 1)):
                for ix in range(max(0, int(b[0] // T)), min(nx, int(b[2] // T) + 1)):
                    f = cache / "tiles" / f"{ix}_{iy}.vox"
                    if f.exists():
                        p = np.fromfile(f, np.float32).reshape(-1, 3)
                        k = (p[:, 0] >= b[0]) & (p[:, 0] < b[2]) & (p[:, 1] >= b[1]) & (p[:, 1] < b[3])
                        acc.append(p[k])
            return np.concatenate(acc).astype(np.float64) if acc else np.zeros((0, 3))
        return query, "tiles/*.vox"
    sys.exit(f"no point cache in {cache} (cloud.npz or tiles/); build the twin first")


def quad_faces(occ):
    """Boundary faces of the occupied voxels. occ is padded by one voxel on every side; returns
    (M,4,3) int32 lattice corners in the un-padded index frame, wound so the normal points outward."""
    inner = occ[1:-1, 1:-1, 1:-1]
    out = []
    for ax in range(3):
        u, w = (ax + 1) % 3, (ax + 2) % 3
        for s in (1, -1):
            sl = [slice(1, -1)] * 3
            sl[ax] = slice(2, None) if s == 1 else slice(0, -2)
            base = np.argwhere(inner & ~occ[tuple(sl)]).astype(np.int32)
            if not len(base):
                continue
            if s == 1:
                base[:, ax] += 1
            q = np.repeat(base[:, None, :], 4, 1)
            du, dw = ((0, 1, 1, 0), (0, 0, 1, 1)) if s == 1 else ((0, 0, 1, 1), (0, 1, 1, 0))
            q[:, :, u] += np.array(du, np.int32); q[:, :, w] += np.array(dw, np.int32)
            out.append(q)
    return np.concatenate(out) if out else np.zeros((0, 4, 3), np.int32)


def voxel_chunk(job):
    """One xy chunk: occupancy of the above-ground points -> morphological closing (fills gaps up to
    ~2 * close) -> drop small blobs -> boundary quads of the voxels inside the chunk core."""
    from scipy import ndimage
    from umass_lidar.structure import bilinear
    cx, cy = job
    v, r = G["voxel"], G["r"]
    pad = 2 * r + 2
    i0, j0, n = cx * CHUNK - pad, cy * CHUNK - pad, CHUNK + 2 * pad
    xyz = G["query"]((i0 * v, j0 * v, (i0 + n) * v, (j0 + n) * v))
    if len(xyz):
        hag = xyz[:, 2] - bilinear(G["ground"], G["res"], np.clip(xyz[:, :2], 0, None))
        xyz = xyz[(hag > G["min_h"]) & (hag < G["max_h"])]
    if len(xyz) < 20:
        return None
    k = np.floor(xyz / v).astype(np.int64) - (i0, j0, 0)
    k = k[(k[:, 0] >= 0) & (k[:, 0] < n) & (k[:, 1] >= 0) & (k[:, 1] < n)]
    if not len(k):
        return None
    k0 = int(k[:, 2].min()) - pad
    nz = int(k[:, 2].max()) - k0 + pad + 1
    cnt = np.bincount((k[:, 0] * n + k[:, 1]) * nz + (k[:, 2] - k0), minlength=n * n * nz)
    occ = (cnt >= G["min_count"]).reshape(n, n, nz)
    n_raw = int(occ[pad:-pad, pad:-pad].sum())
    if r > 0:        # closing with a Euclidean ball of radius r voxels, via two distance transforms
        dil = ndimage.distance_transform_edt(~occ) <= r
        occ |= ndimage.distance_transform_edt(dil) > r
    lab, m = ndimage.label(occ, np.ones((3, 3, 3), bool))
    if m:
        size = np.bincount(lab.ravel(), minlength=m + 1)
        edge = np.zeros(m + 1, bool)          # blobs reaching the padded border continue next door
        for a in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
            edge[np.unique(a)] = True
        drop = (size < G["min_voxels"]) & ~edge
        drop[0] = False
        occ &= ~drop[lab]
    core = occ[pad - 1:n - pad + 1, pad - 1:n - pad + 1]
    n_out = int(core[1:-1, 1:-1].sum())
    if n_out == 0:
        return None
    q = quad_faces(core)
    q += np.array((cx * CHUNK, cy * CHUNK, k0 + 1), np.int32)   # core is not trimmed in z
    return q, n_raw, n_out


def hide_behind(V, mesh_npz, reach):
    """Isaac Gym draws collision triangles flat grey, and a voxel shell sticks out up to one voxel in
    front of the textured surface. Move every vertex that is in front of the mesh-mode collision
    surface (= visual surface pushed 6 cm in, normals facing outside) back onto it, in place.
    Vertices farther than `reach` from that surface -- the holes of the visual mesh -- stay."""
    import open3d as o3d
    if not mesh_npz.exists():
        print(f"[collision] no {mesh_npz.name}: voxel faces are not hidden behind the visual mesh "
              f"(run the default --mode mesh first)", flush=True)
        return 0
    d = np.load(mesh_npz)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(d["V"]), o3d.core.Tensor(d["F"]))
    n_moved = 0
    for i in range(0, len(V), 4_000_000):
        P = V[i:i + 4_000_000]
        r = sc.compute_closest_points(o3d.core.Tensor(P.astype(np.float32)), nthreads=16)
        q, n = r["points"].numpy().astype(np.float64), r["primitive_normals"].numpy().astype(np.float64)
        s = ((P - q) * n).sum(1)                      # > 0: in front of the surface
        k = (s > 0) & (np.linalg.norm(P - q, axis=1) < reach)
        P[k] -= n[k] * s[k, None]
        n_moved += int(k.sum())
    print(f"[collision] {n_moved}/{len(V)} vertices moved behind the visual surface", flush=True)
    return n_moved


def build_voxel(cfg, a, twin):
    from multiprocessing import Pool
    t0 = time.time()
    root = twin.parent
    gj = json.loads((root / "ground.json").read_text())
    res, (sx, sy) = gj["mesh_res"], gj["size_xy"]
    v = a.voxel
    query, src = point_source(cfg)
    G.update(query=query, ground=np.load(root / "ground_ground.npy").astype(np.float64), res=res, voxel=v,
             r=int(round(a.close / v)), min_h=a.min_h, max_h=a.max_h, min_count=a.min_count, min_voxels=a.min_voxels)
    x0, y0, x1, y1 = a.region or (0.0, 0.0, sx, sy)
    L = CHUNK * v
    jobs = [(cx, cy) for cy in range(int(y0 // L), int(np.ceil(y1 / L))) for cx in range(int(x0 // L), int(np.ceil(x1 / L)))]
    print(f"[collision] voxel {v} m, close {G['r']} voxels ({G['r'] * v:.2f} m), {len(jobs)} chunks of {L:.0f} m "
          f"from {src}", flush=True)
    quads, n_raw, n_out = [], 0, 0
    with Pool(min(a.workers, len(jobs))) as pool:          # fork: workers share G
        for i, r in enumerate(pool.imap_unordered(voxel_chunk, jobs)):
            if r is not None:
                quads.append(r[0]); n_raw += r[1]; n_out += r[2]
            if (i + 1) % 20 == 0:
                print(f"[collision] {i + 1}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    if not quads:
        sys.exit("[collision] no occupied voxels")
    Q = np.concatenate(quads)
    if a.region:                                           # tests: keep only faces inside the box
        c = Q.mean(1) * v
        Q = Q[(c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)]
    # weld on the integer lattice (exact), quads -> triangles
    P = Q.reshape(-1, 3).astype(np.int64)
    lo = P.min(0); P -= lo
    dims = P.max(0) + 1
    key = (P[:, 0] * dims[1] + P[:, 1]) * dims[2] + P[:, 2]
    u, inv = np.unique(key, return_inverse=True)
    V = np.stack([u // (dims[1] * dims[2]), (u // dims[2]) % dims[1], u % dims[2]], 1) + lo
    V = V.astype(np.float64) * v
    I = inv.reshape(-1, 4)
    F = np.concatenate([I[:, [0, 1, 2]], I[:, [0, 2, 3]]])
    n_faces_raw = len(F)
    print(f"[collision] {n_out} voxels ({n_out - n_raw} added by closing), {n_faces_raw} faces, "
          f"{time.time() - t0:.0f}s; simplifying", flush=True)
    n_hidden = hide_behind(V, twin / "structure_collision.npz", 2.0 * v) if a.hide else 0
    if a.ratio < 1.0 and len(F) > 1000:
        import fast_simplification
        V, F = fast_simplification.simplify(V, F.astype(np.int64), target_reduction=1.0 - a.ratio)
    inset = a.inset if a.inset is not None else 0.0
    if inset:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
        mesh.compute_vertex_normals()
        V = V - inset * np.asarray(mesh.vertex_normals)
    out = Path(a.out) if a.out else twin / "structure_collision_voxel.npz"
    np.savez_compressed(out, V=V.astype(np.float32), F=F.astype(np.uint32))
    stats = dict(mode="voxel", points=src, voxel_m=v, close_m=G["r"] * v, min_h_m=a.min_h, min_count=a.min_count,
                 min_voxels=a.min_voxels, region=a.region, voxels_raw=n_raw, voxels_out=n_out,
                 faces_voxel=int(n_faces_raw), faces_out=int(len(F)), vertices_out=int(len(V)), ratio=a.ratio,
                 inset_m=inset, vertices_moved_behind_visual=n_hidden, seconds=round(time.time() - t0, 1))
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(f"[collision] voxel: {n_faces_raw} -> {len(F)} faces, {time.time() - t0:.0f}s -> {out}")


def build_mesh(cfg, a, twin):
    files = sorted(twin.glob("s_*.glb"))
    if not files:
        sys.exit(f"no structure tiles in {twin}")
    inset = a.inset if a.inset is not None else 0.06
    t0 = time.time()
    Vs, Fs, n_in, n_drop, off = [], [], 0, 0, 0
    for i, f in enumerate(files):
        r, n0, dr = simplify_tile(*load_tile(f), a.ratio, a.min_faces, a.min_extent, inset)
        n_in += n0; n_drop += dr
        if r is not None:
            Vs.append(r[0].astype(np.float32)); Fs.append((r[1] + off).astype(np.uint32)); off += len(r[0])
        if (i + 1) % 20 == 0:
            print(f"[collision] {i + 1}/{len(files)} tiles, {time.time() - t0:.0f}s", flush=True)
    V, F = np.concatenate(Vs), np.concatenate(Fs)
    out = Path(a.out) if a.out else twin / "structure_collision.npz"
    np.savez_compressed(out, V=V, F=F)
    stats = dict(mode="mesh", tiles=len(files), faces_in=int(n_in), faces_dropped_small_pieces=int(n_drop),
                 faces_out=int(len(F)), vertices_out=int(len(V)), ratio=a.ratio, min_faces=a.min_faces,
                 min_extent_m=a.min_extent, inset_m=inset, seconds=round(time.time() - t0, 1))
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(f"[collision] {n_in} -> {len(F)} faces ({n_drop} in small pieces dropped), "
          f"{time.time() - t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/blue_wall.json"))
    ap.add_argument("--variant", default="full")
    ap.add_argument("--mode", choices=("mesh", "voxel"), default="mesh",
                    help="mesh: simplify the visual structure tiles; voxel: closed voxel shell from the point cloud")
    ap.add_argument("--ratio", type=float, default=0.12, help="keep this fraction of faces")
    ap.add_argument("--inset", type=float, default=None,
                    help="push collision against the surface normal [m] (default: mesh 0.06, voxel 0)")
    ap.add_argument("--out", default=None, help="output .npz (default: <variant>/structure_collision[_voxel].npz)")
    g = ap.add_argument_group("mesh mode")
    g.add_argument("--min-faces", type=int, default=60, help="pieces with fewer faces are dropped")
    g.add_argument("--min-extent", type=float, default=0.4, help="pieces smaller than this [m] are dropped")
    g = ap.add_argument_group("voxel mode")
    g.add_argument("--voxel", type=float, default=0.125, help="voxel size [m]")
    g.add_argument("--close", type=float, default=0.375, help="closing radius [m]: gaps up to ~2x this are filled")
    g.add_argument("--min-h", type=float, default=0.3, help="points lower than this above the ground are ignored")
    g.add_argument("--max-h", type=float, default=300.0, help="points higher than this above the ground are ignored")
    g.add_argument("--min-count", type=int, default=1, help="points needed to occupy a voxel")
    g.add_argument("--min-voxels", type=int, default=12, help="connected blobs with fewer voxels are dropped")
    g.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="only this local-metre box (tests)")
    g.add_argument("--no-hide", dest="hide", action="store_false",
                   help="keep the raw voxel shell (default: vertices in front of the visual surface are moved "
                        "behind it using structure_collision.npz, so Isaac Gym does not show them grey)")
    g.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    twin = ROOT / "assets/twin" / cfg["name"] / a.variant
    (build_voxel if a.mode == "voxel" else build_mesh)(cfg, a, twin)


if __name__ == "__main__":
    main()
