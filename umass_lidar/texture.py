"""Texture baking for the digital twin: photos -> texels, with LiDAR fallbacks.

Per texel (world position P, normal N) the colour comes from, in order:
    1. photo  : calibrated photos that see the texel (LiDAR depth test for occlusion), blended
                with weight cos(incidence) / distance^2 * border fade
    2. las_rgb: colour PCMaster stored on nearby LiDAR points (covers farther surfaces)
    3. material: no colour measured. 905 nm LiDAR intensity does not predict visible brightness
                (corr ~ -0.07 on this dataset), so it is used only (a) to pick a material bin
                per surface type and (b) as fine-scale shading detail:
                    colour = mean photo colour of (surface type, intensity bin) * detail
The lighting is whatever the overcast-day photos contain (baked, no specular / shadows added).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .camera import clean_depth, lidar_view, load_color
from .mesh import ZUP_TO_YUP

SRC_NONE, SRC_PHOTO, SRC_LAS, SRC_MATERIAL, SRC_NODATA, SRC_LAS_LOW = 0, 1, 2, 3, 4, 5
N_TYPES, N_BINS = 4, 12          # surface types: 0 flat ground, 1 sloped ground, 2 vertical, 3 other


# --------------------------------------------------------------------------- intensity
class IntensityModel:
    """Per-laser gain equalisation + percentile stretch + gamma -> [0, 1]."""

    def __init__(self, gains, lo, hi, gamma=0.5):
        self.gains, self.lo, self.hi, self.gamma = np.asarray(gains, np.float32), lo, hi, gamma

    @classmethod
    def fit(cls, sub_npz, gamma=0.5):
        d = np.load(sub_npz)
        I, las = d["intensity"].astype(np.float32), d["laser"]
        med = np.array([np.median(I[las == k]) if (las == k).any() else np.nan for k in range(256)])
        g = np.nanmedian(med) / np.where(np.isfinite(med) & (med > 0), med, np.nanmedian(med))
        In = I * g[las]
        lo, hi = np.percentile(In, [1, 99.5])
        return cls(g, float(lo), float(hi), gamma)

    def __call__(self, intensity, laser):
        x = intensity.astype(np.float32) * self.gains[laser]
        return np.clip((x - self.lo) / (self.hi - self.lo), 0, 1) ** self.gamma

    def to_dict(self):
        return dict(per_laser_gain=self.gains[:32].round(4).tolist(), stretch=[self.lo, self.hi],
                    gamma=self.gamma)


# --------------------------------------------------------------------------- photos
class PhotoBank:
    """All photos (1/reduce) + LiDAR depth maps (1/depth_scale) in RAM, with ground footprints."""

    def __init__(self, cfg, session, model, tiles, reduce=4, depth_scale=8, max_dist=12.0,
                 verbose=True):
        self.s, self.m, self.reduce, self.ds, self.max_dist = session, model, reduce, depth_scale, max_dist
        M = len(session.photos)
        self.R, self.C, self.tc = model.extrinsics(session, np.arange(M))
        ddir = Path(cfg["cache_dir"]) / f"depth_s{depth_scale}"
        ddir.mkdir(parents=True, exist_ok=True)
        self.img, self.depth, self.aabb = [], [], np.zeros((M, 4))
        h, w = model.height // depth_scale, model.width // depth_scale
        gu, gv = np.meshgrid((np.arange(0, w, 6) + 0.5) * depth_scale, (np.arange(0, h, 6) + 0.5) * depth_scale)
        guv = np.stack([gu.ravel(), gv.ravel()], -1)
        for i in range(M):
            f = ddir / f"{i:04d}.npy"
            if f.exists():
                d = np.load(f)
            else:
                d = clean_depth(lidar_view(model, session, i, tiles, depth_scale)["depth"]).astype(np.float16)
                np.save(f, d)
            self.depth.append(d)
            self.img.append(load_color(session.photos[i], reduce))
            z = d[::6, ::6].astype(np.float64).ravel()
            k = np.isfinite(z) & (z < max_dist)
            if k.sum() < 10:
                self.aabb[i] = (1, 1, 0, 0)
                continue
            X = model.backproject(session, i, guv[k], z[k])
            self.aabb[i] = (X[:, 0].min() - 0.3, X[:, 1].min() - 0.3, X[:, 0].max() + 0.3, X[:, 1].max() + 0.3)
            if verbose and i % 100 == 0:
                print(f"[texture] photo bank {i}/{M}", flush=True)
        self.gain = np.ones(M, np.float32)

    def overlapping(self, bounds):
        x0, y0, x1, y1 = bounds
        a = self.aabb
        return np.nonzero((a[:, 0] < x1) & (a[:, 2] > x0) & (a[:, 1] < y1) & (a[:, 3] > y0))[0]

    def sample(self, i, P, N):
        """Colour + weight of texels (P, N) seen in photo i. Returns (rgb float32, w float32)."""
        m = self.m
        d = P - self.C[i]
        Xc = d @ self.R[i].T + self.tc[i]
        z = Xc[:, 2]
        w = np.zeros(len(P), np.float32)
        rgb = np.zeros((len(P), 3), np.float32)
        ok = (z > 0.4) & (z < self.max_dist)
        if not ok.any():
            return rgb, w
        uv, _ = m.pix(Xc[ok])
        k = (uv[:, 0] >= 1) & (uv[:, 0] < m.width - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < m.height - 2)
        idx = np.nonzero(ok)[0][k]
        uv = uv[k]; z = z[idx]
        dm = self.depth[i][(uv[:, 1] / self.ds).astype(int), (uv[:, 0] / self.ds).astype(int)].astype(np.float32)
        dist = np.linalg.norm(d[idx], axis=1)
        cos = -np.einsum("ij,ij->i", N[idx], d[idx]) / np.maximum(dist, 1e-6)
        vis = np.isfinite(dm) & (np.abs(z - dm) < 0.04 * z + 0.06) & (cos > 0.15)
        idx, uv, z, cos = idx[vis], uv[vis], z[vis], cos[vis]
        if not len(idx):
            return rgb, w
        fade = np.minimum(np.minimum(uv[:, 0], m.width - uv[:, 0]) / (0.08 * m.width),
                          np.minimum(uv[:, 1], m.height - uv[:, 1]) / (0.08 * m.height))
        im = self.img[i]
        px = np.clip((uv / self.reduce).astype(int), 0, [im.shape[1] - 1, im.shape[0] - 1])
        rgb[idx] = im[px[:, 1], px[:, 0]].astype(np.float32) * self.gain[i]
        w[idx] = (cos * np.clip(fade, 0, 1) / (z * z)).astype(np.float32) + 1e-9
        return rgb, w


def project_photos(bank, P, N, bounds, cell=2.0, power=3.0, stats=None):
    """Blend all photos into texels. P, N: (K,3) float32. Returns (rgb (K,3), wsum (K,))."""
    K = len(P)
    acc = np.zeros((K, 3), np.float32); wsum = np.zeros(K, np.float32)
    x0, y0, x1, y1 = bounds
    ncx, ncy = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
    cid = np.clip(((P[:, 1] - y0) / cell).astype(np.int64), 0, ncy - 1) * ncx + \
        np.clip(((P[:, 0] - x0) / cell).astype(np.int64), 0, ncx - 1)
    order = np.argsort(cid, kind="stable")
    starts = np.searchsorted(cid[order], np.arange(ncx * ncy + 1))
    for i in bank.overlapping(bounds):
        a = bank.aabb[i]
        cx0, cx1 = max(0, int((a[0] - x0) // cell)), min(ncx - 1, int((a[2] - x0) // cell))
        cy0, cy1 = max(0, int((a[1] - y0) // cell)), min(ncy - 1, int((a[3] - y0) // cell))
        if cx1 < cx0 or cy1 < cy0:
            continue
        segs = [order[starts[cy * ncx + cx0]:starts[cy * ncx + cx1 + 1]] for cy in range(cy0, cy1 + 1)]
        idx = np.concatenate(segs)
        if not len(idx):
            continue
        rgb, w = bank.sample(i, P[idx], N[idx])
        k = w > 0
        if not k.any():
            continue
        wp = w[k] ** power
        acc[idx[k]] += rgb[k] * wp[:, None]
        wsum[idx[k]] += wp
        if stats is not None:
            stats[i] = stats.get(i, 0) + int(k.sum())
    good = wsum > 0
    acc[good] /= wsum[good, None]
    return acc, wsum


# --------------------------------------------------------------------------- material table
class MaterialTable:
    """Mean photo colour per (surface type, intensity bin), accumulated over all texels."""

    def __init__(self):
        self.sum = np.zeros((N_TYPES, N_BINS, 3)); self.cnt = np.zeros((N_TYPES, N_BINS))

    def add(self, stype, inten, rgb):
        b = np.clip((inten * N_BINS).astype(int), 0, N_BINS - 1)
        lin = stype.astype(int) * N_BINS + b
        for c in range(3):
            self.sum[..., c] += np.bincount(lin, rgb[:, c], N_TYPES * N_BINS).reshape(N_TYPES, N_BINS)
        self.cnt += np.bincount(lin, minlength=N_TYPES * N_BINS).reshape(N_TYPES, N_BINS)

    def table(self, min_count=2000):
        t = self.sum / np.maximum(self.cnt, 1)[..., None]
        allmean = self.sum.sum((0, 1)) / max(self.cnt.sum(), 1)
        for s in range(N_TYPES):                       # fill empty bins from neighbours / type mean
            ok = self.cnt[s] >= min_count
            tm = self.sum[s].sum(0) / max(self.cnt[s].sum(), 1) if self.cnt[s].sum() > min_count else allmean
            if ok.any():
                xs = np.nonzero(ok)[0]
                for c in range(3):
                    t[s, :, c] = np.interp(np.arange(N_BINS), xs, t[s, xs, c])
            else:
                t[s] = tm
        return t

    def color(self, table, stype, inten_smooth, detail):
        x = np.clip(inten_smooth * N_BINS - 0.5, 0, N_BINS - 1.001)
        b0 = x.astype(int); a = (x - b0)[:, None]
        base = table[stype, b0] * (1 - a) + table[stype, b0 + 1] * a
        return base * detail[:, None]

    def save(self, path):
        np.savez(path, sum=self.sum, cnt=self.cnt)

    @classmethod
    def load(cls, path):
        z = np.load(path); m = cls(); m.sum, m.cnt = z["sum"], z["cnt"]
        return m


def _detail(i_fine, i_smooth):
    return np.clip(np.sqrt((i_fine + 0.05) / (i_smooth + 0.05)), 0.7, 1.35)


# --------------------------------------------------------------------------- ground tiles
def _norm_conv(sum_img, cnt_img, sigma):
    s = ndimage.gaussian_filter(sum_img, sigma, mode="nearest")
    c = ndimage.gaussian_filter(cnt_img, sigma, mode="nearest")
    return s / np.maximum(c, 1e-6), c


def ground_texels(surf, mesh_res, tex_res, cells):
    """World position / normal of every texel of one ground tile.
    cells = (j0, j1, i0, i1): mesh vertex index ranges (inclusive) of the tile.
    Returns (P (K,3), N (K,3), z (th,tw), coords, bounds, (th, tw))."""
    j0, j1, i0, i1 = cells
    X0, X1 = (j0 + 0.5) * mesh_res, (j1 + 0.5) * mesh_res
    Y0, Y1 = (i0 + 0.5) * mesh_res, (i1 + 0.5) * mesh_res
    tw, th = int(round((X1 - X0) / tex_res)), int(round((Y1 - Y0) / tex_res))
    xs = X0 + (np.arange(tw) + 0.5) * (X1 - X0) / tw
    ys = Y0 + (np.arange(th) + 0.5) * (Y1 - Y0) / th
    # surface height / normal at texels (bilinear from the mesh grid)
    gy = (ys / mesh_res - 0.5); gx = (xs / mesh_res - 0.5)
    coords = np.meshgrid(gy, gx, indexing="ij")
    z = ndimage.map_coordinates(surf, coords, order=1, mode="nearest").astype(np.float32)
    dzy, dzx = np.gradient(surf, mesh_res)
    nx_ = -ndimage.map_coordinates(dzx, coords, order=1, mode="nearest")
    ny_ = -ndimage.map_coordinates(dzy, coords, order=1, mode="nearest")
    nrm = np.stack([nx_, ny_, np.ones_like(nx_)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    XX, YY = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    P = np.stack([XX, YY, z], -1).reshape(-1, 3)
    N = nrm.reshape(-1, 3).astype(np.float32)
    bounds = (X0, Y0, X1, Y1)
    return P, N, z, coords, bounds, (th, tw)


def ground_tile_stage_a(bank, tiles, imodel, surf, ground, sem, mesh_res, tex_res, cells, out_npz,
                        table):
    """Photo + LAS colour and intensity for one ground texture tile.
    cells = (j0, j1, i0, i1): mesh vertex index ranges (inclusive) of the tile."""
    P, N, z, coords, bounds, (th, tw) = ground_texels(surf, mesh_res, tex_res, cells)
    X0, Y0, X1, Y1 = bounds
    rgb, wsum = project_photos(bank, P, N, bounds)
    src = np.where(wsum > 0, SRC_PHOTO, SRC_NONE).astype(np.uint8)

    # LiDAR points near the surface -> per-texel colour and intensity
    csum = np.zeros((th * tw, 3)); ccnt = np.zeros(th * tw); isum = np.zeros(th * tw); icnt = np.zeros(th * tw)
    T = tiles.tile
    for ty in range(max(0, int(Y0 // T)), min(tiles.ny, int(Y1 // T) + 1)):
        for tx in range(max(0, int(X0 // T)), min(tiles.nx, int(X1 // T) + 1)):
            d = tiles.load(tx, ty, ("xyz", "rgb", "intensity", "laser"))
            xyz = np.asarray(d["xyz"])
            if not len(xyz):
                continue
            u = ((xyz[:, 0] - X0) / (X1 - X0) * tw).astype(np.int64)
            v = ((xyz[:, 1] - Y0) / (Y1 - Y0) * th).astype(np.int64)
            k = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
            lin = v[k] * tw + u[k]
            k2 = np.abs(xyz[k, 2] - z.ravel()[lin]) < 0.15
            sel = np.nonzero(k)[0][k2]; lin = lin[k2]
            In = imodel(np.asarray(d["intensity"])[sel], np.asarray(d["laser"])[sel])
            isum += np.bincount(lin, In, th * tw); icnt += np.bincount(lin, minlength=th * tw)
            c = np.asarray(d["rgb"])[sel]
            col = c.sum(1) > 0
            for ch in range(3):
                csum[:, ch] += np.bincount(lin[col], c[col, ch].astype(np.float64), th * tw)
            ccnt += np.bincount(lin[col], minlength=th * tw)
    i_fine, ic = _norm_conv(isum.reshape(th, tw), icnt.reshape(th, tw), 1.0)
    i_smooth, _ = _norm_conv(isum.reshape(th, tw), icnt.reshape(th, tw), 0.12 / tex_res)
    # LAS colour: sharp where coloured points are dense; where they are sparse (far-range colouring)
    # only their low-frequency colour is trusted and the fine structure comes from intensity
    cc2 = ccnt.reshape(th, tw); ic2 = icnt.reshape(th, tw)
    las_rgb = np.stack([_norm_conv(csum[:, ch].reshape(th, tw), cc2, 0.8)[0] for ch in range(3)], -1)
    dense = (ndimage.gaussian_filter(cc2, 3.0) > 0.5 * ndimage.gaussian_filter(ic2, 3.0)) & \
        (ndimage.gaussian_filter(cc2, 0.8) > 0.05)
    use_las = (src == SRC_NONE) & dense.ravel()
    rgb[use_las] = las_rgb.reshape(-1, 3)[use_las]
    src[use_las] = SRC_LAS
    s_low = 0.25 / tex_res
    low = np.stack([_norm_conv(csum[:, ch].reshape(th, tw), cc2, s_low)[0] for ch in range(3)], -1)
    low_ok = ndimage.gaussian_filter(cc2, s_low) * (2 * np.pi * s_low ** 2) > 3.0      # >= ~3 coloured pts nearby
    use_low = (src == SRC_NONE) & low_ok.ravel()
    rgb[use_low] = (low * _detail(i_fine, i_smooth)[..., None]).reshape(-1, 3)[use_low]
    src[use_low] = SRC_LAS_LOW
    # texels with no LiDAR support at all (outside the scan): no-data
    semt = ndimage.map_coordinates(sem.astype(np.float32), coords, order=0, mode="nearest").astype(np.uint8)
    nodata = (semt.ravel() == 3) & (src == SRC_NONE)
    src[nodata] = SRC_NODATA
    stype = np.where(N[:, 2] > 0.9, 0, 1).astype(np.uint8)
    known = src == SRC_PHOTO
    table.add(stype[known], i_smooth.ravel()[known], rgb[known])
    np.savez(out_npz, rgb=np.clip(rgb, 0, 255).astype(np.uint8).reshape(th, tw, 3), src=src.reshape(th, tw),
             i_fine=(i_fine * 255).astype(np.uint8), i_smooth=(i_smooth * 255).astype(np.uint8),
             stype=stype.reshape(th, tw), bounds=np.array(bounds))
    return dict(shape=(th, tw), photo=float((src == SRC_PHOTO).mean()), las=float((src == SRC_LAS).mean()))


def finish_texture(npz, table_rgb, feather_px=6, nodata_rgb=(96, 96, 96)):
    """Fill unmeasured texels with material colour and feather the seam. Returns (rgb, src)."""
    z = np.load(npz)
    rgb = z["rgb"].astype(np.float32); src = z["src"].copy()
    th, tw = src.shape
    mt = MaterialTable()
    i_f = z["i_fine"].astype(np.float32) / 255; i_s = z["i_smooth"].astype(np.float32) / 255
    mat = mt.color(table_rgb, z["stype"].ravel(), i_s.ravel(), _detail(i_f, i_s).ravel()).reshape(th, tw, 3)
    known = (src == SRC_PHOTO) | (src == SRC_LAS) | (src == SRC_LAS_LOW)
    a = np.clip(ndimage.distance_transform_edt(known) / feather_px, 0, 1)[..., None]
    out = a * rgb + (1 - a) * mat
    empty = src == SRC_NONE
    src[empty] = SRC_MATERIAL
    if "valid" in z.files:                       # atlas padding: dilate valid texels outwards
        valid = z["valid"]
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        out = out[tuple(idx)]
    nd = src == SRC_NODATA
    if nd.any():
        soft = ndimage.gaussian_filter(nd.astype(np.float32), 3)[..., None]
        out = out * (1 - soft) + np.array(nodata_rgb, np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8), src


def apply_grade(rgb, grade):
    """Global colour grade of a finished texture, in linear RGB, identical to the Gain / Saturation /
    Gamma knobs of the Blender "TwinShade" node group (scripts/blender_twin.py).
    grade = dict(gain=1.0, saturation=1.0, gamma=1.0) or None."""
    if not grade:
        return rgb
    c = rgb.astype(np.float32) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) * grade.get("gain", 1.0)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lin = lum[..., None] + (lin - lum[..., None]) * grade.get("saturation", 1.0)
    lin = np.clip(lin, 0, None) ** grade.get("gamma", 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.clip(lin, 1e-9, None) ** (1 / 2.4) - 0.055)
    return np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- structure atlases
def unwrap(V, F, texel=0.03, max_size=4096):
    """xatlas UV unwrap. Returns (V2, F2, UV in [0,1], tex_size)."""
    import xatlas
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    tpu = 1.0 / texel
    need = np.sqrt(area / 0.55) * tpu
    if need > max_size:
        tpu *= max_size / need
    atlas = xatlas.Atlas()
    atlas.add_mesh(V.astype(np.float32), F.astype(np.uint32))
    po = xatlas.PackOptions(); po.texels_per_unit = float(tpu); po.padding = 3; po.resolution = 0
    po.bilinear = True
    co = xatlas.ChartOptions(); co.max_iterations = 1
    atlas.generate(chart_options=co, pack_options=po)
    vmap, F2, uv = atlas[0]
    return V[vmap], F2.astype(np.int64), uv.astype(np.float64), (atlas.width, atlas.height)


def atlas_texels(V, F, UV, size):
    """Rasterise faces in UV space -> per-texel world position / normal. Returns
    (valid (h,w), P (K,3), N (K,3)) for valid texels in row-major order (row 0 = v=0)."""
    import cv2
    w, h = size
    fid = np.full((h, w), -1, np.int32)
    pix = UV * [w, h]
    tri = np.round(pix[F] * 16).astype(np.int32)       # 4 fractional bits
    for k in range(len(F)):
        cv2.fillConvexPoly(fid, tri[k], int(k), lineType=cv2.LINE_8, shift=4)
    valid = fid >= 0
    vy, vx = np.nonzero(valid)
    f = fid[vy, vx]
    a, b, c = pix[F[f, 0]], pix[F[f, 1]], pix[F[f, 2]]
    p = np.stack([vx + 0.5, vy + 0.5], -1)
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    wb = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    wc = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    wb = np.clip(wb, -0.2, 1.2); wc = np.clip(wc, -0.2, 1.2); wa = 1 - wb - wc
    P = wa[:, None] * V[F[f, 0]] + wb[:, None] * V[F[f, 1]] + wc[:, None] * V[F[f, 2]]
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    return valid, P.astype(np.float32), fn[f].astype(np.float32)


def structure_tile_stage_a(bank, tiles, imodel, V, F, UV, size, bounds, out_npz, table):
    from scipy.spatial import cKDTree
    from .structure import voxel_reduce
    valid, P, N = atlas_texels(V, F, UV, size)
    h, w = valid.shape
    rgb, wsum = project_photos(bank, P, N, bounds)
    src = np.where(wsum > 0, SRC_PHOTO, SRC_NONE).astype(np.uint8)
    # nearest LiDAR points: colour + intensity
    x0, y0, x1, y1 = bounds
    zlo, zhi = P[:, 2].min() - 0.3, P[:, 2].max() + 0.3
    acc = []
    T = tiles.tile
    for ty in range(max(0, int((y0 - 0.5) // T)), min(tiles.ny, int((y1 + 0.5) // T) + 1)):
        for tx in range(max(0, int((x0 - 0.5) // T)), min(tiles.nx, int((x1 + 0.5) // T) + 1)):
            d = tiles.load(tx, ty, ("xyz", "rgb", "intensity", "laser"))
            xyz = np.asarray(d["xyz"])
            if not len(xyz):
                continue
            k = (xyz[:, 0] >= x0 - 0.5) & (xyz[:, 0] < x1 + 0.5) & (xyz[:, 1] >= y0 - 0.5) & \
                (xyz[:, 1] < y1 + 0.5) & (xyz[:, 2] > zlo) & (xyz[:, 2] < zhi)
            c = np.asarray(d["rgb"])[k].astype(np.float64)
            col = (c.sum(1) > 0).astype(np.float64)
            In = imodel(np.asarray(d["intensity"])[k], np.asarray(d["laser"])[k]).astype(np.float64)
            acc.append(np.c_[xyz[k].astype(np.float64), c * col[:, None], col, In])
    i_fine = np.full(len(P), 0.3, np.float32); i_smooth = i_fine.copy()
    if acc:
        a = np.concatenate(acc)
        pts, ex, cnt = voxel_reduce(a[:, :3], 0.04, a[:, 3:])
        tree = cKDTree(pts)
        dist, nn = tree.query(P, k=4, distance_upper_bound=0.12, workers=8)
        ok = np.isfinite(dist); nnc = np.where(ok, nn, 0)
        wgt = ok / (dist ** 2 + 1e-4)
        wgt = np.where(ok, wgt, 0.0)
        ws = wgt.sum(1)
        colw = (ex[nnc, 3] * wgt)
        cs = colw.sum(1)
        las_rgb = np.einsum("nkc,nk->nc", ex[nnc, :3], wgt) / np.maximum(cs, 1e-9)[:, None]
        inten = (ex[nnc, 4] * wgt).sum(1) / np.maximum(ws, 1e-9)
        i_fine = np.where(ws > 0, inten, 0.3).astype(np.float32)
        use_las = (src == SRC_NONE) & (cs > 1e-6)
        rgb[use_las] = las_rgb[use_las]; src[use_las] = SRC_LAS
        # material-scale intensity: mean over ~15 cm
        nb = tree.query(P, k=16, distance_upper_bound=0.2, workers=8)[1]
        okb = nb < len(pts)
        i_smooth = (np.where(okb, ex[np.where(okb, nb, 0), 4], 0).sum(1) / np.maximum(okb.sum(1), 1)).astype(np.float32)
        i_smooth = np.where(okb.any(1), i_smooth, i_fine)
    stype = np.where(np.abs(N[:, 2]) < 0.5, 2, 3).astype(np.uint8)
    known = src == SRC_PHOTO
    table.add(stype[known], i_smooth[known], rgb[known])

    def img(vals, dtype, ch=None):
        out = np.zeros((h, w) if ch is None else (h, w, ch), dtype)
        out[valid] = vals
        return out
    np.savez(out_npz, rgb=img(np.clip(rgb, 0, 255), np.uint8, 3), src=img(src, np.uint8),
             i_fine=img(i_fine * 255, np.uint8), i_smooth=img(i_smooth * 255, np.uint8),
             stype=img(stype, np.uint8), valid=valid, bounds=np.array(bounds))
    return dict(shape=(h, w), photo=float((src == SRC_PHOTO).mean()), las=float((src == SRC_LAS).mean()))


# --------------------------------------------------------------------------- export
def write_glb(path, V, F, UV, image, name, double_sided=False):
    """Textured mesh (Z-up input) -> GLB in glTF Y-up. image row 0 must be v = 1 (top).
    double_sided duplicates the faces reversed (Isaac Gym culls back faces regardless of the
    material flag), which matters for thin things such as branches and railings."""
    import trimesh
    from PIL import Image
    if double_sided:
        F = np.concatenate([F, F[:, ::-1]])
    mat = trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=Image.fromarray(image),
                                              metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
    mesh = trimesh.Trimesh(V, F, process=False, visual=trimesh.visual.TextureVisuals(uv=UV, material=mat))
    mesh.apply_transform(ZUP_TO_YUP)
    mesh.export(path)


SRC_COLORS = np.array([[0, 0, 0], [60, 200, 90], [70, 130, 230], [230, 170, 50], [90, 90, 90],
                       [150, 190, 240]], np.uint8)
SRC_NAMES = ["none", "photo", "las_rgb", "material", "no_data", "las_rgb_lowfreq_x_intensity"]


def write_sky(path, center, radius=400.0, n_az=48, n_el=12):
    """Inward-facing overcast sky dome (the capture day was overcast; Isaac Gym has no sky)."""
    az = np.linspace(0, 2 * np.pi, n_az + 1); el = np.linspace(-0.12, np.pi / 2, n_el + 1)
    A, E = np.meshgrid(az, el)
    V = np.stack([center[0] + radius * np.cos(E) * np.cos(A), center[1] + radius * np.cos(E) * np.sin(A),
                  center[2] + radius * np.sin(E)], -1).reshape(-1, 3)
    UV = np.stack([A / (2 * np.pi), (E - el[0]) / (el[-1] - el[0])], -1).reshape(-1, 2)
    i = (np.arange(n_el)[:, None] * (n_az + 1) + np.arange(n_az)[None, :]).ravel()
    F = np.concatenate([np.stack([i, i + n_az + 2, i + 1], 1), np.stack([i, i + n_az + 1, i + n_az + 2], 1)])
    t = np.linspace(0, 1, 256)[:, None]                       # row 0 = zenith
    rng = np.random.default_rng(0)
    base = 228 - 26 * (1 - t) ** 1.5 + ndimage.gaussian_filter(rng.normal(0, 30, (256, 1024)), (10, 40))
    img = np.clip(np.stack([base * 0.985, base * 0.995, base * 1.02], -1), 0, 255).astype(np.uint8)
    write_glb(path, V, F, UV, img, "sky")
