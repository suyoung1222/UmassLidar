"""Agisoft Metashape project (.psx / .files) reader for aerial photogrammetry twins.

No Metashape licence needed: everything is read from the project's zip / xml / exr files.

    <project>.files/<chunk>/chunk.zip/doc.xml     sensors (adjusted calibration), cameras
                                                  (camera -> chunk transform), chunk -> ECEF transform
    <project>.files/<chunk>/0/depth_maps/data*.zip  per-photo depth maps (EXR, undistorted pinhole,
                                                    chunk units) + their pinhole calibration
    (dense_cloud.oc3 is a proprietary octree and is not read; the cloud is rebuilt from depth maps)

Conventions: camera axes x right, y down, z forward. Metashape image model
    x' = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 x^2) + 2 p2 x y
    y' = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p2 (r^2 + 2 y^2) + 2 p1 x y
    u = w/2 + cx + f x',  v = h/2 + cy + f y'        (pixel centre of pixel (0,0) is (0.5, 0.5))
World frame = UTM easting / northing / height, local to cfg["origin"], after the constant
cfg["align_shift"] (aerial -> LiDAR frame).
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import cv2                                    # noqa: E402
import numpy as np                            # noqa: E402

from .resepi import utm_forward               # noqa: E402


def _fl(s):
    return np.array(s.split(), float)


def ecef_to_geodetic(X):
    a = 6378137.0; f = 1 / 298.257223563; e2 = f * (2 - f)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h


class Chunk:
    """Aligned cameras of one Metashape chunk component, expressed in the local world frame."""

    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        self.dir = Path(cfg["chunk_dir"]).expanduser()
        self.photos_dir = Path(cfg["photos_dir"]).expanduser()
        self.origin = np.array(cfg["origin"], float)
        self.shift = np.array(cfg.get("align_shift", [0, 0, 0]), float)
        d = zipfile.ZipFile(self.dir / "chunk.zip").read("doc.xml").decode()
        comp = str(cfg.get("component", 1))
        m = re.search(r'<component id="%s".*?<rotation[^>]*>(.*?)</rotation>\s*<translation[^>]*>(.*?)'
                      r'</translation>\s*<scale[^>]*>(.*?)</scale>' % comp, d, re.S)
        self.Rc, self.Tc, self.S = _fl(m.group(1)).reshape(3, 3), _fl(m.group(2)), float(m.group(3))
        # sensors
        self.sensors = {}
        for m in re.finditer(r'<sensor id="(\d+)".*?<calibration type="frame" class="adjusted">(.*?)</calibration>',
                             d, re.S):
            c = m.group(2)
            g = lambda k, default=0.0: float(re.search(r"<%s>(.*?)</%s>" % (k, k), c).group(1)) \
                if re.search(r"<%s>" % k, c) else default   # noqa: E731
            wh = re.search(r'<resolution width="(\d+)" height="(\d+)"', c)
            self.sensors[int(m.group(1))] = dict(
                w=int(wh.group(1)), h=int(wh.group(2)), f=g("f"), cx=g("cx"), cy=g("cy"), k1=g("k1"),
                k2=g("k2"), k3=g("k3"), p1=g("p1"), p2=g("p2"))
        ids, labels, sens, T = [], [], [], []
        for m in re.finditer(r'<camera id="(\d+)" sensor_id="(\d+)" component_id="(\d+)" label="([^"]*)">\s*'
                             r'<transform>(.*?)</transform>', d, re.S):
            if m.group(3) != comp:
                continue
            ids.append(int(m.group(1))); sens.append(int(m.group(2))); labels.append(m.group(4))
            T.append(_fl(m.group(5)).reshape(4, 4))
        self.ids = np.array(ids); self.labels = labels; self.sensor_id = np.array(sens)
        self.index = {cid: k for k, cid in enumerate(ids)}
        T = np.array(T)
        Cl = T[:, :3, 3]
        # camera centre + axes in world: the ECEF -> UTM map is treated as locally affine per camera
        self.C = self._to_world(Cl)
        ax = np.stack([self._to_world(Cl + T[:, :3, k]) - self.C for k in range(3)], -1) / self.S   # (M,3,3)
        U, _, Vt = np.linalg.svd(ax)
        self.R = U @ Vt                                   # world <- camera
        self.depth_index = None
        if verbose:
            print(f"[metashape] {self.dir.name}: {len(ids)} aligned cameras, scale {self.S:.3f} m/unit, "
                  f"sensors {list(self.sensors)}", flush=True)

    def _to_world(self, Xl):
        lat, lon, h = ecef_to_geodetic(self.S * Xl @ self.Rc.T + self.Tc)
        E, N = utm_forward(lat, lon, 18)
        return np.stack([E, N, h], -1) + self.shift - self.origin

    def photo_path(self, k):
        return self.photos_dir / f"{self.labels[k]}.JPG"

    # ---- projection
    def cam(self, k):
        return self.sensors[int(self.sensor_id[k])]

    def to_cam(self, k, P):
        return (P - self.C[k]) @ self.R[k]

    def pix(self, k, Xc):
        """Camera-frame points -> (uv in the full-resolution distorted photo, normalised xy)."""
        s = self.cam(k)
        x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        r2 = x * x + y * y
        rad = 1 + r2 * (s["k1"] + r2 * (s["k2"] + r2 * s["k3"]))
        xd = x * rad + s["p1"] * (r2 + 2 * x * x) + 2 * s["p2"] * x * y
        yd = y * rad + s["p2"] * (r2 + 2 * y * y) + 2 * s["p1"] * x * y
        return np.stack([s["w"] / 2 + s["cx"] + s["f"] * xd, s["h"] / 2 + s["cy"] + s["f"] * yd], -1), \
            np.stack([x, y], -1)

    # ---- depth maps
    def _build_depth_index(self):
        self.depth_index = {}
        for bz in sorted((self.dir / "0/depth_maps").glob("data*.zip")):
            x = zipfile.ZipFile(bz).read("doc.xml").decode()
            for m in re.finditer(r'<depth_map camera_id="(\d+)" level="0" path="([^"]*)">.*?<resolution width="(\d+)" '
                                 r'height="(\d+)"/>\s*<f>(.*?)</f>\s*<cx>(.*?)</cx>\s*<cy>(.*?)</cy>', x, re.S):
                cid = int(m.group(1))
                if cid in self.index:
                    self.depth_index[self.index[cid]] = (bz, m.group(2), int(m.group(3)), int(m.group(4)),
                                                         float(m.group(5)), float(m.group(6)), float(m.group(7)))

    def depth(self, k):
        """Depth along the optical axis in metres (0 = invalid) + pinhole (w, h, f, cx, cy)."""
        if self.depth_index is None:
            self._build_depth_index()
        if k not in self.depth_index:
            return None, None
        bz, path, w, h, f, cx, cy = self.depth_index[k]
        # the single channel is named "Z", which OpenCV's EXR reader ignores -> use OpenEXR
        import OpenEXR, Imath, tempfile
        tmp = "/dev/shm" if os.path.isdir("/dev/shm") else None
        with tempfile.NamedTemporaryFile(suffix=".exr", dir=tmp) as tf:
            tf.write(zipfile.ZipFile(bz).read(path)); tf.flush()
            e = OpenEXR.InputFile(tf.name)
            ch = list(e.header()["channels"].keys())[0]
            d = np.frombuffer(e.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)
            e.close()
        return d.astype(np.float32) * self.S, (w, h, f, cx, cy)

    def backproject_depth(self, k, with_color=True):
        """All valid depth-map pixels of camera k as world points (+ rgb from the 1/8 photo)."""
        d, cal = self.depth(k)
        if d is None:
            return None
        w, h, f, cx, cy = cal
        vv, uu = np.nonzero(d > 0)
        z = d[vv, uu].astype(np.float64)
        Xc = np.stack([(uu + 0.5 - (w / 2 + cx)) / f * z, (vv + 0.5 - (h / 2 + cy)) / f * z, z], -1)
        P = Xc @ self.R[k].T + self.C[k]
        rgb = None
        if with_color:
            im = cv2.imread(str(self.photo_path(k)), cv2.IMREAD_REDUCED_COLOR_8)
            if im is not None:
                im = cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im
                rgb = im[vv, uu, ::-1]
        return P, rgb


def build_cloud(chunk, bbox_local, out_npz, margin=5.0, cam_range=250.0, tile=20.0, verbose=True):
    """Back-project every depth map near the bbox -> points (local xyz f32, rgb u8, camera index u16),
    sorted by 20 m tile for fast spatial access."""
    x0, y0, x1, y1 = bbox_local
    P, C, K = [], [], []
    near = np.nonzero((chunk.C[:, 0] > x0 - cam_range) & (chunk.C[:, 0] < x1 + cam_range) &
                      (chunk.C[:, 1] > y0 - cam_range) & (chunk.C[:, 1] < y1 + cam_range))[0]
    for n, k in enumerate(near):
        r = chunk.backproject_depth(int(k))
        if r is None or r[1] is None:
            continue
        p, c = r
        m = (p[:, 0] > x0 - margin) & (p[:, 0] < x1 + margin) & (p[:, 1] > y0 - margin) & (p[:, 1] < y1 + margin)
        if m.any():
            P.append(p[m].astype(np.float32)); C.append(c[m]); K.append(np.full(int(m.sum()), k, np.uint16))
        if verbose and n % 100 == 0:
            print(f"[metashape] depth maps {n}/{len(near)}", flush=True)
    P, C, K = np.concatenate(P), np.concatenate(C), np.concatenate(K)
    nx = int(np.ceil((x1 - x0 + 2 * margin) / tile)); ny = int(np.ceil((y1 - y0 + 2 * margin) / tile))
    tid = np.clip(((P[:, 1] - y0 + margin) / tile).astype(np.int64), 0, ny - 1) * nx + \
        np.clip(((P[:, 0] - x0 + margin) / tile).astype(np.int64), 0, nx - 1)
    o = np.argsort(tid, kind="stable")
    starts = np.searchsorted(tid[o], np.arange(nx * ny + 1))
    np.savez(out_npz, xyz=P[o], rgb=C[o], cam=K[o], starts=starts, grid=np.array([x0 - margin, y0 - margin, tile, nx, ny]))
    if verbose:
        print(f"[metashape] cloud: {len(P)/1e6:.1f} M points from {len(near)} depth maps -> {out_npz}", flush=True)
    return out_npz


class Cloud:
    def __init__(self, npz):
        z = np.load(npz)
        self.xyz, self.rgb, self.cam, self.starts = z["xyz"], z["rgb"], z["cam"], z["starts"]
        self.x0, self.y0, self.tile, nx, ny = z["grid"]
        self.nx, self.ny = int(nx), int(ny)

    def query(self, bounds, margin=0.0):
        """Index array of points inside bounds (+margin)."""
        x0, y0, x1, y1 = bounds
        T = self.tile
        out = []
        for ty in range(max(0, int((y0 - margin - self.y0) // T)), min(self.ny, int((y1 + margin - self.y0) // T) + 1)):
            for tx in range(max(0, int((x0 - margin - self.x0) // T)), min(self.nx, int((x1 + margin - self.x0) // T) + 1)):
                s, e = self.starts[ty * self.nx + tx], self.starts[ty * self.nx + tx + 1]
                p = self.xyz[s:e]
                k = (p[:, 0] >= x0 - margin) & (p[:, 0] < x1 + margin) & (p[:, 1] >= y0 - margin) & (p[:, 1] < y1 + margin)
                out.append(s + np.nonzero(k)[0])
        return np.concatenate(out) if out else np.zeros(0, np.int64)


def write_las(cloud, origin, path):
    """Cloud -> LAS (UTM, point format 2) so las_terrain.build_terrain_from_las can consume it."""
    import laspy
    hd = laspy.LasHeader(point_format=2, version="1.2")
    hd.scales = [0.001] * 3; hd.offsets = [float(v) for v in origin]
    las = laspy.LasData(hd)
    las.x = cloud.xyz[:, 0].astype(np.float64) + origin[0]
    las.y = cloud.xyz[:, 1].astype(np.float64) + origin[1]
    las.z = cloud.xyz[:, 2].astype(np.float64) + origin[2]
    las.red, las.green, las.blue = (cloud.rgb[:, k].astype(np.uint16) * 256 for k in range(3))
    las.write(str(path))
    return path


class AerialBank:
    """Photos selected for one tile (full-resolution crops) + Metashape depth maps for occlusion.
    Duck-types texture.PhotoBank for texture.project_photos (aabb / overlapping / sample)."""

    def __init__(self, chunk, bounds, zlo, zhi, per_sector=5, sectors=8, pad_px=48):
        self.c = chunk
        x0, y0, x1, y1 = bounds
        ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (zlo + zhi) / 2])
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (zlo, zhi)] + [ctr])
        cand = []
        for k in range(len(chunk.ids)):
            d = ctr - chunk.C[k]
            dist = np.linalg.norm(d)
            if dist > 320 or d[2] > -20:
                continue
            Xc = chunk.to_cam(k, ctr[None])
            if Xc[0, 2] <= 1:
                continue
            uv, _ = chunk.pix(k, Xc)
            s = chunk.cam(k)
            if not (0 < uv[0, 0] < s["w"] and 0 < uv[0, 1] < s["h"]):
                continue
            off = np.hypot(uv[0, 0] - s["w"] / 2, uv[0, 1] - s["h"] / 2) / s["w"]   # prefer tile near image centre
            cand.append((int((np.degrees(np.arctan2(d[1], d[0])) % 360) // (360 / sectors)), dist * (1 + off), k))
        cand.sort()
        sel = []
        for sct in range(sectors):
            sel += [k for s_, _, k in cand if s_ == sct][:per_sector]
        self.k, self.img, self.off, self.depth, self.dcal, self.edge = [], [], [], [], [], []
        for k in sel:
            Xc = chunk.to_cam(k, corners)
            if (Xc[:, 2] <= 1).any():
                continue
            uv, _ = chunk.pix(k, Xc)
            s = chunk.cam(k)
            u0, v0 = np.floor(uv.min(0)).astype(int) - pad_px
            u1, v1 = np.ceil(uv.max(0)).astype(int) + pad_px
            u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, s["w"]), min(v1, s["h"])
            if u1 - u0 < 16 or v1 - v0 < 16:
                continue
            d, cal = chunk.depth(k)
            if d is None:
                continue
            im = cv2.imread(str(chunk.photo_path(k)), cv2.IMREAD_COLOR)
            if im is None:
                continue
            self.k.append(k); self.img.append(im[v0:v1, u0:u1, ::-1].copy()); self.off.append((u0, v0))
            dv = np.where(d > 0, d, np.nan)
            big = np.where(np.isnan(dv), -1e6, dv); small = np.where(np.isnan(dv), 1e6, dv)
            rng = cv2.dilate(big, np.ones((3, 3))) - cv2.erode(small, np.ones((3, 3)))
            self.depth.append(dv); self.dcal.append(cal); self.edge.append(rng)
        M = len(self.k)
        self.aabb = np.tile(np.array([x0 - 1, y0 - 1, x1 + 1, y1 + 1], float), (M, 1))

    def overlapping(self, bounds):
        return np.arange(len(self.k))

    def sample(self, i, P, N):
        c, k = self.c, self.k[i]
        s = c.cam(k)
        w = np.zeros(len(P), np.float32); rgb = np.zeros((len(P), 3), np.float32)
        d = P.astype(np.float64) - c.C[k]
        Xc = d @ c.R[k]
        ok = Xc[:, 2] > 2.0
        if not ok.any():
            return rgb, w
        idx = np.nonzero(ok)[0]
        uv, xy = c.pix(k, Xc[idx])
        im = self.img[i]; u0, v0 = self.off[i]
        px = np.floor(uv).astype(int) - [u0, v0]
        inb = (px[:, 0] >= 0) & (px[:, 0] < im.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < im.shape[0])
        idx, uv, xy, px = idx[inb], uv[inb], xy[inb], px[inb]
        if not len(idx):
            return rgb, w
        z = Xc[idx, 2]
        dw, dh, df, dcx, dcy = self.dcal[i]
        du = np.clip((dw / 2 + dcx + df * xy[:, 0]).astype(int), 0, dw - 1)
        dv = np.clip((dh / 2 + dcy + df * xy[:, 1]).astype(int), 0, dh - 1)
        dm = self.depth[i][dv, du]
        dist = np.linalg.norm(d[idx], axis=1)
        cos = -np.einsum("ij,ij->i", N[idx].astype(np.float64), d[idx]) / np.maximum(dist, 1e-6)
        tol = 0.5 + 0.006 * z
        # near a depth discontinuity the 1/8 depth map cannot tell roof from ground: demand a closer match
        vis = np.isfinite(dm) & (np.abs(z - dm) < np.where(self.edge[i][dv, du] > 2.0, 0.35 * tol, tol)) & (cos > 0.1)
        idx, uv, px, z, cos = idx[vis], uv[vis], px[vis], z[vis], cos[vis]
        if not len(idx):
            return rgb, w
        fade = np.minimum(np.minimum(uv[:, 0], s["w"] - uv[:, 0]) / (0.06 * s["w"]),
                          np.minimum(uv[:, 1], s["h"] - uv[:, 1]) / (0.06 * s["h"]))
        rgb[idx] = im[px[:, 1], px[:, 0]].astype(np.float32)
        w[idx] = (cos * np.clip(fade, 0.02, 1) * (100.0 / z) ** 2).astype(np.float32) + 1e-9
        return rgb, w
