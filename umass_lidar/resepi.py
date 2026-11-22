"""Inertial Labs RESEPI session loader (LiDAR + camera + post-processed INS trajectory).

Session layout (PCMaster project):
    ins/ppk.txt            trajectory, 125 Hz: Week GPSTime UTCTime Lon Lat H-Ell Heading Pitch Roll
    camera/<serial>_<t0>_<t1>.jpg   t in microseconds, t/1e6 - 1e9 = LAS gps_time (adjusted standard)
    processed_ppk.pcmp / .pcpp      lidar / camera lever arm + boresight, CRS
The georeferenced cloud exported by PCMaster (LAS) lives elsewhere and is linked by a config
json (configs/*.json). LAS point_source_id = laser number, gps_time -> sensor pose per point.

Frames:
    world : local metric frame = UTM (EPSG:32618) - origin, Z up (ellipsoidal height - z0)
    body  : IMU frame, x right, y forward, z up (Inertial Labs convention)
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GPS_WEEK = 604800.0
ADJ = 1e9   # LAS "adjusted standard GPS time" offset


def load_config(path=None):
    path = Path(path or ROOT / "configs/blue_wall.json")
    cfg = json.loads(path.read_text())
    cfg["session_dir"] = str(Path(cfg["session_dir"]).expanduser())
    cfg["las"] = str(Path(cfg["las"]).expanduser())
    cfg["twin_dir"] = str(ROOT / "assets/twin" / cfg["name"])
    cfg["cache_dir"] = str(ROOT / "data" / cfg["name"])
    return cfg


def utm_forward(lat_deg, lon_deg, zone=18):
    """WGS84 lat/lon (deg) -> UTM easting/northing (Transverse Mercator series, mm-level)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    lon0 = np.radians(zone * 6 - 183)
    a = 6378137.0; f = 1 / 298.257223563; k0 = 0.9996
    e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2; C = ep2 * np.cos(lat) ** 2; A = (lon - lon0) * np.cos(lat)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * np.tan(lat) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                     + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def grid_convergence(lat_deg, lon_deg, zone=18):
    """Angle (rad) from grid north to true north, positive east of the central meridian."""
    return np.arctan(np.tan(np.radians(lon_deg - (zone * 6 - 183))) * np.sin(np.radians(lat_deg)))


def rot_body_to_world(heading, pitch, roll):
    """(N,3,3) body(x right, y fwd, z up) -> ENU. Angles in rad, heading clockwise from north."""
    ch, sh = np.cos(heading), np.sin(heading)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    z, o = np.zeros_like(ch), np.ones_like(ch)
    Rz = np.stack([ch, sh, z, -sh, ch, z, z, z, o], -1).reshape(-1, 3, 3)      # yaw = -heading
    Rx = np.stack([o, z, z, z, cp, -sp, z, sp, cp], -1).reshape(-1, 3, 3)
    Ry = np.stack([cr, z, sr, z, o, z, -sr, z, cr], -1).reshape(-1, 3, 3)
    return Rz @ Rx @ Ry


@dataclass
class Session:
    cfg: dict
    origin: np.ndarray                 # UTM x, y, ellipsoidal z subtracted from everything
    t: np.ndarray                      # (N,) LAS-style gps_time
    pos: np.ndarray                    # (N,3) body position, local world
    hpr: np.ndarray                    # (N,3) heading, pitch, roll (rad, heading unwrapped, grid)
    photos: list = field(default_factory=list)      # file paths
    photo_t: np.ndarray = None         # (M,) mid-exposure gps_time
    cam_offsets: dict = field(default_factory=dict)
    lidar_offsets: dict = field(default_factory=dict)

    def pose(self, t):
        """Body pose at times t -> (pos (K,3), R_world_body (K,3,3))."""
        t = np.atleast_1d(np.asarray(t, np.float64))
        p = np.stack([np.interp(t, self.t, self.pos[:, k]) for k in range(3)], -1)
        a = np.stack([np.interp(t, self.t, self.hpr[:, k]) for k in range(3)], -1)
        return p, rot_body_to_world(a[:, 0], a[:, 1], a[:, 2])

    def velocity(self, t, dt=0.02):
        p0, _ = self.pose(np.asarray(t) - dt); p1, _ = self.pose(np.asarray(t) + dt)
        return (p1 - p0) / (2 * dt)


def _offsets(xml_path):
    out = {}
    if not Path(xml_path).exists():
        return out
    node = ET.parse(xml_path).getroot().find("Offsets")
    if node is None:
        return out
    for tag in ("linear", "angular", "orientation", "lens"):
        e = node.find(tag)
        if e is not None:
            out[tag] = {k: float(v) for k, v in e.attrib.items()}
    return out


def load_session(cfg=None):
    cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
    d = Path(cfg["session_dir"])
    origin = np.array(cfg["origin"], np.float64)
    cache = Path(cfg["cache_dir"]) / "trajectory.npy"
    if cache.exists():
        a = np.load(cache)
    else:
        a = np.loadtxt(d / "ins/ppk.txt", skiprows=1)   # week, gps, utc, lon, lat, h, hdg, pitch, roll
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, a)
    t = a[:, 0] * GPS_WEEK + a[:, 1] - ADJ
    x, y = utm_forward(a[:, 4], a[:, 3])
    pos = np.stack([x, y, a[:, 5]], 1) - origin
    shift_f = Path(cfg["cache_dir"]) / "las_shift.json"
    if shift_f.exists():       # constant LAS <-> trajectory offset, see estimate_las_shift
        pos = pos + np.array(json.loads(shift_f.read_text())["shift"])
    # heading is w.r.t. true north; the map frame is the UTM grid
    hdg = np.unwrap(np.radians(a[:, 6])) - grid_convergence(a[:, 4], a[:, 3])
    hpr = np.stack([hdg, np.radians(a[:, 7]), np.radians(a[:, 8])], 1)

    files = sorted((d / "camera").glob("*.jpg"))
    pt = []
    for f in files:
        m = re.match(r".*_(\d+)_(\d+)\.jpg$", f.name)
        pt.append((int(m.group(1)) + int(m.group(2))) / 2e6 - ADJ)
    pt = np.array(pt)
    keep = (pt > t[0] + 0.1) & (pt < t[-1] - 0.1)
    files = [f for f, k in zip(files, keep) if k]
    pcpp = d / "processed_ppk.pcpp"
    pcmp = d / "processed_ppk.pcmp"
    return Session(cfg, origin, t, pos, hpr, [str(f) for f in files], pt[keep],
                   _offsets(pcpp if pcpp.exists() else d / "ppk.pcpp"),
                   _offsets(pcmp if pcmp.exists() else d / "ppk.pcmp"))


# --------------------------------------------------------------------------- point caches
def build_subsample(cfg, every=20, verbose=True):
    """Random 1/every subsample of the LAS inside the twin bbox (for calibration / previews).
    Saves xyz (local, float32), rgb uint8, intensity uint8, gps_time float64, laser uint8."""
    import laspy
    out = Path(cfg["cache_dir"]) / f"points_sub{every}.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = cfg["bbox_utm"]
    org = np.array(cfg["origin"])
    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("xyz", "rgb", "intensity", "gps_time", "laser")}
    with laspy.open(cfg["las"]) as f:
        n = 0
        for p in f.chunk_iterator(20_000_000):
            idx = np.nonzero(rng.random(len(p)) < 1.0 / every)[0]
            x, y, z = np.asarray(p.x)[idx], np.asarray(p.y)[idx], np.asarray(p.z)[idx]
            m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            idx = idx[m]
            acc["xyz"].append(np.stack([x[m] - org[0], y[m] - org[1], z[m] - org[2]], 1).astype(np.float32))
            rgb = np.stack([np.asarray(p.red)[idx], np.asarray(p.green)[idx], np.asarray(p.blue)[idx]], 1)
            acc["rgb"].append((rgb >> 8).astype(np.uint8) if rgb.max() > 255 else rgb.astype(np.uint8))
            acc["intensity"].append(np.clip(np.asarray(p.intensity)[idx], 0, 255).astype(np.uint8))
            acc["gps_time"].append(np.asarray(p.gps_time)[idx])
            acc["laser"].append(np.asarray(p.point_source_id)[idx].astype(np.uint8))
            n += len(p)
            if verbose:
                print(f"[resepi] subsample: {n/1e6:.0f} M read", flush=True)
    np.savez(out, **{k: np.concatenate(v) for k, v in acc.items()})
    return out


def estimate_las_shift(cfg, channels=(2, 9, 16, 23, 30), verbose=True):
    """Constant offset between the exported LAS and the INS trajectory.

    Every laser of a spinning lidar sweeps a cone around the spin axis, so in the body frame
    (dir . axis) must be constant per laser. Solving for the world offset that makes this hold
    recovers how the LAS was shifted after georeferencing (blue_wall: ~(-3.5, -4.8, -0.1) m)
    and doubles as a check of the attitude convention (residual ~0.01 deg, axis = 45 deg)."""
    from scipy.optimize import least_squares
    out = Path(cfg["cache_dir"]) / "las_shift.json"
    if out.exists():
        out.unlink()
    s = load_session(cfg)
    d = np.load(build_subsample(cfg))
    las, t = d["laser"], d["gps_time"]
    chs = list(channels)
    idx = np.concatenate([np.nonzero(las == c)[0][::60] for c in chs])
    ci = np.searchsorted(chs, las[idx])
    p, R = s.pose(t[idx])
    X = d["xyz"][idx].astype(np.float64)

    def resid(x):
        n = x[3:6] / np.linalg.norm(x[3:6])
        dv = np.einsum("nji,nj->ni", R, X - p - x[:3])
        return (dv @ n) / np.linalg.norm(dv, axis=1) - x[6:][ci]

    x0 = np.r_[0, 0, 0, 0, 0.707, -0.707, np.radians(np.linspace(13, -15, len(chs)))]
    r = least_squares(resid, x0, loss="soft_l1", f_scale=0.005)
    axis = r.x[3:6] / np.linalg.norm(r.x[3:6])
    res = dict(shift=r.x[:3].tolist(), spin_axis_body=axis.tolist(),
               laser_elev_deg=dict(zip(map(str, chs), np.degrees(r.x[6:]).round(3).tolist())),
               median_abs_resid_deg=float(np.degrees(np.median(np.abs(r.fun)))))
    out.write_text(json.dumps(res, indent=2))
    if verbose:
        print("[resepi] LAS shift", np.round(res["shift"], 3), "spin axis", axis.round(4),
              "resid deg", round(res["median_abs_resid_deg"], 4))
    return res


# --------------------------------------------------------------------------- full-density tiles
TILE_FIELDS = (("xyz", np.float32, 3), ("rgb", np.uint8, 3), ("intensity", np.uint8, 1),
               ("gps_time", np.float64, 1), ("laser", np.uint8, 1))


def build_tiles(cfg, tile=20.0, verbose=True):
    """Split the LAS (inside the twin bbox) into xy tiles of raw binary arrays so later stages
    can random-access the full-density cloud: data/<name>/tiles/<ix>_<iy>.<field>."""
    import laspy
    out = Path(cfg["cache_dir"]) / "tiles"
    done = out / "index.json"
    if done.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*_*.*"):
        f.unlink()
    x0, y0, x1, y1 = cfg["bbox_utm"]
    org = np.array(cfg["origin"])
    nx, ny = int(np.ceil((x1 - x0) / tile)), int(np.ceil((y1 - y0) / tile))
    counts = np.zeros(nx * ny, np.int64)
    with laspy.open(cfg["las"]) as f:
        n = 0
        for p in f.chunk_iterator(20_000_000):
            x, y = np.asarray(p.x), np.asarray(p.y)
            m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            x, y = x[m], y[m]
            rgb = np.stack([np.asarray(p.red)[m], np.asarray(p.green)[m], np.asarray(p.blue)[m]], 1)
            arr = dict(
                xyz=np.stack([x - org[0], y - org[1], np.asarray(p.z)[m] - org[2]], 1).astype(np.float32),
                rgb=(rgb >> 8).astype(np.uint8) if rgb.max() > 255 else rgb.astype(np.uint8),
                intensity=np.clip(np.asarray(p.intensity)[m], 0, 255).astype(np.uint8),
                gps_time=np.asarray(p.gps_time)[m].astype(np.float64),
                laser=np.asarray(p.point_source_id)[m].astype(np.uint8))
            tid = np.minimum(((y - y0) / tile).astype(np.int64), ny - 1) * nx + \
                np.minimum(((x - x0) / tile).astype(np.int64), nx - 1)
            order = np.argsort(tid, kind="stable")
            tid_s = tid[order]
            cuts = np.flatnonzero(np.diff(tid_s)) + 1
            for seg in np.split(order, cuts):
                k = int(tid[seg[0]])
                for name, _, _ in TILE_FIELDS:
                    with open(out / f"{k % nx}_{k // nx}.{name}", "ab") as fh:
                        fh.write(np.ascontiguousarray(arr[name][seg]).tobytes())
                counts[k] += len(seg)
            n += len(p)
            if verbose:
                print(f"[resepi] tiles: {n/1e6:.0f} M read", flush=True)
    done.write_text(json.dumps(dict(tile=tile, nx=nx, ny=ny, counts=counts.tolist())))
    return out


class Tiles:
    def __init__(self, cfg):
        self.dir = Path(cfg["cache_dir"]) / "tiles"
        idx = json.loads((self.dir / "index.json").read_text())
        self.tile, self.nx, self.ny = idx["tile"], idx["nx"], idx["ny"]
        self.counts = np.array(idx["counts"]).reshape(self.ny, self.nx)

    def load(self, ix, iy, fields=("xyz", "rgb", "intensity", "gps_time", "laser"), mmap=True):
        out = {}
        for name, dt, k in TILE_FIELDS:
            if name not in fields:
                continue
            f = self.dir / f"{ix}_{iy}.{name}"
            if not f.exists():
                out[name] = np.zeros((0, k) if k > 1 else (0,), dt)
                continue
            a = np.memmap(f, dt, "r") if mmap else np.fromfile(f, dt)
            out[name] = a.reshape(-1, k) if k > 1 else a
        return out

    def query(self, center, radius, fields=("xyz", "rgb", "intensity", "gps_time", "laser"),
              stride=1):
        """Points within `radius` (xy) of center, concatenated over the touched tiles."""
        cx, cy = center[:2]
        acc = {f: [] for f in fields}
        for iy in range(max(0, int((cy - radius) // self.tile)), min(self.ny, int((cy + radius) // self.tile) + 1)):
            for ix in range(max(0, int((cx - radius) // self.tile)), min(self.nx, int((cx + radius) // self.tile) + 1)):
                d = self.load(ix, iy, tuple(set(fields) | {"xyz"}))
                xyz = np.asarray(d["xyz"][::stride])
                m = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2 < radius ** 2
                for f in fields:
                    acc[f].append(xyz[m] if f == "xyz" else np.asarray(d[f][::stride])[m])
        return {f: np.concatenate(v) if v else np.zeros(0) for f, v in acc.items()}


def _vox_one(args):
    d, ix, iy, voxel = args
    f = Path(d) / f"{ix}_{iy}.xyz"
    if not f.exists():
        return 0
    xyz = np.fromfile(f, np.float32).reshape(-1, 3)
    rgb = np.fromfile(Path(d) / f"{ix}_{iy}.rgb", np.uint8).reshape(-1, 3).astype(np.float64)
    inten = np.fromfile(Path(d) / f"{ix}_{iy}.intensity", np.uint8).astype(np.float64)
    key = np.floor(xyz / voxel).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    u, inv = np.unique(lin, return_inverse=True)
    n = len(u)
    cnt = np.bincount(inv, minlength=n)
    v = np.stack([np.bincount(inv, xyz[:, k].astype(np.float64), n) / cnt for k in range(3)], 1)
    col = rgb.sum(1) > 0
    ccnt = np.bincount(inv[col], minlength=n)
    c = np.stack([np.bincount(inv[col], rgb[col, k], n) / np.maximum(ccnt, 1) for k in range(3)], 1)
    c = np.where(ccnt[:, None] > 0, np.maximum(c, 1), 0)          # 0,0,0 = no colour measured
    att = np.c_[c, np.bincount(inv, inten, n) / cnt].astype(np.uint8)
    keep = cnt >= 2                                              # single-hit voxels are mostly noise
    v[keep].astype(np.float32).tofile(Path(d) / f"{ix}_{iy}.vox")
    att[keep].tofile(Path(d) / f"{ix}_{iy}.voxc")
    return int(keep.sum())


def build_vox(cfg, voxel=0.03, workers=12, verbose=True):
    """Voxel-reduced xyz per tile (<ix>_<iy>.vox) for fast depth rendering."""
    from multiprocessing import Pool
    t = Tiles(cfg)
    flag = t.dir / "vox.json"
    if flag.exists():
        return
    jobs = [(str(t.dir), ix, iy, voxel) for iy in range(t.ny) for ix in range(t.nx)]
    with Pool(workers) as p:
        n = p.map(_vox_one, jobs, chunksize=1)
    flag.write_text(json.dumps(dict(voxel=voxel, points=int(sum(n)))))
    if verbose:
        print(f"[resepi] vox: {sum(n)/1e6:.1f} M voxels", flush=True)


def _tiles_query_vox(self, center, radius, attrs=False):
    """Voxel-reduced points within radius (xy). attrs=True also returns (N,4) uint8 r,g,b,intensity."""
    cx, cy = center[:2]
    acc, acc_a = [], []
    for iy in range(max(0, int((cy - radius) // self.tile)), min(self.ny, int((cy + radius) // self.tile) + 1)):
        for ix in range(max(0, int((cx - radius) // self.tile)), min(self.nx, int((cx + radius) // self.tile) + 1)):
            key = (ix, iy)
            if key not in self._vox:
                f = self.dir / f"{ix}_{iy}.vox"
                if f.exists():
                    self._vox[key] = (np.fromfile(f, np.float32).reshape(-1, 3),
                                      np.fromfile(f.with_suffix(".voxc"), np.uint8).reshape(-1, 4))
                else:
                    self._vox[key] = (np.zeros((0, 3), np.float32), np.zeros((0, 4), np.uint8))
                if len(self._vox) > 40:
                    self._vox.pop(next(iter(self._vox)))
            xyz, att = self._vox[key]
            m = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2 < radius ** 2
            acc.append(xyz[m])
            if attrs:
                acc_a.append(att[m])
    xyz = np.concatenate(acc) if acc else np.zeros((0, 3), np.float32)
    if attrs:
        return xyz, (np.concatenate(acc_a) if acc_a else np.zeros((0, 4), np.uint8))
    return xyz


Tiles._vox = {}
Tiles.query_vox = _tiles_query_vox
