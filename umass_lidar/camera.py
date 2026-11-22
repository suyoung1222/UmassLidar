"""Photo camera model for a RESEPI session + LiDAR-anchored calibration / pose refinement.

    Xc = dR_i * R(rvec) * R_CB0 * R_wb(t_i + dt)^T * (Xw - (p_b(t_i + dt) + R_wb * lever)) + dt_i
    u  = f * xn * (1 + k1 r^2 + k2 r^4) + cx          (same for v)

R_CB0 is the nominal mount (looking forward, 45 deg down, upside-down) found by brute-force
search over axis conventions (see find_mount). Global parameters (f, cx, cy, k1, k2, rvec,
lever, dt) and per-photo corrections (dR_i, dt_i) are estimated from 2D-3D correspondences
between each photo and the LiDAR map (block matching against the LiDAR colour render), i.e. the
LiDAR map is the fixed structure and the INS trajectory is the prior (map-based visual
localisation). Photo-to-photo feature matching is avoided on purpose: the site is paved with a
regular tile grid and SIFT matches alias by whole tiles.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# rows = camera axes (x right, y down, z forward of the EXIF-upright image) in the body frame
R_CB0 = np.diag([-1.0, 1.0, -1.0]) @ _rx(np.radians(-45.0)).T


@dataclass
class CameraModel:
    width: int = 6000
    height: int = 4000
    f: float = 4085.0
    cx: float = 3000.0
    cy: float = 2000.0
    k1: float = 0.0
    k2: float = 0.0
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lever: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dt: float = 0.0
    deltas: np.ndarray = None          # (M, 6) per-photo [rotvec, translation] in camera frame

    GLOBAL = ("f", "cx", "cy", "k1", "k2", "rvec", "lever", "dt")

    # ---- (de)serialisation
    def to_json(self, path, extra=None):
        d = dict(width=self.width, height=self.height, f=self.f, cx=self.cx, cy=self.cy,
                 k1=self.k1, k2=self.k2, rvec=list(map(float, self.rvec)),
                 lever=list(map(float, self.lever)), dt=self.dt, R_CB0=R_CB0.tolist())
        d.update(extra or {})
        Path(path).write_text(json.dumps(d, indent=2))
        if self.deltas is not None:
            np.save(Path(path).with_suffix(".deltas.npy"), self.deltas)

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        m = cls(**{k: d[k] for k in ("width", "height", "f", "cx", "cy", "k1", "k2", "dt")})
        m.rvec, m.lever = np.array(d["rvec"]), np.array(d["lever"])
        dp = Path(path).with_suffix(".deltas.npy")
        if dp.exists():
            m.deltas = np.load(dp)
        return m

    def get_vec(self):
        return np.r_[self.f, self.cx, self.cy, self.k1, self.k2, self.rvec, self.lever, self.dt]

    def set_vec(self, x):
        self.f, self.cx, self.cy, self.k1, self.k2 = map(float, x[:5])
        self.rvec, self.lever, self.dt = np.array(x[5:8]), np.array(x[8:11]), float(x[11])

    # ---- geometry
    def extrinsics(self, session, idx, use_deltas=True):
        """World->camera rotation (K,3,3), camera centre (K,3), per-photo translation (K,3)."""
        idx = np.atleast_1d(idx)
        p, Rwb = session.pose(session.photo_t[idx] + self.dt)
        C = p + Rwb @ self.lever
        R = (Rot.from_rotvec(self.rvec).as_matrix() @ R_CB0) @ np.transpose(Rwb, (0, 2, 1))
        tc = np.zeros((len(idx), 3))
        if use_deltas and self.deltas is not None:
            R = Rot.from_rotvec(self.deltas[idx, :3]).as_matrix() @ R
            tc = self.deltas[idx, 3:]
        return R, C, tc

    def pix(self, Xc):
        z = Xc[..., 2]
        xn, yn = Xc[..., 0] / z, Xc[..., 1] / z
        r2 = xn * xn + yn * yn
        d = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return np.stack([self.f * xn * d + self.cx, self.f * yn * d + self.cy], -1), z

    def rays(self, uv, iters=5):
        """Unit-z camera rays for pixels (undistort by fixed-point iteration)."""
        xd, yd = (uv[..., 0] - self.cx) / self.f, (uv[..., 1] - self.cy) / self.f
        xn, yn = xd.copy(), yd.copy()
        for _ in range(iters):
            r2 = xn * xn + yn * yn
            d = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xn, yn = xd / d, yd / d
        return np.stack([xn, yn, np.ones_like(xn)], -1)

    def project(self, session, i, Xw):
        R, C, tc = self.extrinsics(session, i)
        return self.pix((Xw - C[0]) @ R[0].T + tc[0])

    def backproject(self, session, i, uv, depth):
        R, C, tc = self.extrinsics(session, i)
        Xc = self.rays(uv) * depth[..., None]
        return (Xc - tc[0]) @ R[0] + C[0]


# --------------------------------------------------------------------------- LiDAR renders
def lidar_view(model, session, i, tiles, scale=16, radius=16.0, want=("depth",), max_depth=18.0,
               full_density=False):
    """Z-buffer the LiDAR cloud into photo i at 1/scale resolution.
    Returns dict: depth (nan = empty) and optionally 'rgb' (0 = no colour) / 'intensity'.
    Uses the 3 cm voxel cloud (fast) unless full_density."""
    R, C, tc = model.extrinsics(session, i)
    attrs = any(w in ("rgb", "intensity") for w in want)
    if full_density:
        d = tiles.query(C[0], radius, fields=("xyz", "rgb", "intensity"))
        xyz, att = d["xyz"], np.c_[d["rgb"], d["intensity"]]
    elif attrs:
        xyz, att = tiles.query_vox(C[0], radius, attrs=True)
    else:
        xyz, att = tiles.query_vox(C[0], radius), None
    Xc = (xyz.astype(np.float64) - C[0]) @ R[0].T + tc[0]
    ok = (Xc[:, 2] > 0.3) & (Xc[:, 2] < max_depth)
    uv, z = model.pix(Xc[ok])
    w, h = model.width // scale, model.height // scale
    u = np.floor(uv[:, 0] / scale).astype(np.int64); v = np.floor(uv[:, 1] / scale).astype(np.int64)
    k = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    lin = v[k] * w + u[k]; z = z[k]
    order = np.argsort(-z)                      # far first, nearest written last
    out = {}
    depth = np.full(w * h, np.nan); depth[lin[order]] = z[order]
    out["depth"] = depth.reshape(h, w)
    if attrs:
        sel = np.nonzero(ok)[0][k][order]
        img = np.zeros((w * h, 4), np.uint8); img[lin[order]] = att[sel]
        out["rgb"] = img[:, :3].reshape(h, w, 3)
        out["intensity"] = img[:, 3].reshape(h, w)
    return out


def clean_depth(depth, max_fill=2):
    """Remove see-through pixels (far points seen through gaps) and fill small holes."""
    from scipy import ndimage
    d = depth.copy()
    big = np.where(np.isnan(d), np.inf, d)
    near = ndimage.minimum_filter(big, 5)
    d[(big > near * 1.15 + 0.1) & np.isfinite(big)] = np.nan       # occluded background
    hole = np.isnan(d)
    idx = ndimage.distance_transform_edt(hole, return_distances=True, return_indices=True)
    filled = d[tuple(idx[1])]
    d[hole & (idx[0] <= max_fill)] = filled[hole & (idx[0] <= max_fill)]
    return d


def sample_depth(depth, uv, scale):
    """Bilinear depth lookup at full-res pixels uv; nan where any neighbour is missing or the
    local depth is discontinuous (occlusion edge)."""
    h, w = depth.shape
    x = uv[:, 0] / scale - 0.5; y = uv[:, 1] / scale - 0.5
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    fx, fy = x - x0c, y - y0c
    d00, d01 = depth[y0c, x0c], depth[y0c, x0c + 1]
    d10, d11 = depth[y0c + 1, x0c], depth[y0c + 1, x0c + 1]
    st = np.stack([d00, d01, d10, d11])
    val = d00 * (1 - fx) * (1 - fy) + d01 * fx * (1 - fy) + d10 * (1 - fx) * fy + d11 * fx * fy
    rng = st.max(0) - st.min(0)
    bad = ~ok | np.isnan(val) | (rng > 0.08 * st.min(0) + 0.05)
    val[bad] = np.nan
    return val


# --------------------------------------------------------------------------- photo features
def load_gray(path, reduce=4):
    import cv2
    flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
            4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduce]
    return cv2.imread(path, flag)


def load_color(path, reduce=4):
    import cv2
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
    return cv2.imread(path, flag)[..., ::-1]


# --------------------------------------------------------------------------- photo <-> LiDAR registration
def register_photo(model, session, i, tiles, scale=16, block=40, stride=24, search=20, min_corr=0.45):
    """2D-3D correspondences for photo i by block-matching the LiDAR render (points carry the
    colour PCMaster gave them) against the photo. Photo-to-photo SIFT is NOT used: the site is
    paved with a regular tile grid and feature matches alias by whole tiles.
    Returns (uv_observed (K,2) full-res px, Xw (K,3), corr (K,))."""
    import cv2
    v = lidar_view(model, session, i, tiles, scale, want=("depth", "rgb"))
    depth = clean_depth(v["depth"], max_fill=0)
    g_l = v["rgb"].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    valid = (v["rgb"].sum(-1) > 0) & np.isfinite(depth)
    ph = load_gray(session.photos[i], 8).astype(np.float32)
    h, w = depth.shape
    ph = cv2.resize(ph, (w, h), interpolation=cv2.INTER_AREA)
    # local contrast normalisation makes the match insensitive to exposure / colour processing
    def lcn(a, m=None):
        if m is not None:
            mean = cv2.blur(a * m, (15, 15)) / np.maximum(cv2.blur(m, (15, 15)), 1e-3)
            a = np.where(m > 0, a, mean)
        a = a - cv2.GaussianBlur(a, (0, 0), 6)
        return a / (np.sqrt(cv2.GaussianBlur(a * a, (0, 0), 6)) + 2.0)
    L = lcn(g_l, valid.astype(np.float32)); P = lcn(ph)
    uv_o, uv_r, zz, cc = [], [], [], []
    for y0 in range(search, h - block - search, stride):
        for x0 in range(search, w - block - search, stride):
            m = valid[y0:y0 + block, x0:x0 + block]
            if m.mean() < 0.6:
                continue
            t = L[y0:y0 + block, x0:x0 + block]
            if t.std() < 0.25:
                continue
            win = P[y0 - search:y0 + block + search, x0 - search:x0 + block + search]
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx < min_corr or loc[0] in (0, 2 * search) or loc[1] in (0, 2 * search):
                continue
            # the peak must be unique (repetitive paving): second best outside 3 px clearly lower
            r2 = r.copy(); cv2.circle(r2, loc, 4, -1.0, -1)
            if r2.max() > 0.85 * mx:
                continue
            dx = loc[0] - search; dy = loc[1] - search
            cx_, cy_ = loc
            sx = 0.5 * (r[cy_, cx_ - 1] - r[cy_, cx_ + 1]) / (r[cy_, cx_ - 1] - 2 * mx + r[cy_, cx_ + 1] - 1e-9)
            sy = 0.5 * (r[cy_ - 1, cx_] - r[cy_ + 1, cx_]) / (r[cy_ - 1, cx_] - 2 * mx + r[cy_ + 1, cx_] - 1e-9)
            # anchor = valid LiDAR pixels of the block centre region (median depth pixel)
            ys, xs = np.nonzero(m[block // 4:-block // 4, block // 4:-block // 4])
            if len(ys) < 10:
                continue
            ys = ys + y0 + block // 4; xs = xs + x0 + block // 4
            d = depth[ys, xs]
            j = np.argsort(d)[len(d) // 2]
            if d.max() - d.min() > 0.25 * d[j] + 0.3:      # block straddles a depth edge
                continue
            pr = (np.array([xs[j], ys[j]]) + 0.5) * scale
            uv_r.append(pr); zz.append(d[j]); cc.append(mx)
            uv_o.append(pr + np.array([dx + np.clip(sx, -1, 1), dy + np.clip(sy, -1, 1)]) * scale)
    if not uv_o:
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)
    uv_r = np.array(uv_r); zz = np.array(zz)
    return np.array(uv_o), model.backproject(session, i, uv_r, zz), np.array(cc)


def build_observations(model, session, tiles, photo_ids, verbose=True):
    oi, ouv, oX = [], [], []
    for n, i in enumerate(photo_ids):
        uv, X, c = register_photo(model, session, i, tiles)
        if len(uv):
            oi.append(np.full(len(uv), i)); ouv.append(uv); oX.append(X)
        if verbose and n % 100 == 0:
            print(f"[camera] register {n}/{len(photo_ids)} ({len(uv)} blocks)", flush=True)
    return np.concatenate(oi), np.concatenate(ouv), np.concatenate(oX), None


def _residual(model, session, oi, ouv, oX, R, C, tc):
    Xc = np.einsum("nij,nj->ni", R[oi], oX - C[oi]) + tc[oi]
    uv, z = model.pix(Xc)
    r = uv - ouv
    r[z < 0.1] = 500.0
    return r


def reproj_stats(model, session, obs):
    oi, ouv, oX, _ = obs
    R, C, tc = model.extrinsics(session, np.arange(len(session.photos)))
    e = np.linalg.norm(_residual(model, session, oi, ouv, oX, R, C, tc), axis=1)
    return dict(n=int(len(e)), photos_with_obs=int(len(np.unique(oi))), median_px=float(np.median(e)),
                p90_px=float(np.percentile(e, 90)), inlier_frac_20px=float((e < 20).mean()))


def optimize_global(model, session, obs, free=("f", "cx", "cy", "k1", "k2", "rvec", "lever", "dt"),
                    verbose=True):
    """Global intrinsics + boresight + lever arm + time offset (robust least squares)."""
    from scipy.optimize import least_squares
    oi, ouv, oX, _ = obs
    names = ["f", "cx", "cy", "k1", "k2", "rvec", "rvec", "rvec", "lever", "lever", "lever", "dt"]
    fr = np.array([n in free for n in names])
    x0 = model.get_vec()
    sc = np.array([100, 50, 50, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.01])
    allp = np.arange(len(session.photos))

    def fun(y):
        x = x0.copy(); x[fr] = x0[fr] + y * sc[fr]
        model.set_vec(x)
        R, C, tc = model.extrinsics(session, allp)
        return _residual(model, session, oi, ouv, oX, R, C, tc).ravel()

    r = least_squares(fun, np.zeros(fr.sum()), loss="cauchy", f_scale=8.0, x_scale=1.0,
                      diff_step=1e-3, max_nfev=60)
    fun(r.x)
    if verbose:
        print("[camera] global:", {k: np.round(getattr(model, k), 5).tolist() for k in model.GLOBAL})
    return model


def optimize_photos(model, session, obs, sigma_rot_deg=0.3, sigma_pos=0.03, px_sigma=8.0,
                    verbose=True):
    """Per-photo pose corrections with an INS prior, solved independently per photo
    (structure is fixed LiDAR, so photos decouple given the anchors)."""
    from scipy.optimize import least_squares
    oi, ouv, oX, _ = obs
    M = len(session.photos)
    base = CameraModel(**{k: getattr(model, k) for k in ("width", "height", "f", "cx", "cy", "k1", "k2", "dt")})
    base.rvec, base.lever = model.rvec, model.lever
    R0, C0, _ = base.extrinsics(session, np.arange(M))
    deltas = np.zeros((M, 6))
    order = np.argsort(oi, kind="stable")
    cuts = np.searchsorted(oi[order], np.arange(M + 1))
    w_prior = np.r_[np.full(3, px_sigma / np.radians(sigma_rot_deg)), np.full(3, px_sigma / sigma_pos)]
    w_prior = w_prior * np.sqrt(12.0)       # prior counts like ~12 block observations
    c = 2.0 * px_sigma
    nfix = 0
    for i in range(M):
        sel = order[cuts[i]:cuts[i + 1]]
        if len(sel) < 12:
            continue
        X0 = (oX[sel] - C0[i]) @ R0[i].T
        uv_t = ouv[sel]

        def fun(d):
            Xc = X0 @ Rot.from_rotvec(d[:3]).as_matrix().T + d[3:]
            uv, _ = base.pix(Xc)
            r = (uv - uv_t).ravel() / c
            # Cauchy pseudo-residual on the data only; the INS prior stays quadratic so that
            # rotation / translation cannot trade off against each other unpenalised
            return np.r_[c * np.sign(r) * np.sqrt(np.log1p(r * r)), d * w_prior]

        r = least_squares(fun, np.zeros(6))
        dr = np.degrees(np.linalg.norm(r.x[:3])); dtm = np.linalg.norm(r.x[3:])
        if dr > 4 * sigma_rot_deg or dtm > 4 * sigma_pos:      # implausible vs INS: keep INS pose
            continue
        deltas[i] = r.x; nfix += 1
    model.deltas = deltas
    if verbose:
        rot = np.degrees(np.linalg.norm(deltas[:, :3], axis=1)); tr = np.linalg.norm(deltas[:, 3:], axis=1)
        print(f"[camera] per-photo: {nfix}/{M} refined, |rot| median {np.median(rot):.3f} deg "
              f"max {rot.max():.2f}, |t| median {np.median(tr)*100:.1f} cm max {tr.max()*100:.1f}")
    return model


# --------------------------------------------------------------------------- mount search
def find_mount(session, sub_npz, photo_ids=(100, 250, 400, 550, 700), f=4085.0):
    """Brute-force the camera mount among 24 axis permutations x pitch {-45, 0, 45} by the
    correlation between photo grey and the grey of LAS points coloured by PCMaster."""
    import itertools
    d = np.load(sub_npz)
    col = d["rgb"].sum(1) > 0
    xyz, g, tt = d["xyz"][col], d["rgb"][col].astype(np.float32).mean(1), d["gps_time"][col]
    cands = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s_) in enumerate(zip(perm, sg)):
                R[i, p] = s_
            if np.linalg.det(R) > 0:
                cands.append(R)
    data = []
    for i in photo_ids:
        im = load_gray(session.photos[i], 4).astype(np.float32)
        p, R = session.pose(session.photo_t[i])
        m = (np.linalg.norm(xyz - p[0], axis=1) < 25) & (np.abs(tt - session.photo_t[i]) < 20)
        data.append((im, (xyz[m] - p[0]) @ R[0], g[m]))
    best = []
    for C in cands:
        for pitch in (-45, 0, 45):
            Rcb = C @ _rx(np.radians(pitch)).T
            sc = []
            for im, xb, gg in data:
                xc = xb @ Rcb.T
                ok = xc[:, 2] > 1
                u = f * xc[ok, 0] / xc[ok, 2] + 3000; v = f * xc[ok, 1] / xc[ok, 2] + 2000
                k = (u >= 0) & (u < 6000) & (v >= 0) & (v < 4000)
                sc.append(np.corrcoef(im[(v[k] / 4).astype(int), (u[k] / 4).astype(int)], gg[ok][k])[0, 1]
                          if k.sum() > 500 else 0.0)
            best.append((float(np.mean(sc)), Rcb))
    best.sort(key=lambda b: -b[0])
    return best[:4]


# --------------------------------------------------------------------------- pipeline entry
def calibrate(cfg, session, tiles, rounds=3, calib_step=6, verbose=True, refine_only=False):
    """Camera calibration + per-photo refinement against the LiDAR map. Writes
    data/<name>/camera.json (+deltas) and outputs/calib/<name>/report.json."""
    out = Path(cfg["cache_dir"]) / "camera.json"
    model = CameraModel()
    cam_lin = session.cam_offsets.get("linear", {})
    model.lever = np.array([cam_lin.get("x", 0.0), cam_lin.get("y", 0.0), cam_lin.get("z", 0.0)])
    M = len(session.photos)
    sub = np.arange(0, M, calib_step)
    report = dict(photos=M, method="block matching of photo vs LiDAR colour render (2D-3D), robust LSQ",
                  stages=[])
    rep_file = Path(cfg["cache_dir"]).parents[1] / "outputs/calib" / cfg["name"] / "report.json"
    if refine_only:                 # reuse the global stage of a previous run
        model = CameraModel.from_json(out); model.deltas = None
        report["stages"] = [st for st in json.loads(rep_file.read_text())["stages"]
                            if not st["stage"].startswith("all photos")]
    else:
        obs = build_observations(model, session, tiles, sub, verbose=verbose)
        report["stages"].append(dict(stage="nominal (pcpp lever arm, f = 16 mm)", **reproj_stats(model, session, obs)))
        if verbose:
            print("[camera]", report["stages"][-1], flush=True)
        # cx, cy are degenerate with boresight pitch / yaw for these observations and stay fixed.
        schedule = [("f", "rvec"), ("f", "rvec"), ("f", "rvec", "k1", "dt")][:rounds]
        for r, free in enumerate(schedule):
            prev, prev_stats = model.get_vec(), report["stages"][-1]
            optimize_global(model, session, obs, free=free, verbose=verbose)
            new_obs = build_observations(model, session, tiles, sub, verbose=False)
            st = reproj_stats(model, session, new_obs)
            ok = st["n"] >= 0.7 * prev_stats["n"] and st["median_px"] <= prev_stats["median_px"] * 1.05
            report["stages"].append(dict(stage=f"global round {r+1} ({', '.join(free)})", accepted=bool(ok), **st))
            if verbose:
                print("[camera]", report["stages"][-1], flush=True)
            if not ok:                    # worse than before: keep the previous model and stop
                model.set_vec(prev)
                report["stages"].pop()
                report["stages"].append(dict(prev_stats, accepted=False,
                                             stage=f"global round {r+1} ({', '.join(free)}): rejected, reverted"))
                break
            obs = new_obs
    allp = np.arange(M)
    obs = build_observations(model, session, tiles, allp, verbose=verbose)
    report["stages"].append(dict(stage="all photos, global model", **reproj_stats(model, session, obs)))
    # the observations were anchored with the global model; refining once from them is enough
    optimize_photos(model, session, obs, verbose=verbose)
    obs = build_observations(model, session, tiles, allp, verbose=False)
    report["stages"].append(dict(stage="all photos, per-photo refined", **reproj_stats(model, session, obs)))
    if verbose:
        print("[camera]", report["stages"][-1], flush=True)
    d = model.deltas
    report["global"] = {k: np.round(getattr(model, k), 6).tolist() for k in CameraModel.GLOBAL}
    report["per_photo"] = dict(
        rot_deg_median=float(np.median(np.degrees(np.linalg.norm(d[:, :3], axis=1)))),
        rot_deg_p95=float(np.percentile(np.degrees(np.linalg.norm(d[:, :3], axis=1)), 95)),
        trans_cm_median=float(np.median(np.linalg.norm(d[:, 3:], axis=1)) * 100),
        trans_cm_p95=float(np.percentile(np.linalg.norm(d[:, 3:], axis=1), 95) * 100),
        photos_refined=int((np.abs(d).sum(1) > 0).sum()))
    model.to_json(out)
    rep_dir = Path(cfg["cache_dir"]).parents[1] / "outputs/calib" / cfg["name"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "report.json").write_text(json.dumps(report, indent=2))
    return model
