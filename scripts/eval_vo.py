#!/usr/bin/env python
"""Independent visual SfM / VO baseline: photos only (no INS, no LiDAR) -> COLMAP -> compare the
camera trajectory against the PPK INS trajectory.

The RESEPI camera takes a 24 MP still every ~2.5 s (~2.7 m baseline, looking 45 deg down), so
this is sparse sequential SfM rather than video VO. Reported per reconstructed component:
registered photos, Sim(3)-aligned ATE and relative pose error vs INS.

    python scripts/eval_vo.py                    # processing env `umasslidar`
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umass_lidar import resepi                     # noqa: E402
from umass_lidar.camera import CameraModel         # noqa: E402


def umeyama(src, dst):
    """Sim(3) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-size", type=int, default=1500, help="longest image side for SIFT")
    ap.add_argument("--overlap", type=int, default=6, help="sequential matching window")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    import cv2
    import pycolmap

    cfg = resepi.load_config(a.config)
    s = resepi.load_session(cfg)
    work = Path(cfg["cache_dir"]) / "sfm"
    img_dir = work / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    out = resepi.ROOT / "outputs/vo" / cfg["name"]; out.mkdir(parents=True, exist_ok=True)
    names = [f"{i:04d}.jpg" for i in range(len(s.photos))]
    for n, p in zip(names, s.photos):
        if not (img_dir / n).exists():
            im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
            cv2.imwrite(str(img_dir / n), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    db = work / "database.db"; sparse = work / "sparse"
    if a.redo or not sparse.exists() or not any(sparse.iterdir()):
        if db.exists():
            db.unlink()
        sparse.mkdir(exist_ok=True)
        eo = pycolmap.FeatureExtractionOptions(); eo.max_image_size = a.max_size; eo.num_threads = a.threads
        eo.use_gpu = False
        ro = pycolmap.ImageReaderOptions(); ro.camera_model = "SIMPLE_RADIAL"
        pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                                  extraction_options=eo)
        po = pycolmap.SequentialPairingOptions(); po.overlap = a.overlap; po.quadratic_overlap = True
        po.loop_detection = False
        mo = pycolmap.FeatureMatchingOptions(); mo.use_gpu = False; mo.num_threads = a.threads
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po)
        io = pycolmap.IncrementalPipelineOptions(); io.num_threads = a.threads; io.min_model_size = 5
        io.max_num_models = 100; io.multiple_models = True
        pycolmap.incremental_mapping(db, img_dir, sparse, options=io)

    # ---- INS reference (camera centres from the calibrated model if available)
    cj = Path(cfg["cache_dir"]) / "camera.json"
    model = CameraModel.from_json(cj) if cj.exists() else CameraModel()
    Rw, C, _ = model.extrinsics(s, np.arange(len(s.photos)))
    comps = []
    for d in sorted(sparse.iterdir()):
        if not (d / "images.bin").exists():
            continue
        rec = pycolmap.Reconstruction(d)
        idx, cen, Rs = [], [], []
        for im in rec.images.values():
            if not im.has_pose:
                continue
            T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R = T.rotation.matrix()
            idx.append(int(im.name[:4])); cen.append(-R.T @ T.translation); Rs.append(R)
        if len(idx) < 5:
            continue
        o = np.argsort(idx); idx = np.array(idx)[o]; cen = np.array(cen)[o]; Rs = np.array(Rs)[o]
        sc, R, t = umeyama(cen, C[idx])
        al = sc * cen @ R.T + t
        err = np.linalg.norm(al - C[idx], axis=1)
        # relative errors between consecutive registered photos
        dtr = np.linalg.norm(np.diff(al, axis=0), axis=1) - np.linalg.norm(np.diff(C[idx], axis=0), axis=1)
        Rrel_v = np.einsum("nij,nkj->nik", Rs[1:], Rs[:-1])
        Rrel_i = np.einsum("nij,nkj->nik", Rw[idx][1:], Rw[idx][:-1])
        dR = np.einsum("nij,nkj->nik", Rrel_v, Rrel_i)
        ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        cam = next(iter(rec.cameras.values()))
        comps.append(dict(model=d.name, photos=len(idx), first=int(idx[0]), last=int(idx[-1]),
                          path_len_m=float(np.linalg.norm(np.diff(C[idx], axis=0), axis=1).sum()),
                          scale=float(sc), ate_rmse_m=float(np.sqrt((err ** 2).mean())),
                          ate_median_m=float(np.median(err)),
                          rpe_trans_rmse_m=float(np.sqrt((dtr ** 2).mean())),
                          rpe_rot_median_deg=float(np.median(ang)),
                          focal_px_fullres=float(cam.params[0] * 4), points3D=len(rec.points3D),
                          _idx=idx, _al=al))
    comps.sort(key=lambda c: -c["photos"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].plot(s.pos[::25, 0], s.pos[::25, 1], color="0.75", lw=1, label="INS trajectory (PPK)")
    ax[0].scatter(C[:, 0], C[:, 1], s=4, color="0.4", label=f"photos ({len(C)})")
    cm = plt.get_cmap("tab10")
    for k, c in enumerate(comps[:10]):
        ax[0].plot(c["_al"][:, 0], c["_al"][:, 1], ".-", ms=3, lw=1, color=cm(k),
                   label=f"SfM model {c['model']}: {c['photos']} photos, ATE {c['ate_rmse_m']:.2f} m")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=7); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].set_title("Photo-only SfM (COLMAP, Sim(3)-aligned per model) vs INS")
    reg = np.zeros(len(C), bool)
    for c in comps:
        reg[c["_idx"]] = True
    ax[1].bar(np.arange(len(C)), reg, width=1.0, color="tab:green")
    ax[1].set_title(f"registered photos: {reg.sum()}/{len(C)} in {len(comps)} models")
    ax[1].set_xlabel("photo index"); ax[1].set_yticks([])
    fig.tight_layout(); fig.savefig(out / "sfm_vs_ins.png", dpi=110)
    for c in comps:
        c.pop("_idx"); c.pop("_al")
    rep = dict(photos=len(C), registered=int(reg.sum()), models=len(comps), components=comps,
               settings=dict(max_size=a.max_size, overlap=a.overlap, matcher="sequential, no loop detection"))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "components"}, indent=1))
    for c in comps[:8]:
        print(c)


if __name__ == "__main__":
    main()
