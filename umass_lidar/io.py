"""Point cloud I/O for the legacy airborne path (crop a big LAS into a compact .npz).

All loaders return a dict with:
    xyz:    (N, 3) float64, original CRS coordinates (NAD83 / UTM 18N for Airborne.las)
    rgb:    (N, 3) uint8
    cls:    (N,)   uint8 ASPRS classification (0 if unknown)
    normal: (N, 3) float32 or None
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _rgb16_to_8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.size and rgb.max() > 255:
        rgb = rgb >> 8
    return rgb.astype(np.uint8)


def crop_las(path: str | Path, bbox_xy: tuple[float, float, float, float],
             chunk_size: int = 20_000_000, verbose: bool = True) -> dict:
    """Stream a (huge) LAS file and keep points inside bbox_xy = (xmin, ymin, xmax, ymax).

    Filtering happens on the raw int32 coordinates so each chunk costs little memory.
    """
    import laspy

    xmin, ymin, xmax, ymax = bbox_xy
    out = {"xyz": [], "rgb": [], "cls": [], "normal": []}
    t0 = time.time()
    with laspy.open(str(path)) as f:
        h = f.header
        sx, sy, _ = h.scales
        ox, oy, _ = h.offsets
        ix0, ix1 = int(np.floor((xmin - ox) / sx)), int(np.ceil((xmax - ox) / sx))
        iy0, iy1 = int(np.floor((ymin - oy) / sy)), int(np.ceil((ymax - oy) / sy))
        dims = set(h.point_format.dimension_names)
        has_normal = "normal x" in dims
        done = 0
        for pts in f.chunk_iterator(chunk_size):
            X, Y = pts.X, pts.Y
            m = (X >= ix0) & (X <= ix1) & (Y >= iy0) & (Y <= iy1)
            done += len(pts)
            if m.any():
                p = pts[m]
                out["xyz"].append(np.stack([p.x, p.y, p.z], 1))
                out["rgb"].append(_rgb16_to_8(np.stack([p.red, p.green, p.blue], 1)))
                out["cls"].append(np.asarray(p.classification, np.uint8))
                if has_normal:
                    out["normal"].append(np.stack([p["normal x"], p["normal y"], p["normal z"]], 1)
                                         .astype(np.float32))
            if verbose:
                kept = sum(len(a) for a in out["xyz"])
                print(f"[crop_las] {done/1e6:.0f}/{h.point_count/1e6:.0f} M scanned, "
                      f"{kept/1e6:.2f} M kept, {time.time()-t0:.0f}s", flush=True)
    res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    if res["xyz"] is None:
        raise ValueError(f"No points in bbox {bbox_xy}")
    return res


def save_npz(path: str | Path, cloud: dict) -> None:
    np.savez(path, **{k: v for k, v in cloud.items() if v is not None})


def load_npz(path: str | Path) -> dict:
    z = np.load(path)
    return {"xyz": z["xyz"], "rgb": z["rgb"], "cls": z["cls"],
            "normal": z["normal"] if "normal" in z else None}
