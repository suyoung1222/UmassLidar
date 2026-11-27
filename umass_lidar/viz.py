"""Top-down terrain map with a local-metre grid (for picking start / goal points)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import nav


def write_terrain_map(prefix, grid_m=10.0, px_per_m=5.0, robot_radius=0.45, points=None, path=None,
                      out=None, image=None):
    """Albedo with drivable cells tinted green, blocked cells dimmed, grid in local metres.

    points: optional {label: (x, y)} to mark;  path: optional (N, 2) polyline.
    """
    prefix = Path(prefix)
    meta = json.loads(prefix.with_suffix(".json").read_text())
    h = np.load(str(prefix) + "_height.npy")
    res = meta["mesh_res"]
    sx, sy = meta["size_xy"]
    W, H = int(sx * px_per_m), int(sy * px_per_m)
    src = image if image is not None and Path(image).exists() else str(prefix) + "_albedo.png"
    alb = cv2.imread(str(src))                                 # row 0 = north (max y), BGR
    img = cv2.resize(alb, (W, H), interpolation=cv2.INTER_AREA)
    sem = Path(str(prefix) + "_semantic.npy")
    blocked = (np.load(sem) != 0) if sem.exists() else None    # obstacle / roof / unobserved
    trav = nav.traversable_mask(h, res, 15.0, robot_radius, 0.0, blocked)
    block = cv2.resize((~trav[::-1]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    img[block] = (0.6 * img[block]).astype(np.uint8)                       # blocked: dimmed
    img[~block] = (0.65 * img[~block] + 0.35 * np.array([60, 230, 60])).astype(np.uint8)  # drivable: green

    px = lambda x, y: (int(x * px_per_m), int(H - y * px_per_m))
    for gx in np.arange(0, sx, grid_m):
        major = int(round(gx / grid_m)) % 5 == 0
        cv2.line(img, px(gx, 0), px(gx, sy), (0, 200, 255) if major else (90, 110, 110), 1)
        if major:
            cv2.putText(img, f"x={gx:.0f}", (px(gx, 0)[0] + 3, H - 6), 0, 0.5, (0, 255, 255), 2)
    for gy in np.arange(0, sy, grid_m):
        major = int(round(gy / grid_m)) % 5 == 0
        cv2.line(img, px(0, gy), px(sx, gy), (0, 200, 255) if major else (90, 110, 110), 1)
        if major:
            cv2.putText(img, f"y={gy:.0f}", (4, px(0, gy)[1] - 4), 0, 0.5, (0, 255, 255), 2)
    if path is not None:
        cv2.polylines(img, [np.array([px(*p) for p in path], np.int32)], False, (255, 120, 0), 3)
    for label, (x, y) in (points or {}).items():
        cv2.circle(img, px(x, y), 8, (0, 255, 0), -1)
        cv2.circle(img, px(x, y), 8, (0, 0, 0), 2)
        cv2.putText(img, f"{label} ({x:.0f},{y:.0f})", (px(x, y)[0] + 10, px(x, y)[1] + 5), 0, 0.6,
                    (0, 255, 0), 2)
    out = Path(out) if out else Path(str(prefix) + "_map.png")
    cv2.imwrite(str(out), img)
    return out


def depth_vis(d16, max_m=10.0):
    v = np.clip(d16.astype(np.float32) / (max_m * 1000.0), 0, 1)
    img = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    img[d16 == 0] = 0
    return img


def label(img, text, org=(10, 24), scale=0.65):
    cv2.putText(img, text, org, 0, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, 0, scale, (255, 255, 255), 2, cv2.LINE_AA)
