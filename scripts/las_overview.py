"""Render a top-down overview map of the whole LAS so crop regions can be picked by eye.

Streams the file once and writes:
    <out>.png            orthophoto (mean point color per cell) with a UTM grid drawn on it
    <out>_raw.png        same without grid
    <out>.json           georeference: pixel (col, row) <-> UTM

    python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview --res 1.0

Optionally mark candidate regions (center x, center y, size in metres):
    python scripts/las_overview.py ... --mark 703437 4696145 176
"""
import argparse
import json
import time
from pathlib import Path

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--las", required=True)
ap.add_argument("--out", default="data/overview")
ap.add_argument("--res", type=float, default=1.0, help="metres per pixel")
ap.add_argument("--grid", type=float, default=500.0, help="grid spacing in metres")
ap.add_argument("--mark", type=float, nargs=3, action="append", default=[],
                metavar=("CX", "CY", "SIZE"), help="draw a square region (repeatable)")
args = ap.parse_args()

t0 = time.time()
with laspy.open(args.las) as f:
    h = f.header
    x0, y0 = h.mins[0], h.mins[1]
    x1, y1 = h.maxs[0], h.maxs[1]
    nx, ny = int(np.ceil((x1 - x0) / args.res)), int(np.ceil((y1 - y0) / args.res))
    acc = np.zeros((ny * nx, 3), np.float64)
    cnt = np.zeros(ny * nx, np.int64)
    done = 0
    for pts in f.chunk_iterator(20_000_000):
        ix = np.clip(((pts.x - x0) / args.res).astype(np.int64), 0, nx - 1)
        iy = np.clip(((pts.y - y0) / args.res).astype(np.int64), 0, ny - 1)
        lin = iy * nx + ix
        cnt += np.bincount(lin, minlength=ny * nx)
        for c, name in enumerate(("red", "green", "blue")):
            acc[:, c] += np.bincount(lin, np.asarray(pts[name], np.float64), ny * nx)
        done += len(pts)
        print(f"[overview] {done/1e6:.0f}/{h.point_count/1e6:.0f} M  {time.time()-t0:.0f}s", flush=True)

img = acc / np.maximum(cnt, 1)[:, None]
if img.max() > 255:
    img /= 256.0
img = img.reshape(ny, nx, 3).astype(np.uint8)[::-1]  # row 0 = north
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(img).save(f"{out}_raw.png")


def to_px(x, y):
    return (x - x0) / args.res, (y1 - y) / args.res


im = Image.fromarray(img).convert("RGB")
d = ImageDraw.Draw(im)
font = ImageFont.load_default(size=max(12, nx // 100))
for gx in np.arange(np.ceil(x0 / args.grid) * args.grid, x1, args.grid):
    px, _ = to_px(gx, y0)
    d.line([(px, 0), (px, ny)], fill=(255, 255, 0), width=1)
    d.text((px + 3, 3), f"E {gx:.0f}", fill=(255, 255, 0), font=font)
for gy in np.arange(np.ceil(y0 / args.grid) * args.grid, y1, args.grid):
    _, py = to_px(x0, gy)
    d.line([(0, py), (nx, py)], fill=(255, 255, 0), width=1)
    d.text((3, py + 3), f"N {gy:.0f}", fill=(255, 255, 0), font=font)
for i, (cx, cy, s) in enumerate(args.mark):
    a, b = to_px(cx - s / 2, cy + s / 2), to_px(cx + s / 2, cy - s / 2)
    d.rectangle([a, b], outline=(255, 0, 0), width=3)
    d.text((a[0] + 4, a[1] + 4), f"#{i} {s:.0f} m", fill=(255, 0, 0), font=font)
im.save(f"{out}.png")

Path(f"{out}.json").write_text(json.dumps({
    "res": args.res, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "width": nx, "height": ny,
    "pixel_to_utm": "x = xmin + col*res ; y = ymax - row*res",
    "density_pts_per_m2_mean": float(cnt[cnt > 0].mean() / args.res ** 2),
    "covered_fraction": float((cnt > 0).mean())}, indent=2))
print(f"wrote {out}.png ({nx}x{ny}px) in {time.time()-t0:.0f}s")
