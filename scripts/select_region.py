"""Pick a region on the overview map with the mouse and cut it out of the big LAS.

    python scripts/select_region.py                     # ~/Desktop/Airborne.las, data/overview*
    python scripts/select_region.py --name library --max-side 400

Drag a box on the map (drag its handles to adjust, scroll to zoom, right-drag to pan). The title
shows size / area / estimated points; a box over the limit turns red and cannot be confirmed.
Enter = crop and write <out>/<name>.las + .json, Esc or q = quit without writing.

Needs a display. Without one give the box directly:
    python scripts/select_region.py --bbox 703349 4696065 703525 4696225 --name b1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check(bbox, a, density):
    """(ok, one-line description) for a UTM bbox against the size limits."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    area = w * h
    msg = f"{w:.0f} x {h:.0f} m = {area / 1e4:.1f} ha, ~{area * density / 1e6:.0f} M pts"
    if max(w, h) > a.max_side:
        return False, msg + f"  TOO LARGE: side > {a.max_side:.0f} m, draw a smaller box"
    if area > a.max_area * 1e4:
        return False, msg + f"  TOO LARGE: area > {a.max_area:.0f} ha, draw a smaller box"
    if min(w, h) < a.min_side:
        return False, msg + f"  too small: side < {a.min_side:.0f} m"
    return True, msg + "  OK: press Enter to crop"


def pick(a, ov):
    """Interactive box on the overview image. Returns a UTM bbox or None."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(f"{a.overview}_raw.png"))
    full = (ov["xmin"], ov["xmax"], ov["ymin"], ov["ymax"])
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(img, extent=full, interpolation="nearest")
    ax.set_xlabel("UTM E (m)"); ax.set_ylabel("UTM N (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="y", lw=0.3, alpha=0.5)
    st = dict(bbox=None, ok=False, done=False, pan=None)

    def show(text, color="k"):
        ax.set_title(text, color=color, fontsize=11)
        fig.canvas.draw_idle()

    def on_select(e0, e1):
        x0, x1 = sorted((e0.xdata, e1.xdata)); y0, y1 = sorted((e0.ydata, e1.ydata))
        st["bbox"] = [round(v, 2) for v in (x0, y0, x1, y1)]
        st["ok"], msg = check(st["bbox"], a, ov["density_pts_per_m2_mean"])
        col = "tab:green" if st["ok"] else "tab:red"
        for art in sel.artists[:1]:
            art.set_edgecolor(col)
        show(msg, col)

    def on_key(ev):
        if ev.key == "enter":
            if st["ok"]:
                st["done"] = True
                plt.close(fig)
            elif st["bbox"] is not None:
                show("not accepted: " + check(st["bbox"], a, ov["density_pts_per_m2_mean"])[1], "tab:red")
        elif ev.key in ("escape", "q"):
            plt.close(fig)

    def on_scroll(ev):
        if ev.inaxes is not ax:
            return
        k = 0.8 if ev.button == "up" else 1.25
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ev.xdata + (x0 - ev.xdata) * k, ev.xdata + (x1 - ev.xdata) * k)
        ax.set_ylim(ev.ydata + (y0 - ev.ydata) * k, ev.ydata + (y1 - ev.ydata) * k)
        fig.canvas.draw_idle()

    def on_press(ev):
        if ev.button == 3 and ev.inaxes is ax:
            st["pan"] = (ev.x, ev.y, ax.get_xlim(), ax.get_ylim())

    def on_move(ev):
        if st["pan"] is None or ev.x is None:
            return
        px, py, xl, yl = st["pan"]
        bb = ax.get_window_extent()
        dx = (ev.x - px) * (xl[1] - xl[0]) / bb.width; dy = (ev.y - py) * (yl[1] - yl[0]) / bb.height
        ax.set_xlim(xl[0] - dx, xl[1] - dx); ax.set_ylim(yl[0] - dy, yl[1] - dy)
        fig.canvas.draw_idle()

    def on_release(ev):
        if ev.button == 3:
            st["pan"] = None

    sel = RectangleSelector(ax, on_select, useblit=False, button=[1], interactive=True, minspanx=1, minspany=1,
                            props=dict(facecolor="none", edgecolor="tab:green", lw=2))
    for name, fn in (("key_press_event", on_key), ("scroll_event", on_scroll), ("button_press_event", on_press),
                     ("motion_notify_event", on_move), ("button_release_event", on_release)):
        fig.canvas.mpl_connect(name, fn)
    show(f"drag a box (limit {a.max_side:.0f} m side, {a.max_area:.0f} ha)   scroll = zoom, right-drag = pan, "
         "Enter = crop, Esc = quit")
    fig.tight_layout()
    if a._hook:                      # tests drive the callbacks instead of the window
        a._hook(fig, ax, sel, st, on_select, on_key)
    else:
        plt.show()
    return st["bbox"] if st["done"] else None


def crop_to_las(las, bbox, out, chunk=20_000_000):
    """Stream `las`, keep points inside bbox (xmin, ymin, xmax, ymax), write them in the same
    point format / scale / offset / CRS. Returns the number of points written."""
    import laspy
    t0 = time.time()
    with laspy.open(str(las)) as f:
        h = f.header
        hdr = laspy.LasHeader(point_format=h.point_format, version=h.version)
        hdr.scales, hdr.offsets = h.scales, h.offsets
        hdr.vlrs.extend(h.vlrs)
        (sx, sy, _), (ox, oy, _) = h.scales, h.offsets
        ix0, ix1 = int(np.floor((bbox[0] - ox) / sx)), int(np.ceil((bbox[2] - ox) / sx))
        iy0, iy1 = int(np.floor((bbox[1] - oy) / sy)), int(np.ceil((bbox[3] - oy) / sy))
        n = done = 0
        with laspy.open(str(out), mode="w", header=hdr) as w:
            for pts in f.chunk_iterator(chunk):
                X, Y = pts.X, pts.Y
                m = (X >= ix0) & (X <= ix1) & (Y >= iy0) & (Y <= iy1)
                done += len(pts)
                if m.any():
                    w.write_points(pts[m]); n += int(m.sum())
                print(f"[crop] {done / 1e6:.0f}/{h.point_count / 1e6:.0f} M scanned, {n / 1e6:.2f} M kept, "
                      f"{time.time() - t0:.0f}s", flush=True)
    return n


def main(argv=None, hook=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las", default="~/Desktop/Airborne.las")
    ap.add_argument("--overview", default=str(ROOT / "data/overview"),
                    help="prefix written by las_overview.py (<prefix>_raw.png, <prefix>.json)")
    ap.add_argument("--out", default=str(ROOT / "data/sites"), help="output directory")
    ap.add_argument("--name", default=None, help="output name (default: site_<E>_<N>_<W>x<H>)")
    ap.add_argument("--max-side", type=float, default=500.0, help="largest allowed box side (m)")
    ap.add_argument("--max-area", type=float, default=25.0, help="largest allowed area (ha)")
    ap.add_argument("--min-side", type=float, default=20.0)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="skip the window and use this UTM box")
    a = ap.parse_args(argv)
    a._hook = hook
    las = Path(a.las).expanduser()
    jf = Path(f"{a.overview}.json")
    if not jf.exists() or not Path(f"{a.overview}_raw.png").exists():
        sys.exit(f"overview not found ({a.overview}*). Make it once:\n"
                 f"  python scripts/las_overview.py --las {a.las} --out {a.overview}")
    ov = json.loads(jf.read_text())
    if a.bbox:
        bbox = a.bbox
        ok, msg = check(bbox, a, ov["density_pts_per_m2_mean"])
        if not ok:
            sys.exit(f"[select] {msg}")
    else:
        bbox = pick(a, ov)
        if bbox is None:
            sys.exit("[select] nothing selected")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    name = a.name or f"site_{(bbox[0] + bbox[2]) / 2:.0f}_{(bbox[1] + bbox[3]) / 2:.0f}_{w:.0f}x{h:.0f}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[select] {name}: bbox {bbox}  ({w:.0f} x {h:.0f} m)")
    n = crop_to_las(las, bbox, out / f"{name}.las")
    if n == 0:
        (out / f"{name}.las").unlink()
        sys.exit("[select] no points inside the box (outside the scan?)")
    meta = dict(name=name, source=str(las), bbox_utm=bbox, size_xy=[w, h], area_ha=w * h / 1e4, points=n,
                density_pts_per_m2=n / (w * h))
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[select] wrote {out / name}.las  ({n / 1e6:.2f} M pts, {n / (w * h):.0f} pts/m2) and {name}.json")


if __name__ == "__main__":
    main()
