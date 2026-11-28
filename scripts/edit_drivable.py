"""Correct the drivable area of a twin by hand: draw polygons that are forced drivable or forbidden.

    conda activate umasstwin
    python scripts/edit_drivable.py --config configs/pond.json

The automatic drivable area (semantic grid + slope limit + robot footprint) stays as it is; the polygons
drawn here are stored next to the twin in assets/twin/<name>/drivable_edits.json and laid over it whenever
a UGV route is planned (run_fleet.py, run_nav.py, run_team.py) and in the waypoint window. Delete the file
to go back to the automatic result.

Window:  red = not drivable, orange = drivable but cut off from the largest drivable area.
    A            start an "allow" polygon (forces the cells inside to drivable)
    B            start a "block" polygon (forbids the cells inside)
    left click   add a corner            right click / Backspace   remove the last corner
    Enter        finish the polygon (saved at once, the colours are recomputed)
    D            then click inside a finished polygon to delete it
    Esc          drop the polygon being drawn          Q   quit

"allow" only changes where routes may go. The collision ground keeps its shape: obstacle cells are raised
by up to 0.6 m there and 3D structures keep their collision, so the robot can still bump into something
real. The window warns when an "allow" polygon covers raised cells.

Without a display:
    python scripts/edit_drivable.py --config configs/pond.json --allow X0 Y0 X1 Y1 X2 Y2 ...
    python scripts/edit_drivable.py --config configs/pond.json --block X0 Y0 X1 Y1 X2 Y2 ...
    python scripts/edit_drivable.py --config configs/pond.json --list | --clear
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umass_lidar import nav  # noqa: E402

COL = dict(allow="#19d219", block="#ff2a2a")


def robot_radius(name):
    f = ROOT / "assets/robots/robots.json"
    reg = json.loads(f.read_text()) if f.exists() else {}
    return (reg[name]["track"] * 0.5 + 0.35) if name in reg else 0.92


class Site:
    def __init__(self, config, robot, max_slope):
        cfg = json.loads(Path(config).read_text())
        self.root = ROOT / "assets/twin" / cfg["name"]
        if not (self.root / "ground.json").exists():
            sys.exit(f"[edit] {self.root} has no ground: build the twin first")
        self.meta = json.loads((self.root / "ground.json").read_text())
        self.h = np.load(self.root / "ground_height.npy")
        self.sem = np.load(self.root / "ground_semantic.npy")
        g = self.root / "ground_ground.npy"
        self.raise_ = self.h - np.load(g) if g.exists() else np.zeros_like(self.h)   # stubs above the bare ground
        self.res, self.size = self.meta["mesh_res"], self.meta["size_xy"]
        self.radius, self.max_slope = robot_radius(robot), max_slope
        self.file = self.root / nav.EDITS_FILE
        self.edits = nav.load_drivable_edits(self.root)

    def save(self):
        if self.edits:
            self.file.write_text(json.dumps(dict(
                frame="local metres (x east, y north) from the twin origin",
                note="applied in order over the automatic drivable area; see scripts/edit_drivable.py",
                edits=self.edits), indent=2))
        elif self.file.exists():
            self.file.unlink()

    def masks(self):
        """(drivable, connected to the largest drivable area) at the planner's 0.5 m resolution."""
        coarse, cres = nav.coarse_mask(self.h, self.res, 0.5, max_slope_deg=self.max_slope, robot_radius=self.radius,
                                       blocked=self.sem != 0, edits=self.edits)
        lab, n = ndimage.label(coarse, structure=np.ones((3, 3)))
        if n == 0:
            return coarse, coarse
        area = ndimage.sum(coarse, lab, np.arange(1, n + 1))
        return coarse, lab == 1 + int(np.argmax(area))

    def stats(self):
        coarse, main = self.masks()
        return dict(drivable_ha=float(coarse.sum() * 0.25 / 1e4), main_ha=float(main.sum() * 0.25 / 1e4))

    def warn(self, polygon):
        """Text if an allow polygon covers cells that stick out of the collision ground, else ''."""
        m = nav.polygon_mask(self.h.shape, self.res, polygon)
        up = self.raise_[m]
        n = int((up > 0.1).sum())
        if n == 0:
            return ""
        return (f"{n * self.res ** 2:.1f} m2 of this polygon is raised by up to {float(up.max()):.2f} m in the collision "
                f"ground: routes may cross it, but the robot can bump there")


def window(site):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.path import Path as MPath

    img = cv2.imread(str(site.root / "map_base.png"))[..., ::-1]
    s = max(1, int(np.ceil(max(img.shape[:2]) / 2400)))
    sx, sy = site.size
    fig, ax = plt.subplots(figsize=(11, 11 * sy / sx if sy <= sx else 11))
    fig.canvas.manager.set_window_title("Edit drivable area")
    ax.imshow(img[::s, ::s], extent=[0, sx, 0, sy], origin="upper", interpolation="bilinear")
    shade = ax.imshow(np.zeros((2, 2, 4), np.float32), extent=[0, sx, 0, sy], origin="lower", interpolation="nearest", zorder=2)
    ax.set_xlim(0, sx); ax.set_ylim(0, sy); ax.set_aspect("equal")
    ax.set_xlabel("x east [m]"); ax.set_ylabel("y north [m]")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.06)
    banner = fig.text(0.5, 0.86, "", ha="center", va="top", fontsize=11, weight="bold", color="w", visible=False,
                      bbox=dict(boxstyle="round,pad=0.6", fc="firebrick", ec="k", alpha=0.92), zorder=10)
    st = dict(mode=None, pts=[], delete=False)
    art = []

    def say(text, color="firebrick"):
        banner.set_text(text); banner.set_visible(bool(text)); banner.get_bbox_patch().set_facecolor(color)
        if text:
            print(f"[edit] {text}", flush=True)

    def recompute():
        drv, main = site.masks()
        k = max(1, int(np.ceil(max(drv.shape) / 1600)))
        rgba = np.zeros((*drv[::k, ::k].shape, 4), np.float32)
        rgba[~drv[::k, ::k]] = (0.9, 0.05, 0.05, 0.42)
        rgba[drv[::k, ::k] & ~main[::k, ::k]] = (1.0, 0.6, 0.0, 0.5)
        shade.set_data(rgba)
        st["ha"] = (drv.sum() * 0.25 / 1e4, main.sum() * 0.25 / 1e4)

    def redraw():
        for a_ in art:
            a_.remove()
        art.clear()
        for i, e in enumerate(site.edits):
            art.append(ax.add_patch(Polygon(e["polygon"], closed=True, fill=False, ec=COL[e["type"]], lw=2.5, zorder=4)))
            c = np.mean(e["polygon"], axis=0)
            art.append(ax.annotate(f"{i + 1} {e['type']}", c, ha="center", color="w", fontsize=8, weight="bold", zorder=5,
                                   bbox=dict(boxstyle="round,pad=0.15", fc=COL[e["type"]], ec="k", alpha=0.85)))
        if st["pts"]:
            P = np.array(st["pts"] + st["pts"][:1])
            art.extend(ax.plot(P[:, 0], P[:, 1], "--o", color=COL[st["mode"]], lw=2, ms=6, mec="k", zorder=6))
        what = ("click a polygon to delete it" if st["delete"] else
                f"drawing {st['mode'].upper()} polygon: click corners, Enter = finish, Esc = drop" if st["mode"] else
                "A = allow polygon   B = block polygon   D = delete a polygon   Q = quit")
        ax.set_title(f"{site.root.name}: drivable {st['ha'][0]:.2f} ha, largest connected area {st['ha'][1]:.2f} ha "
                     f"(robot radius {site.radius:.2f} m, slope < {site.max_slope:.0f} deg)   |   {len(site.edits)} edits\n"
                     f"red = not drivable, orange = cut off from the largest area   |   {what}", fontsize=10)
        fig.canvas.draw_idle()

    def changed(msg=""):
        site.save()
        say("recomputing ...", "0.25"); fig.canvas.draw(); fig.canvas.flush_events()
        recompute()
        say(msg, "darkorange" if msg else "firebrick")
        redraw()

    def on_click(e):
        if e.inaxes is not ax or getattr(fig.canvas.toolbar, "mode", ""):
            return
        if st["delete"]:
            for i in range(len(site.edits) - 1, -1, -1):
                if MPath(site.edits[i]["polygon"]).contains_point((e.xdata, e.ydata)):
                    site.edits.pop(i); st["delete"] = False
                    changed(); return
            say("no polygon under the click"); redraw(); return
        if st["mode"] is None:
            say("press A (allow) or B (block) first, then click the corners"); redraw(); return
        if e.button == 1:
            st["pts"].append([round(float(e.xdata), 2), round(float(e.ydata), 2)])
        elif e.button == 3 and st["pts"]:
            st["pts"].pop()
        say(""); redraw()

    def on_key(e):
        if e.key in ("a", "b"):
            st.update(mode="allow" if e.key == "a" else "block", pts=[], delete=False)
        elif e.key == "d":
            st.update(mode=None, pts=[], delete=bool(site.edits))
        elif e.key == "escape":
            st.update(mode=None, pts=[], delete=False)
        elif e.key == "backspace" and st["pts"]:
            st["pts"].pop()
        elif e.key in ("enter", "return", " ") and st["mode"]:
            if len(st["pts"]) < 3:
                say("a polygon needs at least 3 corners"); redraw(); return
            edit = dict(type=st["mode"], polygon=list(st["pts"]))
            site.edits.append(edit)
            st.update(mode=None, pts=[])
            changed(site.warn(edit["polygon"]) if edit["type"] == "allow" else ""); return
        elif e.key == "q":
            plt.close(fig); return
        else:
            return
        say(""); redraw()

    for k in ("keymap.home", "keymap.quit", "keymap.back", "keymap.save", "keymap.fullscreen", "keymap.pan",
              "keymap.zoom", "keymap.grid", "keymap.yscale", "keymap.xscale"):
        plt.rcParams[k] = []
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    recompute(); redraw()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--robot", default="warthog", help="footprint used for the preview (the planner uses each robot's own)")
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--allow", type=float, nargs="+", metavar="X Y", help="add an allow polygon (local metres) and exit")
    ap.add_argument("--block", type=float, nargs="+", metavar="X Y", help="add a block polygon (local metres) and exit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clear", action="store_true", help="remove every edit (back to the automatic result)")
    a = ap.parse_args()
    site = Site(a.config, a.robot, a.max_slope)

    if a.clear:
        site.edits = []; site.save()
        print(f"[edit] removed {site.file}")
        return
    if a.list:
        for i, e in enumerate(site.edits):
            print(f"{i + 1}: {e['type']:5s} {len(e['polygon'])} corners, centre {np.round(np.mean(e['polygon'], axis=0), 1)}")
        print(f"[edit] {len(site.edits)} edits in {site.file}; {site.stats()}")
        return
    if a.allow or a.block:
        before = site.stats()
        for kind, v in (("allow", a.allow), ("block", a.block)):
            if v:
                if len(v) < 6 or len(v) % 2:
                    sys.exit(f"[edit] --{kind} needs at least 3 x y pairs")
                poly = [[float(v[i]), float(v[i + 1])] for i in range(0, len(v), 2)]
                site.edits.append(dict(type=kind, polygon=poly))
                w = site.warn(poly) if kind == "allow" else ""
                if w:
                    print(f"[edit] WARNING: {w}")
        site.save()
        print(f"[edit] {len(site.edits)} edits -> {site.file}\n[edit] before {before}\n[edit] after  {site.stats()}")
        return
    window(site)
    print(f"[edit] {len(site.edits)} edits" + (f" in {site.file}" if site.edits else " (no file: automatic drivable area)"))


if __name__ == "__main__":
    main()
