"""Click waypoints for several robots on the twin's top-view map (matplotlib window).

    routes = pick_routes(map_png, size_xy, robots, colors, validate=fn, blocked=grid)

robots: [dict(name="ugv1", type="ugv" | "uav", model=...)], handled one after the other.
Returns {name: [(x, y), ...]} in local metres (first point = start), or None if the window was
closed before every robot had a route.
"""
from __future__ import annotations

import numpy as np

HELP = "left click: add waypoint (first = start)   right click / Backspace: undo   R: clear   Enter / Space / button: next robot   Q: quit"


def pick_routes(map_png, size_xy, robots, colors, validate=None, selectable=None, min_points=2, max_px=2400):
    """validate(robot, pts) -> (ok, message, path_xy or None): called on Enter; a failed route stays
    editable. selectable(robot, start_xy or None) -> (drivable, reachable) bool grids (row 0 = south),
    or None = everywhere. Ground the robot cannot drive on is covered in translucent red; drivable
    ground with no connection to the start (reachable is None until the start is placed) in orange."""
    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread(str(map_png))[..., ::-1]
    s = max(1, int(np.ceil(max(img.shape[:2]) / max_px)))
    img = img[::s, ::s]
    sx, sy = size_xy
    fig, ax = plt.subplots(figsize=(11, 11 * sy / sx if sy <= sx else 11))
    fig.canvas.manager.set_window_title("Select waypoints")
    ax.imshow(img, extent=[0, sx, 0, sy], origin="upper", interpolation="bilinear")
    shade = ax.imshow(np.zeros((2, 2, 4), np.float32), extent=[0, sx, 0, sy], origin="lower",
                      interpolation="nearest", visible=False, zorder=2)
    shade_key = [None]

    def update_shade():
        """Red over the cells the current robot cannot be sent to (depends on its start once placed)."""
        r = robots[st["i"]]
        start = st["pts"][0] if st["pts"] else None
        key = (st["i"], start)
        if key == shade_key[0]:
            return
        shade_key[0] = key
        ok = selectable(r, start) if selectable else None
        if ok is None:
            shade.set_visible(False); return
        drv, reach = ok
        k = max(1, int(np.ceil(max(drv.shape) / 1600)))
        rgba = np.zeros((*drv[::k, ::k].shape, 4), np.float32)
        rgba[~drv[::k, ::k]] = (0.9, 0.05, 0.05, 0.42)
        if reach is not None:
            rgba[drv[::k, ::k] & ~reach[::k, ::k]] = (1.0, 0.6, 0.0, 0.5)
        shade.set_data(rgba); shade.set_visible(True)

    ax.set_xlim(0, sx); ax.set_ylim(0, sy); ax.set_aspect("equal")
    ax.set_xlabel("x east [m]"); ax.set_ylabel("y north [m]")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.11)
    from matplotlib.widgets import Button
    btn = {k: Button(fig.add_axes(rect), text, color=col, hovercolor="0.95") for k, (rect, text, col) in dict(
        next=([0.70, 0.015, 0.28, 0.05], "Next robot  (Enter)", "#b9f0b9"),
        undo=([0.07, 0.015, 0.14, 0.05], "Undo", "0.85"), clear=([0.22, 0.015, 0.14, 0.05], "Clear", "0.85")).items()}
    banner = fig.text(0.5, 0.86, "", ha="center", va="top", fontsize=11, weight="bold", color="w", visible=False,
                      bbox=dict(boxstyle="round,pad=0.6", fc="firebrick", ec="k", alpha=0.92), zorder=10)

    def say(text, color="firebrick"):
        """Message in the middle of the map (and in the terminal); empty text hides it."""
        banner.set_text(text); banner.set_visible(bool(text)); banner.get_bbox_patch().set_facecolor(color)
        if text:
            print(f"[select] {text}".replace("\n", " "), flush=True)

    st = dict(i=0, pts=[], msg="", done=False)
    routes, drawn, cur = {}, [], []

    def redraw():
        for a_ in cur:
            a_.remove()
        cur.clear()
        if st["i"] >= len(robots):
            return
        r = robots[st["i"]]; col = colors[r["name"]]
        update_shade()
        if st["pts"]:
            P = np.array(st["pts"])
            cur.extend(ax.plot(P[:, 0], P[:, 1], "-o", color=col, lw=2, ms=7, mec="k", mew=1.2))
            cur.append(ax.plot(P[0, 0], P[0, 1], "s", color=col, ms=12, mec="k", mew=1.5)[0])
            for j, p in enumerate(P):
                cur.append(ax.annotate("start" if j == 0 else str(j), p, xytext=(6, 6), textcoords="offset points",
                                       color="w", fontsize=9, weight="bold",
                                       bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.6)))
        kind = ("UGV: red = not drivable" + (", orange = drivable but not connected to this start" if st["pts"] else "")) if r["type"] == "ugv" else "UAV: flies straight legs above everything"
        ax.set_title(f"[{st['i'] + 1}/{len(robots)}] {r['name']} ({r['model']})  -  {kind}\n{HELP}", fontsize=10)
        last = st["i"] == len(robots) - 1
        btn["next"].label.set_text("Start simulation  (Enter)" if last else f"Next: {robots[st['i'] + 1]['name']}  (Enter)")
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes is not ax or st["i"] >= len(robots) or getattr(fig.canvas.toolbar, "mode", ""):
            return                                             # toolbar zoom / pan is active
        if e.button == 1:
            st["pts"].append((float(e.xdata), float(e.ydata)))
        elif e.button == 3 and st["pts"]:
            st["pts"].pop()
        say("")
        redraw()

    def advance(_=None):
        if st["i"] >= len(robots) or st.get("busy"):
            return
        r = robots[st["i"]]
        if len(st["pts"]) < min_points:
            say(f"{r['name']} needs at least {min_points} point{'s (start + goal)' if min_points > 1 else ' (its start)'}."
                "\nClick on the map first.")
            redraw(); return
        st["busy"] = True
        say(f"planning the route of {r['name']} ...", "0.25")
        fig.canvas.draw(); fig.canvas.flush_events()
        try:
            ok, msg, path = validate(r, list(st["pts"])) if validate else (True, "", None)
        except Exception as e:                                  # never leave the window stuck on an error
            ok, msg, path = False, f"{r['name']}: {type(e).__name__}: {e}", None
        st["busy"] = False
        if not ok:
            say(f"{msg}\nUndo it (right click), click on ground with no red / orange, then press Enter again.")
            redraw(); return
        say("")
        col = colors[r["name"]]
        P = np.array(st["pts"])
        if path is not None:
            drawn.extend(ax.plot(path[:, 0], path[:, 1], "-", color=col, lw=2.5, alpha=0.9))
        drawn.extend(ax.plot(P[:, 0], P[:, 1], "o" if path is not None else "-o", color=col, ms=6, mec="k", lw=2.5))
        drawn.append(ax.annotate(r["name"], P[0], xytext=(8, -14), textcoords="offset points", color=col,
                                 fontsize=10, weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="k", ec="none", alpha=0.7)))
        routes[r["name"]] = list(st["pts"])
        st.update(i=st["i"] + 1, pts=[], msg="")
        if st["i"] >= len(robots):
            st["done"] = True
            plt.close(fig); return
        redraw()

    def undo(_=None):
        if st["pts"]:
            st["pts"].pop()
        say(""); redraw()

    def clear(_=None):
        st["pts"].clear()
        say(""); redraw()

    def on_key(e):
        if e.key in ("enter", "return", " ", "space", "n"):
            advance()
        elif e.key == "backspace":
            undo()
        elif e.key == "r":
            clear()
        elif e.key == "q":
            plt.close(fig)

    btn["next"].on_clicked(advance); btn["undo"].on_clicked(undo); btn["clear"].on_clicked(clear)

    # matplotlib's own key bindings would steal r (home), q (quit) and backspace (back)
    for k in ("keymap.home", "keymap.quit", "keymap.back", "keymap.save", "keymap.fullscreen"):
        plt.rcParams[k] = []
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
    return routes if st["done"] else None
