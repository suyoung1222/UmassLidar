"""Heightmap-based traversability, spawn sampling, A* planning and pure pursuit.

Pure numpy/scipy so it runs in both the processing env and the Isaac Gym env.
Grid convention: h[row=y, col=x], cell centre at ((col+0.5)*res, (row+0.5)*res).
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage


def slope_deg(h, res):
    gy, gx = np.gradient(h, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


EDITS_FILE = "drivable_edits.json"


def load_drivable_edits(twin_root):
    """Manual corrections of the drivable area drawn with scripts/edit_drivable.py:
    [dict(type="allow" | "block", polygon=[[x, y], ...])] in local metres, applied in order. [] if none."""
    import json
    from pathlib import Path
    f = Path(twin_root) / EDITS_FILE
    return json.loads(f.read_text()).get("edits", []) if f.exists() else []


def polygon_mask(shape, res, polygon):
    """Bool grid (row 0 = south) of the cells whose centre region a polygon in metres covers."""
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(polygon, float) / res).astype(np.int32)], 1)
    return m.astype(bool)


def traversable_mask(h, res, max_slope_deg=15.0, robot_radius=0.4, margin_m=2.0, blocked=None, edits=None):
    """Cells the robot footprint can occupy: every cell within robot_radius is below max slope
    and not in `blocked` (e.g. unobserved cells of the semantic grid).
    edits (load_drivable_edits) override both tests cell by cell before the footprint is applied:
    "allow" polygons clear them, "block" polygons set them."""
    k = max(1, int(np.ceil(robot_radius / res)))
    bad = slope_deg(h, res) >= max_slope_deg
    if blocked is not None:
        bad |= blocked
    for e in edits or []:
        bad[polygon_mask(bad.shape, res, e["polygon"])] = e["type"] == "block"
    yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
    ok = ~ndimage.binary_dilation(bad, structure=(xx * xx + yy * yy) <= k * k)
    m = int(margin_m / res)
    if m > 0:
        ok[:m] = ok[-m:] = False
        ok[:, :m] = ok[:, -m:] = False
    return ok


def ground_mask(h, res, window_m=20.0, tol=0.5):
    """Cells near the local minimum height (rejects flat roofs / canopy tops in DSM terrains)."""
    return (h - ndimage.minimum_filter(h, int(window_m / res))) < tol


def find_spawn_points(h, res, n, rng, max_slope_deg=4.0, radius_m=1.0, margin_m=5.0, blocked=None):
    valid = traversable_mask(h, res, max_slope_deg, radius_m, margin_m, blocked) & ground_mask(h, res)
    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        raise RuntimeError("no flat ground cell found for spawning")
    sel = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [((xs[i] + 0.5) * res, (ys[i] + 0.5) * res, float(h[ys[i], xs[i]])) for i in sel]


def largest_ground_component(mask, h, res):
    """Largest connected traversable region that is at ground level."""
    lab, n = ndimage.label(mask)
    if n == 0:
        raise RuntimeError("no traversable cells")
    g = ground_mask(h, res)
    sizes = ndimage.sum(mask & g, lab, np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def astar(mask, start_rc, goal_rc, cost=None):
    """8-connected A* on a boolean grid; optional per-cell cost multiplier (>= 1).
    Returns list of (row, col) or None."""
    H, W = mask.shape
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    g = {start_rc: 0.0}
    parent = {}
    pq = [(0.0, start_rc)]
    closed = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal_rc:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        for dr, dc, c in nbrs:
            r, q = cur[0] + dr, cur[1] + dc
            if not (0 <= r < H and 0 <= q < W) or not mask[r, q]:
                continue
            ng = g[cur] + (c if cost is None else c * cost[r, q])
            if ng < g.get((r, q), np.inf):
                g[(r, q)] = ng
                parent[(r, q)] = cur
                f = ng + np.hypot(goal_rc[0] - r, goal_rc[1] - q)
                heapq.heappush(pq, (f, (r, q)))
    return None


def coarse_mask(h, res, plan_res=0.5, **trav_kw):
    """Traversability on the planning grid: a coarse cell is free only if all its cells are."""
    mask = traversable_mask(h, res, **trav_kw)
    s = max(1, int(round(plan_res / res)))
    H, W = (mask.shape[0] // s) * s, (mask.shape[1] // s) * s
    return mask[:H, :W].reshape(H // s, s, W // s, s).all((1, 3)), res * s


def snap_points(h, res, pts, min_area_m2=100.0, max_dist=20.0, plan_res=0.5, clearance=0.0,
                nudge_m=2.5, **trav_kw):
    """Snap the first point to the nearest sizeable ground-level drivable region and the
    remaining points to that same connected region, so a path between them exists.
    Returns [(xy, snap_distance), ...]."""
    coarse, cres = coarse_mask(h, res, plan_res, **trav_kw)
    lab, n = ndimage.label(coarse, structure=np.ones((3, 3)))
    if n == 0:
        raise ValueError("terrain has no drivable cells")
    s = int(round(cres / res))
    g = ground_mask(h, res)[:coarse.shape[0] * s, :coarse.shape[1] * s]
    g = g.reshape(coarse.shape[0], s, coarse.shape[1], s).all((1, 3))
    area = ndimage.sum(coarse & g, lab, np.arange(1, n + 1)) * cres ** 2
    big = np.isin(lab, 1 + np.nonzero(area >= min_area_m2)[0])
    if not big.any():
        big = lab == 1 + int(np.argmax(area))
    out = [snap_to_mask(big, cres, pts[0], max_dist)]
    p0 = out[0][0]
    region = lab == lab[int(p0[1] / cres), int(p0[0] / cres)]
    out += [snap_to_mask(region, cres, p, max_dist) for p in pts[1:]]
    if clearance > 0:
        # nudge each point (within nudge_m) to a cell of the same region that keeps `clearance`
        # from blocked cells, so the robot never has to stop or turn right next to an obstacle
        clear = ndimage.distance_transform_edt(coarse) * cres
        k = int(np.ceil(nudge_m / cres))
        res_out = []
        for (q, d), p in zip(out, pts):
            r, c = int(q[1] / cres), int(q[0] / cres)
            if clear[r, c] >= clearance:
                res_out.append((q, d)); continue
            r0, r1 = max(0, r - k), min(coarse.shape[0], r + k + 1)
            c0, c1 = max(0, c - k), min(coarse.shape[1], c + k + 1)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            ok = region[r0:r1, c0:c1]
            dist = np.hypot(rr - r, cc - c) * cres
            score = np.where(ok, np.minimum(clear[r0:r1, c0:c1], clearance) * 10 - dist, -np.inf)
            j = np.unravel_index(np.argmax(score), score.shape)
            q2 = ((cc[j] + 0.5) * cres, (rr[j] + 0.5) * cres)
            res_out.append((q2, float(np.hypot(q2[0] - p[0], q2[1] - p[1]))))
        out = res_out
    return out


def plan_path(h, res, start_xy, goal_xy, plan_res=0.5, **trav_kw):
    """A* path in metres (N, 2) from start to goal on a coarsened traversability grid."""
    coarse, cres = coarse_mask(h, res, plan_res, **trav_kw)
    to_rc = lambda p: (int(p[1] / cres), int(p[0] / cres))
    # prefer paths that keep clear of obstacles: cost rises within ~1.5 m of blocked cells
    clear = ndimage.distance_transform_edt(coarse) * cres
    cost = 1.0 + 4.0 * np.exp(-clear / 0.75)
    path = astar(coarse, to_rc(start_xy), to_rc(goal_xy), cost)
    if path is None:
        return None
    xy = np.array([((c + 0.5) * cres, (r + 0.5) * cres) for r, c in path])
    if len(xy) > 5:  # smooth the staircase
        xy[1:-1] = ndimage.uniform_filter1d(xy, 5, axis=0, mode="nearest")[1:-1]
    return xy


def sample_route(h, res, rng, min_len=40.0, tries=30, **trav_kw):
    """Random start/goal pair in the largest ground region with a path of at least min_len."""
    mask = traversable_mask(h, res, **trav_kw)
    comp = largest_ground_component(mask, h, res)
    flat = comp & traversable_mask(h, res, 5.0, 1.0, 5.0, blocked=trav_kw.get("blocked"), edits=trav_kw.get("edits"))
    ys, xs = np.nonzero(flat)
    best = None
    for _ in range(tries):
        i, j = rng.choice(len(ys), 2, replace=False)
        a = ((xs[i] + 0.5) * res, (ys[i] + 0.5) * res)
        b = ((xs[j] + 0.5) * res, (ys[j] + 0.5) * res)
        if np.hypot(a[0] - b[0], a[1] - b[1]) < min_len * 0.7:
            continue
        p = plan_path(h, res, a, b, **trav_kw)
        if p is None:
            continue
        L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        if best is None or L > best[1]:
            best = (p, L)
        if L >= min_len:
            break
    if best is None:
        raise RuntimeError("could not sample a route")
    return best[0]


def pure_pursuit(pos_xy, yaw, path, lookahead=1.0, v_max=1.0, w_max=1.5, goal_tol=0.5):
    """Returns (v, w, done). Slows down for large heading errors."""
    d = np.linalg.norm(path - pos_xy, axis=1)
    i = int(np.argmin(d))
    if d[-1] < goal_tol:
        return 0.0, 0.0, True
    j = i
    while j < len(path) - 1 and np.linalg.norm(path[j] - pos_xy) < lookahead:
        j += 1
    tgt = path[j]
    err = np.arctan2(tgt[1] - pos_xy[1], tgt[0] - pos_xy[0]) - yaw
    err = (err + np.pi) % (2 * np.pi) - np.pi
    w = float(np.clip(2.0 * err, -w_max, w_max))
    # keep creeping forward while turning: small skid-steer robots stall when pivoting in place
    # on a high-friction mesh; only a near-reversal turns on the spot
    v = float(v_max * max(0.2 if abs(err) < 2.3 else 0.0, max(0.0, np.cos(err)) ** 2))
    return v, w, False


def snap_to_mask(mask, res, xy, max_dist=15.0):
    """Nearest True cell centre to xy (metres); raises if farther than max_dist."""
    r, c = int(xy[1] / res), int(xy[0] / res)
    if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]:
        return (float(xy[0]), float(xy[1])), 0.0
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    r = int(np.clip(r, 0, mask.shape[0] - 1)); c = int(np.clip(c, 0, mask.shape[1] - 1))
    rr, cc = idx[0][r, c], idx[1][r, c]
    p = ((cc + 0.5) * res, (rr + 0.5) * res)
    d = float(np.hypot(p[0] - xy[0], p[1] - xy[1]))
    if d > max_dist:
        raise ValueError(f"point {xy} is {d:.1f} m from the nearest drivable cell")
    return p, d
