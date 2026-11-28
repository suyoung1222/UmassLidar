"""Teleoperation: drive one UGV (Warthog) or fly one UAV (Starling 2 Max) in a twin with the keyboard.

    conda activate umasstwin                      # environment.yml
    python scripts/run_teleop.py --config configs/pond.json --platform ugv      # click the start on the map
    python scripts/run_teleop.py --config configs/pond.json --platform uav --start 340 450

Without --start a window shows the twin's top view: click where the robot starts, then Enter.
The Isaac Gym window must have the keyboard focus (click it once).

Teleop keys           UGV                           UAV
    I / K             forward / back                forward / back (along its heading)
    J / L             turn left / right             yaw left / right
    U / O             -                             strafe left / right
    Y / H             -                             climb / descend
    = / -             speed x1.25 / x0.8
With no key held the UGV stops and the UAV holds its position. The UAV takes off to --agl by itself.

View keys (same as run_fleet.py): 1 follow the robot, M fixed top view of the whole map, 0 free camera,
N trajectory on / off, arrows orbit / zoom, P pause. Close the window (or Ctrl-C in the terminal) to stop.

Outputs in --out (default outputs/teleop/<twin>_<platform>_<time>):
    trajectory.csv      per frame (30 Hz): pose, UTM, command
    trajectory.png      the driven / flown path on the map
    result.json         distance, duration, start, settings
"""
from isaacgym import gymapi  # noqa: F401, I001

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umass_lidar import nav, sim as S, uav as U  # noqa: E402

FPS = 30
MODEL = dict(ugv="warthog", uav="starling")


def snap_start(kind, pt, h, res, sem, size, rcfg, max_slope, edits):
    """Start point checked for the platform: a UGV is moved to the nearest drivable cell (within 20 m)."""
    if not (0 <= pt[0] < size[0] and 0 <= pt[1] < size[1]):
        raise ValueError(f"start {np.round(pt, 1)} is outside the twin (0..{size[0]:.0f}, 0..{size[1]:.0f})")
    if kind == "uav":
        return tuple(map(float, pt))
    (q, d), = nav.snap_points(h, res, [tuple(pt)], clearance=1.0, max_slope_deg=max_slope,
                              robot_radius=rcfg["track"] * 0.5 + 0.35, blocked=sem != 0, edits=edits)
    if d > 0:
        print(f"[teleop] start {np.round(pt, 1)} moved {d:.1f} m to drivable ground {np.round(q, 1)}")
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="twin config (default configs/blue_wall.json)")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--platform", required=True, choices=("ugv", "uav"))
    ap.add_argument("--model", default=None, help="robot model (default: warthog for ugv, starling for uav)")
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"), help="start in local metres (default: click on the map)")
    ap.add_argument("--yaw", type=float, default=0.0, help="start heading [deg], 0 = east, 90 = north")
    ap.add_argument("--speed", type=float, default=None, help="top speed m/s at gain 1 (default 3 for both)")
    ap.add_argument("--climb", type=float, default=2.0, help="UAV vertical speed m/s")
    ap.add_argument("--turn", type=float, default=1.2, help="turn rate rad/s")
    ap.add_argument("--agl", type=float, default=3.0, help="UAV take-off height above ground [m]")
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--collision", choices=("mesh", "voxel", "none"), default="mesh")
    ap.add_argument("--view", default="robot", choices=("robot", "top", "free"), help="viewer camera at start")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop by itself after this long (default: until the window is closed)")
    ap.add_argument("--debug-axes", type=float, nargs=4, default=None, help=argparse.SUPPRESS)   # constant fwd left up yaw
    ap.add_argument("--debug-shot", type=float, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    kind, model = a.platform, a.model or MODEL[a.platform]
    name = f"{kind}1"

    twin, cfg = S.resolve_twin(a.config, a.variant)
    gprefix = twin.parent / "ground"
    meta, h = S.load_terrain_data(gprefix)
    sem = np.load(str(gprefix) + "_semantic.npy")
    res, origin, size = meta["mesh_res"], np.array(meta["origin"]), np.array(meta["size_xy"], float)
    base_png = twin.parent / "map_base.png"
    edits = nav.load_drivable_edits(twin.parent)
    rcfg = S.robot_config(model) if kind == "ugv" else None
    color = S.FollowViewer.COLORS[0 if kind == "ugv" else 1]
    v_max = a.speed or 3.0

    # ---- where does it start
    start = [None]

    def check(_, pts):
        try:
            start[0] = snap_start(kind, pts[0], h, res, sem, size, rcfg, a.max_slope, edits)
        except ValueError as e:
            return False, f"{name}: {e}", None
        return True, "", None

    if a.start:
        ok, msg, _ = check(None, [tuple(a.start)])
        if not ok:
            sys.exit(f"[teleop] {msg}")
    else:
        from umass_lidar.waypoint_ui import pick_routes

        def selectable(r, _start):
            if kind != "ugv":
                return None
            coarse, _ = nav.coarse_mask(h, res, 0.5, max_slope_deg=a.max_slope, robot_radius=rcfg["track"] * 0.5 + 0.35,
                                        blocked=sem != 0, edits=edits)
            return coarse, None
        print("[teleop] click the start of the robot in the window, then Enter", flush=True)
        if pick_routes(base_png, size, [dict(name=name, type=kind, model=model)], {name: color}, validate=check,
                       selectable=selectable, min_points=1) is None:
            sys.exit("[teleop] start selection cancelled")
    xy = np.array(start[0], float)
    yaw0 = float(np.radians(a.yaw))
    gz0 = float(h[int(xy[1] / res), int(xy[0] / res)])
    out = Path(a.out) if a.out else ROOT / "outputs/teleop" / f"{cfg['name']}_{kind}_{time.strftime('%y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- sim
    print("[teleop] loading the twin (a large twin takes 1-2 minutes)", flush=True)
    gym, sim, sp = S.make_sim()
    env, _ = S.add_twin(gym, sim, twin, meta, a.collision)
    if kind == "ugv":
        robot = S.WheeledRobot(gym, sim, env, model, xy, yaw0, h, res, actor_name=name)
        pose = lambda: (robot.state()[0], robot.state()[2])                       # noqa: E731
        view_dist = max(3.0, 14.0 * rcfg["wheel_radius"])
        top_z = float(h.max()) + 30.0
    else:
        quad = U.Quadrotor(gym, sim, env, [xy[0], xy[1], gz0 + 0.12], yaw0, sp.dt, p=U.MODELS[model], actor_name=name)
        ctl = U.QuadController(p=U.MODELS[model])
        pose = lambda: (quad.state()["pos"], quad.state()["yaw"])                 # noqa: E731
        view_dist = 3.0
        top_z = float(np.nanmax(U.surface_model(h, U.structure_top(twin, h.shape, res), res)))
        hold = np.array([xy[0], xy[1], gz0 + a.agl])           # position held while no key is pressed
        yaw_cmd = yaw0
    ground = h[sem == 0] if (sem == 0).any() else h
    overview = dict(center=size / 2, span=size, ground_z=float(np.median(ground)), top_z=top_z)
    viewer = S.FollowViewer(gym, sim, env, {name: (pose, view_dist)}, {"robot": name, "top": "top"}.get(a.view),
                            overview=overview, colors={name: color})
    print("[teleop] keys: I/K forward/back  J/L turn" + ("  U/O strafe  Y/H up/down" if kind == "uav" else "") +
          "  =/- speed   (click the Isaac Gym window first)", flush=True)

    f = open(out / "trajectory.csv", "w", newline="")
    log = csv.writer(f)
    log.writerow(["frame", "t", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "utm_e", "utm_n", "h_ell",
                  "cmd_forward", "cmd_left", "cmd_up", "cmd_yaw_rate"])
    every, settle = int(round(1 / (sp.dt * FPS))), int(1.0 / sp.dt)
    traj, dist, frame, k, t = [], 0.0, 0, 0, 0.0
    try:
        while a.max_seconds is None or t < a.max_seconds:
            t = (k - settle) * sp.dt
            fwd, left, up, turn = a.debug_axes if a.debug_axes else viewer.axes()
            if k < settle:
                fwd = left = up = turn = 0.0
            if kind == "ugv":
                pos, quat, yaw, _, _ = robot.state()
                robot.command(v_max * fwd, a.turn * turn)
            else:
                s = quad.state()
                pos, quat, yaw = s["pos"], s["quat"], s["yaw"]
                if k < settle:
                    quad.apply_motors(np.zeros(4), s)
                else:
                    yaw_cmd += a.turn * turn * sp.dt
                    c, sn = np.cos(yaw_cmd), np.sin(yaw_cmd)
                    if fwd or left or up:
                        v_des = np.array([v_max * (c * fwd - sn * left), v_max * (sn * fwd + c * left), a.climb * up])
                        hold = pos + 0.3 * v_des                # where it will stop when the keys are released
                    else:
                        d = hold - pos
                        v_des = d * min(1.5, 2.0 / max(np.linalg.norm(d), 1e-6))    # P hold, at most 2 m/s
                    quad.apply_motors(ctl.velocity(s, v_des, yaw_cmd), s)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if k >= settle and (k - settle) % every == 0:
                if traj:
                    dist += float(np.linalg.norm(pos - traj[-1]))
                traj.append(np.array(pos))
                viewer.track(name, pos)
                log.writerow([frame, f"{t:.3f}", *np.round(pos, 4), *np.round(quat, 5), f"{yaw:.4f}",
                              f"{pos[0] + origin[0]:.3f}", f"{pos[1] + origin[1]:.3f}", f"{pos[2] + origin[2]:.3f}",
                              f"{fwd:.2f}", f"{left:.2f}", f"{up:.2f}", f"{turn:.2f}"])
                frame += 1
            if not viewer.step():
                break
            if a.debug_shot is not None and t >= a.debug_shot:
                gym.write_viewer_image_to_file(viewer.viewer, str(out / "viewer.png")); a.debug_shot = None
            k += 1
    except KeyboardInterrupt:
        print("\n[teleop] stopped")
    f.close()

    img = cv2.imread(str(base_png))
    sc, H = img.shape[1] / size[0], img.shape[0]
    th = max(2, int(round(img.shape[1] / 700)))
    if len(traj) > 1:
        P = np.array(traj)
        px = np.column_stack([P[:, 0] * sc, H - P[:, 1] * sc]).astype(np.int32)
        bgr = tuple(int(255 * c) for c in color[::-1])
        cv2.polylines(img, [px], False, (0, 0, 0), th + 3, cv2.LINE_AA); cv2.polylines(img, [px], False, bgr, th, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, px[0])), th * 3, (60, 220, 60), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, px[-1])), th * 3, bgr, -1, cv2.LINE_AA)
    cv2.imwrite(str(out / "trajectory.png"), img)
    (out / "result.json").write_text(json.dumps(dict(
        twin=str(twin), platform=kind, model=model, start=[float(xy[0]), float(xy[1])], start_yaw_deg=a.yaw,
        structure_collision=a.collision, time_s=frame / FPS, travelled_m=dist, top_speed=v_max,
        end=[float(v) for v in traj[-1]] if traj else None, origin_utm=meta["origin"], crs=cfg.get("crs")), indent=2))
    print(f"[done] {name} ({model}): {dist:.1f} m in {frame / FPS:.1f} s -> {out}", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # destroying the viewer of a large twin segfaults inside Isaac Gym; everything is on disk


if __name__ == "__main__":
    main()
