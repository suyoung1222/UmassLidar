"""UGV + UAV in one sim: a wheeled robot (default LIMO) and the quadrotor follow the same waypoints
along the same A* ground path. The quad flies above that path at
    z = max(ground + --agl, structure top + --clearance)   (smoothed along the path).

    conda activate gleam
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python scripts/run_team.py --config configs/library.json --waypoints 150 140 190 130 215 100 240 60 \
        --out outputs/team/library_000 --headless

Outputs in --out:
    plan.png                        map with waypoints and the shared A* path
    topview.mp4 chase.mp4           top-down (both tracks) / third-person framing both robots
    ugv_rgb.mp4 ugv_depth.mp4       LIMO D455
    uav_rgb.mp4 uav_depth.mp4       quad D455 (pitched down)
    depth_ugv/ depth_uav/           uint16 mm PNG per frame
    combined.mp4                    all views in one frame
    trajectory_ugv.csv trajectory_uav.csv, result.json, route.npy (ground), route_uav.npy (3D)
"""
from isaacgym import gymapi  # noqa: I001

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umass_lidar import nav, sim as S, uav as U  # noqa: E402
from umass_lidar.viz import depth_vis, label, write_terrain_map  # noqa: E402
from run_nav import plan  # noqa: E402

FPS = 30


def flight_path(path, ends, h, dsm, res, agl, clearance, start_xy, step=0.5, window=4.0):
    """Ground path (N,2) -> 3D flight path over it. Returns (P (M,3), indices of the waypoint
    vertices in P). P[0] = take-off point on the ground, P[1] = straight above it."""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    s = np.arange(0, seg[-1], step)
    s = np.unique(np.r_[s, seg[ends], seg[-1]])                 # keep the waypoints as vertices
    xy = np.column_stack([np.interp(s, seg, path[:, k]) for k in range(2)])
    r = np.clip((xy[:, 1] / res).astype(int), 0, h.shape[0] - 1)
    c = np.clip((xy[:, 0] / res).astype(int), 0, h.shape[1] - 1)
    z = np.maximum(h[r, c] + agl, dsm[r, c] + clearance)
    n = max(1, int(round(window / step)))
    # max over +-window, then mean over the same window: smooth and never below the requirement
    z = ndimage.uniform_filter1d(ndimage.maximum_filter1d(z, 2 * n + 1, mode="nearest"), 2 * n + 1, mode="nearest")
    z0 = float(h[int(start_xy[1] / res), int(start_xy[0] / res)]) + 0.12
    P = np.vstack([[start_xy[0], start_xy[1], z0], [start_xy[0], start_xy[1], z[0]], np.column_stack([xy, z])])
    return P, [int(np.argmin(np.abs(s - seg[e]))) + 2 for e in ends]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="session config (default configs/blue_wall.json)")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--robot", default="limo")
    ap.add_argument("--waypoints", type=float, nargs="+", metavar="X Y", required=True, help="x0 y0 x1 y1 ... (start first)")
    ap.add_argument("--utm", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="UGV m/s (default: robot max, capped 1.0)")
    ap.add_argument("--uav-speed", type=float, default=None, help="UAV cruise m/s (default: same as the UGV)")
    ap.add_argument("--agl", type=float, default=5.0, help="UAV height above ground [m]")
    ap.add_argument("--clearance", type=float, default=2.0, help="UAV min height above structures [m]")
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--goal-tol", type=float, default=0.4)
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0, help="UGV D455 pitch, positive = down")
    ap.add_argument("--uav-cam-pitch-deg", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--no-depth-png", action="store_true")
    ap.add_argument("--collision", choices=("mesh", "voxel", "none"), default="mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    twin, cfg = S.resolve_twin(a.config, a.variant)
    gprefix = twin.parent / "ground"
    meta, h = S.load_terrain_data(gprefix)
    sem = np.load(str(gprefix) + "_semantic.npy")
    res = meta["mesh_res"]; origin = np.array(meta["origin"])
    rcfg = S.robot_config(a.robot)
    out = Path(a.out)
    for d in ("depth_ugv", "depth_uav"):
        (out / d).mkdir(parents=True, exist_ok=True)

    assert len(a.waypoints) >= 4 and len(a.waypoints) % 2 == 0, "--waypoints needs >= 2 x y pairs"
    pts = [tuple(a.waypoints[i:i + 2]) for i in range(0, len(a.waypoints), 2)]
    if a.utm:
        pts = [(p[0] - origin[0], p[1] - origin[1]) for p in pts]
    try:
        path, wps, ends = plan(h, res, sem, pts, rcfg, a.max_slope, np.random.default_rng(0))   # A* per leg
    except ValueError as e:
        sys.exit(f"[plan] {e}")
    yaw0 = float(np.arctan2(*(path[min(5, len(path) - 1)] - path[0])[::-1]))
    uav_start = path[0] - 1.5 * np.array([np.cos(yaw0), np.sin(yaw0)])        # on the ground behind the UGV
    dsm = U.surface_model(h, U.structure_top(twin, h.shape, res), res)
    P, uends = flight_path(path, ends, h, dsm, res, a.agl, a.clearance, uav_start)
    route_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    names = {("start" if i == 0 else "goal" if i == len(wps) - 1 else f"wp{i}"): p for i, p in enumerate(wps)}
    write_terrain_map(gprefix, points=names, path=path, out=out / "plan.png", image=twin.parent / "map_base.png")
    print(f"[plan] {len(wps)} waypoints, shared A* path {route_len:.1f} m; UAV z {P[2:, 2].min():.1f}..{P[2:, 2].max():.1f} "
          f"(ground along path {min(h[int(p[1] / res), int(p[0] / res)] for p in path):.1f}.."
          f"{max(h[int(p[1] / res), int(p[0] / res)] for p in path):.1f})")
    if a.plan_only:
        return

    # ---- sim
    gym, sim, sp = S.make_sim()
    env, _ = S.add_twin(gym, sim, twin, meta, a.collision)
    robot = S.WheeledRobot(gym, sim, env, a.robot, path[0], yaw0, h, res)
    quad = U.Quadrotor(gym, sim, env, P[0], yaw0, sp.dt)
    ctl = U.QuadController()
    v_max = a.speed or min(rcfg["max_speed"], 1.0)
    u_speed = a.uav_speed or v_max
    look = float(np.clip(0.8 * v_max, 0.4, 0.8))
    print(f"[team] {a.robot} {robot.mass:.1f} kg at {v_max:.1f} m/s, quad {quad.mass:.2f} kg at {u_speed:.1f} m/s, "
          f"collision {a.collision}")

    lo, hi = path.min(0), path.max(0)
    span = np.maximum(hi - lo + 16.0, 30.0)
    path_z = np.array([h[int(p[1] / res), int(p[0] / res)] for p in path])
    gz = float(np.median(path_z))
    top = S.TopDownCamera(gym, env, (lo + hi) / 2, span, gz)
    cc = gymapi.CameraProperties()
    cc.width, cc.height, cc.horizontal_fov = 960, 540, 70
    chase = gym.create_camera_sensor(env, cc)
    ego_g = S.RGBDCamera(gym, env, robot, a.cam_pitch_deg)
    ego_a = S.RGBDCamera(gym, env, quad, a.uav_cam_pitch_deg)
    viewer = None if a.headless else gym.create_viewer(sim, gymapi.CameraProperties())

    vids = ("topview", "chase", "ugv_rgb", "ugv_depth", "uav_rgb", "uav_depth", "combined")
    writers = {k: imageio.get_writer(out / f"{k}.mp4", fps=FPS, quality=7, macro_block_size=1) for k in vids}
    fg = open(out / "trajectory_ugv.csv", "w", newline=""); wg = csv.writer(fg)
    fa = open(out / "trajectory_uav.csv", "w", newline=""); wa = csv.writer(fa)
    head = ["frame", "t", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "utm_e", "utm_n", "h_ell",
            "cam_x", "cam_y", "cam_z", "cam_qx", "cam_qy", "cam_qz", "cam_qw"]
    wg.writerow(head + ["cmd_v", "cmd_w", "meas_v", "meas_w", "target_wp"])
    wa.writerow(head + ["vx", "vy", "vz", "f1", "f2", "f3", "f4", "agl", "target_wp"])

    path_px = top.project(np.column_stack([path, path_z]))
    every = int(round(1 / (sp.dt * FPS)))
    settle = int(1.0 / sp.dt)
    body_env = gym.find_actor_rigid_body_index(env, quad.actor, "base_link", gymapi.DOMAIN_ENV)
    # UGV state
    g_i, g_reached, g_done, g_stuck, g_flip, g_dist, g_traj = 0, [None] * len(ends), False, False, False, 0.0, []
    g_prog = (0.0, None)
    v_cmd = w_cmd = 0.0
    # UAV state: targets = take-off vertex, then the waypoint vertices
    targets = [1] + uends
    u_i, hold, u_reached, u_done, u_crash, u_dist, u_traj = 0, 0.0, [None] * len(targets), False, False, 0.0, []
    contacts, first_contact, max_err, yaw_cmd = 0, None, 0.0, yaw0
    cam_yaw, frame, k = yaw0, 0, 0
    valid_g, valid_a, sep = [], [], []
    while k < int(a.max_seconds / sp.dt) + settle and not (g_done and u_done) and not (g_stuck or g_flip or u_crash):
        pos, quat, yaw, v, w = robot.state()
        s = quad.state()
        upos = s["pos"]
        t = (k - settle) * sp.dt
        f_cmd = np.zeros(4)
        r_, c_ = min(int(upos[1] / res), h.shape[0] - 1), min(int(upos[0] / res), h.shape[1] - 1)
        if k >= settle:
            # ---- UGV: pure pursuit on the current leg
            if not g_done:
                sub = path[(ends[g_i - 1] if g_i else 0):ends[g_i] + 1]
                v_cmd, w_cmd, arrived = nav.pure_pursuit(pos[:2], yaw, sub, lookahead=look, v_max=v_max, goal_tol=a.goal_tol)
                if arrived:
                    g_reached[g_i] = round(t, 2)
                    print(f"[ugv] reached waypoint {g_i + 1}/{len(ends)} at t={t:.1f}s", flush=True)
                    g_i += 1
                    g_done = g_i >= len(ends)
                if 1 - 2 * (quat[0] ** 2 + quat[1] ** 2) < 0.3:
                    g_flip = True
                if g_prog[1] is None or np.linalg.norm(pos[:2] - g_prog[1]) > 0.2:
                    g_prog = (t, pos[:2].copy())
                elif t - g_prog[0] > 8.0:
                    g_stuck = True
            if g_done:
                v_cmd = w_cmd = 0.0
            # ---- UAV: velocity control towards a carrot on the 3D path (hovers at the goal when done)
            ti_ = min(u_i, len(targets) - 1)
            goal = P[targets[ti_]]
            cpt, err = U.carrot(P, 0 if ti_ == 0 else targets[ti_ - 1], targets[ti_], upos, lookahead=max(1.5, u_speed))
            if ti_ > 0 and not u_done:
                max_err = max(max_err, err)
            dg = np.linalg.norm(goal - upos)
            to_c = (goal if u_done else cpt) - upos
            vmax = min(u_speed, np.sqrt(2 * 1.5 * dg) + 0.2) if ti_ > 0 else min(2.0, dg + 0.3)
            v_des = to_c / max(np.linalg.norm(to_c), 1e-6) * min(vmax, np.linalg.norm(to_c) * 1.5)
            if np.linalg.norm(v_des[:2]) > 0.5:
                yaw_cmd = float(np.arctan2(v_des[1], v_des[0]))
            f_cmd = ctl.velocity(s, v_des, yaw_cmd)
            if not u_done:
                hold = hold + sp.dt if dg < max(a.goal_tol, 0.5) else 0.0
                if hold >= 0.3:
                    u_reached[u_i] = round(t, 2)
                    print(f"[uav] reached {'take-off' if u_i == 0 else f'waypoint {u_i}/{len(uends)}'} at t={t:.1f}s "
                          f"(z={upos[2]:.1f})", flush=True)
                    u_i += 1; hold = 0.0
                    u_done = u_i >= len(targets)
            if s["R"][2, 2] < 0.2 and t > 1.0:
                u_crash = True
            fc = gym.get_env_rigid_contact_forces(env)[body_env]
            if t > 1.0 and np.linalg.norm([fc["x"], fc["y"], fc["z"]]) > 0.5 and upos[2] - h[r_, c_] > 0.5:
                contacts += 1
                if first_contact is None:
                    first_contact = (round(t, 2), [round(float(x), 2) for x in upos])
                    print(f"[uav] CONTACT at t={t:.1f}s pos={np.round(upos, 1)}", flush=True)
        robot.command(v_cmd, w_cmd)
        quad.apply_motors(f_cmd, s)
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if k >= settle and (k - settle) % every == 0:
            if g_traj:
                g_dist += float(np.linalg.norm(pos[:2] - g_traj[-1][:2])); u_dist += float(np.linalg.norm(upos - u_traj[-1]))
            g_traj.append(pos.copy()); u_traj.append(upos.copy())
            sep.append(float(np.linalg.norm(upos[:2] - pos[:2])))
            # third-person camera: behind the UGV, far enough back to keep the quad in frame
            cam_yaw += 0.06 * ((yaw - cam_yaw + np.pi) % (2 * np.pi) - np.pi)
            c, sn = np.cos(cam_yaw), np.sin(cam_yaw)
            mid = (pos + upos) / 2
            back = 6.0 + 1.2 * np.linalg.norm(upos - pos)
            gym.set_camera_location(chase, env, gymapi.Vec3(pos[0] - back * c, pos[1] - back * sn, pos[2] + 0.45 * back),
                                    gymapi.Vec3(mid[0], mid[1], mid[2]))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            ti = top.image(sim)
            ci = gym.get_camera_image(sim, env, chase, gymapi.IMAGE_COLOR).reshape(cc.height, cc.width, 4)[..., :3].copy()
            rg, dg16, vg = ego_g.read(sim)
            ra, da16, va = ego_a.read(sim)
            valid_g.append(vg); valid_a.append(va)
            if not a.no_depth_png:
                cv2.imwrite(str(out / "depth_ugv" / f"{frame:06d}.png"), dg16)
                cv2.imwrite(str(out / "depth_uav" / f"{frame:06d}.png"), da16)
            for wr, p_, q_, y_, ego, extra in (
                    (wg, pos, quat, yaw, ego_g, [f"{v_cmd:.3f}", f"{w_cmd:.3f}", f"{v:.3f}", f"{w:.3f}", min(g_i + 1, len(ends))]),
                    (wa, upos, s["quat"], s["yaw"], ego_a, [*np.round(s["vel"], 3), *np.round(quad.f, 3),
                                                            f"{upos[2] - h[r_, c_]:.3f}", min(u_i, len(uends))])):
                cp_, cq_ = ego.pose()
                wr.writerow([frame, f"{t:.3f}", *np.round(p_, 4), *np.round(q_, 5), f"{y_:.4f}",
                             f"{p_[0] + origin[0]:.3f}", f"{p_[1] + origin[1]:.3f}", f"{p_[2] + origin[2]:.3f}",
                             *np.round(cp_, 4), *np.round(cq_, 5), *extra])

            # ---- overlays (top view: UGV red track / yellow marker, UAV cyan)
            cv2.polylines(ti, [path_px], False, (255, 200, 0), 2, cv2.LINE_AA)
            flat = lambda X: np.column_stack([np.asarray(X)[:, :2], np.full(len(X), gz)])  # noqa: E731
            if len(g_traj) > 1:
                cv2.polylines(ti, [top.project(flat(u_traj))], False, (0, 220, 255), 5, cv2.LINE_AA)
                cv2.polylines(ti, [top.project(flat(g_traj))], False, (255, 40, 40), 2, cv2.LINE_AA)
            for i, p in enumerate(wps):
                nm = "START" if i == 0 else ("GOAL" if i == len(wps) - 1 else f"WP{i}")
                col = (40, 220, 40) if i == 0 else ((255, 60, 255) if i == len(wps) - 1 else (255, 200, 0))
                u = tuple(int(x) for x in top.project([p[0], p[1], gz])[0])
                cv2.circle(ti, u, 8, col, -1, cv2.LINE_AA); cv2.circle(ti, u, 8, (0, 0, 0), 2, cv2.LINE_AA)
                label(ti, nm, (u[0] + 11, u[1] + 5), 0.55)
            for p_, col, nm in ((upos, (0, 220, 255), "UAV"), (pos, (255, 255, 0), "UGV")):
                u = tuple(int(x) for x in top.project([p_[0], p_[1], gz])[0])
                cv2.circle(ti, u, 7, col, -1, cv2.LINE_AA); cv2.circle(ti, u, 7, (0, 0, 0), 2, cv2.LINE_AA)
                label(ti, nm, (u[0] + 10, u[1] - 8), 0.5)
            cv2.rectangle(ti, (0, 0), (ti.shape[1], 34), (0, 0, 0), -1)
            label(ti, f"t={t:5.1f}s  UGV {g_dist:5.1f} m wp {min(g_i + 1, len(ends))}/{len(ends)}  |  "
                      f"UAV {u_dist:5.1f} m AGL {upos[2] - h[r_, c_]:4.1f} m wp {min(max(u_i, 1), len(uends))}/{len(uends)}",
                  scale=0.6)
            label(ci, f"third-person  |  collision {a.collision}")
            if contacts:
                label(ci, f"UAV CONTACT x{contacts}", (10, 52))

            def tile(rgb, d16, text):
                im = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
                dv = cv2.resize(depth_vis(d16, max_m=20.0), (480, 270), interpolation=cv2.INTER_NEAREST)
                label(im, text, scale=0.5); label(dv, "depth 0-20 m", scale=0.5)
                return np.vstack([im, dv])
            writers["topview"].append_data(ti)
            writers["chase"].append_data(ci)
            writers["ugv_rgb"].append_data(rg); writers["ugv_depth"].append_data(depth_vis(dg16))
            writers["uav_rgb"].append_data(ra); writers["uav_depth"].append_data(depth_vis(da16, max_m=20.0))
            writers["combined"].append_data(np.hstack([ti, np.vstack([ci, np.hstack([
                tile(rg, dg16, f"UGV ({a.robot}) D455"), tile(ra, da16, f"UAV D455, {a.uav_cam_pitch_deg:.0f} deg down")])])]))
            frame += 1
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                break
            gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
        k += 1

    for w_ in writers.values():
        w_.close()
    fg.close(); fa.close()
    g_err = float(np.linalg.norm(robot.state()[0][:2] - path[-1]))
    u_err = float(np.linalg.norm(quad.state()["pos"] - P[-1]))
    result = dict(success=bool(g_done and u_done and not contacts), twin=str(twin), structure_collision=a.collision,
                  waypoints=[list(map(float, p)) for p in wps], planned_length_m=route_len, time_s=frame / FPS,
                  frames=frame, fps=FPS, origin_utm=meta["origin"], crs=cfg.get("crs"),
                  horizontal_separation_m=dict(mean=float(np.mean(sep)), max=float(np.max(sep))) if sep else None,
                  ugv=dict(robot=a.robot, done=bool(g_done), stuck=bool(g_stuck), flipped=bool(g_flip), speed=v_max,
                           waypoint_reached_t=g_reached, driven_m=g_dist, goal_error_m=g_err,
                           valid_depth_fraction=float(np.mean(valid_g)) if valid_g else 0.0, K=ego_g.K,
                           cam_pitch_down_deg=a.cam_pitch_deg),
                  uav=dict(done=bool(u_done), crashed=bool(u_crash), speed=u_speed, agl=a.agl, clearance=a.clearance,
                           takeoff_t=u_reached[0], waypoint_reached_t=u_reached[1:], flown_m=u_dist, goal_error_m=u_err,
                           max_cross_track_m=max_err, contact_steps=contacts, first_contact=first_contact,
                           valid_depth_fraction=float(np.mean(valid_a)) if valid_a else 0.0, K=ego_a.K,
                           cam_pitch_down_deg=a.uav_cam_pitch_deg),
                  d455=dict(width=S.D455["width"], height=S.D455["height"], hfov_deg=S.D455["hfov"],
                            depth="uint16 mm, 0 = invalid", depth_range_m=[S.D455["depth_min"], S.D455["depth_max"]]))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    np.save(out / "route.npy", path); np.save(out / "route_uav.npy", P)
    print(f"[done] UGV {'REACHED' if g_done else 'FLIPPED' if g_flip else 'STUCK' if g_stuck else 'TIMEOUT'} "
          f"({g_dist:.1f} m, goal error {g_err:.2f} m) | UAV {'REACHED' if u_done else 'CRASHED' if u_crash else 'TIMEOUT'} "
          f"({u_dist:.1f} m, goal error {u_err:.2f} m, contacts {contacts}) in {frame / FPS:.1f} s -> {out}")
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
