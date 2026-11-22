"""Main experiment: waypoint navigation of one wheeled robot (default LIMO) in the digital twin,
recorded from three viewpoints: top-down, third-person chase, and a RealSense D455-like RGB-D
camera on the robot.

    conda activate gleam
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python scripts/run_nav.py --waypoints 96 185  110 150  102 62 --out outputs/nav/run_000 --headless

Waypoints are local metres (read them off assets/twin/<name>/map.png; --utm for UTM E N).
The first waypoint is the start. Without --waypoints a random reachable route is sampled.

Outputs in --out:
    plan.png                    map with waypoints and planned path
    topview.mp4 chase.mp4       top-down / third-person videos
    d455_rgb.mp4 d455_depth.mp4 ego RGB and colourised depth; depth/*.png = uint16 mm
    combined.mp4                all views in one frame
    trajectory.csv              per frame: robot + camera pose, UTM, commands
    result.json                 success per waypoint, distances, camera intrinsics
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umass_lidar import nav, sim as S  # noqa: E402
from umass_lidar.viz import write_terrain_map  # noqa: E402

FPS = 30


def plan(h, res, sem, pts, rcfg, max_slope, rng, min_len=40.0):
    trav = dict(max_slope_deg=max_slope, robot_radius=rcfg["track"] * 0.5 + 0.35, blocked=sem != 0)
    if pts is None:
        path = nav.sample_route(h, res, rng, min_len=min_len, **trav)
        return path, [tuple(path[0]), tuple(path[-1])], [len(path) - 1]
    snap = nav.snap_points(h, res, pts, clearance=1.0, **trav)
    for p, (q, d) in zip(pts, snap):
        if d > 0:
            print(f"[plan] waypoint {np.round(p, 1)} snapped {d:.1f} m to drivable {np.round(q, 1)}")
    wps = [q for q, _ in snap]
    legs, ends = [], []
    for a, b in zip(wps[:-1], wps[1:]):
        leg = nav.plan_path(h, res, a, b, **trav)
        if leg is None:
            raise ValueError(f"no drivable path {np.round(a, 1)} -> {np.round(b, 1)}")
        legs.append(leg if not legs else leg[1:])
        ends.append(sum(len(l) for l in legs) - 1)
    return np.concatenate(legs), wps, ends


def depth_vis(d16, max_m=10.0):
    v = np.clip(d16.astype(np.float32) / (max_m * 1000.0), 0, 1)
    img = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    img[d16 == 0] = 0
    return img


def label(img, text, org=(10, 24), scale=0.65):
    cv2.putText(img, text, org, 0, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, 0, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="session config (default configs/blue_wall.json)")
    ap.add_argument("--variant", default="full", help="twin variant built by build_twin.py")
    ap.add_argument("--robot", default="limo")
    ap.add_argument("--waypoints", type=float, nargs="+", metavar="X Y", help="x0 y0 x1 y1 ... (start first)")
    ap.add_argument("--utm", action="store_true", help="waypoints are UTM E N")
    ap.add_argument("--speed", type=float, default=None, help="m/s (default: robot max, capped 1.0)")
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--goal-tol", type=float, default=0.4)
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0, help="D455 pitch, positive = down")
    ap.add_argument("--no-depth-png", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    twin, cfg = S.resolve_twin(a.config, a.variant)
    gprefix = twin.parent / "ground"
    meta, h = S.load_terrain_data(gprefix)
    sem = np.load(str(gprefix) + "_semantic.npy")
    res = meta["mesh_res"]; origin = np.array(meta["origin"])
    rcfg = S.robot_config(a.robot)
    out = Path(a.out); (out / "depth").mkdir(parents=True, exist_ok=True)

    pts = None
    if a.waypoints:
        assert len(a.waypoints) >= 4 and len(a.waypoints) % 2 == 0, "--waypoints needs >= 2 x y pairs"
        pts = [tuple(a.waypoints[i:i + 2]) for i in range(0, len(a.waypoints), 2)]
        if a.utm:
            pts = [(p[0] - origin[0], p[1] - origin[1]) for p in pts]
    try:
        path, wps, ends = plan(h, res, sem, pts, rcfg, a.max_slope, np.random.default_rng(a.seed))
    except ValueError as e:
        write_terrain_map(gprefix, points={f"wp{i}": p for i, p in enumerate(pts or [])}, out=out / "plan.png",
                          image=twin.parent / "map_base.png")
        sys.exit(f"[plan] {e}; see {out / 'plan.png'} (green = drivable)")
    route_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    names = {("start" if i == 0 else "goal" if i == len(wps) - 1 else f"wp{i}"): p for i, p in enumerate(wps)}
    write_terrain_map(gprefix, points=names, path=path, out=out / "plan.png", image=twin.parent / "map_base.png")
    print(f"[plan] {len(wps)} waypoints, {route_len:.1f} m, map: {out / 'plan.png'}")
    if a.plan_only:
        return

    # ---- sim
    gym, sim, sp = S.make_sim()
    env, _ = S.add_twin(gym, sim, twin, meta)
    yaw0 = float(np.arctan2(*(path[min(5, len(path) - 1)] - path[0])[::-1]))
    robot = S.WheeledRobot(gym, sim, env, a.robot, path[0], yaw0, h, res)
    print(f"[robot] {a.robot} mass={robot.mass:.1f} kg")

    lo, hi = path.min(0), path.max(0)
    span = np.maximum(hi - lo + 16.0, 30.0)
    path_z = np.array([h[int(p[1] / res), int(p[0] / res)] for p in path])
    top = S.TopDownCamera(gym, env, (lo + hi) / 2, span, float(np.median(path_z)))
    cc = gymapi.CameraProperties()
    cc.width, cc.height, cc.horizontal_fov = 960, 540, 70
    chase = gym.create_camera_sensor(env, cc)
    ego = S.RGBDCamera(gym, env, robot, a.cam_pitch_deg)
    viewer = None if a.headless else gym.create_viewer(sim, gymapi.CameraProperties())

    writers = {k: imageio.get_writer(out / f"{k}.mp4", fps=FPS, quality=q, macro_block_size=1)
               for k, q in (("topview", 6), ("chase", 7), ("d455_rgb", 8), ("d455_depth", 7), ("combined", 7))}
    fcsv = open(out / "trajectory.csv", "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["frame", "t", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "utm_e", "utm_n", "h_ell",
                 "cam_x", "cam_y", "cam_z", "cam_qx", "cam_qy", "cam_qz", "cam_qw",
                 "cmd_v", "cmd_w", "meas_v", "meas_w", "target_wp"])

    v_max = a.speed or min(rcfg["max_speed"], 1.0)
    look = float(np.clip(0.8 * v_max, 0.4, 0.8))
    # scale of the chase camera follows robot size
    size = max(0.35, rcfg["track"] * 2.2)
    back, up = 5.5 * size, 3.0 * size
    path3 = np.column_stack([path, path_z])
    path_px = top.project(path3)
    every = int(round(1 / (sp.dt * FPS)))
    settle = int(1.0 / sp.dt)
    traj, reached, stuck, k, dist, frame = [], [None] * len(ends), False, 0, 0.0, 0
    wp_i = 0
    cam_yaw = yaw0
    v_cmd = w_cmd = 0.0
    last_progress = (0.0, None)
    valid = []
    done = flipped = False
    while k < int(a.max_seconds / sp.dt) + settle and not done and not stuck:
        pos, quat, yaw, v, w = robot.state()
        t = (k - settle) * sp.dt
        if k >= settle:
            # follow only the current leg: legs may overlap (out-and-back) and the nearest-point
            # search must not jump back onto a leg that is already done
            sub = path[(ends[wp_i - 1] if wp_i else 0):ends[wp_i] + 1]
            v_cmd, w_cmd, arrived = nav.pure_pursuit(pos[:2], yaw, sub, lookahead=look, v_max=v_max,
                                                     goal_tol=a.goal_tol)
            if arrived:
                reached[wp_i] = round(t, 2)
                print(f"[nav] reached waypoint {wp_i + 1}/{len(ends)} at t={t:.1f}s", flush=True)
                wp_i += 1
                done = wp_i >= len(ends)
                if done:
                    v_cmd = w_cmd = 0.0
            up_z = 1 - 2 * (quat[0] ** 2 + quat[1] ** 2)          # body z axis . world z
            if up_z < 0.3:
                flipped = True
                break
            if last_progress[1] is None or np.linalg.norm(pos[:2] - last_progress[1]) > 0.2:
                last_progress = (t, pos[:2].copy())
            elif t - last_progress[0] > 8.0:
                stuck = True
        robot.command(v_cmd, w_cmd)
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if k >= settle and (k - settle) % every == 0:
            if traj:
                dist += float(np.linalg.norm(pos[:2] - traj[-1][:2]))
            traj.append(pos.copy())
            cam_yaw += 0.08 * ((yaw - cam_yaw + np.pi) % (2 * np.pi) - np.pi)
            c, s = np.cos(cam_yaw), np.sin(cam_yaw)
            gym.set_camera_location(chase, env,
                                    gymapi.Vec3(pos[0] - back * c, pos[1] - back * s, pos[2] + up),
                                    gymapi.Vec3(pos[0] + back * 0.4 * c, pos[1] + back * 0.4 * s, pos[2] + 0.1))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            ti = top.image(sim)
            ci = gym.get_camera_image(sim, env, chase, gymapi.IMAGE_COLOR).reshape(cc.height, cc.width, 4)[..., :3].copy()
            rgb, d16, vf = ego.read(sim)
            valid.append(vf)
            if not a.no_depth_png:
                cv2.imwrite(str(out / "depth" / f"{frame:06d}.png"), d16)
            cp_, cq_ = ego.pose()
            wr.writerow([frame, f"{t:.3f}", *np.round(pos, 4), *np.round(quat, 5), f"{yaw:.4f}",
                         f"{pos[0] + origin[0]:.3f}", f"{pos[1] + origin[1]:.3f}", f"{pos[2] + origin[2]:.3f}",
                         *np.round(cp_, 4), *np.round(cq_, 5),
                         f"{v_cmd:.3f}", f"{w_cmd:.3f}", f"{v:.3f}", f"{w:.3f}", min(wp_i + 1, len(ends))])

            # ---- overlays
            cv2.polylines(ti, [path_px], False, (255, 200, 0), 2, cv2.LINE_AA)
            if len(traj) > 1:
                cv2.polylines(ti, [top.project(np.array(traj))], False, (255, 40, 40), 3, cv2.LINE_AA)
            for i, p in enumerate(wps):
                nm = "START" if i == 0 else ("GOAL" if i == len(wps) - 1 else f"WP{i}")
                col = (40, 220, 40) if i == 0 else ((255, 60, 255) if i == len(wps) - 1 else (255, 200, 0))
                if 0 < i and reached[i - 1] is not None:
                    col = (150, 150, 150)
                u = tuple(top.project([p[0], p[1], float(np.median(path_z))])[0])
                cv2.circle(ti, u, 8, col, -1, cv2.LINE_AA); cv2.circle(ti, u, 8, (0, 0, 0), 2, cv2.LINE_AA)
                label(ti, nm, (u[0] + 11, u[1] + 5), 0.55)
            u = top.project(pos)[0]
            tip = top.project(pos + np.array([np.cos(yaw), np.sin(yaw), 0]) * max(1.2, span.max() / 35))[0]
            cv2.arrowedLine(ti, tuple(u), tuple(tip), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.4)
            cv2.circle(ti, tuple(u), 6, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(ti, tuple(u), 6, (0, 0, 0), 2, cv2.LINE_AA)
            remain = float(np.linalg.norm(pos[:2] - path[ends[min(wp_i, len(ends) - 1)]]))
            cv2.rectangle(ti, (0, 0), (ti.shape[1], 34), (0, 0, 0), -1)
            label(ti, f"{a.robot}  t={t:5.1f}s  v={v:4.2f} m/s  driven={dist:5.1f} m  "
                      f"wp {min(wp_i + 1, len(ends))}/{len(ends)}  to wp={remain:4.1f} m")
            cv2.putText(ti, "N", (top.w - 34, 70), 0, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(ti, (top.w - 24, 120), (top.w - 24, 80), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
            dv = depth_vis(d16)
            label(ci, "third-person")
            rgb_s = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
            pip = cv2.resize(dv, (320, 180), interpolation=cv2.INTER_NEAREST)
            rgb_s[540 - 180 - 8:540 - 8, 960 - 320 - 8:960 - 8] = pip
            cv2.rectangle(rgb_s, (960 - 328, 540 - 188), (960 - 8, 540 - 8), (255, 255, 255), 1)
            label(rgb_s, "D455 RGB (inset: depth 0-10 m)")
            writers["topview"].append_data(ti)
            writers["chase"].append_data(ci)
            writers["d455_rgb"].append_data(rgb)
            writers["d455_depth"].append_data(dv)
            writers["combined"].append_data(np.hstack([ti, np.vstack([ci, rgb_s])]))
            frame += 1
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                break
            gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
        k += 1

    for w_ in writers.values():
        w_.close()
    fcsv.close()
    pos = robot.state()[0]
    err = float(np.linalg.norm(pos[:2] - path[-1]))
    result = dict(success=bool(done), stuck=bool(stuck), flipped=bool(flipped), robot=a.robot, twin=str(twin),
                  waypoints=[list(map(float, p)) for p in wps], waypoint_reached_t=reached,
                  planned_length_m=route_len, driven_m=dist, time_s=frame / FPS, goal_error_m=err,
                  frames=frame, fps=FPS, origin_utm=meta["origin"], crs=cfg.get("crs"),
                  d455=dict(width=S.D455["width"], height=S.D455["height"], hfov_deg=S.D455["hfov"], K=ego.K,
                            depth="uint16 mm, 0 = invalid", depth_range_m=[S.D455["depth_min"], S.D455["depth_max"]],
                            mount_xyz_in_base=rcfg["camera_xyz"], pitch_down_deg=a.cam_pitch_deg,
                            valid_depth_fraction=float(np.mean(valid)) if valid else 0.0,
                            pose_columns="cam_* in trajectory.csv: link frame x forward, y left, z up"))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    np.save(out / "route.npy", path)
    status = "REACHED ALL WAYPOINTS" if done else ("FLIPPED" if flipped else "STUCK" if stuck else "TIMEOUT")
    print(f"[done] {status}: planned {route_len:.1f} m, driven {dist:.1f} m in {frame / FPS:.1f} s, "
          f"goal error {err:.2f} m -> {out}")
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
