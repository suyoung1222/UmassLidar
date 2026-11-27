"""UAV experiment: quadrotor waypoint flight over the digital twin with a forward-down D455-like
RGB-D camera, recorded like run_nav.py (top-down, chase, ego RGB / depth).

    conda activate gleam
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python scripts/run_uav.py --waypoints 95 150  103 166  125 172  138 168 --out outputs/uav/run_000 --headless

Waypoints are local metres (x y, from assets/twin/<name>/map.png; --utm for UTM E N). The drone
takes off at the first one and flies each leg at a constant altitude: --agl above the ground and
at least --clearance above the twin's surface model (ground mesh + 3D structure tiles, dilated by
1.5 m). Structure tiles are visual-only in the sim, so the surface model is what keeps the
drone above trees / buildings; flying into one is reported as `structure_hit`.

Outputs in --out:
    plan.png                    map with waypoints and flight path (altitude per leg in the label)
    topview.mp4 chase.mp4       top-down / third-person videos
    d455_rgb.mp4 d455_depth.mp4 ego RGB and colourised depth; depth/*.png = uint16 mm
    combined.mp4                all views in one frame
    trajectory.csv              per frame: body + camera pose, UTM, velocity, rotor thrusts, AGL
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
from umass_lidar import sim as S, uav as U  # noqa: E402
from umass_lidar.viz import depth_vis, label, write_terrain_map  # noqa: E402

FPS = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="session config (default configs/blue_wall.json)")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--waypoints", type=float, nargs="+", metavar="X Y", required=True)
    ap.add_argument("--utm", action="store_true")
    ap.add_argument("--agl", type=float, default=6.0, help="flight height above ground [m]")
    ap.add_argument("--clearance", type=float, default=2.0, help="min height above structures [m]")
    ap.add_argument("--speed", type=float, default=3.0, help="cruise speed [m/s]")
    ap.add_argument("--goal-tol", type=float, default=0.5)
    ap.add_argument("--hover", type=float, default=1.0, help="hover time at each waypoint [s]")
    ap.add_argument("--cam-pitch-deg", type=float, default=30.0, help="D455 pitch, positive = down")
    ap.add_argument("--depth-max", type=float, default=S.D455["depth_max"])
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--no-depth-png", action="store_true")
    ap.add_argument("--collision", choices=("mesh", "voxel", "none"), default="mesh",
                    help="3D structure collision from build_collision.py: mesh = simplified visual mesh, "
                         "voxel = closed voxel shell from the points (--mode voxel), none = visual-only")
    ap.add_argument("--no-structure-collision", action="store_true", help="same as --collision none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--uav", default="quad", choices=sorted(U.MODELS), help="UAV model")
    a = ap.parse_args()
    QP = U.MODELS[a.uav]
    if a.no_structure_collision:
        a.collision = "none"

    twin, cfg = S.resolve_twin(a.config, a.variant)
    gprefix = twin.parent / "ground"
    meta, h = S.load_terrain_data(gprefix)
    res = meta["mesh_res"]; origin = np.array(meta["origin"])
    out = Path(a.out); (out / "depth").mkdir(parents=True, exist_ok=True)

    assert len(a.waypoints) >= 4 and len(a.waypoints) % 2 == 0, "--waypoints needs >= 2 x y pairs"
    pts = [tuple(a.waypoints[i:i + 2]) for i in range(0, len(a.waypoints), 2)]
    if a.utm:
        pts = [(p[0] - origin[0], p[1] - origin[1]) for p in pts]
    size = np.array(meta["size_xy"])
    for p in pts:
        if not (0 <= p[0] < size[0] and 0 <= p[1] < size[1]):
            sys.exit(f"[plan] waypoint {p} outside the twin (0..{size[0]:.0f}, 0..{size[1]:.0f})")
    top_s = U.structure_top(twin, h.shape, res)
    dsm = U.surface_model(h, top_s, res)
    path, ends, alts = U.plan_flight(h, dsm, res, pts, a.agl, a.clearance)
    path[0, 2] += 0.12                              # take-off point = body centre resting on ground
    route_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    names = {("start" if i == 0 else "goal" if i == len(pts) - 1 else f"wp{i}"): p for i, p in enumerate(pts)}
    write_terrain_map(gprefix, points=names, path=path[1:, :2], out=out / "plan.png",
                      image=twin.parent / "map_base.png")
    print(f"[plan] {len(pts)} waypoints, {route_len:.1f} m 3D, leg altitude (local z) "
          f"{[round(z, 1) for z in alts]}, ground at start {path[0, 2]:.1f}")
    if a.plan_only:
        return

    # ---- sim
    gym, sim, sp = S.make_sim()
    env, _ = S.add_twin(gym, sim, twin, meta, a.collision)
    yaw0 = float(np.arctan2(*(path[2, :2] - path[1, :2])[::-1]))
    quad = U.Quadrotor(gym, sim, env, path[0], yaw0, sp.dt, p=QP)
    ctl = U.QuadController(p=QP)
    print(f"[uav] quad mass={quad.mass:.2f} kg, hover thrust {quad.mass * 9.81 / 4:.2f} N/rotor "
          f"(max {QP['f_max']} N)")

    lo, hi = path[:, :2].min(0), path[:, :2].max(0)
    span = np.maximum(hi - lo + 16.0, 30.0)
    gz = float(np.median([U._sample(h, res, p[:2], p[:2])[0] for p in path]))
    top = S.TopDownCamera(gym, env, (lo + hi) / 2, span, gz)
    cc = gymapi.CameraProperties()
    cc.width, cc.height, cc.horizontal_fov = 960, 540, 70
    chase = gym.create_camera_sensor(env, cc)
    spec = dict(S.D455, depth_max=a.depth_max)
    ego = S.RGBDCamera(gym, env, quad, a.cam_pitch_deg, spec)
    viewer = None if a.headless else gym.create_viewer(sim, gymapi.CameraProperties())

    writers = {k: imageio.get_writer(out / f"{k}.mp4", fps=FPS, quality=q, macro_block_size=1)
               for k, q in (("topview", 6), ("chase", 7), ("d455_rgb", 8), ("d455_depth", 7), ("combined", 7))}
    fcsv = open(out / "trajectory.csv", "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["frame", "t", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "utm_e", "utm_n", "h_ell",
                 "cam_x", "cam_y", "cam_z", "cam_qx", "cam_qy", "cam_qz", "cam_qw",
                 "vx", "vy", "vz", "f1", "f2", "f3", "f4", "agl", "target_wp"])

    path_px = top.project(path)
    every = int(round(1 / (sp.dt * FPS)))
    settle = int(0.5 / sp.dt)
    # targets: index 0 = take-off vertex (path[1]), then ends[i]
    targets = [1] + ends
    tgt_i, hold, reached = 0, 0.0, [None] * len(targets)
    traj, dist, frame, k = [], 0.0, 0, 0
    cam_yaw = yaw_cmd = yaw0
    valid, done, crashed, hit, max_err = [], False, False, False, 0.0
    body_env = gym.find_actor_rigid_body_index(env, quad.actor, "base_link", gymapi.DOMAIN_ENV)
    contacts, first_contact, blocked, last_progress = 0, None, False, (0.0, None)
    while k < int(a.max_seconds / sp.dt) + settle and not done and not crashed and not blocked:
        s = quad.state()
        pos = s["pos"]
        t = (k - settle) * sp.dt
        if k < settle:
            f_cmd = np.zeros(4)
        else:
            goal = path[targets[tgt_i]]
            seg_lo = 0 if tgt_i == 0 else targets[tgt_i - 1]
            c, err = U.carrot(path, seg_lo, targets[tgt_i], pos, lookahead=max(1.5, a.speed))
            if tgt_i > 0:
                max_err = max(max_err, err)
            dg = np.linalg.norm(goal - pos)
            to_c = c - pos
            # slow down for the goal: v <= sqrt(2 a d) with a = 1.5 m/s^2
            vmax = min(a.speed, np.sqrt(2 * 1.5 * dg) + 0.2) if tgt_i > 0 else min(2.0, dg + 0.3)
            v_des = to_c / max(np.linalg.norm(to_c), 1e-6) * min(vmax, np.linalg.norm(to_c) * 1.5)
            if np.linalg.norm(v_des[:2]) > 0.8:
                yaw_cmd = float(np.arctan2(v_des[1], v_des[0]))
            f_cmd = ctl.velocity(s, v_des, yaw_cmd)
            if dg < a.goal_tol:
                hold += sp.dt
                if hold >= (a.hover if tgt_i > 0 else 0.3):
                    reached[tgt_i] = round(t, 2)
                    nm = "take-off" if tgt_i == 0 else f"waypoint {tgt_i}/{len(ends)}"
                    print(f"[uav] reached {nm} at t={t:.1f}s (z={pos[2]:.1f})", flush=True)
                    tgt_i += 1; hold = 0.0
                    done = tgt_i >= len(targets)
            else:
                hold = 0.0
            if s["R"][2, 2] < 0.2 and t > 1.0:
                crashed = True
            r_, c_ = min(int(pos[1] / res), h.shape[0] - 1), min(int(pos[0] / res), h.shape[1] - 1)
            if np.isfinite(top_s[r_, c_]) and pos[2] < top_s[r_, c_] and t > 2.0:
                hit = True
            # contact with anything after take-off (ground excluded by the 1 s grace + 0.5 m AGL)
            fc = gym.get_env_rigid_contact_forces(env)[body_env]
            if t > 1.0 and np.linalg.norm([fc["x"], fc["y"], fc["z"]]) > 0.5 and \
                    pos[2] - h[r_, c_] > 0.5:
                contacts += 1
                if first_contact is None:
                    first_contact = (round(t, 2), [round(float(v), 2) for v in pos])
                    print(f"[uav] CONTACT at t={t:.1f}s pos={np.round(pos, 1)}", flush=True)
            if last_progress[1] is None or np.linalg.norm(pos - last_progress[1]) > 0.3:
                last_progress = (t, pos.copy())
            elif t - last_progress[0] > 5.0:
                blocked = True
        quad.apply_motors(f_cmd, s)
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if k >= settle and (k - settle) % every == 0:
            if traj:
                dist += float(np.linalg.norm(pos - traj[-1]))
            traj.append(pos.copy())
            yaw = s["yaw"]
            cam_yaw += 0.08 * ((yaw - cam_yaw + np.pi) % (2 * np.pi) - np.pi)
            cy, sy = np.cos(cam_yaw), np.sin(cam_yaw)
            gym.set_camera_location(chase, env, gymapi.Vec3(pos[0] - 4.0 * cy, pos[1] - 4.0 * sy, pos[2] + 1.8),
                                    gymapi.Vec3(pos[0] + 1.5 * cy, pos[1] + 1.5 * sy, pos[2] - 0.5))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            ti = top.image(sim)
            ci = gym.get_camera_image(sim, env, chase, gymapi.IMAGE_COLOR).reshape(cc.height, cc.width, 4)[..., :3].copy()
            rgb, d16, vf = ego.read(sim)
            valid.append(vf)
            if not a.no_depth_png:
                cv2.imwrite(str(out / "depth" / f"{frame:06d}.png"), d16)
            cp_, cq_ = ego.pose()
            gr = float(U._sample(h, res, pos[:2], pos[:2])[0])
            wr.writerow([frame, f"{t:.3f}", *np.round(pos, 4), *np.round(s["quat"], 5), f"{yaw:.4f}",
                         f"{pos[0] + origin[0]:.3f}", f"{pos[1] + origin[1]:.3f}", f"{pos[2] + origin[2]:.3f}",
                         *np.round(cp_, 4), *np.round(cq_, 5), *np.round(s["vel"], 3),
                         *np.round(quad.f, 3), f"{pos[2] - gr:.3f}", min(tgt_i, len(ends))])

            cv2.polylines(ti, [path_px], False, (255, 200, 0), 2, cv2.LINE_AA)
            if len(traj) > 1:
                cv2.polylines(ti, [top.project(np.array(traj))], False, (255, 40, 40), 3, cv2.LINE_AA)
            for i, p in enumerate(pts):
                nm = "START" if i == 0 else ("GOAL" if i == len(pts) - 1 else f"WP{i}")
                col = (40, 220, 40) if i == 0 else ((255, 60, 255) if i == len(pts) - 1 else (255, 200, 0))
                if i > 0 and reached[i] is not None:
                    col = (150, 150, 150)
                u = tuple(top.project([p[0], p[1], gz])[0])
                cv2.circle(ti, u, 8, col, -1, cv2.LINE_AA); cv2.circle(ti, u, 8, (0, 0, 0), 2, cv2.LINE_AA)
                label(ti, nm, (u[0] + 11, u[1] + 5), 0.55)
            u = top.project([pos[0], pos[1], gz])[0]
            tip = top.project([pos[0] + np.cos(yaw) * max(1.5, span.max() / 35),
                               pos[1] + np.sin(yaw) * max(1.5, span.max() / 35), gz])[0]
            cv2.arrowedLine(ti, tuple(u), tuple(tip), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.4)
            cv2.circle(ti, tuple(u), 6, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.rectangle(ti, (0, 0), (ti.shape[1], 34), (0, 0, 0), -1)
            label(ti, f"quad  t={t:5.1f}s  |v|={np.linalg.norm(s['vel']):4.2f} m/s  AGL={pos[2] - gr:4.1f} m  "
                      f"flown={dist:5.1f} m  wp {min(max(tgt_i, 1), len(ends))}/{len(ends)}")
            dv = depth_vis(d16, max_m=a.depth_max)
            label(ci, "third-person  |  structure collision " + a.collision)
            if contacts:
                label(ci, f"CONTACT x{contacts} (first t={first_contact[0]:.1f}s)", (10, 52))
            rgb_s = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
            pip = cv2.resize(dv, (320, 180), interpolation=cv2.INTER_NEAREST)
            rgb_s[540 - 188:540 - 8, 960 - 328:960 - 8] = pip
            cv2.rectangle(rgb_s, (960 - 328, 540 - 188), (960 - 8, 540 - 8), (255, 255, 255), 1)
            label(rgb_s, f"D455 RGB (pitch {a.cam_pitch_deg:.0f} deg down; inset: depth 0-{a.depth_max:.0f} m)")
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
    pos = quad.state()["pos"]
    err = float(np.linalg.norm(pos - path[-1]))
    result = dict(success=bool(done and not hit and not contacts), crashed=bool(crashed), blocked=bool(blocked),
                  structure_hit=bool(hit), contact_steps=contacts, first_contact=first_contact,
                  structure_collision=a.collision, robot=a.uav,
                  twin=str(twin), waypoints=[list(map(float, p)) for p in pts], leg_altitude_z=alts,
                  agl=a.agl, clearance=a.clearance, waypoint_reached_t=reached[1:], takeoff_t=reached[0],
                  planned_length_m=route_len, flown_m=dist, time_s=frame / FPS, goal_error_m=err,
                  max_cross_track_m=max_err, frames=frame, fps=FPS, origin_utm=meta["origin"],
                  crs=cfg.get("crs"), quad=QP,
                  d455=dict(width=spec["width"], height=spec["height"], hfov_deg=spec["hfov"], K=ego.K,
                            depth="uint16 mm, 0 = invalid", depth_range_m=[spec["depth_min"], spec["depth_max"]],
                            mount_xyz_in_base=list(QP["camera_xyz"]), pitch_down_deg=a.cam_pitch_deg,
                            valid_depth_fraction=float(np.mean(valid)) if valid else 0.0,
                            pose_columns="cam_* in trajectory.csv: link frame x forward, y left, z up"))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    np.save(out / "route.npy", path)
    status = ("REACHED ALL WAYPOINTS" if result["success"] else "CRASHED" if crashed
              else "BLOCKED" if blocked else "STRUCTURE HIT" if hit else "CONTACT" if contacts
              else "TIMEOUT")
    print(f"[done] {status}: planned {route_len:.1f} m, flown {dist:.1f} m in {frame / FPS:.1f} s, "
          f"goal error {err:.2f} m, max cross-track {max_err:.2f} m -> {out}")
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
