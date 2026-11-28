"""Main experiment: a fleet of UGVs (Warthog) and UAVs (Starling 2 Max) in one digital twin.

    conda activate umasstwin                      # environment.yml

    # 1) pick the waypoints of every robot on the top-view map, then watch them navigate
    python scripts/run_fleet.py --config configs/pond.json --ugv 2 --uav 1

    # 2) replay routes saved by an earlier run (routes.json is written into every --out)
    python scripts/run_fleet.py --config configs/pond.json --mode file --routes outputs/fleet/<run>/routes.json

Modes (--mode):
    waypoint_select   (default) a window shows the twin's top view; click the waypoints of ugv1, ugv2,
                      ..., uav1, ... one robot after the other (first click = start). Then the sim opens.
    file              read the routes from --routes (same JSON that waypoint_select writes).
    algorithm         FUTURE WORK: robots driven by an experiment algorithm instead of fixed waypoints.
                      See `AlgorithmPolicy` below for the interface that is reserved for it.

UGVs follow an A* ground path through their waypoints (pure pursuit); UAVs take off, fly straight legs
at  z = max(ground + --agl, structure top + --clearance)  and hover at the last waypoint.

Viewer keys: 1..9 (top row or numpad) follow that robot in third person, M fixed top view of the whole
map, 0 free camera, N trajectories on / off, arrows orbit / zoom, P pause. Every robot's travelled
path is drawn in its own colour (also in the top view).

Outputs in --out (default outputs/fleet/<twin>_<time>):
    routes.json                 robots + waypoints (replay with --mode file)
    plan.png                    planned routes on the map
    trajectories.png            travelled paths on the map, one colour per robot
    trajectory_<robot>.csv      per frame: pose, UTM, target waypoint
    result.json                 per robot: reached waypoints, distance, goal error, status
    topview.mp4                 with --record: fixed top view of the whole map with the trajectories
    <robot>_rgb.mp4 / _depth.mp4   with --record-ego: D455 on every robot (slow)
"""
from isaacgym import gymapi  # noqa: I001

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from umass_lidar import nav, sim as S, uav as U  # noqa: E402
from umass_lidar.viz import depth_vis, label  # noqa: E402
from run_nav import plan as plan_ground  # noqa: E402

FPS = 30
UGV_DEFAULT, UAV_DEFAULT = "warthog", "starling"


# ---------------------------------------------------------------- policies

class WaypointPolicy:
    """Built-in behaviour: every robot follows its own planned route (see UGV.follow / UAV.follow)."""

    def reset(self, world, robots):
        pass

    def step(self, t, robots):
        return {r.name: r.follow(t) for r in robots}


class AlgorithmPolicy:
    """FUTURE WORK (--mode algorithm): plug the experiment algorithm in here.

    reset(world, robots)   world = dict(h, sem, dsm, res, origin, size_xy, twin); robots = the fleet.
    step(t, robots)        -> {robot.name: command}, called every physics step (120 Hz), with
                              UGV command = (v [m/s], w [rad/s])
                              UAV command = (v_des xyz [m/s], yaw [rad])
    Robot state is available as robot.pose() (pos xyz, yaw) and robot.raw (full simulator state); the
    D455 images need --record-ego style cameras (S.RGBDCamera) created for the robots.
    """

    def reset(self, world, robots):
        raise NotImplementedError("--mode algorithm is reserved for the experiment algorithm (future work)")

    def step(self, t, robots):
        raise NotImplementedError


# ---------------------------------------------------------------- robots

class UGV:
    kind = "ugv"

    def __init__(self, name, model, pts, world, a):
        self.name, self.model, self.rcfg = name, model, S.robot_config(model)
        h, res = world["h"], world["res"]
        self.path, self.wps, self.ends = plan_ground(h, res, world["sem"], pts, self.rcfg, a.max_slope,
                                                     np.random.default_rng(0), edits=world["edits"])
        self.yaw0 = float(np.arctan2(*(self.path[min(5, len(self.path) - 1)] - self.path[0])[::-1]))
        self.v_max = a.speed or min(self.rcfg["max_speed"], 1.0)
        self.look = float(np.clip(0.8 * self.v_max, 0.4, 2.5))
        self.tol, self.world = a.goal_tol, world
        self.route_len = float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())
        self.plan_xyz = np.column_stack([self.path, [h[int(p[1] / res), int(p[0] / res)] for p in self.path]])
        self.i, self.reached, self.status, self.dist, self.traj = 0, [None] * len(self.ends), "running", 0.0, []
        self._prog, self.cmd = (0.0, None), (0.0, 0.0)

    def spawn(self, gym, sim, env, idx):
        self.robot = S.WheeledRobot(gym, sim, env, self.model, self.path[0], self.yaw0, self.world["h"],
                                    self.world["res"], actor_name=self.name, collision_filter=1 << (idx % 16))
        self.view_dist = max(3.0, 14.0 * self.rcfg["wheel_radius"])

    def read(self):
        self.raw = self.robot.state()

    def pose(self):
        return self.raw[0], self.raw[2]

    def quat(self):
        return self.raw[1]

    def follow(self, t):
        """Pure pursuit on the current leg -> (v, w)."""
        pos, quat, yaw = self.raw[0], self.raw[1], self.raw[2]
        if self.status != "running":
            return 0.0, 0.0
        sub = self.path[(self.ends[self.i - 1] if self.i else 0):self.ends[self.i] + 1]
        v, w, arrived = nav.pure_pursuit(pos[:2], yaw, sub, lookahead=self.look, v_max=self.v_max, goal_tol=self.tol)
        if arrived:
            self.reached[self.i] = round(t, 2)
            print(f"[{self.name}] reached waypoint {self.i + 1}/{len(self.ends)} at t={t:.1f}s", flush=True)
            self.i += 1
            if self.i >= len(self.ends):
                self.status = "reached"
                return 0.0, 0.0
        if 1 - 2 * (quat[0] ** 2 + quat[1] ** 2) < 0.3:
            self.status = "flipped"
        if self._prog[1] is None or np.linalg.norm(pos[:2] - self._prog[1]) > 0.2:
            self._prog = (t, pos[:2].copy())
        elif t - self._prog[0] > 8.0:
            self.status = "stuck"
        return v, w

    def apply(self, cmd):
        self.cmd = cmd
        self.robot.command(*cmd)

    def target_wp(self):
        return min(self.i + 1, len(self.ends))

    def goal_error(self):
        return float(np.linalg.norm(self.robot.state()[0][:2] - self.path[-1]))


class UAV:
    kind = "uav"

    def __init__(self, name, model, pts, world, a):
        self.name, self.model, self.p = name, model, U.MODELS[model]
        h, res, size = world["h"], world["res"], world["size_xy"]
        for q in pts:
            if not (0 <= q[0] < size[0] and 0 <= q[1] < size[1]):
                raise ValueError(f"waypoint {np.round(q, 1)} is outside the twin")
        self.wps = [tuple(map(float, q)) for q in pts]
        self.path, self.ends, self.alts = U.plan_flight(h, world["dsm"], res, pts, a.agl, a.clearance)
        self.path[0, 2] += 0.12                               # body centre resting on the ground
        self.targets = [1] + self.ends                        # take-off vertex, then the waypoints
        self.yaw0 = float(np.arctan2(*(self.path[2, :2] - self.path[1, :2])[::-1]))
        self.speed, self.tol, self.world = a.uav_speed, max(a.goal_tol, 0.5), world
        self.route_len = float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())
        self.plan_xyz = self.path
        self.i, self.reached, self.status, self.dist, self.traj = 0, [None] * len(self.targets), "running", 0.0, []
        self._hold, self.yaw_cmd, self.contacts, self.view_dist = 0.0, self.yaw0, 0, 3.0

    def spawn(self, gym, sim, env, idx):
        self.gym, self.env = gym, env
        dt = gym.get_sim_params(sim).dt
        self.dt = dt
        self.quad = U.Quadrotor(gym, sim, env, self.path[0], self.yaw0, dt, p=self.p, actor_name=self.name,
                                collision_filter=1 << (idx % 16))
        self.ctl = U.QuadController(p=self.p)
        self.robot = self.quad                                 # for S.RGBDCamera
        self.body_env = gym.find_actor_rigid_body_index(env, self.quad.actor, "base_link", gymapi.DOMAIN_ENV)

    def read(self):
        self.raw = self.quad.state()

    def pose(self):
        return self.raw["pos"], self.raw["yaw"]

    def quat(self):
        return self.raw["quat"]

    def follow(self, t):
        """Velocity command towards a carrot on the 3D route -> (v_des, yaw). Hovers at the goal."""
        s, P, tg = self.raw, self.path, self.targets
        pos = s["pos"]
        done = self.status == "reached"
        if self.status == "crashed":
            return None
        ti = min(self.i, len(tg) - 1)
        goal = P[tg[ti]]
        cpt, _ = U.carrot(P, 0 if ti == 0 else tg[ti - 1], tg[ti], pos, lookahead=max(1.5, self.speed))
        dg = np.linalg.norm(goal - pos)
        to_c = (goal if done else cpt) - pos
        vmax = min(self.speed, np.sqrt(2 * 1.5 * dg) + 0.2) if ti > 0 else min(2.0, dg + 0.3)
        v_des = to_c / max(np.linalg.norm(to_c), 1e-6) * min(vmax, np.linalg.norm(to_c) * 1.5)
        if np.linalg.norm(v_des[:2]) > 0.5:
            self.yaw_cmd = float(np.arctan2(v_des[1], v_des[0]))
        if not done:
            self._hold = self._hold + self.dt if dg < self.tol else 0.0
            if self._hold >= 0.3:
                self.reached[self.i] = round(t, 2)
                what = "take-off" if self.i == 0 else f"waypoint {self.i}/{len(self.ends)}"
                print(f"[{self.name}] reached {what} at t={t:.1f}s (z={pos[2]:.1f})", flush=True)
                self.i += 1; self._hold = 0.0
                if self.i >= len(tg):
                    self.status = "reached"
        h, res = self.world["h"], self.world["res"]
        r_, c_ = min(int(pos[1] / res), h.shape[0] - 1), min(int(pos[0] / res), h.shape[1] - 1)
        if s["R"][2, 2] < 0.2 and t > 1.0:
            self.status = "crashed"
            print(f"[{self.name}] CRASHED at t={t:.1f}s pos={np.round(pos, 1)}", flush=True)
        fc = self.gym.get_env_rigid_contact_forces(self.env)[self.body_env]
        if t > 1.0 and np.linalg.norm([fc["x"], fc["y"], fc["z"]]) > 0.5 and pos[2] - h[r_, c_] > 0.5:
            if not self.contacts:
                print(f"[{self.name}] CONTACT at t={t:.1f}s pos={np.round(pos, 1)}", flush=True)
            self.contacts += 1
        return v_des, self.yaw_cmd

    def apply(self, cmd):
        f = np.zeros(4) if cmd is None else self.ctl.velocity(self.raw, np.asarray(cmd[0], float), float(cmd[1]))
        self.quad.apply_motors(f, self.raw)

    def idle(self):
        self.quad.apply_motors(np.zeros(4), self.raw)

    def target_wp(self):
        return min(self.i, len(self.ends))

    def goal_error(self):
        return float(np.linalg.norm(self.quad.state()["pos"] - self.path[-1]))


# ---------------------------------------------------------------- helpers

def fleet_spec(n_ugv, n_uav, ugv_model, uav_model):
    return [dict(name=f"ugv{i + 1}", type="ugv", model=ugv_model) for i in range(n_ugv)] + \
           [dict(name=f"uav{i + 1}", type="uav", model=uav_model) for i in range(n_uav)]


def draw_map(base_png, size_xy, robots, colors, out, key="plan_xyz", px_per_m=None):
    """Routes (key="plan_xyz") or travelled paths (key="traj") on the twin's map, one colour per robot."""
    img = cv2.imread(str(base_png))
    s = img.shape[1] / size_xy[0] if px_per_m is None else px_per_m
    H = img.shape[0]
    px = lambda P: np.column_stack([np.asarray(P)[:, 0] * s, H - np.asarray(P)[:, 1] * s]).astype(np.int32)   # noqa: E731
    th = max(2, int(round(img.shape[1] / 700)))
    for r in robots:
        P = np.asarray(getattr(r, key))
        if len(P) < 2:
            continue
        bgr = tuple(int(255 * c) for c in colors[r.name][::-1])
        cv2.polylines(img, [px(P)], False, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.polylines(img, [px(P)], False, bgr, th, cv2.LINE_AA)
        for j, w in enumerate(r.wps):
            u = tuple(int(v) for v in px([w])[0])
            cv2.circle(img, u, th * 3, bgr, -1, cv2.LINE_AA); cv2.circle(img, u, th * 3, (0, 0, 0), 2, cv2.LINE_AA)
            if j == 0:
                for col, t_ in (((0, 0, 0), th + 3), ((255, 255, 255), max(1, th - 1))):
                    cv2.putText(img, r.name, (u[0] + th * 4, u[1] + th * 2), 0, 0.3 * th, col, t_, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="twin config (default configs/blue_wall.json)")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--ugv", type=int, default=1, help="number of UGVs")
    ap.add_argument("--uav", type=int, default=1, help="number of UAVs")
    ap.add_argument("--ugv-model", default=UGV_DEFAULT, help="UGV model from assets/robots/robots.json")
    ap.add_argument("--uav-model", default=UAV_DEFAULT, choices=sorted(U.MODELS))
    ap.add_argument("--mode", default="waypoint_select", choices=("waypoint_select", "file", "algorithm"))
    ap.add_argument("--routes", default=None, help="routes.json for --mode file")
    ap.add_argument("--speed", type=float, default=None, help="UGV m/s (default: robot max, capped 1.0)")
    ap.add_argument("--uav-speed", type=float, default=3.0, help="UAV cruise m/s")
    ap.add_argument("--agl", type=float, default=5.0, help="UAV height above ground [m]")
    ap.add_argument("--clearance", type=float, default=2.0, help="UAV min height above structures [m]")
    ap.add_argument("--max-slope", type=float, default=15.0)
    ap.add_argument("--goal-tol", type=float, default=0.4)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--collision", choices=("mesh", "voxel", "none"), default="mesh")
    ap.add_argument("--out", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--view", default="top", help="viewer camera at start: top, free, or a robot name (ugv1, uav1, ...)")
    ap.add_argument("--hold", action="store_true", help="keep the viewer open after every robot has finished")
    ap.add_argument("--record", action="store_true", help="write topview.mp4 (whole map + trajectories)")
    ap.add_argument("--record-ego", action="store_true", help="also write D455 RGB / depth videos of every robot (slow)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--debug-shot", type=float, nargs="+", default=None, help=argparse.SUPPRESS)  # viewer screenshots at t
    a = ap.parse_args()
    a.record = a.record or a.record_ego

    twin, cfg = S.resolve_twin(a.config, a.variant)
    gprefix = twin.parent / "ground"
    meta, h = S.load_terrain_data(gprefix)
    sem = np.load(str(gprefix) + "_semantic.npy")
    res, origin, size = meta["mesh_res"], np.array(meta["origin"]), np.array(meta["size_xy"], float)
    base_png = twin.parent / "map_base.png"
    out = Path(a.out) if a.out else ROOT / "outputs/fleet" / f"{cfg['name']}_{time.strftime('%y%m%d_%H%M%S')}"

    # ---- who is in the fleet, and where do they go
    if a.mode == "algorithm":
        sys.exit("[fleet] --mode algorithm is future work: it is reserved for the experiment algorithm.\n"
                 "        The interface kept for it is AlgorithmPolicy in scripts/run_fleet.py.")
    if a.mode == "file":
        if not a.routes:
            sys.exit("[fleet] --mode file needs --routes routes.json")
        saved = json.loads(Path(a.routes).read_text())
        spec = [dict(name=r["name"], type=r["type"], model=r["model"]) for r in saved["robots"]]
        pts_of = {r["name"]: [tuple(p) for p in r["waypoints"]] for r in saved["robots"]}
    else:
        spec = fleet_spec(a.ugv, a.uav, a.ugv_model, a.uav_model)
        pts_of = None
    if not spec:
        sys.exit("[fleet] no robots: use --ugv N and / or --uav N")
    colors = {r["name"]: S.FollowViewer.COLORS[i % len(S.FollowViewer.COLORS)] for i, r in enumerate(spec)}
    print(f"[fleet] {cfg['name']}: " + ", ".join(f"{r['name']} ({r['model']})" for r in spec))
    world = dict(h=h, sem=sem, res=res, origin=origin, size_xy=size, twin=twin, dsm=None,
                 edits=nav.load_drivable_edits(twin.parent))
    if world["edits"]:
        print(f"[fleet] {len(world['edits'])} manual drivable-area edits ({twin.parent / nav.EDITS_FILE})")
    if any(r["type"] == "uav" for r in spec):
        world["dsm"] = U.surface_model(h, U.structure_top(twin, h.shape, res), res)

    built = {}

    def build(r, pts):
        """Plan the route of one robot. (ok, message, ground path for the picker)."""
        try:
            built[r["name"]] = (UGV if r["type"] == "ugv" else UAV)(r["name"], r["model"], pts, world, a)
        except (ValueError, KeyError) as e:
            msg = str(e)
            m = re.search(r"point \(([-\d.]+), ([-\d.]+)\) is", msg)
            if m and r["type"] == "ugv":                       # snap failure: say which click it was
                k = int(np.argmin([np.hypot(p[0] - float(m.group(1)), p[1] - float(m.group(2))) for p in pts]))
                what = "the start" if k == 0 else f"waypoint {k}"
                msg = f"{what} is not on drivable ground" if k == 0 else f"{what} cannot be reached from the start (red / orange area)"
            return False, f"{r['name']}: {msg}", None
        ag = built[r["name"]]
        print(f"[plan] {ag.name}: {len(pts)} waypoints, {ag.route_len:.1f} m", flush=True)
        return True, "", ag.plan_xyz[:, :2]

    reach = {}

    def selectable(r, start):
        """Cells a UGV can be sent to: drivable, and once its start is placed, connected to that start."""
        if r["type"] != "ugv":
            return None
        if r["model"] not in reach:
            rc = S.robot_config(r["model"])
            coarse, cres = nav.coarse_mask(h, res, 0.5, max_slope_deg=a.max_slope,
                                           robot_radius=rc["track"] * 0.5 + 0.35, blocked=sem != 0, edits=world["edits"])
            lab, _ = ndimage.label(coarse, structure=np.ones((3, 3)))
            near = ndimage.distance_transform_edt(~coarse, return_distances=False, return_indices=True)
            reach[r["model"]] = (coarse, cres, lab, near)
        coarse, cres, lab, near = reach[r["model"]]
        if start is None:
            return coarse, None
        i = int(np.clip(start[1] / cres, 0, coarse.shape[0] - 1)); j = int(np.clip(start[0] / cres, 0, coarse.shape[1] - 1))
        return coarse, lab == lab[near[0][i, j], near[1][i, j]]  # region of the drivable cell nearest to the start

    if pts_of is None:
        from umass_lidar.waypoint_ui import pick_routes
        print("[fleet] waypoint_select: click the waypoints of each robot in the window, Enter = next robot", flush=True)
        pts_of = pick_routes(base_png, size, spec, colors, validate=build, selectable=selectable)
        if pts_of is None:
            sys.exit("[fleet] waypoint selection cancelled")
    else:
        for r in spec:
            ok, msg, _ = build(r, pts_of[r["name"]])
            if not ok:
                sys.exit(f"[plan] {msg}")
    robots = [built[r["name"]] for r in spec]
    starts = np.array([r.plan_xyz[0, :2] for r in robots])
    for i in range(len(robots)):
        for j in range(i):
            if np.linalg.norm(starts[i] - starts[j]) < 2.0:
                print(f"[fleet] WARNING: {robots[i].name} and {robots[j].name} start {np.linalg.norm(starts[i] - starts[j]):.1f} m apart")

    out.mkdir(parents=True, exist_ok=True)
    (out / "routes.json").write_text(json.dumps(dict(
        twin=cfg["name"], frame="local metres (x east, y north) from the twin origin", origin_utm=meta["origin"],
        robots=[dict(r, waypoints=[list(map(float, p)) for p in pts_of[r["name"]]]) for r in spec]), indent=2))
    draw_map(base_png, size, robots, colors, out / "plan.png")
    print(f"[fleet] routes -> {out / 'routes.json'}", flush=True)
    if a.plan_only:
        return

    # ---- sim
    print("[fleet] loading the twin (a large twin takes 1-2 minutes)", flush=True)
    gym, sim, sp = S.make_sim()
    env, _ = S.add_twin(gym, sim, twin, meta, a.collision)
    for i, r in enumerate(robots):
        r.spawn(gym, sim, env, i)
    policy = WaypointPolicy()
    policy.reset(world, robots)

    ground = h[sem == 0] if (sem == 0).any() else h
    gz = float(np.median(ground))
    top_z = float(np.nanmax(world["dsm"])) if world["dsm"] is not None else float(h.max()) + 30.0
    overview = dict(center=size / 2, span=size, ground_z=gz, top_z=top_z)
    viewer = None
    if not a.headless:
        targets = {r.name: ((lambda r=r: r.pose()), r.view_dist) for r in robots}
        follow = a.view if a.view in targets or a.view == "top" else None
        viewer = S.FollowViewer(gym, sim, env, targets, follow, overview=overview, colors=colors)
    top = writer = None
    egos, ego_w = {}, {}
    if a.record:
        import imageio
        top = S.TopDownCamera(gym, env, size / 2, size, gz, hfov=30.0)
        writer = imageio.get_writer(out / "topview.mp4", fps=FPS, quality=7, macro_block_size=1)
        if a.record_ego:
            for r in robots:
                egos[r.name] = S.RGBDCamera(gym, env, r.robot, 0.0 if r.kind == "ugv" else 30.0)
                ego_w[r.name] = [imageio.get_writer(out / f"{r.name}_{k}.mp4", fps=FPS, quality=7, macro_block_size=1)
                                 for k in ("rgb", "depth")]

    files, logs = {}, {}
    for r in robots:
        files[r.name] = open(out / f"trajectory_{r.name}.csv", "w", newline="")
        logs[r.name] = csv.writer(files[r.name])
        logs[r.name].writerow(["frame", "t", "x", "y", "z", "qx", "qy", "qz", "qw", "yaw", "utm_e", "utm_n", "h_ell", "target_wp"])

    every, settle = int(round(1 / (sp.dt * FPS))), int(1.0 / sp.dt)
    hold_open = a.hold and viewer is not None
    frame, k = 0, 0
    t0 = time.time()
    while hold_open or (k < int(a.max_seconds / sp.dt) + settle and any(r.status == "running" for r in robots)):
        t = (k - settle) * sp.dt
        for r in robots:
            r.read()
        if k >= settle:
            cmds = policy.step(t, robots)
            if viewer is not None and viewer.cur in built and built[viewer.cur].kind == "ugv":
                man = viewer.teleop(built[viewer.cur].v_max)
                if man is not None:
                    cmds[viewer.cur] = man                     # T: manual drive of the followed UGV
            for r in robots:
                r.apply(cmds[r.name])
        else:
            for r in robots:
                r.apply((0.0, 0.0)) if r.kind == "ugv" else r.idle()
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if k >= settle and (k - settle) % every == 0:
            for r in robots:
                pos, yaw = r.pose()
                if r.traj:
                    r.dist += float(np.linalg.norm(pos - r.traj[-1]))
                r.traj.append(np.array(pos))
                logs[r.name].writerow([frame, f"{t:.3f}", *np.round(pos, 4), *np.round(r.quat(), 5), f"{yaw:.4f}",
                                       f"{pos[0] + origin[0]:.3f}", f"{pos[1] + origin[1]:.3f}", f"{pos[2] + origin[2]:.3f}",
                                       r.target_wp()])
                if viewer is not None:
                    viewer.track(r.name, pos)
            if a.record:
                gym.step_graphics(sim)
                gym.render_all_camera_sensors(sim)
                ti = top.image(sim)
                flat = lambda X: np.column_stack([np.asarray(X)[:, :2], np.full(len(X), gz)])  # noqa: E731
                for r in robots:
                    rgb = tuple(int(255 * c) for c in colors[r.name])
                    cv2.polylines(ti, [top.project(flat(r.plan_xyz))], False, tuple(int(0.55 * c) for c in rgb), 1, cv2.LINE_AA)
                    if len(r.traj) > 1:
                        cv2.polylines(ti, [top.project(flat(r.traj))], False, (0, 0, 0), 5, cv2.LINE_AA)
                        cv2.polylines(ti, [top.project(flat(r.traj))], False, rgb, 3, cv2.LINE_AA)
                    u = tuple(int(x) for x in top.project(flat(r.traj[-1:]))[0])
                    cv2.circle(ti, u, 7, rgb, -1, cv2.LINE_AA); cv2.circle(ti, u, 7, (0, 0, 0), 2, cv2.LINE_AA)
                    label(ti, r.name, (u[0] + 10, u[1] - 8), 0.5)
                cv2.rectangle(ti, (0, 0), (ti.shape[1], 34), (0, 0, 0), -1)
                label(ti, f"t={t:6.1f}s   " + "   ".join(f"{r.name} {r.dist:.0f} m" for r in robots), scale=0.55)
                writer.append_data(ti)
                for r in robots:
                    if r.name in egos:
                        rgb_, d16, _ = egos[r.name].read(sim)
                        ego_w[r.name][0].append_data(rgb_); ego_w[r.name][1].append_data(depth_vis(d16, max_m=20.0))
            frame += 1
        if viewer is not None and not viewer.step():
            break
        if viewer is not None and a.debug_shot and t >= a.debug_shot[0]:
            gym.write_viewer_image_to_file(viewer.viewer, str(out / f"viewer_{a.debug_shot.pop(0):05.1f}.png"))
        k += 1

    for f in files.values():
        f.close()
    for w_ in ([writer] if writer else []) + [w_ for ws in ego_w.values() for w_ in ws]:
        w_.close()
    draw_map(base_png, size, robots, colors, out / "trajectories.png", key="traj")
    result = dict(success=all(r.status == "reached" for r in robots), mode=a.mode, twin=str(twin),
                  structure_collision=a.collision, time_s=frame / FPS, wall_s=round(time.time() - t0, 1), fps=FPS,
                  origin_utm=meta["origin"], crs=cfg.get("crs"),
                  robots={r.name: dict(type=r.kind, model=r.model, status=r.status if r.status != "running" else "timeout",
                                       color_rgb=[round(c, 2) for c in colors[r.name]],
                                       waypoints=[list(map(float, p)) for p in pts_of[r.name]],
                                       planned_length_m=r.route_len, travelled_m=r.dist,
                                       waypoint_reached_t=r.reached, goal_error_m=r.goal_error(),
                                       **(dict(contact_steps=r.contacts) if r.kind == "uav" else {})) for r in robots})
    (out / "result.json").write_text(json.dumps(result, indent=2))
    for r in robots:
        st = result["robots"][r.name]
        print(f"[done] {r.name}: {st['status'].upper()} ({r.dist:.1f} / {r.route_len:.1f} m, goal error {st['goal_error_m']:.2f} m)")
    print(f"[done] {frame / FPS:.1f} s simulated -> {out}", flush=True)
    if viewer is not None:
        # everything is on disk; destroying the viewer of a large twin segfaults inside Isaac Gym, so leave directly
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
