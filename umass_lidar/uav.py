"""Quadrotor (UAV) for the digital twin: URDF, rotor-level dynamics, geometric controller,
3D waypoint route over the twin's surface model.

Isaac Gym has no rotor model, so the drone is a single rigid body and every physics step we apply
the total thrust / body torque that four first-order motors produce. Action interfaces, from low
to high level (the usual RL action spaces):

    Quadrotor.apply_motors(f)          4 rotor thrusts [N]           (direct)
    QuadController.ctbr(T, rates)      collective thrust + body rates -> rotor thrusts
    QuadController.velocity(v, yaw)    world velocity + yaw          -> rotor thrusts

Everything except the Quadrotor actor itself is numpy-only (no isaacgym import at module level),
so the controller / route code can be reused by a batched RL env.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]

# ~1 kg X-quad with a D455 under the nose (250-450 class); thrust/weight ~2.4
QUAD = dict(
    mass=1.0,
    inertia=(0.0110, 0.0110, 0.0190),
    arm=0.17,                 # centre -> rotor [m]
    f_max=6.0,                # per rotor [N]
    k_m=0.016,                # rotor reaction torque / thrust [m]
    tau_motor=0.03,           # rotor first-order time constant [s]
    drag=(0.25, 0.25, 0.35),  # linear body drag [N / (m/s)]
    camera_link="d455_link",
    camera_xyz=(0.13, 0.0, -0.03),
)

# ModalAI Starling 2 Max. Published (docs.modalai.com/starling-2-max-datasheet): take-off weight
# 566 g, motor-to-motor 260 mm wide x 190 mm long (322 mm diagonal), 180 mm tri-blade props,
# 2203.5 1500 KV motors, 500 g payload. ModalAI publishes no URDF, thrust curve or inertia.
# Geometry, meshes and inertia follow the community model github.com/BjarkeHJ/AIRLAB_Starling2Max
# (fetched by scripts/prepare_robots.py --starling; inertia there is a solid-box approximation of the
# 0.217 x 0.287 x 0.129 m body). f_max, k_m, tau_motor and drag are ESTIMATES, not measurements
# (that model assumes thrust/weight 2.58; f_max here gives 3.7).
STARLING = dict(
    name="starling",
    mass=0.566,
    inertia=(0.00441, 0.00284, 0.00576),
    arm=0.161,                # centre -> rotor [m]
    rotor_xy=(0.095, 0.130),  # rotor offset forward / sideways [m]: the frame is wider than long
    rotor_z=0.024,
    f_max=5.2,                # per rotor [N], estimate
    k_m=0.016,
    tau_motor=0.03,
    drag=(0.20, 0.20, 0.28),
    prop_radius=0.090,
    body=(0.17, 0.08, 0.05),
    mesh=dict(body="meshes/Starling2Max_body.obj", prop="meshes/Starling2Max_7040prop.obj"),
    camera_link="d455_link",
    camera_xyz=(0.13, 0.0, -0.01),
)
MODELS = {"quad": QUAD, "starling": STARLING}


def rotor_xy(p):
    """Rotor positions (x fwd, y left) in the order FL, RL, RR, FR."""
    rx, ry = p.get("rotor_xy", (p["arm"] / np.sqrt(2),) * 2)
    return np.array([[rx, ry], [-rx, ry], [-rx, -ry], [rx, -ry]])


def write_quad_urdf(path=None, p=STARLING):
    """Body + 4 rotors (visual only) + fixed d455_link. Uses the meshes in p["mesh"] when they have
    been fetched (scripts/prepare_robots.py --starling), otherwise boxes and discs."""
    name = p.get("name", "quad")
    path = Path(path or ROOT / f"assets/robots/{name}/{name}.urdf")
    bx, by, bz = p.get("body", (0.16, 0.10, 0.06))
    pr = p.get("prop_radius", 0.065)
    path.parent.mkdir(parents=True, exist_ok=True)
    ixx, iyy, izz = p["inertia"]
    rxy = rotor_xy(p)
    mesh = p.get("mesh")
    if mesh and all((path.parent / f).exists() for f in mesh.values()):
        vis = [f'<visual><geometry><mesh filename="{mesh["body"]}"/></geometry>'
               '<material name="body"><color rgba="0.12 0.12 0.13 1"/></material></visual>']
        for k, (x, y) in enumerate(rxy):
            col = "0.9 0.25 0.15 1" if x > 0 else "0.2 0.6 0.95 1"        # red = front
            vis.append(f'<visual><origin xyz="{x:.4f} {y:.4f} {p.get("rotor_z", 0.03)}" rpy="0 0 {0.7 * k:.2f}"/>'
                       f'<geometry><mesh filename="{mesh["prop"]}"/></geometry>'
                       f'<material name="rotor{k}"><color rgba="{col}"/></material></visual>')
        cam_box = "0.012 0.03 0.012"
    else:
        vis = [f'<visual><geometry><box size="{bx} {by} {bz}"/></geometry>'
               '<material name="body"><color rgba="0.15 0.15 0.17 1"/></material></visual>']
        for k, (x, y) in enumerate(rxy):
            yaw, ln = np.arctan2(y, x), float(np.hypot(x, y))
            vis.append(f'<visual><origin xyz="{x / 2:.4f} {y / 2:.4f} 0" rpy="0 0 {yaw:.4f}"/>'
                       f'<geometry><box size="{ln:.3f} 0.025 0.02"/></geometry>'
                       '<material name="arm"><color rgba="0.25 0.25 0.28 1"/></material></visual>')
            col = "0.9 0.25 0.15 1" if x > 0 else "0.2 0.6 0.95 1"       # red = front
            vis.append(f'<visual><origin xyz="{x:.4f} {y:.4f} 0.03"/>'
                       f'<geometry><cylinder radius="{pr}" length="0.008"/></geometry>'
                       f'<material name="rotor{k}"><color rgba="{col}"/></material></visual>')
        cam_box = "0.03 0.12 0.03"
    cbx, cby = 2 * rxy[0] + 0.1
    cx, cy, cz = p["camera_xyz"]
    urdf = f"""<robot name="{name}">
<link name="base_link">
  <inertial><mass value="{p['mass']}"/>
    <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/></inertial>
  {''.join(vis)}
  <collision><geometry><box size="{cbx:.3f} {cby:.3f} 0.08"/></geometry></collision>
</link>
<link name="{p['camera_link']}">
  <inertial><mass value="0.001"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><geometry><box size="{cam_box}"/></geometry>
    <material name="cam"><color rgba="0.05 0.05 0.05 1"/></material></visual>
</link>
<joint name="d455_joint" type="fixed"><parent link="base_link"/><child link="{p['camera_link']}"/>
  <origin xyz="{cx} {cy} {cz}"/></joint>
</robot>
"""
    path.write_text(urdf)
    return path


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mixer(p=STARLING):
    """A with [T, tau_x, tau_y, tau_z] = A @ f.  Rotors (x, y): FL, RL, RR, FR; FL/RR spin CCW."""
    xy = rotor_xy(p)
    spin = np.array([1, -1, 1, -1])          # tau_z = k_m * sum(spin_i * f_i)
    return np.vstack([np.ones(4), xy[:, 1], -xy[:, 0], spin * p["k_m"]])


class QuadController:
    """Lee et al. SO(3) geometric controller. All inputs world frame, z up."""

    def __init__(self, p=STARLING, max_tilt_deg=35.0, max_acc=(4.0, 3.0)):
        self.p, self.m, self.J = p, p["mass"], np.diag(p["inertia"])
        self.A = mixer(p)
        self.Ainv = np.linalg.inv(self.A)
        self.kR = np.array([4.0, 4.0, 1.2])
        self.kW = np.array([0.38, 0.38, 0.30])
        self.kv = 2.5
        self.max_tilt = np.radians(max_tilt_deg)
        self.max_acc = max_acc

    def _motors(self, T, tau):
        return np.clip(self.Ainv @ np.r_[T, tau], 0.0, self.p["f_max"])

    def velocity(self, state, v_des, yaw_des, a_ff=None):
        """World velocity + yaw target -> rotor thrusts (N)."""
        pos, R, v, w_body = state["pos"], state["R"], state["vel"], state["w_body"]
        a = self.kv * (np.asarray(v_des) - v) + (0 if a_ff is None else a_ff)
        ah = np.linalg.norm(a[:2])
        if ah > self.max_acc[0]:
            a[:2] *= self.max_acc[0] / ah
        a[2] = np.clip(a[2], -self.max_acc[1], self.max_acc[1])
        F = self.m * (a + np.array([0, 0, 9.81]))
        # tilt limit: keep the vertical component, shrink the horizontal one
        hmax = F[2] * np.tan(self.max_tilt)
        if np.linalg.norm(F[:2]) > hmax:
            F[:2] *= hmax / np.linalg.norm(F[:2])
        b3 = F / np.linalg.norm(F)
        b1c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        b2 = np.cross(b3, b1c); b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        T = float(F @ R[:, 2])
        E = Rd.T @ R - R.T @ Rd
        eR = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        tau = -self.kR * eR - self.kW * w_body + np.cross(w_body, self.J @ w_body)
        return self._motors(T, tau)

    def ctbr(self, state, T, rates):
        """Collective thrust (N) + body rates (rad/s) -> rotor thrusts."""
        w = state["w_body"]
        tau = self.J @ (12.0 * (np.asarray(rates) - w)) + np.cross(w, self.J @ w)
        return self._motors(T, tau)


class Quadrotor:
    """Isaac Gym actor + rotor dynamics. Call apply_motors() before every gym.simulate()."""

    def __init__(self, gym, sim, env, xyz, yaw, dt, p=STARLING, actor_name=None, collision_filter=1):
        from isaacgym import gymapi
        self.gymapi, self.gym, self.env, self.p, self.dt = gymapi, gym, env, p, dt
        self.cfg = dict(camera_link=p["camera_link"], camera_xyz=list(p["camera_xyz"]))
        urdf = write_quad_urdf(p=p)
        o = gymapi.AssetOptions()
        o.collapse_fixed_joints = False
        o.angular_damping = 0.0
        o.linear_damping = 0.0
        o.max_angular_velocity = 100.0
        asset = gym.load_asset(sim, str(urdf.parent), urdf.name, o)
        t = gymapi.Transform()
        t.p = gymapi.Vec3(*xyz)
        t.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), yaw)
        self.actor = gym.create_actor(env, asset, t, actor_name or p.get("name", "quad"), 0, collision_filter)
        self.mass = sum(b.mass for b in gym.get_actor_rigid_body_properties(env, self.actor))
        self.body = gym.find_actor_rigid_body_handle(env, self.actor, "base_link")
        self.base = gym.find_actor_rigid_body_index(env, self.actor, "base_link", gymapi.DOMAIN_ACTOR)
        self.A = mixer(p)
        self.f = np.zeros(4)          # actual rotor thrusts (after motor lag)

    def state(self):
        st = self.gym.get_actor_rigid_body_states(self.env, self.actor, self.gymapi.STATE_ALL)
        p, q = st["pose"]["p"][self.base], st["pose"]["r"][self.base]
        lv, av = st["vel"]["linear"][self.base], st["vel"]["angular"][self.base]
        quat = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
        R = quat_to_R(quat)
        w = np.array([av["x"], av["y"], av["z"]], dtype=np.float64)
        return dict(pos=np.array([p["x"], p["y"], p["z"]], dtype=np.float64), quat=quat, R=R,
                    vel=np.array([lv["x"], lv["y"], lv["z"]], dtype=np.float64), w_body=R.T @ w,
                    yaw=float(np.arctan2(R[1, 0], R[0, 0])))

    def apply_motors(self, f_cmd, state=None):
        gymapi = self.gymapi
        a = self.dt / (self.p["tau_motor"] + self.dt)
        self.f += a * (np.clip(f_cmd, 0, self.p["f_max"]) - self.f)
        s = state or self.state()
        T, tx, ty, tz = self.A @ self.f
        R, v = s["R"], s["vel"]
        drag_b = -np.asarray(self.p["drag"]) * (R.T @ v)
        F = R @ (np.array([0, 0, T]) + drag_b)
        tau = R @ np.array([tx, ty, tz])
        self.gym.apply_body_forces(self.env, self.body, gymapi.Vec3(*F), gymapi.Vec3(*tau), gymapi.ENV_SPACE)


# ---------------------------------------------------------------- surface model + route

def structure_top(twin_dir, shape, res):
    """Max z per grid cell of the twin's 3D structure tiles (s_*.glb, visual only in the sim:
    trees / walls / buildings the drone must fly over). Cached next to the tiles."""
    twin_dir = Path(twin_dir)
    cache = twin_dir / "structure_top.npy"
    if cache.exists():
        top = np.load(cache)
        if top.shape == tuple(shape):
            return top
    from .mesh import load_glb_zup
    top = np.full(shape, -np.inf, np.float32)
    files = sorted(twin_dir.glob("s_*.glb"))
    for i, f in enumerate(files):
        V = np.asarray(load_glb_zup(f).vertices)
        r = np.clip((V[:, 1] / res).astype(np.int64), 0, shape[0] - 1)
        c = np.clip((V[:, 0] / res).astype(np.int64), 0, shape[1] - 1)
        np.maximum.at(top, (r, c), V[:, 2].astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"[uav] structure top {i + 1}/{len(files)}", flush=True)
    np.save(cache, top)
    return top


def surface_model(h, top, res, clearance_radius=1.5):
    """DSM = max(ground mesh, structure tiles), dilated horizontally by clearance_radius."""
    dsm = np.maximum(h, np.where(np.isfinite(top), top, -np.inf))
    k = max(1, int(round(clearance_radius / res)))
    return ndimage.maximum_filter(dsm, size=2 * k + 1)


def _sample(grid, res, a, b, step=0.5):
    n = max(2, int(np.linalg.norm(np.subtract(b, a)) / step) + 1)
    xy = np.linspace(a, b, n)
    r = np.clip((xy[:, 1] / res).astype(int), 0, grid.shape[0] - 1)
    c = np.clip((xy[:, 0] / res).astype(int), 0, grid.shape[1] - 1)
    return grid[r, c]


def plan_flight(h, dsm, res, wps_xy, agl=6.0, clearance=2.0, takeoff_z=None):
    """3D polyline: vertical take-off, then for each leg a constant altitude
        z = max(ground_along_leg + agl, dsm_along_leg + clearance),
    climbing / descending at the waypoint before flying the leg.
    Returns (path (N,3), indices of waypoint vertices, per-leg altitude)."""
    wps_xy = [np.asarray(p, float) for p in wps_xy]
    z0 = float(_sample(h, res, wps_xy[0], wps_xy[0])[0]) if takeoff_z is None else takeoff_z
    pts, ends, alts = [np.r_[wps_xy[0], z0]], [], []
    for a, b in zip(wps_xy[:-1], wps_xy[1:]):
        z = max(float(_sample(h, res, a, b).max()) + agl, float(_sample(dsm, res, a, b).max()) + clearance)
        alts.append(z)
        pts.append(np.r_[a, z])
        pts.append(np.r_[b, z])
        ends.append(len(pts) - 1)
    return np.array(pts), ends, alts


def carrot(path, seg_lo, seg_hi, pos, lookahead):
    """Point `lookahead` metres ahead of the projection of pos onto path[seg_lo..seg_hi]."""
    best, best_d = None, np.inf
    for i in range(seg_lo, seg_hi):
        a, b = path[i], path[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-9 else float(np.clip((pos - a) @ ab / L2, 0, 1))
        d = np.linalg.norm(a + t * ab - pos)
        if d < best_d:
            best_d, best = d, (i, t)
    i, t = best
    remain = lookahead
    p = path[i] + t * (path[i + 1] - path[i])
    while True:
        seg = path[i + 1] - p
        L = np.linalg.norm(seg)
        if L >= remain or i + 1 >= seg_hi:
            return (p + seg * min(1.0, remain / max(L, 1e-9))), best_d
        remain -= L
        i += 1
        p = path[i]
