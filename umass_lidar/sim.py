"""Isaac Gym helpers shared by the experiment scripts (import isaacgym before torch/this module)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from isaacgym import gymapi

from .mesh import load_glb_zup

ROOT = Path(__file__).resolve().parents[1]


def resolve_twin(config=None, variant="full"):
    """(twin_dir, cfg) for a session config; the twin must have been built with
    scripts/build_twin.py in the processing env."""
    path = Path(config or ROOT / "configs/blue_wall.json")
    cfg = json.loads(path.read_text())
    twin = ROOT / "assets/twin" / cfg["name"] / variant
    if not (twin / "twin_visual.urdf").exists():
        raise FileNotFoundError(
            f"{twin} not built. Run (processing env):  python scripts/build_twin.py --config {path}")
    return twin, cfg


def load_terrain_data(prefix):
    prefix = Path(prefix)
    meta = json.loads(prefix.with_suffix(".json").read_text())
    h = np.load(str(prefix) + "_height.npy")
    return meta, h


def make_sim(dt=1 / 120):
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.dt = dt
    sp.substeps = 2
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 6
    sp.physx.num_velocity_iterations = 1
    sp.physx.use_gpu = True
    sp.use_gpu_pipeline = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    assert sim is not None, "create_sim failed"
    return gym, sim, sp


def set_baked_lighting(gym, sim, directional=0.04):
    """Render textures (almost) unlit. Isaac Gym shades as  tex * intensity * (ambient + max(0, n.l))
    (measured), so a tiny intensity with ambient = 1/intensity reproduces the baked colours on
    every surface orientation; `directional` is the residual top-light fraction that keeps the
    robot model readable."""
    i = directional
    gym.set_light_parameters(sim, 0, gymapi.Vec3(i, i, i), gymapi.Vec3(1 / i, 1 / i, 1 / i),
                             gymapi.Vec3(0.3, 0.2, 1.0))
    for k in (1, 2, 3):
        gym.set_light_parameters(sim, k, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 1))


def add_twin(gym, sim, twin_dir, meta):
    """Collision trimesh (2.5D ground + low obstacle stubs) + textured visual tiles
    (ground tiles and 3D structure tiles, see scripts/build_twin.py). Returns (env, actor)."""
    twin_dir = Path(twin_dir)
    mesh = load_glb_zup(twin_dir.parent / "ground.glb")
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    F = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    tp = gymapi.TriangleMeshParams()
    tp.nb_vertices, tp.nb_triangles = len(V), len(F)
    tp.static_friction = tp.dynamic_friction = 1.0
    tp.restitution = 0.0
    gym.add_triangle_mesh(sim, V.flatten(), F.flatten(), tp)

    o = gymapi.AssetOptions()
    o.fix_base_link = True
    o.disable_gravity = True
    vis = gym.load_asset(sim, str(twin_dir), "twin_visual.urdf", o)
    assert vis is not None, "twin visual failed to load"
    env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(*meta["size_xy"], 60), 1)
    actor = gym.create_actor(env, vis, gymapi.Transform(), "twin_visual", -1, 0xFFFF)
    set_baked_lighting(gym, sim)
    return env, actor


def robot_config(name):
    reg = json.loads((ROOT / "assets/robots/robots.json").read_text())
    if name not in reg:
        raise KeyError(f"robot '{name}' not prepared; run scripts/prepare_robots.py "
                       f"(available: {list(reg)})")
    return reg[name]


class WheeledRobot:
    """Skid-steer robot actor driven by (v, w) commands."""

    def __init__(self, gym, sim, env, name, xy, yaw, h, res):
        self.gym, self.env, self.cfg = gym, env, robot_config(name)
        ro = gymapi.AssetOptions()
        ro.default_dof_drive_mode = int(gymapi.DOF_MODE_VEL)
        ro.collapse_fixed_joints = False
        ro.density = 10.0   # links without <inertial> (decorations) must not add fake mass
        asset = gym.load_asset(sim, str(ROOT / "assets/robots"), self.cfg["urdf"], ro)
        dofs = gym.get_asset_dof_names(asset)
        self.is_left = np.array([n in self.cfg["left_wheels"] for n in dofs])
        self.is_right = np.array([n in self.cfg["right_wheels"] for n in dofs])
        t = gymapi.Transform()
        z = float(h[int(xy[1] / res), int(xy[0] / res)])
        t.p = gymapi.Vec3(xy[0], xy[1], z + self.cfg["wheel_radius"] * 2 + 0.15)
        t.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), yaw)
        self.actor = gym.create_actor(env, asset, t, name, 0, 1)   # filter=1: no self-collision
        self.mass = sum(b.mass for b in gym.get_actor_rigid_body_properties(env, self.actor))
        props = gym.get_actor_dof_properties(env, self.actor)
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.0)
        props["damping"].fill(max(20.0, self.mass * 3.0))
        # per-wheel torque limit ~ robot weight at the rim: enough to skid-turn, not to climb walls
        props["effort"].fill(self.mass * 9.81 * self.cfg["wheel_radius"] * 1.0)
        gym.set_actor_dof_properties(env, self.actor, props)
        self.base = gym.find_actor_rigid_body_index(env, self.actor, "base_link", gymapi.DOMAIN_ACTOR)

    def state(self):
        """(pos xyz, quat xyzw, yaw, forward speed, yaw rate)"""
        st = self.gym.get_actor_rigid_body_states(self.env, self.actor, gymapi.STATE_ALL)
        p, q = st["pose"]["p"][self.base], st["pose"]["r"][self.base]
        lv, av = st["vel"]["linear"][self.base], st["vel"]["angular"][self.base]
        pos = np.array([p["x"], p["y"], p["z"]])
        quat = np.array([q["x"], q["y"], q["z"], q["w"]])
        yaw = np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]),
                         1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        v = lv["x"] * np.cos(yaw) + lv["y"] * np.sin(yaw)
        return pos, quat, float(yaw), float(v), float(av["z"])

    def command(self, v, w):
        r, tr = self.cfg["wheel_radius"], self.cfg["track"]
        wl, wr = (v - w * tr / 2) / r, (v + w * tr / 2) / r
        tgt = np.where(self.is_left, wl, np.where(self.is_right, wr, 0.0)).astype(np.float32)
        self.gym.set_actor_dof_velocity_targets(self.env, self.actor, tgt)


class TopDownCamera:
    """Perspective camera looking straight down, north up / east right, with a simple
    world->pixel projection for drawing overlays."""

    def __init__(self, gym, env, center_xy, span_xy, ground_z, width=1080, height=1080, hfov=40.0):
        self.gym, self.env, self.w, self.h = gym, env, width, height
        cp = gymapi.CameraProperties()
        cp.width, cp.height, cp.horizontal_fov = width, height, hfov
        cp.near_plane, cp.far_plane = 1.0, 2000.0
        self.handle = gym.create_camera_sensor(env, cp)
        self.f = width / (2 * np.tan(np.radians(hfov) / 2))
        need = max(span_xy[0] / width, span_xy[1] / height)          # metres per pixel
        self.cx, self.cy = center_xy
        self.z = ground_z + need * self.f
        self.ground_z = ground_z
        t = gymapi.Transform()
        t.p = gymapi.Vec3(self.cx, self.cy, self.z)
        # camera axes (x fwd, y left, z up) -> world: fwd = -Z, left = -X, up = +Y
        t.r = gymapi.Quat(-0.5, 0.5, 0.5, 0.5)
        gym.set_camera_transform(self.handle, env, t)

    def project(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        z = xyz[:, 2] if xyz.shape[1] > 2 else np.full(len(xyz), self.ground_z)
        d = self.z - z
        u = self.w / 2 + self.f * (xyz[:, 0] - self.cx) / d
        v = self.h / 2 - self.f * (xyz[:, 1] - self.cy) / d
        return np.stack([u, v], 1).astype(np.int32)

    def image(self, sim):
        return self.gym.get_camera_image(sim, self.env, self.handle, gymapi.IMAGE_COLOR) \
            .reshape(self.h, self.w, 4)[..., :3].copy()


D455 = dict(width=1280, height=720, fps=30, hfov=87.0, depth_min=0.52, depth_max=20.0)


class RGBDCamera:
    """RealSense D455-like pinhole RGB-D camera rigidly attached to a robot link
    (ideal depth: no stereo noise / holes; depth saved as uint16 mm, 0 = out of range)."""

    def __init__(self, gym, env, robot, pitch_down_deg=0.0, spec=D455):
        self.gym, self.env, self.spec = gym, env, spec
        cp = gymapi.CameraProperties()
        cp.width, cp.height, cp.horizontal_fov = spec["width"], spec["height"], spec["hfov"]
        cp.near_plane, cp.far_plane = 0.05, 1500.0
        self.handle = gym.create_camera_sensor(env, cp)
        link = robot.cfg["camera_link"]
        body = gym.find_actor_rigid_body_handle(env, robot.actor, link)
        assert body >= 0, f"camera link {link} not found"
        lt = gymapi.Transform()
        lt.p = gymapi.Vec3(0.015, 0, 0)
        lt.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(pitch_down_deg))
        gym.attach_camera_to_body(self.handle, env, body, lt, gymapi.FOLLOW_TRANSFORM)
        self.body = gym.find_actor_rigid_body_index(env, robot.actor, link, gymapi.DOMAIN_ACTOR)
        self.robot = robot
        fx = spec["width"] / (2 * np.tan(np.radians(spec["hfov"]) / 2))
        self.K = [[fx, 0, spec["width"] / 2], [0, fx, spec["height"] / 2], [0, 0, 1]]

    def read(self, sim):
        """(rgb uint8 HxWx3, depth uint16 mm HxW, valid fraction)"""
        sp = self.spec
        rgb = self.gym.get_camera_image(sim, self.env, self.handle, gymapi.IMAGE_COLOR) \
            .reshape(sp["height"], sp["width"], 4)[..., :3].copy()
        d = -self.gym.get_camera_image(sim, self.env, self.handle, gymapi.IMAGE_DEPTH)
        ok = np.isfinite(d) & (d >= sp["depth_min"]) & (d <= sp["depth_max"])
        return rgb, np.where(ok, d * 1000.0, 0).astype(np.uint16), float(ok.mean())

    def pose(self):
        st = self.gym.get_actor_rigid_body_states(self.env, self.robot.actor, gymapi.STATE_POS)
        p, q = st["pose"]["p"][self.body], st["pose"]["r"][self.body]
        return [float(p["x"]), float(p["y"]), float(p["z"])], [float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])]
