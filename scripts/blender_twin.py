"""Open a digital twin (assets/twin/<name>/<variant>) in Blender to inspect and tune textures and
the simulated D455 camera. Runs inside Blender (tested with 5.3), not in the conda envs:

    blender --python scripts/blender_twin.py -- --twin assets/twin/blue_wall/full \
        --run outputs/nav/run_000 --region 80 130 150 190

    # headless stills from the D455 pose of trajectory frames 0, 300, 600, ...
    blender -b --python scripts/blender_twin.py -- --twin assets/twin/blue_wall/full \
        --run outputs/nav/run_000 --render outputs/blender/run_000 --frames 0 3000 300

What it sets up (see docs/blender_tuning.md):
    * every g_* / s_* tile, in the same local metres (x east, y north, z up) as the simulator
    * textures linked to the tile PNGs on disk, so a re-export shows up after Image > Reload
    * one shared node group "TwinShade" with the knobs Gain / Gamma / Saturation / Lit / ShowSource
    * camera "D455" under the empty "d455_link", animated from <run>/trajectory.csv
      (no run yet: --at X Y YAW_DEG places it at LIMO camera height on the ground)
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

GROUND_Z = 0.015          # same lift as twin_visual.urdf (visual ground above the collision mesh)
STRUCT_TILE = 20.0        # structure tile size (m), s_<ix>_<iy>


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", required=True, help="assets/twin/<name>/<variant>")
    ap.add_argument("--run", help="outputs/nav/<run> (trajectory.csv + result.json) for the D455 camera")
    ap.add_argument("--at", type=float, nargs=3, action="append", metavar=("X", "Y", "YAW_DEG"), default=[],
                    help="without --run: put the D455 here (local m, yaw 0 = east, 90 = north) at "
                         "--cam-height above the ground; repeat for more viewpoints (frame 0, 1, ...)")
    ap.add_argument("--cam-height", type=float, default=0.28, help="D455 height above ground for --at (LIMO, measured in runs)")
    ap.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="only load tiles overlapping this local-metre box")
    ap.add_argument("--no-structure", action="store_true")
    ap.add_argument("--packed", action="store_true", help="keep the textures packed in the GLBs")
    ap.add_argument("--lit", type=float, default=0.0, help="0 = baked colours (as in Isaac Gym), 1 = sun + sky")
    ap.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    ap.add_argument("--render", help="output directory: render stills from the D455 camera and quit")
    ap.add_argument("--frames", type=int, nargs=3, metavar=("START", "STOP", "STEP"), default=[0, 1, 1])
    ap.add_argument("--depth", action="store_true", help="also write <frame>_depth.exr (metres, planar z)")
    ap.add_argument("--save", help="save the scene as this .blend")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- materials
def shade_group():
    """Shared shader: grade (linear RGB) -> mix(Emission, Principled). One set of knobs for all tiles."""
    ng = bpy.data.node_groups.new("TwinShade", "ShaderNodeTree")
    ng.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Source", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    def value(name, v):
        n = N.new("ShaderNodeValue"); n.name = n.label = name; n.outputs[0].default_value = v
        return n.outputs[0]

    def mix(a, b, fac, blend="MIX"):
        n = N.new("ShaderNodeMix"); n.data_type = "RGBA"; n.blend_type = blend
        n.clamp_factor = False; n.clamp_result = False
        L.new(fac, n.inputs[0]) if hasattr(fac, "node") else setattr(n.inputs[0], "default_value", fac)
        L.new(a, n.inputs[6]); L.new(b, n.inputs[7])
        return n.outputs[2]

    gain = N.new("ShaderNodeVectorMath"); gain.operation = "SCALE"
    L.new(gi.outputs["Color"], gain.inputs[0]); L.new(value("Gain", 1.0), gain.inputs[3])
    bw = N.new("ShaderNodeRGBToBW"); L.new(gain.outputs[0], bw.inputs[0])
    sat = mix(bw.outputs[0], gain.outputs[0], value("Saturation", 1.0))
    gam = N.new("ShaderNodeGamma"); L.new(sat, gam.inputs[0]); L.new(value("Gamma", 1.0), gam.inputs[1])
    col = mix(gam.outputs[0], gi.outputs["Source"], value("ShowSource", 0.0))
    em = N.new("ShaderNodeEmission"); L.new(col, em.inputs[0])
    pb = N.new("ShaderNodeBsdfPrincipled"); L.new(col, pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 1.0
    ms = N.new("ShaderNodeMixShader")
    L.new(value("Lit", 0.0), ms.inputs[0]); L.new(em.outputs[0], ms.inputs[1]); L.new(pb.outputs[0], ms.inputs[2])
    L.new(ms.outputs[0], go.inputs[0])
    return ng


def retarget_material(mat, group, twin, name, packed):
    nt = mat.node_tree
    img = next((n.image for n in nt.nodes if n.type == "TEX_IMAGE" and n.image), None)
    png, src_png = twin / f"{name}.png", twin / f"{name}_source.png"
    if not packed and png.exists():
        img = bpy.data.images.load(str(png), check_existing=True)
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; tex.name = "albedo"
    g = nt.nodes.new("ShaderNodeGroup"); g.node_tree = group
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs[0], g.inputs["Color"])
    if src_png.exists():
        st = nt.nodes.new("ShaderNodeTexImage"); st.name = "source"
        st.image = bpy.data.images.load(str(src_png), check_existing=True)
        st.interpolation = "Closest"
        nt.links.new(st.outputs[0], g.inputs["Source"])
    nt.links.new(g.outputs[0], out.inputs[0])
    mat.use_backface_culling = False


# --------------------------------------------------------------------------- tiles
def tile_bounds(name, size_xy, n_split):
    kind, a, b = re.match(r"([gs])_(\d+)_(\d+)$", name).groups()
    a, b = int(a), int(b)
    if kind == "s":
        return a * STRUCT_TILE, b * STRUCT_TILE, (a + 1) * STRUCT_TILE, (b + 1) * STRUCT_TILE
    sx, sy = size_xy[0] / n_split[0], size_xy[1] / n_split[1]
    return a * sx, b * sy, (a + 1) * sx, (b + 1) * sy


def load_tiles(twin, region, no_structure, packed):
    meta = json.loads((twin.parent / "ground.json").read_text())
    names = sorted(p.stem for p in twin.glob("[gs]_*.glb"))
    gs = [tuple(map(int, n.split("_")[1:])) for n in names if n[0] == "g"]
    n_split = (max(a for a, _ in gs) + 1, max(b for _, b in gs) + 1)
    group = shade_group()
    cols = {k: bpy.data.collections.new(k) for k in ("ground", "structure")}
    for c in cols.values():
        bpy.context.scene.collection.children.link(c)
    n = 0
    for name in names:
        if no_structure and name[0] == "s":
            continue
        b = tile_bounds(name, meta["size_xy"], n_split)
        if region and not (b[0] < region[2] and b[2] > region[0] and b[1] < region[3] and b[3] > region[1]):
            continue
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(twin / f"{name}.glb"))      # Y-up glTF -> Blender Z-up
        for ob in set(bpy.data.objects) - before:
            for c in list(ob.users_collection):
                c.objects.unlink(ob)
            cols["ground" if name[0] == "g" else "structure"].objects.link(ob)
            if ob.type != "MESH":
                continue
            ob.name = name
            if name[0] == "g":
                ob.location.z += GROUND_Z
            for m in ob.data.materials:
                retarget_material(m, group, twin, name, packed)
        n += 1
    print(f"[blender] {n} tiles from {twin}")
    return meta


# --------------------------------------------------------------------------- camera
def add_d455(run, at=(), cam_height=0.28, twin=None):
    res = json.loads((run / "result.json").read_text()) if run and (run / "result.json").exists() else {}
    d = res.get("d455", {})
    w, h, hfov = d.get("width", 1280), d.get("height", 720), d.get("hfov_deg", 87.0)
    pitch = math.radians(d.get("pitch_down_deg", 0.0))
    sc = bpy.context.scene
    link = bpy.data.objects.new("d455_link", None)          # link frame: x forward, y left, z up
    link.empty_display_type = "ARROWS"; link.empty_display_size = 0.3
    link.rotation_mode = "QUATERNION"
    cam = bpy.data.objects.new("D455", bpy.data.cameras.new("D455"))
    for o in (link, cam):
        sc.collection.objects.link(o)
    cam.parent = link
    # Blender camera (x right, y up, looks along -z) expressed in the link frame, after the same
    # local transform as umass_lidar/sim.py RGBDCamera: +0.015 m forward, pitch about +y (down)
    axes = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0)))
    cam.matrix_local = Matrix.Translation((0.015, 0, 0)) @ (Matrix.Rotation(pitch, 3, "Y") @ axes).to_4x4()
    cam.data.sensor_fit = "HORIZONTAL"; cam.data.angle_x = math.radians(hfov)
    cam.data.clip_start, cam.data.clip_end = 0.05, 1500.0
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.fps = res.get("fps", 30)
    rows = []
    if run and (run / "trajectory.csv").exists():
        rows = list(csv.DictReader(open(run / "trajectory.csv")))
        for r in rows:
            f = int(r["frame"])
            link.location = Vector((float(r["cam_x"]), float(r["cam_y"]), float(r["cam_z"])))
            link.rotation_quaternion = Quaternion((float(r["cam_qw"]), float(r["cam_qx"]), float(r["cam_qy"]),
                                                   float(r["cam_qz"])))
            link.keyframe_insert("location", frame=f); link.keyframe_insert("rotation_quaternion", frame=f)
        sc.frame_start, sc.frame_end = 0, len(rows) - 1
        sc.frame_set(0)
    elif at:
        import numpy as np
        meta = json.loads((twin.parent / "ground.json").read_text())
        hm = np.load(str(twin.parent / "ground_height.npy")); r = meta["mesh_res"]
        for f, (x, y, yaw) in enumerate(at):
            z = float(hm[min(int(y / r), hm.shape[0] - 1), min(int(x / r), hm.shape[1] - 1)])
            link.location = Vector((x, y, z + cam_height))
            link.rotation_quaternion = Quaternion((0, 0, 1), math.radians(yaw))
            link.keyframe_insert("location", frame=f); link.keyframe_insert("rotation_quaternion", frame=f)
        rows = list(at)
        sc.frame_start, sc.frame_end = 0, len(at) - 1
        sc.frame_set(0)
    else:
        link.location = (0, 0, 1.0)
    print(f"[blender] D455 {w}x{h} hfov {hfov} deg, {len(rows)} trajectory frames")


def setup_scene(a, meta):
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        if o.name in ("Cube", "Light", "Camera"):
            bpy.data.objects.remove(o)
    for eng in (("CYCLES",) if a.engine == "cycles" else ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            pass
    if a.engine == "cycles":
        sc.cycles.samples = 32
    sc.view_settings.view_transform = "Standard"       # AgX / Filmic would change the baked colours
    sc.view_settings.look = "None"
    sc.unit_settings.system = "METRIC"
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.76, 0.77, 0.80, 1.0)     # overcast sky, as assets sky.glb
    bg.inputs[1].default_value = 1.0
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(150))
    sc.collection.objects.link(sun)
    bpy.data.node_groups["TwinShade"].nodes["Lit"].outputs[0].default_value = a.lit
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type == "VIEW_3D":
            sp = area.spaces.active
            sp.clip_end = 3000.0
            sp.shading.type = "MATERIAL"


def render(a):
    sc = bpy.context.scene
    out = Path(a.render).resolve(); out.mkdir(parents=True, exist_ok=True)
    ims, vl = sc.render.image_settings, bpy.context.view_layer
    if a.depth:
        # depth as an image: every surface emits its camera-space z (planar depth in metres, the
        # same definition as the Isaac Gym depth image); alpha = 0 where nothing was hit
        dm = bpy.data.materials.new("DepthOverride"); dm.use_nodes = True
        nt = dm.node_tree; nt.nodes.clear()
        cd, em, om = (nt.nodes.new(t) for t in ("ShaderNodeCameraData", "ShaderNodeEmission",
                                                "ShaderNodeOutputMaterial"))
        nt.links.new(cd.outputs["View Z Depth"], em.inputs[0]); nt.links.new(em.outputs[0], om.inputs[0])
        dm.use_backface_culling = False
    for f in range(a.frames[0], min(a.frames[1], sc.frame_end + 1), a.frames[2]):
        sc.frame_set(f)
        ims.file_format = "PNG"
        sc.render.filepath = str(out / f"{f:06d}.png")
        bpy.ops.render.render(write_still=True)
        if a.depth:
            samples = sc.eevee.taa_render_samples
            sc.eevee.taa_render_samples = 1                 # no anti-aliasing: it would mix depths
            vl.material_override, sc.render.film_transparent = dm, True
            ims.file_format, ims.color_mode, ims.color_depth = "OPEN_EXR", "RGBA", "32"
            sc.render.filepath = str(out / f"{f:06d}_depth.exr")
            bpy.ops.render.render(write_still=True)
            vl.material_override, sc.render.film_transparent = None, False
            sc.eevee.taa_render_samples = samples
            ims.file_format, ims.color_mode, ims.color_depth = "PNG", "RGB", "8"
        print(f"[blender] rendered frame {f} -> {out}", flush=True)


def main():
    a = parse()
    twin = Path(a.twin).resolve()
    meta = load_tiles(twin, a.region, a.no_structure, a.packed)
    add_d455(Path(a.run).resolve() if a.run else None, a.at, a.cam_height, twin)
    setup_scene(a, meta)
    if a.save:
        Path(a.save).resolve().parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.save).resolve()))
    if a.render:
        render(a)


main()
