"""Convert open-source wheeled-robot xacro descriptions into plain URDFs Isaac Gym can load,
and mount an Intel RealSense D455 placeholder link (`d455_link`) for camera attachment.

Sources (cloned into third_party/ if missing):
    jackal : https://github.com/jackal/jackal   (jackal_description, BSD)
    husky  : https://github.com/husky/husky     (husky_description, BSD)
    limo   : AgileX limo_description; pass --limo-desc <path to limo_description>

    python scripts/prepare_robots.py                       # jackal + husky
    python scripts/prepare_robots.py --limo-desc ~/Documents/limo/agilex_open_class/limo/limo_description

Output: assets/robots/<name>/<name>.urdf + meshes/ and assets/robots/robots.json
(wheel radius, track width, left/right wheel joints, camera mount pose).

No ROS needed: `$(find pkg)` is resolved by text substitution, includes of packages we
do not have (sensor drivers, gazebo plugins) are dropped, `package://` becomes relative.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"
OUT = ROOT / "assets/robots"

# D455 body: 124 x 26 x 29 mm. Camera frame here is ROS body convention: x fwd, y left, z up.
D455_SIZE = (0.026, 0.124, 0.029)

ROBOTS = {
    "jackal": dict(repo="https://github.com/jackal/jackal.git", branch="noetic-devel",
                   pkg="jackal_description", xacro="urdf/jackal.urdf.xacro",
                   wheel_radius=0.098, track=0.37559, cam_parent="base_link",
                   cam_xyz=(0.20, 0.0, 0.25), max_speed=2.0),
    "husky": dict(repo="https://github.com/husky/husky.git", branch="noetic-devel",
                  pkg="husky_description", xacro="urdf/husky.urdf.xacro",
                  wheel_radius=0.1651, track=0.5708, cam_parent="base_link",
                  cam_xyz=(0.40, 0.0, 0.32), max_speed=1.0),
    "limo": dict(repo=None, pkg="limo_description", xacro="urdf/limo_four_diff.xacro",
                 wheel_radius=0.045, track=0.172, cam_parent="base_link",
                 cam_xyz=(0.10, 0.0, 0.13), max_speed=1.0),
}


def preprocess(src_dir: Path, pkg: str, tmp: Path) -> Path:
    """Copy the package and make its xacro files ROS-free."""
    dst = tmp / pkg
    shutil.copytree(src_dir, dst)
    for f in list(dst.rglob("*.xacro")) + list(dst.rglob("*.urdf")) + list(dst.rglob("*.gazebo")):
        s = f.read_text()
        s = s.replace(f"$(find {pkg})", str(dst))
        # drop includes that need other ROS packages
        s = re.sub(r"<xacro:include[^>]*\$\(find [^)]+\)[^>]*/>", "", s)
        s = s.replace("file://", "")
        f.write_text(s)
    return dst


def convert(name: str, cfg: dict, pkg_dir: Path):
    out = OUT / name
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        dst = preprocess(pkg_dir, cfg["pkg"], Path(t))
        xml = subprocess.run([sys.executable, "-c", "import xacro; xacro.main()", str(dst / cfg["xacro"])],
                             capture_output=True, text=True, cwd=dst / "urdf")
        if xml.returncode:
            raise RuntimeError(xml.stderr[-2000:])
        xml = xml.stdout
        root = ET.fromstring(xml)
        # strip gazebo / transmission blocks, relocate meshes
        for tag in ("gazebo", "transmission"):
            for e in root.findall(tag):
                root.remove(e)
        for m in root.iter("mesh"):
            fn = m.get("filename")
            p = Path(fn.replace(f"package://{cfg['pkg']}", str(dst)))
            if not p.exists():
                raise FileNotFoundError(fn)
            if p.suffix.lower() == ".dae":
                # Isaac Gym's assimp re-orients Z_UP COLLADA files (robot parts end up rotated).
                # Bake to OBJ with raw coordinates, which is how ROS tools interpret them.
                import trimesh
                tm = trimesh.load(p, force="scene")
                geo = tm.dump(concatenate=True)
                name_out = p.stem + "_dae.obj"
                geo.export(out / "meshes" / name_out, mtl_name=p.stem + "_dae.mtl")
            else:
                name_out = p.name
                shutil.copy(p, out / "meshes" / name_out)
            m.set("filename", f"meshes/{name_out}")

        # D455 mount
        link = ET.SubElement(root, "link", name="d455_link")
        vis = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(vis, "geometry"), "box", size=" ".join(map(str, D455_SIZE)))
        ET.SubElement(ET.SubElement(vis, "material", name="d455_grey"), "color", rgba="0.75 0.75 0.78 1")
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "mass", value="0.103")
        ET.SubElement(inert, "inertia", ixx="1e-4", iyy="1e-5", izz="1e-4", ixy="0", ixz="0", iyz="0")
        j = ET.SubElement(root, "joint", name="d455_joint", type="fixed")
        ET.SubElement(j, "parent", link=cfg["cam_parent"])
        ET.SubElement(j, "child", link="d455_link")
        ET.SubElement(j, "origin", xyz=" ".join(map(str, cfg["cam_xyz"])), rpy="0 0 0")

        ET.indent(root)
        (out / f"{name}.urdf").write_text(ET.tostring(root, encoding="unicode"))

    wheels = [j.get("name") for j in root.findall("joint")
              if j.get("type") == "continuous" and "wheel" in j.get("name")]
    info = dict(urdf=f"{name}/{name}.urdf", wheel_radius=cfg["wheel_radius"], track=cfg["track"],
                left_wheels=[w for w in wheels if "left" in w],
                right_wheels=[w for w in wheels if "right" in w],
                camera_link="d455_link", camera_xyz=cfg["cam_xyz"], max_speed=cfg["max_speed"])
    assert info["left_wheels"] and info["right_wheels"], wheels
    print(f"[{name}] links={len(root.findall('link'))} wheels={wheels}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limo-desc", default=None, help="path to AgileX limo_description package")
    args = ap.parse_args()
    reg_path = OUT / "robots.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    for name, cfg in ROBOTS.items():
        if cfg["repo"]:
            repo_dir = TP / name
            if not repo_dir.exists():
                TP.mkdir(exist_ok=True)
                subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", cfg["branch"],
                                cfg["repo"], str(repo_dir)], check=True)
            pkg_dir = repo_dir / cfg["pkg"]
        elif args.limo_desc:
            pkg_dir = Path(args.limo_desc).expanduser()
        else:
            continue
        reg[name] = convert(name, cfg, pkg_dir)
    reg_path.write_text(json.dumps(reg, indent=2))
    print(f"wrote {reg_path}")


if __name__ == "__main__":
    main()
