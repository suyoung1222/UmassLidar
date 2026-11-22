"""[Legacy / airborne data] Crop a region out of the big Airborne.las into a compact .npz and
optionally mesh it as a 2.5D terrain GLB (umass_lidar/mesh.py; DSM / DTM / hybrid height modes).
The main pipeline for ground sessions is scripts/build_twin.py.

Region is given either as a UTM bbox or as center + square size (metres):
    python scripts/crop_las.py --las ~/Desktop/Airborne.las \
        --bbox 703349.19 4696065.5 703525.0 4696225.0 --out data/b1_las.npz
    python scripts/crop_las.py --las ~/Desktop/Airborne.las \
        --center 703437 4696145 --size 200 --out data/site1.npz --glb site1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umass_lidar.io import crop_las, save_npz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--las", required=True)
ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
ap.add_argument("--center", type=float, nargs=2, metavar=("CX", "CY"))
ap.add_argument("--size", type=float, nargs="+", default=[200.0], help="square side, or SX SY (m)")
ap.add_argument("--out", required=True)
ap.add_argument("--glb", default=None, metavar="NAME", help="also write assets/terrain/NAME.glb (2.5D)")
ap.add_argument("--mode", default="auto", choices=["auto", "dsm", "dtm", "hybrid"])
ap.add_argument("--mesh-res", type=float, default=0.25)
ap.add_argument("--tex-res", type=float, default=0.1)
args = ap.parse_args()

if (args.bbox is None) == (args.center is None):
    ap.error("give exactly one of --bbox or --center")
if args.bbox is None:
    sx, sy = args.size[0], args.size[-1]
    cx, cy = args.center
    args.bbox = [cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2]
print("bbox:", args.bbox)

cloud = crop_las(args.las, tuple(args.bbox))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
save_npz(args.out, cloud)
print(f"saved {len(cloud['xyz'])} points -> {args.out}")
if args.glb:
    from umass_lidar.mesh import build_terrain
    glb, meta, mesh = build_terrain(cloud, "assets/terrain", args.glb, mesh_res=args.mesh_res,
                                    tex_res=args.tex_res, mode=args.mode)
    print(f"mode={meta.mode} size={meta.size_xy} F={len(mesh.faces)} -> {glb}")
