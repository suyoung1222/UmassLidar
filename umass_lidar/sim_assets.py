"""Asset glue that must work in both envs (no isaacgym / laspy imports)."""
from __future__ import annotations

from pathlib import Path

# Isaac Gym also renders the static collision trimesh (flat grey); the textured ground copy is
# lifted slightly so it always wins the depth test.
VISUAL_Z_OFFSET = 0.015


def write_visual_urdf(twin_dir, ground, structure, name="twin_visual"):
    """One fixed link whose visuals are all tile GLBs (GLB is Y-up -> rotate back to Z-up)."""
    twin_dir = Path(twin_dir)
    vis = []
    for n in ground:
        vis.append(f'<visual><origin xyz="0 0 {VISUAL_Z_OFFSET}" rpy="1.5707963 0 0"/>'
                   f'<geometry><mesh filename="{n}.glb"/></geometry></visual>')
    for n in structure:
        vis.append('<visual><origin xyz="0 0 0" rpy="1.5707963 0 0"/>'
                   f'<geometry><mesh filename="{n}.glb"/></geometry></visual>')
    urdf = twin_dir / f"{name}.urdf"
    urdf.write_text(f'<robot name="{name}"><link name="world_visual">\n' + "\n".join(vis) +
                    "\n</link></robot>\n")
    return urdf
