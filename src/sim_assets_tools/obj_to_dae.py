#!/usr/bin/env python3
"""
Convert MeshLab-colored OBJ to Collada DAE with embedded vertex colors.
DAE files with vertex colors work directly in URDF (RViz2 / Gazebo).

Usage:
  python3 obj_to_dae.py                          # process all in color/
  python3 obj_to_dae.py astribot_head_link_2     # process one

Input:  color/*.obj  (MeshLab vertex-colored OBJ: "v x y z r g b")
Output: dae/*.dae    (Collada with embedded vertex colors)

In URDF, replace:
  <mesh filename="package://xxx/meshes/s1_head/astribot_head_link_2.STL"/>
with:
  <mesh filename="package://xxx/meshes/dae/astribot_head_link_2.dae"/>
and remove the <material> tag (color is in DAE).
"""

import os
import re
import sys

from lib.obj_io import parse_colored_obj, find_color_obj
from lib.dae_writer import write_dae_vertex_color

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLOR_DIR = os.path.join(BASE_DIR, "color")
DAE_DIR = os.path.join(BASE_DIR, "dae")


def convert_one(mesh_name):
    """Convert one colored OBJ to DAE."""
    color_obj = find_color_obj(mesh_name, COLOR_DIR)

    if color_obj is None:
        print(f"  ERROR: Colored OBJ not found for '{mesh_name}' in {COLOR_DIR}")
        print(f"    Expected: {mesh_name}.obj or {mesh_name}_color.obj")
        return False

    print(f"  Reading: {color_obj}")
    positions, colors, normals, faces, has_colors = parse_colored_obj(color_obj)
    print(f"    {len(positions)} verts, {len(faces)} faces, has_colors={has_colors}")

    if not has_colors:
        print("  WARNING: No vertex colors in OBJ. Using default white.")

    os.makedirs(DAE_DIR, exist_ok=True)
    out_path = os.path.join(DAE_DIR, f"{mesh_name}.dae")
    print(f"  Writing DAE: {out_path}")
    write_dae_vertex_color(out_path, positions, colors, normals, faces)
    print("  Done!")
    return True


def main():
    os.makedirs(DAE_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        mesh_names = sys.argv[1:]
    else:
        # Auto-detect: extract clean mesh names from color/ filenames
        mesh_names = []
        if os.path.exists(COLOR_DIR):
            for f in sorted(os.listdir(COLOR_DIR)):
                if f.endswith('.obj'):
                    name = f[:-4]
                    name = re.sub(r'\s*_color$', '', name)
                    if name and name not in mesh_names:
                        mesh_names.append(name)

    if not mesh_names:
        print("No OBJ files found in color/ directory.")
        return

    print(f"Converting {len(mesh_names)} mesh(es) to DAE...\n")

    for name in mesh_names:
        print(f"[{name}]")
        convert_one(name)
        print()

    print("=" * 50)
    print(f"DAE files saved to: {DAE_DIR}")
    print("\nIn URDF, replace STL references with DAE:")
    print(f'  <mesh filename="package://xxx/meshes/dae/{mesh_names[0]}.dae"/>')


if __name__ == "__main__":
    main()
