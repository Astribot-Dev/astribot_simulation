#!/usr/bin/env python3
"""
Convert STL files to single-color Collada DAE.

Uses the same multi-material DAE format as glb_to_dae.py
(proven compatible with SAPIEN/ManiSkill, matching Unitree Go2 approach).

Usage:
  python3 stl_to_dae.py                           # convert all STL in gripper_stl/ (black)
  python3 stl_to_dae.py astribot_gripper_L1_Link  # convert one
  python3 stl_to_dae.py --color white              # use white shell color instead
"""

import os
import argparse
import numpy as np
import trimesh

from lib.constants import WHITE_SHELL, BLACK_ACCENT
from lib.dae_writer import write_dae_flat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STL_DIR = os.path.join(BASE_DIR, "gripper_stl")
DAE_DIR = os.path.join(BASE_DIR, "dae_copy")

COLOR_MAP = {
    'black': ('black_accent', BLACK_ACCENT),
    'white': ('white_shell', WHITE_SHELL),
}


def convert_stl_to_dae(stl_path, dae_path, color_name='black'):
    """Convert a single STL to a colored DAE."""
    mesh = trimesh.load(stl_path)

    positions = np.array(mesh.vertices)
    normals = np.array(mesh.vertex_normals)
    faces = np.array(mesh.faces)

    print(f"  {len(positions)} verts, {len(faces)} faces")

    material_id, diffuse = COLOR_MAP[color_name]
    material_groups = [{
        'material_id': material_id,
        'material_name': material_id,
        'diffuse': diffuse,
        'positions': positions,
        'normals': normals,
        'faces': faces,
    }]

    write_dae_flat(dae_path, material_groups)
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert STL to colored DAE")
    parser.add_argument("names", nargs='*', help="STL file basenames (without .STL)")
    parser.add_argument("--color", choices=list(COLOR_MAP.keys()), default='black',
                        help="Material color (default: black)")
    parser.add_argument("--input-dir", default=STL_DIR,
                        help="STL input directory")
    parser.add_argument("--output-dir", default=DAE_DIR,
                        help="DAE output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.names:
        stl_files = []
        for name in args.names:
            if not name.upper().endswith('.STL'):
                name = name + '.STL'
            path = os.path.join(args.input_dir, name)
            if os.path.exists(path):
                stl_files.append(path)
            else:
                print(f"ERROR: {path} not found")
    else:
        stl_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith('.stl')
        ])

    if not stl_files:
        print("No STL files found.")
        return

    print(f"Converting {len(stl_files)} STL file(s) to {args.color} DAE...\n")

    for stl_path in stl_files:
        name = os.path.splitext(os.path.basename(stl_path))[0]
        dae_path = os.path.join(args.output_dir, f"{name}.dae")
        print(f"[{name}]")
        try:
            convert_stl_to_dae(stl_path, dae_path, args.color)
            size_kb = os.path.getsize(dae_path) / 1024
            print(f"  Saved: {dae_path} ({size_kb:.0f} KB)\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}\n")

    print("=" * 50)
    print(f"DAE files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
