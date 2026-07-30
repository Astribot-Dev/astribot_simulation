#!/usr/bin/env python3
"""
Batch convert GLB files to Collada DAE with multi-material support.

Each GLB contains two sub-meshes:
  - White shell (main_color ~[102,102,102]) -> mapped to white (248,249,251)
  - Black accent (main_color [0,0,0])       -> mapped to black (20,20,22)

Output DAE uses multiple <triangles> groups with separate materials
(same approach as Unitree Go2 DAE files), which SAPIEN/ManiSkill can render.

Usage:
  python3 glb_to_dae.py                        # convert all in color_glb/
  python3 glb_to_dae.py astribot_head_link_2   # convert one
"""

import os
import sys

import numpy as np
from lib.constants import BLACK_ACCENT, WHITE_SHELL
from lib.dae_writer import write_dae_flat
from lib.mesh_classify import load_glb_classified

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLB_DIR = os.path.join(BASE_DIR, "color_glb")
DAE_DIR = os.path.join(BASE_DIR, "dae_copy")


def convert_glb_to_dae(glb_path, dae_path):
    """Convert a single GLB file to DAE with multi-material triangle groups."""
    white_meshes, black_meshes = load_glb_classified(glb_path)

    # Build material groups
    material_groups = []

    def add_group(meshes, material_id, material_name, diffuse):
        if not meshes:
            return
        all_pos = []
        all_norm = []
        all_faces = []
        v_offset = 0
        for m in meshes:
            all_pos.append(np.array(m.vertices))
            norms = (
                m.vertex_normals
                if hasattr(m, "vertex_normals") and m.vertex_normals is not None
                else np.zeros_like(m.vertices)
            )
            all_norm.append(norms)
            all_faces.append(np.array(m.faces) + v_offset)
            v_offset += len(m.vertices)

        material_groups.append(
            {
                "material_id": material_id,
                "material_name": material_name,
                "diffuse": diffuse,
                "positions": np.vstack(all_pos),
                "normals": np.vstack(all_norm),
                "faces": np.vstack(all_faces),
            }
        )

    add_group(white_meshes, "white_shell", "white_shell", WHITE_SHELL)
    add_group(black_meshes, "black_accent", "black_accent", BLACK_ACCENT)

    if not material_groups:
        print("  WARNING: no geometry found")
        return False

    total_verts = sum(len(g["positions"]) for g in material_groups)
    total_faces = sum(len(g["faces"]) for g in material_groups)
    print(
        f"  Output: {total_verts} verts, {total_faces} faces, "
        f"{len(material_groups)} material(s)"
    )

    write_dae_flat(dae_path, material_groups)
    return True


def main():
    os.makedirs(DAE_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        names = sys.argv[1:]
        glb_files = []
        for name in names:
            if not name.endswith(".glb"):
                name = name + ".glb"
            path = os.path.join(GLB_DIR, name)
            if os.path.exists(path):
                glb_files.append(path)
            else:
                print(f"ERROR: {path} not found")
    else:
        glb_files = sorted(
            [os.path.join(GLB_DIR, f) for f in os.listdir(GLB_DIR) if f.endswith(".glb")]
        )

    if not glb_files:
        print("No GLB files found.")
        return

    print(f"Converting {len(glb_files)} GLB file(s) to DAE (multi-material)...\n")

    for glb_path in glb_files:
        name = os.path.splitext(os.path.basename(glb_path))[0]
        dae_path = os.path.join(DAE_DIR, f"{name}.dae")
        print(f"[{name}]")
        try:
            convert_glb_to_dae(glb_path, dae_path)
            size_mb = os.path.getsize(dae_path) / 1024 / 1024
            print(f"  Saved: {dae_path} ({size_mb:.1f} MB)\n")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ERROR: {e}\n")

    print("=" * 50)
    print(f"DAE files saved to: {DAE_DIR}")


if __name__ == "__main__":
    main()
