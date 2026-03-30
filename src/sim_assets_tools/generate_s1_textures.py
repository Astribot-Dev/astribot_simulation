#!/usr/bin/env python3
"""
Generate textures and OBJ meshes for Astribot S1 multi-material links.
Converts STL -> OBJ with UV coordinates and generates texture PNGs.

Usage: python3 generate_s1_textures.py
"""

import os
import numpy as np
import trimesh
import xatlas

from lib.constants import (COLOR_WHITE_SHELL, COLOR_BLACK_VISOR,
                           COLOR_DARK_MECHANISM, COLOR_CHASSIS_TOP,
                           COLOR_CHASSIS_SIDE)
from lib.texture_utils import rasterize_face_colors, dilate_texture
from lib.obj_io import write_obj

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MESH_BASE = os.path.join(PROJECT_ROOT, "astribot_descriptions/urdf/astribot_s1_urdf/meshes")
TEXTURE_DIR = os.path.join(MESH_BASE, "textures")
OBJ_DIR = os.path.join(MESH_BASE, "obj")

TEX_SIZE = 2048


def uv_unwrap(mesh):
    """UV unwrap mesh using xatlas."""
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces, normals)
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]

    new_verts = verts[vmapping]
    new_normals = normals[vmapping]
    return new_verts, new_faces, new_normals, uvs


def compute_face_props(verts, faces):
    """Compute face centroids and normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    e1 = v1 - v0
    e2 = v2 - v0
    fn = np.cross(e1, e2)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    norms[norms == 0] = 1
    fn = fn / norms
    return centroids, fn


def process_mesh(stl_path, mesh_name, color_func, tex_size=TEX_SIZE,
                 bg_color=COLOR_WHITE_SHELL):
    """Full pipeline: STL -> UV unwrap -> texture -> OBJ."""
    print(f"  Loading {os.path.basename(stl_path)}...")
    mesh = trimesh.load(stl_path)

    print(f"  UV unwrapping ({len(mesh.faces)} faces)...")
    new_verts, new_faces, new_normals, uvs = uv_unwrap(mesh)
    print(f"  UV done: {len(new_verts)} verts, {len(new_faces)} faces")

    centroids, face_normals = compute_face_props(new_verts, new_faces)
    face_colors = color_func(centroids, face_normals)

    print(f"  Rasterizing texture ({tex_size}x{tex_size})...")
    tex, mask = rasterize_face_colors(new_faces, uvs, face_colors, tex_size,
                                       bg_color)
    tex = dilate_texture(tex, mask, iterations=4)

    os.makedirs(OBJ_DIR, exist_ok=True)
    os.makedirs(TEXTURE_DIR, exist_ok=True)

    tex_path = os.path.join(TEXTURE_DIR, f"{mesh_name}.png")
    tex.save(tex_path)
    print(f"  Saved texture: {tex_path}")

    obj_path = os.path.join(OBJ_DIR, f"{mesh_name}.obj")
    write_obj(obj_path, new_verts, new_faces, new_normals, uvs)
    print(f"  Saved OBJ: {obj_path}")

    return tex, new_verts, new_faces, uvs, centroids, face_normals


# ==================== Material zone functions ====================

def head_link_2_colors(centroids, normals):
    """Head: white shell + black visor on front face only."""
    n = len(centroids)
    colors = np.tile(COLOR_WHITE_SHELL, (n, 1)).astype(np.float64)
    is_visor = (normals[:, 1] < -0.5) & (centroids[:, 1] < -0.15)
    colors[is_visor] = COLOR_BLACK_VISOR
    print(f"    Visor faces: {is_visor.sum()} / {n} ({100*is_visor.mean():.1f}%)")
    return colors


def torso_link_4_colors(centroids, normals):
    """Chest/trunk: white body."""
    n = len(centroids)
    return np.tile(COLOR_WHITE_SHELL, (n, 1)).astype(np.float64)


def torso_base_link_colors(centroids, normals):
    """Chassis: white top + dark gray sides/bottom."""
    n = len(centroids)
    colors = np.tile(COLOR_CHASSIS_TOP, (n, 1)).astype(np.float64)
    is_bottom = (normals[:, 2] < -0.3) | (centroids[:, 2] < -0.02)
    is_lower_side = (centroids[:, 2] < 0.03) & (np.abs(normals[:, 2]) < 0.3)
    is_dark = is_bottom | is_lower_side
    colors[is_dark] = COLOR_CHASSIS_SIDE
    print(f"    Dark chassis faces: {is_dark.sum()} / {n} ({100*is_dark.mean():.1f}%)")
    return colors


def torso_link_2_colors(centroids, normals):
    """Waist section: mostly white with limited dark mechanism area."""
    n = len(centroids)
    colors = np.tile(COLOR_WHITE_SHELL, (n, 1)).astype(np.float64)
    is_mechanism = (
        (centroids[:, 1] > 0.08) &
        (np.abs(centroids[:, 2]) < 0.04) &
        (centroids[:, 0] > -0.35) &
        (centroids[:, 0] < -0.1)
    )
    colors[is_mechanism] = COLOR_DARK_MECHANISM
    print(f"    Mechanism faces: {is_mechanism.sum()} / {n} ({100*is_mechanism.mean():.1f}%)")
    return colors


# ==================== Main ====================

def main():
    os.makedirs(OBJ_DIR, exist_ok=True)
    os.makedirs(TEXTURE_DIR, exist_ok=True)

    # --- head_link_2 ---
    print("\n[1/4] Processing head_link_2 (visor)...")
    process_mesh(
        os.path.join(MESH_BASE, "s1_head/astribot_head_link_2.STL"),
        "astribot_head_link_2",
        head_link_2_colors
    )

    # --- torso_link_4 (chest) ---
    print("\n[2/4] Processing torso_link_4 (chest)...")
    process_mesh(
        os.path.join(MESH_BASE, "s1_torso/astribot_torso_link_4.STL"),
        "astribot_torso_link_4",
        torso_link_4_colors
    )
    print("  TIP: Use add_logo_to_texture.py to overlay logo on the texture.")

    # --- torso_base_link (chassis) ---
    print("\n[3/4] Processing torso_base_link (chassis)...")
    process_mesh(
        os.path.join(MESH_BASE, "s1_torso/astribot_torso_base_link.STL"),
        "astribot_torso_base_link",
        torso_base_link_colors,
        bg_color=COLOR_CHASSIS_TOP
    )

    # --- torso_link_2 (waist mechanism) ---
    print("\n[4/4] Processing torso_link_2 (waist)...")
    process_mesh(
        os.path.join(MESH_BASE, "s1_torso/astribot_torso_link_2.STL"),
        "astribot_torso_link_2",
        torso_link_2_colors
    )

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
