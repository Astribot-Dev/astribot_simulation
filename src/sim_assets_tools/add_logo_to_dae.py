#!/usr/bin/env python3
"""
Add Astribot logo to astribot_torso_link_4 DAE using UV texture mapping.

Instead of per-face material assignment (which causes severe aliasing),
this script UV-projects a texture image onto the front chest area.
Logo quality is determined by texture resolution, not mesh density.

Usage:
  python3 add_logo_to_dae.py
"""

import os

import numpy as np
from lib.constants import (
    BLACK_ACCENT,
    DARK_THRESHOLD,
    FRONT_THRESHOLD,
    LOGO_Y_CENTER,
    LOGO_Z_CENTER,
    WHITE_SHELL,
    ZONE_HALF,
)
from lib.dae_writer import write_dae_textured
from lib.mesh_classify import load_glb_classified
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLB_PATH = os.path.join(BASE_DIR, "color_glb", "astribot_torso_link_4.glb")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
OUTPUT_DAE = os.path.join(BASE_DIR, "dae_copy", "astribot_torso_link_4.dae")
TEXTURE_NAME = "logo_texture.png"
TEXTURE_PATH = os.path.join(BASE_DIR, "dae_copy", TEXTURE_NAME)

# Logo rendering size within the zone (physical meters)
LOGO_SIZE = 0.065

# Texture resolution
TEX_SIZE = 1024


def create_logo_texture():
    """Create texture image: logo centered on white background, sized for UV zone."""
    logo = Image.open(LOGO_PATH).convert("RGBA")
    print(f"  Source logo: {logo.size[0]}x{logo.size[1]}")

    bg = tuple(int(c * 255) for c in WHITE_SHELL)
    texture = Image.new("RGB", (TEX_SIZE, TEX_SIZE), bg)

    zone_span = ZONE_HALF * 2
    logo_frac = LOGO_SIZE / zone_span
    logo_px = int(TEX_SIZE * logo_frac)
    logo_resized = logo.resize((logo_px, logo_px), Image.LANCZOS)

    offset = (TEX_SIZE - logo_px) // 2
    texture.paste(logo_resized, (offset, offset), logo_resized)

    # Flip vertically: Collada V=0 is bottom, but PNG row 0 is top
    texture = texture.transpose(Image.FLIP_TOP_BOTTOM)

    os.makedirs(os.path.dirname(TEXTURE_PATH), exist_ok=True)
    texture.save(TEXTURE_PATH)
    print(f"  Texture: {TEX_SIZE}x{TEX_SIZE}, logo {logo_px}px ({logo_frac*100:.0f}%)")


def extract_submesh(verts, normals, faces):
    """Extract submesh with only referenced vertices, remap face indices."""
    used = np.unique(faces.flatten())
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], normals[used], remap[faces]


def main():
    print("Creating logo texture...")
    create_logo_texture()

    print("Loading GLB...")
    white_meshes, black_meshes = load_glb_classified(GLB_PATH, DARK_THRESHOLD)

    # Merge white meshes
    w_verts_list, w_norms_list, w_faces_list = [], [], []
    off = 0
    for m in white_meshes:
        w_verts_list.append(np.array(m.vertices))
        w_norms_list.append(np.array(m.vertex_normals))
        w_faces_list.append(np.array(m.faces) + off)
        off += len(m.vertices)
    w_verts = np.vstack(w_verts_list)
    w_norms = np.vstack(w_norms_list)
    w_faces = np.vstack(w_faces_list)

    # Face normals & centroids (GLB coords: X=front, Y=up, Z=left-right)
    v0 = w_verts[w_faces[:, 0]]
    v1 = w_verts[w_faces[:, 1]]
    v2 = w_verts[w_faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    fn = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(fn, axis=1, keepdims=True)
    mag[mag == 0] = 1
    fn = fn / mag

    # Logo zone: front-facing faces within the UV projection area
    # GLB Y axis = vertical, GLB Z axis = horizontal
    in_zone = (
        (fn[:, 0] > FRONT_THRESHOLD)
        & (centroids[:, 1] > LOGO_Z_CENTER - ZONE_HALF)
        & (centroids[:, 1] < LOGO_Z_CENTER + ZONE_HALF)
        & (np.abs(centroids[:, 2] - LOGO_Y_CENTER) < ZONE_HALF)
    )
    print(f"  Logo zone faces: {in_zone.sum()}")

    # --- Build material groups ---
    flat_groups = []

    # White shell (outside logo zone)
    white_mask = ~in_zone
    if white_mask.any():
        sv, sn, sf = extract_submesh(w_verts, w_norms, w_faces[white_mask])
        flat_groups.append(
            {
                "material_id": "white_shell",
                "material_name": "white_shell",
                "diffuse": WHITE_SHELL,
                "positions": sv,
                "normals": sn,
                "faces": sf,
            }
        )

    # Black accent
    if black_meshes:
        bp, bn, bf = [], [], []
        off = 0
        for m in black_meshes:
            bp.append(np.array(m.vertices))
            bn.append(np.array(m.vertex_normals))
            bf.append(np.array(m.faces) + off)
            off += len(m.vertices)
        flat_groups.append(
            {
                "material_id": "black_accent",
                "material_name": "black_accent",
                "diffuse": BLACK_ACCENT,
                "positions": np.vstack(bp),
                "normals": np.vstack(bn),
                "faces": np.vstack(bf),
            }
        )

    # Texture-mapped logo zone
    sv, sn, sf = extract_submesh(w_verts, w_norms, w_faces[in_zone])
    zone_span = ZONE_HALF * 2
    uvs = np.zeros((len(sv), 2))
    uvs[:, 0] = 0.5 + (sv[:, 2] - LOGO_Y_CENTER) / zone_span
    uvs[:, 1] = 0.5 + (sv[:, 1] - LOGO_Z_CENTER) / zone_span

    tex_group = {
        "material_id": "logo_zone",
        "material_name": "logo_zone",
        "positions": sv,
        "normals": sn,
        "faces": sf,
        "uvs": uvs,
    }

    # --- Write DAE ---
    os.makedirs(os.path.dirname(OUTPUT_DAE), exist_ok=True)
    write_dae_textured(OUTPUT_DAE, flat_groups, tex_group, TEXTURE_NAME)

    size_mb = os.path.getsize(OUTPUT_DAE) / 1024 / 1024
    print(f"\nSaved: {OUTPUT_DAE} ({size_mb:.1f} MB)")
    print(f"Texture: {TEXTURE_PATH}")
    print(f"Note: {TEXTURE_NAME} must be in the same directory as the DAE file.")


if __name__ == "__main__":
    main()
