#!/usr/bin/env python3
"""
GLB -> OBJ + PNG texture pipeline using xatlas (no Blender needed).

Reads GLB files with multiple sub-meshes (white shell + black accent),
performs UV unwrapping with xatlas, and bakes per-face material colors
onto a dense texture map.

Usage:
  python3 glb2uv_xatlas.py                                # process all GLBs
  python3 glb2uv_xatlas.py --single astribot_torso_link_4 # process one
  python3 glb2uv_xatlas.py --tex-size 4096 --dilate 32    # high quality

Options:
  --input-dir DIR    GLB input directory (default: color_glb/)
  --output-dir DIR   OBJ output directory (default: output_obj/)
  --texture-dir DIR  PNG texture output directory (default: output_texture/)
  --tex-size N       Texture resolution (default: 2048)
  --dilate N         Dilation iterations for seam filling (default: 24)
  --single NAME      Process only one GLB file (basename without .glb)
"""

import os
import sys
import argparse
import time
import numpy as np
import trimesh
import xatlas
from PIL import Image, ImageDraw

# Add parent directory so lib/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.constants import COLOR_WHITE_SHELL, COLOR_BLACK_VISOR, DARK_THRESHOLD
from lib.texture_utils import rasterize_face_colors, dilate_texture
from lib.obj_io import write_obj, parse_obj
from lib.mesh_classify import load_glb_with_face_colors


# GLB filename -> OBJ/texture filename mapping (when names differ)
NAME_MAP = {
    "astribot_torso_base": "astribot_torso_base_link",
}


def uv_unwrap(verts, faces):
    """UV unwrap mesh using xatlas. Returns new verts, faces, and UVs."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    atlas.generate()
    vmapping, new_faces, uvs = atlas[0]

    new_verts = verts[vmapping]

    mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    new_normals = np.array(mesh.vertex_normals, dtype=np.float32)

    return new_verts, new_faces, new_normals, uvs, vmapping


def match_faces_by_centroid(glb_verts, glb_faces, obj_verts, obj_faces):
    """Match OBJ faces to GLB faces by nearest face centroid.

    Returns an array of GLB face indices, one per OBJ face.
    """
    from scipy.spatial import KDTree

    glb_centroids = np.zeros((len(glb_faces), 3), dtype=np.float64)
    for i, f in enumerate(glb_faces):
        glb_centroids[i] = glb_verts[f].mean(axis=0)

    obj_centroids = np.zeros((len(obj_faces), 3), dtype=np.float64)
    for i, (fv, _) in enumerate(obj_faces):
        obj_centroids[i] = obj_verts[fv].mean(axis=0)

    tree = KDTree(glb_centroids)
    dists, indices = tree.query(obj_centroids)

    print(f"    Face matching: max dist={dists.max():.6f}, "
          f"mean dist={dists.mean():.6f}")
    if dists.max() > 0.01:
        n_bad = (dists > 0.01).sum()
        print(f"    WARNING: {n_bad} faces have match distance > 0.01")

    return indices


def process_glb_with_existing_uv(glb_path, obj_path, tex_output,
                                  tex_size=2048, dilate_iters=24):
    """Bake GLB material colors onto existing OBJ UV layout."""
    name = os.path.splitext(os.path.basename(glb_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing (existing UV mode): {name}")
    print(f"{'='*60}")

    # Step 1: Load GLB with material colors
    print(f"  Loading GLB: {os.path.basename(glb_path)}")
    glb_verts, glb_faces, glb_face_colors = load_glb_with_face_colors(
        glb_path, COLOR_WHITE_SHELL, COLOR_BLACK_VISOR, DARK_THRESHOLD)
    print(f"  GLB merged: {len(glb_verts)} verts, {len(glb_faces)} faces")

    # Step 2: Load existing OBJ with UV
    print(f"  Loading OBJ UV: {os.path.basename(obj_path)}")
    obj_verts, obj_uvs, _obj_normals, obj_faces = parse_obj(obj_path)
    obj_verts = obj_verts.astype(np.float32)
    obj_uvs = obj_uvs.astype(np.float32)
    print(f"  OBJ: {len(obj_verts)} verts, {len(obj_uvs)} UVs, "
          f"{len(obj_faces)} faces")

    if len(obj_uvs) == 0:
        print(f"  ERROR: No UV coordinates found in {obj_path}")
        return False

    # Step 3: Match OBJ faces to GLB faces by centroid
    # GLB is Y-up, OBJ is Z-up: rotate 90 deg around X
    glb_verts_matched = glb_verts.copy()
    y_old = glb_verts_matched[:, 1].copy()
    glb_verts_matched[:, 1] = -glb_verts_matched[:, 2]
    glb_verts_matched[:, 2] = y_old
    print(f"  Matching faces by centroid (GLB Y-up -> Z-up rotated)...")
    t0 = time.time()
    glb_indices = match_faces_by_centroid(
        glb_verts_matched, glb_faces, obj_verts, obj_faces)
    print(f"  Matching done in {time.time() - t0:.1f}s")

    # Step 4: Rasterize GLB colors onto OBJ UV layout
    print(f"  Rasterizing {tex_size}x{tex_size} texture...")
    t0 = time.time()
    tex = Image.new('RGB', (tex_size, tex_size), tuple(COLOR_WHITE_SHELL))
    mask = Image.new('L', (tex_size, tex_size), 0)
    draw = ImageDraw.Draw(tex)
    mask_draw = ImageDraw.Draw(mask)
    s = tex_size - 1

    for i, (face_v, face_vt) in enumerate(obj_faces):
        if len(face_vt) < 3:
            continue
        glb_idx = glb_indices[i]
        c = glb_face_colors[glb_idx]
        color = (int(c[0]), int(c[1]), int(c[2]))

        uv_coords = []
        for vt_idx in face_vt[:3]:
            if vt_idx < len(obj_uvs):
                uv = obj_uvs[vt_idx]
                uv_coords.append((int(uv[0] * s), int((1 - uv[1]) * s)))
            else:
                uv_coords.append((0, 0))

        draw.polygon(uv_coords, fill=color)
        mask_draw.polygon(uv_coords, fill=255)

    print(f"  Rasterize done in {time.time() - t0:.1f}s")

    # Step 5: Dilate seams
    print(f"  Dilating seams ({dilate_iters} iterations)...")
    t0 = time.time()
    tex = dilate_texture(tex, mask, iterations=dilate_iters)
    print(f"  Dilate done in {time.time() - t0:.1f}s")

    # Step 6: Save
    os.makedirs(os.path.dirname(tex_output) or '.', exist_ok=True)
    tex.save(tex_output)
    print(f"  Saved texture: {tex_output}")
    return True


def process_glb(glb_path, obj_output, tex_output, tex_size=2048, dilate_iters=24):
    """Full pipeline: GLB -> UV unwrap -> texture -> OBJ."""
    name = os.path.splitext(os.path.basename(glb_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # Step 1: Load GLB with material colors
    print(f"  Loading {os.path.basename(glb_path)}...")
    verts, faces, face_colors = load_glb_with_face_colors(
        glb_path, COLOR_WHITE_SHELL, COLOR_BLACK_VISOR, DARK_THRESHOLD)
    print(f"  Merged: {len(verts)} verts, {len(faces)} faces")

    # GLB is Y-up, MuJoCo/OBJ expects Z-up: rotate 90 deg around X
    y_old = verts[:, 1].copy()
    verts[:, 1] = -verts[:, 2]
    verts[:, 2] = y_old

    n_black = np.all(face_colors < 30, axis=1).sum()
    n_white = len(face_colors) - n_black
    print(f"  Colors: {n_white} white faces, {n_black} black faces")

    # Step 2: UV unwrap with xatlas
    print(f"  UV unwrapping with xatlas...")
    t0 = time.time()
    new_verts, new_faces, new_normals, uvs, vmapping = uv_unwrap(verts, faces)
    print(f"  UV done in {time.time() - t0:.1f}s: "
          f"{len(new_verts)} verts, {len(new_faces)} faces")

    # Step 3: Remap face colors (xatlas preserves face order)
    new_face_colors = face_colors

    # Step 4: Rasterize texture
    print(f"  Rasterizing {tex_size}x{tex_size} texture...")
    t0 = time.time()
    tex, mask = rasterize_face_colors(new_faces, uvs, new_face_colors, tex_size)
    print(f"  Rasterize done in {time.time() - t0:.1f}s")

    # Step 5: Dilate to fill seam gaps
    print(f"  Dilating seams ({dilate_iters} iterations)...")
    t0 = time.time()
    tex = dilate_texture(tex, mask, iterations=dilate_iters)
    print(f"  Dilate done in {time.time() - t0:.1f}s")

    # Step 6: Save
    os.makedirs(os.path.dirname(tex_output) or '.', exist_ok=True)
    tex.save(tex_output)
    print(f"  Saved texture: {tex_output}")

    os.makedirs(os.path.dirname(obj_output) or '.', exist_ok=True)
    write_obj(obj_output, new_verts, new_faces, new_normals, uvs)
    print(f"  Saved OBJ: {obj_output}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="GLB -> OBJ + PNG texture using xatlas")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--texture-dir", default=None)
    parser.add_argument("--tex-size", type=int, default=2048)
    parser.add_argument("--dilate", type=int, default=24)
    parser.add_argument("--single", default=None)
    parser.add_argument("--existing-uv", default=None,
                        help="Directory with existing OBJ files (use their UV "
                             "layout instead of xatlas). Only outputs textures.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(base_dir, "..", "color_glb")
    output_dir = args.output_dir or os.path.join(base_dir, "output_obj")
    texture_dir = args.texture_dir or os.path.join(base_dir, "output_texture")

    os.makedirs(texture_dir, exist_ok=True)

    if args.single:
        name = args.single
        if not name.endswith('.glb'):
            name += '.glb'
        glb_files = [os.path.join(input_dir, name)]
        if not os.path.exists(glb_files[0]):
            print(f"ERROR: {glb_files[0]} not found")
            return
    else:
        glb_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith('.glb')
        ])

    if not glb_files:
        print(f"No GLB files found in {input_dir}")
        return

    print(f"Processing {len(glb_files)} GLB file(s)")
    print(f"  Texture size: {args.tex_size}x{args.tex_size}")
    print(f"  Dilate iters: {args.dilate}")
    if args.existing_uv:
        print(f"  Mode: existing UV from {args.existing_uv}")
    else:
        print(f"  Mode: xatlas UV unwrap")
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for glb_path in glb_files:
        glb_name = os.path.splitext(os.path.basename(glb_path))[0]
        out_name = NAME_MAP.get(glb_name, glb_name)
        tex_path = os.path.join(texture_dir, f"{out_name}.png")

        try:
            if args.existing_uv:
                obj_path = os.path.join(args.existing_uv, f"{out_name}.obj")
                if not os.path.exists(obj_path):
                    candidates = [f for f in os.listdir(args.existing_uv)
                                  if f.endswith('.obj') and glb_name in f]
                    if candidates:
                        obj_path = os.path.join(args.existing_uv, candidates[0])
                    else:
                        print(f"  SKIP {glb_name}: no matching OBJ in "
                              f"{args.existing_uv}")
                        results.append((out_name, False))
                        continue
                ok = process_glb_with_existing_uv(
                    glb_path, obj_path, tex_path,
                    tex_size=args.tex_size, dilate_iters=args.dilate)
            else:
                obj_path = os.path.join(output_dir, f"{out_name}.obj")
                ok = process_glb(glb_path, obj_path, tex_path,
                                 tex_size=args.tex_size, dilate_iters=args.dilate)
            results.append((out_name, ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((out_name, False))

    print(f"\n{'='*60}")
    print("Results:")
    for out_name, ok in results:
        print(f"  {out_name}: {'OK' if ok else 'FAILED'}")

    proj_tex = os.path.expanduser(
        "~/ros2/simu/astribot_simulation/astribot_descriptions/"
        "urdf/astribot_s1_urdf/meshes/textures/")
    print(f"\nTo apply to simulation:")
    print(f"  cp {texture_dir}/*.png {proj_tex}")
    if not args.existing_uv:
        proj_obj = os.path.expanduser(
            "~/ros2/simu/astribot_simulation/astribot_descriptions/"
            "urdf/astribot_s1_urdf/meshes/obj/")
        print(f"  cp {output_dir}/*.obj {proj_obj}")


if __name__ == "__main__":
    main()
