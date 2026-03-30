"""GLB sub-mesh classification by material luminance."""

import numpy as np
import trimesh

from .constants import DARK_THRESHOLD


def classify_by_luminance(rgba_color, threshold=DARK_THRESHOLD):
    """Classify a material color as 'white' or 'black' by luminance.

    Args:
        rgba_color: RGBA uint8 array/tuple (at least 3 elements)
        threshold:  uint8 luminance cutoff (below = black)

    Returns:
        'white' or 'black'
    """
    luminance = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
    return 'black' if luminance < threshold else 'white'


def load_glb_classified(glb_path, dark_threshold=DARK_THRESHOLD):
    """Load a GLB file and classify sub-meshes into white/black groups.

    Args:
        glb_path:       path to GLB file
        dark_threshold: uint8 luminance cutoff

    Returns:
        (white_meshes, black_meshes) — lists of trimesh.Trimesh objects
    """
    scene = trimesh.load(glb_path)
    white_meshes = []
    black_meshes = []

    if not isinstance(scene, trimesh.Scene):
        # Single mesh — treat as white
        print(f"  Single mesh: {len(scene.vertices)} verts, "
              f"{len(scene.faces)} faces -> white")
        white_meshes.append(scene)
        return white_meshes, black_meshes

    for name, geom in scene.geometry.items():
        if hasattr(geom.visual, 'material') and geom.visual.material is not None:
            main_color = geom.visual.material.main_color  # RGBA uint8
        elif hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            main_color = geom.visual.vertex_colors.mean(axis=0).astype(np.uint8)
        else:
            main_color = np.array([102, 102, 102, 255], dtype=np.uint8)

        label = classify_by_luminance(main_color, dark_threshold)
        if label == 'black':
            black_meshes.append(geom)
        else:
            white_meshes.append(geom)

        print(f"  {name}: {len(geom.vertices)} verts, {len(geom.faces)} faces, "
              f"color={main_color[:3]} -> {label}")

    return white_meshes, black_meshes


def load_glb_with_face_colors(glb_path, white_color, black_color,
                               dark_threshold=DARK_THRESHOLD):
    """Load GLB and return merged vertices, faces, and per-face colors.

    Args:
        glb_path:       path to GLB file
        white_color:    RGB array/tuple for white faces (e.g. uint8)
        black_color:    RGB array/tuple for black faces
        dark_threshold: uint8 luminance cutoff

    Returns:
        (verts, faces, face_colors) — numpy arrays
    """
    scene = trimesh.load(glb_path)

    all_verts = []
    all_faces = []
    all_face_colors = []
    vert_offset = 0

    if isinstance(scene, trimesh.Scene):
        geometries = list(scene.geometry.items())
    else:
        geometries = [("mesh", scene)]

    white_color = np.array(white_color, dtype=np.uint8)
    black_color = np.array(black_color, dtype=np.uint8)

    for name, geom in geometries:
        mat_color = np.array([102, 102, 102, 255], dtype=np.uint8)
        if hasattr(geom.visual, 'material') and geom.visual.material is not None:
            try:
                mat_color = np.array(geom.visual.material.main_color, dtype=np.uint8)
            except Exception:
                pass

        label = classify_by_luminance(mat_color, dark_threshold)
        face_color = black_color if label == 'black' else white_color

        n_faces = len(geom.faces)
        print(f"    {name}: {len(geom.vertices)} verts, {n_faces} faces, "
              f"color={mat_color[:3]} -> {label}")

        all_verts.append(np.array(geom.vertices))
        all_faces.append(np.array(geom.faces) + vert_offset)
        all_face_colors.append(np.tile(face_color, (n_faces, 1)))
        vert_offset += len(geom.vertices)

    verts = np.vstack(all_verts).astype(np.float32)
    faces = np.vstack(all_faces).astype(np.uint32)
    face_colors = np.vstack(all_face_colors).astype(np.uint8)

    return verts, faces, face_colors
