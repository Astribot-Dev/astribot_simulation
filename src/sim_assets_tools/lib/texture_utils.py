"""Texture rasterization, dilation, and barycentric utilities."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .constants import COLOR_WHITE_SHELL


def rasterize_face_colors(faces, uvs, face_colors, tex_size, bg_color=COLOR_WHITE_SHELL):
    """Rasterize per-face colors onto UV-mapped texture image.

    Args:
        faces:       (F, 3) array of vertex indices
        uvs:         (V, 2) array of UV coordinates
        face_colors: (F, 3) array of RGB colors (uint8 or float)
        tex_size:    texture resolution (square)
        bg_color:    background RGB tuple (uint8)

    Returns:
        (tex, mask) where tex is PIL Image and mask is PIL Image (L mode)
    """
    tex = Image.new("RGB", (tex_size, tex_size), tuple(bg_color))
    mask = Image.new("L", (tex_size, tex_size), 0)
    draw = ImageDraw.Draw(tex)
    mask_draw = ImageDraw.Draw(mask)
    s = tex_size - 1

    for i in range(len(faces)):
        f = faces[i]
        uv0 = uvs[f[0]]
        uv1 = uvs[f[1]]
        uv2 = uvs[f[2]]

        px0 = (int(uv0[0] * s), int((1 - uv0[1]) * s))
        px1 = (int(uv1[0] * s), int((1 - uv1[1]) * s))
        px2 = (int(uv2[0] * s), int((1 - uv2[1]) * s))

        c = face_colors[i]
        color = (int(c[0]), int(c[1]), int(c[2]))
        draw.polygon([px0, px1, px2], fill=color)
        mask_draw.polygon([px0, px1, px2], fill=255)

    return tex, mask


def dilate_texture(tex, mask, iterations=24):
    """Dilate painted regions outward to fill UV seam gaps.

    Args:
        tex:        PIL Image (RGB)
        mask:       PIL Image (L mode, 255=painted, 0=unpainted)
        iterations: number of dilation passes

    Returns:
        PIL Image with dilated regions filled
    """
    tex_arr = np.array(tex)
    mask_arr = np.array(mask)

    for _ in range(iterations):
        dilated_mask = np.array(Image.fromarray(mask_arr).filter(ImageFilter.MaxFilter(3)))
        border = (dilated_mask > 0) & (mask_arr == 0)
        if not border.any():
            break
        blurred = np.array(Image.fromarray(tex_arr).filter(ImageFilter.BoxBlur(1)))
        tex_arr[border] = blurred[border]
        mask_arr = dilated_mask

    return Image.fromarray(tex_arr)


def barycentric(px, py, x0, y0, x1, y1, x2, y2):
    """Compute barycentric coordinates of point (px,py) in triangle (v0,v1,v2).

    Returns (w0, w1, w2) weights. Point is inside if all >= 0.
    """
    v0x = x1 - x0
    v0y = y1 - y0
    v1x = x2 - x0
    v1y = y2 - y0
    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return -1, -1, -1
    inv = 1.0 / denom
    dx = px - x0
    dy = py - y0
    d20 = dx * v0x + dy * v0y
    d21 = dx * v1x + dy * v1y
    w1 = (d11 * d20 - d01 * d21) * inv
    w2 = (d00 * d21 - d01 * d20) * inv
    w0 = 1.0 - w1 - w2
    return w0, w1, w2
