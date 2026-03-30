#!/usr/bin/env python3
"""
Mesh face reduction (decimation) using trimesh quadric decimation.

Usage:
  python3 reduce_faces.py INPUT OUTPUT [--ratio 0.45]

Example:
  python3 reduce_faces.py input.STL output.STL --ratio 0.45
"""

import os
import argparse
import trimesh


def reduce_faces(model_path, output_path, ratio=0.2, scale=None):
    """Reduce mesh face count via quadric decimation.

    Args:
        model_path:  input mesh file (OBJ or STL)
        output_path: output mesh file
        ratio:       target face count as fraction of original
        scale:       optional vertex scale factor (e.g. 0.001 for mm->m)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    mesh = trimesh.load_mesh(model_path)
    original_faces = mesh.faces.shape[0]
    target_faces = int(original_faces * ratio)
    print(f"Original faces: {original_faces}")
    print(f"Target faces: {target_faces} ({ratio*100:.0f}%)")

    print("Decimating...")
    reduced = mesh.simplify_quadric_decimation(face_count=target_faces)
    print(f"Result faces: {reduced.faces.shape[0]}")

    if scale is not None:
        reduced.vertices *= scale
        print(f"Scaled vertices by {scale}")

    reduced.export(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mesh face reduction")
    parser.add_argument("input", help="Input mesh file (STL/OBJ)")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="Target face ratio (default: 0.2)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Vertex scale factor (e.g. 0.001 for mm->m)")
    args = parser.parse_args()

    reduce_faces(args.input, args.output, args.ratio, args.scale)


if __name__ == "__main__":
    main()
