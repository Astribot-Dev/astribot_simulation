#!/usr/bin/env python3
"""
Process MuJoCo scanned object XML files.

Takes a model.xml in each object directory and splits it into:
  - {name}_asset.xml  (for <include> in MuJoCo asset section)
  - {name}_model.xml  (for <include> in MuJoCo worldbody)

Also generates all_asset.xml and all_model.xml that aggregate all objects.

Usage:
  python3 process_mujoco_xml.py                    # process all in default path
  python3 process_mujoco_xml.py --path /some/dir/  # custom path
"""

import argparse
import os


def process_single_object(path, object_name):
    """Split model.xml into asset and model XML files for one object."""
    if object_name.endswith(".xml"):
        return

    model_xml_path = os.path.join(path, object_name, "model.xml")
    if not os.path.exists(model_xml_path):
        print(f"  SKIP {object_name}: no model.xml")
        return

    with open(model_xml_path, "r") as f:
        model_xml = f.read()

    # Generate asset XML
    asset_path = os.path.join(path, object_name, f"{object_name}_asset.xml")
    asset_xml = model_xml.split("<asset>")[1].split("</asset>")[0]
    asset_xml = asset_xml.replace('name="', 'name="' + object_name + "_")
    asset_xml = asset_xml.replace(
        'file="', 'file="worldbody/mujoco_scanned_objects/models/' + object_name + "/"
    )
    asset_xml = asset_xml.replace('texture="', 'texture="' + object_name + "_texture")
    asset_xml = asset_xml.replace(
        'texture="' + object_name + "_texturetexture", 'texture="' + object_name + "_texture"
    )

    with open(asset_path, "w") as f:
        f.write("<mujocoinclude>\n")
        f.write("<asset>\n")
        f.write(asset_xml)
        f.write("</asset>\n")
        f.write("</mujocoinclude>\n")

    # Generate model XML
    model_out_path = os.path.join(path, object_name, f"{object_name}_model.xml")
    worldbody_xml = model_xml.split("<worldbody>")[1].split("</worldbody>")[0]
    worldbody_xml = worldbody_xml.replace('name="', 'name="' + object_name + "_")
    worldbody_xml = worldbody_xml.replace('material="', 'material="' + object_name + "_")
    worldbody_xml = worldbody_xml.replace('mesh="', 'mesh="' + object_name + "_")
    worldbody_xml = worldbody_xml.replace('type="mesh"', 'type="mesh" mass="0.001"')
    worldbody_xml = worldbody_xml.replace(
        "<geom", '<geom friction="0.9" solimp="0.95 0.99 0.001" solref="0.001 1" priority="1"'
    )

    with open(model_out_path, "w") as f:
        f.write("<worldbody>\n")
        f.write(worldbody_xml)
        f.write("</worldbody>\n")

    print(f"  {object_name}: asset + model XML generated")


def process_all_objects(path):
    """Process all object directories."""
    for name in sorted(os.listdir(path)):
        obj_dir = os.path.join(path, name)
        if os.path.isdir(obj_dir):
            process_single_object(path, name)


def calculate_obj_volume(obj_filename):
    """Estimate volume from OBJ vertex positions."""
    vertices = []
    with open(obj_filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) == 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    volume = 0.0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        x1, y1, z1 = vertices[i]
        x2, y2, z2 = vertices[j]
        volume += x1 * y2 * z1 - x2 * y1 * z2
    return abs(volume / 6.0)


def generate_aggregate_xml(path, object_index_start=9, object_index_end=15):
    """Generate all_asset.xml and all_model.xml aggregating selected objects."""
    directory_list = [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and "Shoe" not in d
    ]

    # Sort by volume
    volume_list = []
    for name in directory_list:
        volume = 0
        obj_dir = os.path.join(path, name)
        for fn in os.listdir(obj_dir):
            if fn.endswith(".obj"):
                volume += calculate_obj_volume(os.path.join(obj_dir, fn))
        volume_list.append(volume)

    directory_list = [x for _, x in sorted(zip(volume_list, directory_list))]
    directory_list = directory_list[object_index_start:object_index_end]

    # all_asset.xml
    parent_dir = os.path.dirname(path.rstrip("/"))
    asset_path = os.path.join(parent_dir, "all_asset.xml")
    with open(asset_path, "w") as f:
        f.write("<mujocoinclude>\n")
        for name in directory_list:
            f.write(
                f'<include file="worldbody/mujoco_scanned_objects/models/'
                f'{name}/{name}_asset.xml"/>\n'
            )
        f.write("</mujocoinclude>\n")

    # all_model.xml
    model_path = os.path.join(parent_dir, "all_model.xml")
    distance = 0.3
    with open(model_path, "w") as f:
        f.write("<worldbody>\n")
        for index, name in enumerate(directory_list):
            first_digit = index // 100
            second_digit = (index - first_digit * 100) // 10
            third_digit = index - first_digit * 100 - second_digit * 10

            x = second_digit * distance + 0.5
            z = 1.2 if third_digit % 2 == 1 else 0.8
            y = third_digit * 0.12 - 0.5

            f.write(f'<body name="{name}" pos="{x} {y} {z}" euler="0 0 0">\n')
            f.write(
                f'<include file="worldbody/mujoco_scanned_objects/models/'
                f'{name}/{name}_model.xml"></include>\n'
            )
            f.write("<freejoint/>\n")
            f.write("</body>\n")
        f.write("</worldbody>\n")

    print(f"Generated: {asset_path}")
    print(f"Generated: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Process MuJoCo scanned object XML files")
    parser.add_argument("--path", default=None, help="Path to scanned objects directory")
    parser.add_argument("--start", type=int, default=9, help="Object index start (default: 9)")
    parser.add_argument("--end", type=int, default=15, help="Object index end (default: 15)")
    args = parser.parse_args()

    path = args.path or (
        "/home/robot/astribot_ws/src/astribot_robotics/"
        "simulation_platform/environment/worldbody/"
        "mujoco_scanned_objects/models/"
    )

    process_all_objects(path)
    generate_aggregate_xml(path, args.start, args.end)


if __name__ == "__main__":
    main()
