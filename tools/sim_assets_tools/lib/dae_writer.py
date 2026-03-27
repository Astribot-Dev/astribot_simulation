"""Collada DAE file writers for different material modes."""

import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree


def write_dae_flat(filepath, material_groups):
    """Write Collada DAE with flat-color materials and triangle groups.

    material_groups: list of dicts, each with:
      - 'material_id': str, unique material identifier
      - 'material_name': str, display name
      - 'diffuse': tuple/array (3,), RGB 0-1
      - 'positions': array (N, 3)
      - 'normals': array (N, 3)
      - 'faces': array (M, 3), indices into positions
    """
    collada = Element('COLLADA', {
        'xmlns': 'http://www.collada.org/2005/11/COLLADASchema',
        'version': '1.4.1'
    })

    # Asset
    asset = SubElement(collada, 'asset')
    contributor = SubElement(asset, 'contributor')
    SubElement(contributor, 'authoring_tool').text = 'mesh processing toolkit'
    SubElement(asset, 'up_axis').text = 'Z_UP'
    SubElement(asset, 'unit', {'meter': '1.0', 'name': 'meter'})

    # --- Library effects & materials ---
    lib_effects = SubElement(collada, 'library_effects')
    lib_mats = SubElement(collada, 'library_materials')

    for grp in material_groups:
        mid = grp['material_id']
        diffuse = grp['diffuse']

        effect = SubElement(lib_effects, 'effect', {'id': f'{mid}-effect'})
        profile = SubElement(effect, 'profile_COMMON')
        technique = SubElement(profile, 'technique', {'sid': 'common'})
        lambert = SubElement(technique, 'lambert')
        emission = SubElement(lambert, 'emission')
        SubElement(emission, 'color', {'sid': 'emission'}).text = '0 0 0 1'
        diff_el = SubElement(lambert, 'diffuse')
        SubElement(diff_el, 'color', {'sid': 'diffuse'}).text = (
            f'{diffuse[0]:.7f} {diffuse[1]:.7f} {diffuse[2]:.7f} 1'
        )

        mat = SubElement(lib_mats, 'material', {
            'id': f'{mid}-material', 'name': grp['material_name']
        })
        SubElement(mat, 'instance_effect', {'url': f'#{mid}-effect'})

    # --- Library geometries ---
    lib_geom = SubElement(collada, 'library_geometries')
    geometry = SubElement(lib_geom, 'geometry', {'id': 'mesh0', 'name': 'mesh0'})
    mesh = SubElement(geometry, 'mesh')

    # Merge all positions and normals, track offsets
    all_positions = []
    all_normals = []
    group_offsets = []
    offset = 0
    for grp in material_groups:
        group_offsets.append(offset)
        all_positions.append(grp['positions'])
        all_normals.append(grp['normals'])
        offset += len(grp['positions'])

    positions = np.vstack(all_positions)
    normals = np.vstack(all_normals)
    n_verts = len(positions)

    # Positions source
    pos_source = SubElement(mesh, 'source', {'id': 'mesh0-positions'})
    SubElement(pos_source, 'float_array', {
        'id': 'mesh0-positions-array', 'count': str(n_verts * 3)
    }).text = ' '.join(f'{v:.8f}' for v in positions.flatten())
    pos_tech = SubElement(pos_source, 'technique_common')
    pos_acc = SubElement(pos_tech, 'accessor', {
        'source': '#mesh0-positions-array', 'count': str(n_verts), 'stride': '3'
    })
    for p in 'XYZ':
        SubElement(pos_acc, 'param', {'name': p, 'type': 'float'})

    # Normals source
    norm_source = SubElement(mesh, 'source', {'id': 'mesh0-normals'})
    SubElement(norm_source, 'float_array', {
        'id': 'mesh0-normals-array', 'count': str(n_verts * 3)
    }).text = ' '.join(f'{n:.8f}' for n in normals.flatten())
    norm_tech = SubElement(norm_source, 'technique_common')
    norm_acc = SubElement(norm_tech, 'accessor', {
        'source': '#mesh0-normals-array', 'count': str(n_verts), 'stride': '3'
    })
    for p in 'XYZ':
        SubElement(norm_acc, 'param', {'name': p, 'type': 'float'})

    # Vertices
    vertices = SubElement(mesh, 'vertices', {'id': 'mesh0-vertices'})
    SubElement(vertices, 'input', {
        'semantic': 'POSITION', 'source': '#mesh0-positions'
    })

    # Triangles — one group per material
    for i, grp in enumerate(material_groups):
        mid = grp['material_id']
        faces = grp['faces'] + group_offsets[i]

        triangles = SubElement(mesh, 'triangles', {
            'count': str(len(faces)), 'material': f'{mid}-material'
        })
        SubElement(triangles, 'input', {
            'semantic': 'VERTEX', 'source': '#mesh0-vertices', 'offset': '0'
        })
        SubElement(triangles, 'input', {
            'semantic': 'NORMAL', 'source': '#mesh0-normals', 'offset': '1'
        })

        p_data = []
        for face in faces:
            for vi in face:
                p_data.append(str(vi))
                p_data.append(str(vi))  # normal index = vertex index
        SubElement(triangles, 'p').text = ' '.join(p_data)

    # --- Library visual scenes ---
    lib_scenes = SubElement(collada, 'library_visual_scenes')
    vis_scene = SubElement(lib_scenes, 'visual_scene', {
        'id': 'Scene', 'name': 'Scene'
    })
    node = SubElement(vis_scene, 'node', {
        'id': 'node0', 'name': 'mesh0', 'type': 'NODE'
    })
    inst_geom = SubElement(node, 'instance_geometry', {'url': '#mesh0'})
    bind_mat = SubElement(inst_geom, 'bind_material')
    tech_common = SubElement(bind_mat, 'technique_common')
    for grp in material_groups:
        mid = grp['material_id']
        SubElement(tech_common, 'instance_material', {
            'symbol': f'{mid}-material', 'target': f'#{mid}-material'
        })

    # Scene
    SubElement(
        SubElement(collada, 'scene'),
        'instance_visual_scene', {'url': '#Scene'}
    )

    tree = ElementTree(collada)
    tree.write(filepath, encoding='utf-8', xml_declaration=True)


def write_dae_vertex_color(filepath, positions, colors, normals, faces):
    """Write Collada DAE with per-vertex RGBA colors.

    positions: (N, 3)
    colors:    (N, 3) RGB 0-1
    normals:   (K, 3) or None
    faces:     list of (vert_indices[], normal_indices[]) tuples
    """
    n_verts = len(positions)
    n_faces = len(faces)

    if normals is None or len(normals) == 0:
        normals = np.zeros_like(positions)
        normals[:, 2] = 1.0

    collada = Element('COLLADA', {
        'xmlns': 'http://www.collada.org/2005/11/COLLADASchema',
        'version': '1.4.1'
    })

    # Asset
    asset = SubElement(collada, 'asset')
    SubElement(asset, 'up_axis').text = 'Z_UP'
    SubElement(asset, 'unit', {'meter': '1.0', 'name': 'meter'})

    # Library effects
    lib_effects = SubElement(collada, 'library_effects')
    effect = SubElement(lib_effects, 'effect', {'id': 'effect0'})
    profile = SubElement(effect, 'profile_COMMON')
    technique = SubElement(profile, 'technique', {'sid': 'common'})
    lambert = SubElement(technique, 'lambert')
    diffuse = SubElement(lambert, 'diffuse')
    SubElement(diffuse, 'color').text = '1 1 1 1'

    # Library materials
    lib_mats = SubElement(collada, 'library_materials')
    mat = SubElement(lib_mats, 'material', {'id': 'material0', 'name': 'material0'})
    SubElement(mat, 'instance_effect', {'url': '#effect0'})

    # Library geometries
    lib_geom = SubElement(collada, 'library_geometries')
    geometry = SubElement(lib_geom, 'geometry', {'id': 'mesh0', 'name': 'mesh0'})
    mesh = SubElement(geometry, 'mesh')

    # Positions source
    pos_source = SubElement(mesh, 'source', {'id': 'mesh0-positions'})
    SubElement(pos_source, 'float_array', {
        'id': 'mesh0-positions-array', 'count': str(n_verts * 3)
    }).text = ' '.join(f'{v:.8f}' for v in positions.flatten())
    pos_tech = SubElement(pos_source, 'technique_common')
    pos_acc = SubElement(pos_tech, 'accessor', {
        'source': '#mesh0-positions-array', 'count': str(n_verts), 'stride': '3'
    })
    for p in 'XYZ':
        SubElement(pos_acc, 'param', {'name': p, 'type': 'float'})

    # Normals source
    n_normals = len(normals)
    norm_source = SubElement(mesh, 'source', {'id': 'mesh0-normals'})
    SubElement(norm_source, 'float_array', {
        'id': 'mesh0-normals-array', 'count': str(n_normals * 3)
    }).text = ' '.join(f'{n:.8f}' for n in normals.flatten())
    norm_tech = SubElement(norm_source, 'technique_common')
    norm_acc = SubElement(norm_tech, 'accessor', {
        'source': '#mesh0-normals-array', 'count': str(n_normals), 'stride': '3'
    })
    for p in 'XYZ':
        SubElement(norm_acc, 'param', {'name': p, 'type': 'float'})

    # Colors source (RGBA)
    col_source = SubElement(mesh, 'source', {'id': 'mesh0-colors'})
    rgba = np.column_stack([colors, np.ones(n_verts)])
    SubElement(col_source, 'float_array', {
        'id': 'mesh0-colors-array', 'count': str(n_verts * 4)
    }).text = ' '.join(f'{c:.6f}' for c in rgba.flatten())
    col_tech = SubElement(col_source, 'technique_common')
    col_acc = SubElement(col_tech, 'accessor', {
        'source': '#mesh0-colors-array', 'count': str(n_verts), 'stride': '4'
    })
    for p in 'RGBA':
        SubElement(col_acc, 'param', {'name': p, 'type': 'float'})

    # Vertices
    vertices = SubElement(mesh, 'vertices', {'id': 'mesh0-vertices'})
    SubElement(vertices, 'input', {
        'semantic': 'POSITION', 'source': '#mesh0-positions'
    })

    # Triangles
    triangles = SubElement(mesh, 'triangles', {
        'count': str(n_faces), 'material': 'material0'
    })
    SubElement(triangles, 'input', {
        'semantic': 'VERTEX', 'source': '#mesh0-vertices', 'offset': '0'
    })
    SubElement(triangles, 'input', {
        'semantic': 'NORMAL', 'source': '#mesh0-normals', 'offset': '1'
    })
    SubElement(triangles, 'input', {
        'semantic': 'COLOR', 'source': '#mesh0-colors', 'offset': '0'
    })

    p_data = []
    for face_v, face_vn in faces:
        for i in range(len(face_v)):
            p_data.append(str(face_v[i]))
            p_data.append(str(face_vn[i] if i < len(face_vn) else face_v[i]))
    SubElement(triangles, 'p').text = ' '.join(p_data)

    # Library visual scenes
    lib_scenes = SubElement(collada, 'library_visual_scenes')
    vis_scene = SubElement(lib_scenes, 'visual_scene', {
        'id': 'Scene', 'name': 'Scene'
    })
    node = SubElement(vis_scene, 'node', {
        'id': 'node0', 'name': 'mesh0', 'type': 'NODE'
    })
    inst_geom = SubElement(node, 'instance_geometry', {'url': '#mesh0'})
    bind_mat = SubElement(inst_geom, 'bind_material')
    tech_common = SubElement(bind_mat, 'technique_common')
    SubElement(tech_common, 'instance_material', {
        'symbol': 'material0', 'target': '#material0'
    })

    # Scene
    SubElement(
        SubElement(collada, 'scene'),
        'instance_visual_scene', {'url': '#Scene'}
    )

    tree = ElementTree(collada)
    tree.write(filepath, encoding='utf-8', xml_declaration=True)


def write_dae_textured(filepath, flat_groups, tex_group, tex_filename):
    """Write Collada DAE with flat-color materials + one texture-mapped material.

    flat_groups: list of dicts with material_id, diffuse, positions, normals, faces
    tex_group:   dict with material_id, positions, normals, uvs, faces
    tex_filename: texture image filename (relative to DAE)
    """
    collada = Element('COLLADA', {
        'xmlns': 'http://www.collada.org/2005/11/COLLADASchema',
        'version': '1.4.1'
    })

    # Asset
    asset = SubElement(collada, 'asset')
    contributor = SubElement(asset, 'contributor')
    SubElement(contributor, 'authoring_tool').text = 'mesh processing toolkit'
    SubElement(asset, 'up_axis').text = 'Z_UP'
    SubElement(asset, 'unit', {'meter': '1.0', 'name': 'meter'})

    # --- Library images ---
    lib_images = SubElement(collada, 'library_images')
    img_el = SubElement(lib_images, 'image', {'id': 'logo_img', 'name': 'logo_img'})
    SubElement(img_el, 'init_from').text = tex_filename

    # --- Effects & materials ---
    lib_effects = SubElement(collada, 'library_effects')
    lib_mats = SubElement(collada, 'library_materials')

    # Flat-color effects
    for grp in flat_groups:
        mid = grp['material_id']
        d = grp['diffuse']

        eff = SubElement(lib_effects, 'effect', {'id': f'{mid}-effect'})
        prof = SubElement(eff, 'profile_COMMON')
        tech = SubElement(prof, 'technique', {'sid': 'common'})
        lam = SubElement(tech, 'lambert')
        em = SubElement(lam, 'emission')
        SubElement(em, 'color', {'sid': 'emission'}).text = '0 0 0 1'
        df = SubElement(lam, 'diffuse')
        SubElement(df, 'color', {'sid': 'diffuse'}).text = (
            f'{d[0]:.7f} {d[1]:.7f} {d[2]:.7f} 1'
        )

        mat = SubElement(lib_mats, 'material', {
            'id': f'{mid}-material', 'name': mid
        })
        SubElement(mat, 'instance_effect', {'url': f'#{mid}-effect'})

    # Texture effect (logo zone)
    tmid = tex_group['material_id']
    eff = SubElement(lib_effects, 'effect', {'id': f'{tmid}-effect'})
    prof = SubElement(eff, 'profile_COMMON')

    sp1 = SubElement(prof, 'newparam', {'sid': 'logo_surf'})
    surf = SubElement(sp1, 'surface', {'type': '2D'})
    SubElement(surf, 'init_from').text = 'logo_img'

    sp2 = SubElement(prof, 'newparam', {'sid': 'logo_samp'})
    samp = SubElement(sp2, 'sampler2D')
    SubElement(samp, 'source').text = 'logo_surf'

    tech = SubElement(prof, 'technique', {'sid': 'common'})
    lam = SubElement(tech, 'lambert')
    em = SubElement(lam, 'emission')
    SubElement(em, 'color', {'sid': 'emission'}).text = '0 0 0 1'
    df = SubElement(lam, 'diffuse')
    SubElement(df, 'texture', {'texture': 'logo_samp', 'texcoord': 'UVMap'})

    mat = SubElement(lib_mats, 'material', {
        'id': f'{tmid}-material', 'name': tmid
    })
    SubElement(mat, 'instance_effect', {'url': f'#{tmid}-effect'})

    # --- Geometry ---
    lib_geom = SubElement(collada, 'library_geometries')
    geometry = SubElement(lib_geom, 'geometry', {'id': 'mesh0', 'name': 'mesh0'})
    mesh_el = SubElement(geometry, 'mesh')

    # Merge all groups (flat first, texture last)
    all_groups = flat_groups + [tex_group]
    all_pos, all_norm, offsets = [], [], []
    off = 0
    for g in all_groups:
        offsets.append(off)
        all_pos.append(g['positions'])
        all_norm.append(g['normals'])
        off += len(g['positions'])
    positions = np.vstack(all_pos)
    normals_all = np.vstack(all_norm)
    nv = len(positions)

    # Positions source
    src_p = SubElement(mesh_el, 'source', {'id': 'mesh0-pos'})
    SubElement(src_p, 'float_array', {
        'id': 'mesh0-pos-arr', 'count': str(nv * 3)
    }).text = ' '.join(f'{x:.8f}' for x in positions.flatten())
    tc = SubElement(src_p, 'technique_common')
    acc = SubElement(tc, 'accessor', {
        'source': '#mesh0-pos-arr', 'count': str(nv), 'stride': '3'
    })
    for p in 'XYZ':
        SubElement(acc, 'param', {'name': p, 'type': 'float'})

    # Normals source
    src_n = SubElement(mesh_el, 'source', {'id': 'mesh0-norm'})
    SubElement(src_n, 'float_array', {
        'id': 'mesh0-norm-arr', 'count': str(nv * 3)
    }).text = ' '.join(f'{x:.8f}' for x in normals_all.flatten())
    tc = SubElement(src_n, 'technique_common')
    acc = SubElement(tc, 'accessor', {
        'source': '#mesh0-norm-arr', 'count': str(nv), 'stride': '3'
    })
    for p in 'XYZ':
        SubElement(acc, 'param', {'name': p, 'type': 'float'})

    # UV source (for texture group vertices only)
    tex_off = offsets[-1]
    uvs = tex_group['uvs']
    n_uv = len(uvs)
    src_uv = SubElement(mesh_el, 'source', {'id': 'mesh0-uv'})
    SubElement(src_uv, 'float_array', {
        'id': 'mesh0-uv-arr', 'count': str(n_uv * 2)
    }).text = ' '.join(f'{x:.8f}' for x in uvs.flatten())
    tc = SubElement(src_uv, 'technique_common')
    acc = SubElement(tc, 'accessor', {
        'source': '#mesh0-uv-arr', 'count': str(n_uv), 'stride': '2'
    })
    SubElement(acc, 'param', {'name': 'S', 'type': 'float'})
    SubElement(acc, 'param', {'name': 'T', 'type': 'float'})

    # Vertices
    verts_el = SubElement(mesh_el, 'vertices', {'id': 'mesh0-verts'})
    SubElement(verts_el, 'input', {
        'semantic': 'POSITION', 'source': '#mesh0-pos'
    })

    # Flat-color triangle groups
    for i, g in enumerate(flat_groups):
        mid = g['material_id']
        faces = g['faces'] + offsets[i]
        tri = SubElement(mesh_el, 'triangles', {
            'count': str(len(faces)), 'material': f'{mid}-material'
        })
        SubElement(tri, 'input', {
            'semantic': 'VERTEX', 'source': '#mesh0-verts', 'offset': '0'
        })
        SubElement(tri, 'input', {
            'semantic': 'NORMAL', 'source': '#mesh0-norm', 'offset': '1'
        })
        pd = []
        for f in faces:
            for vi in f:
                pd.append(str(vi))
                pd.append(str(vi))
        SubElement(tri, 'p').text = ' '.join(pd)

    # Texture-mapped triangle group
    tex_faces = tex_group['faces'] + tex_off
    tri = SubElement(mesh_el, 'triangles', {
        'count': str(len(tex_faces)), 'material': f'{tmid}-material'
    })
    SubElement(tri, 'input', {
        'semantic': 'VERTEX', 'source': '#mesh0-verts', 'offset': '0'
    })
    SubElement(tri, 'input', {
        'semantic': 'NORMAL', 'source': '#mesh0-norm', 'offset': '1'
    })
    SubElement(tri, 'input', {
        'semantic': 'TEXCOORD', 'source': '#mesh0-uv', 'offset': '2', 'set': '0'
    })
    pd = []
    for f in tex_faces:
        for vi in f:
            uv_i = vi - tex_off
            pd.append(str(vi))
            pd.append(str(vi))
            pd.append(str(uv_i))
    SubElement(tri, 'p').text = ' '.join(pd)

    # --- Visual scene ---
    lib_sc = SubElement(collada, 'library_visual_scenes')
    vs = SubElement(lib_sc, 'visual_scene', {'id': 'Scene', 'name': 'Scene'})
    nd = SubElement(vs, 'node', {
        'id': 'node0', 'name': 'mesh0', 'type': 'NODE'
    })
    ig = SubElement(nd, 'instance_geometry', {'url': '#mesh0'})
    bm = SubElement(ig, 'bind_material')
    tc2 = SubElement(bm, 'technique_common')
    for g in all_groups:
        mid = g['material_id']
        SubElement(tc2, 'instance_material', {
            'symbol': f'{mid}-material', 'target': f'#{mid}-material'
        })

    SubElement(
        SubElement(collada, 'scene'),
        'instance_visual_scene', {'url': '#Scene'}
    )

    ElementTree(collada).write(filepath, encoding='utf-8', xml_declaration=True)
