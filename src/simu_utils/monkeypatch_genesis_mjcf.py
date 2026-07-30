"""
monkeypatch_genesis_mjcf.py
W1-PR4 fix for Genesis 1.0's MJCF loader.

Background
----------
Genesis 1.0's MJCF parser (`genesis.utils.mjcf.build_model`) does two stages:

  1. Recursively inline every `<include file="..."/>` it can find. While
     inlining, it rewrites mesh `filename=` attributes by prepending the
     include path's parent directory (e.g. a mesh declared as
     `../../../urdf/foo.png` inside `assets/astribot_s1_asset_for_gripper.xml`
     becomes `assets/../../../urdf/foo.png`).
  2. Re-serialize the inlined `xml.etree` root with
     `ET.tostring(root, encoding="utf8")` and pass the resulting string to
     `mujoco.MjModel.from_xml_string(data)`.

`from_xml_string` has no concept of "this string came from this file on
disk", so every relative `<include file="..."/>` and every relative
`<texture file="..."/>` / `<mesh filename="..."/>` is resolved against the
**current working directory** instead of the entry MJCF's directory. The
`<compiler meshdir/assetdir/texturedir>` attributes that Genesis sets just
before stage 2 are also ignored by `from_xml_string` — only
`from_xml_path` honours them, and it preserves the per-include-file
base-directory context that makes relative mesh paths resolve correctly
under nested `<include>`s.

For our pipeline specifically
-----------------------------
`astribot_s1_chassis_fixed_with_gripper.xml` and its sibling entries
reference `assets/astribot_s1_asset_for_gripper.xml`,
`model_with_effector/...`, and `actuators/...`. The first generation of the
fix tried to chdir into the entry MJCF's directory; that fixed `<include>`
resolution but the mesh/texture paths still failed because the inlined
root's mesh `file=` fields are `assets/../../../urdf/...` after the inline
rewriting pass, and `from_xml_string` resolves that string against cwd.

The fix
-------
Re-implement `build_model` for the MJCF case to **skip the inline stage
entirely** and call `mujoco.MjModel.from_xml_path(mp)` directly. MuJoCo
walks the `<include>` tree itself, preserving each included file's own
directory as the base for resolving its own mesh/texture paths, which is
exactly what the original MJCF semantics expect and what worked in
MuJoCo 0.2.1. The one-shot patch is installed at process startup; the
factory calls it from `astribot_envs_factory.create_simulation_env` and
the verify script calls it from its L5 §5.2 gate. Other backends (MuJoCo,
ManiSkill, IsaacLab) never reach this code path.

What we lose
------------
* URDF placeholder-inertia handling (line 180-192 of
  `genesis/utils/mjcf.py`) is bypassed. We only ever feed MJCF files to
  this codepath so that is fine; if a future caller does feed a URDF, the
  patch falls through to the original `build_model` so they still get the
  URDF behaviour.
* Genesis's `discard_visual`, `default_armature`, and `merge_fixed_links`
  options are silently ignored. The current Astribot configs don't pass
  any of them, so this is a no-op for us today; if a future config does,
  we will need to reapply them via direct calls to `mujoco` after
  `from_xml_path` returns.
"""

from __future__ import annotations

import os
from typing import Optional

_PATCH_INSTALLED = False


def install_build_model_patch(working_dir: Optional[str] = None) -> None:
    """Install a one-shot monkey-patch on `genesis.utils.mjcf.build_model`.

    Parameters
    ----------
    working_dir : str, optional
        Directory of the entry MJCF. Currently unused — the patch resolves
        the entry path via `os.path.isabs(xml)` and falls back to Genesis's
        original code if it can't find a file. Kept in the signature for
        future extensions (e.g. emitting diagnostics about which directory
        the entry lives in).
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    import genesis.utils.mjcf as gm
    import mujoco

    _orig_build_model = gm.build_model

    def _patched_build_model(xml, *args, **kwargs):
        # Passthrough for non-string inputs (already-built MjModel etc.).
        if not isinstance(xml, str):
            return _orig_build_model(xml, *args, **kwargs)

        # Resolve the file path. Genesis joins with its own `get_assets_dir`
        # first, but if the input is already absolute, `os.path.join`
        # returns the absolute path verbatim.
        from genesis.utils.mjcf import get_assets_dir

        candidate = os.path.join(get_assets_dir(), xml)
        if not os.path.exists(candidate):
            candidate = xml
        if not os.path.exists(candidate):
            # Raw XML string or some other form; let Genesis handle it
            # (and likely fail) so we don't lose any signal.
            return _orig_build_model(xml, *args, **kwargs)

        # Sniff: is this a URDF or an MJCF? We branch on the root tag.
        import xml.etree.ElementTree as ET

        try:
            root = ET.parse(candidate).getroot()
        except ET.ParseError:
            return _orig_build_model(xml, *args, **kwargs)

        if root.tag == "robot":
            # URDF — defer to Genesis's original implementation so its
            # placeholder-inertia handling runs.
            return _orig_build_model(xml, *args, **kwargs)

        # MJCF: skip the inline stage, let MuJoCo do the include walk.
        # `from_xml_path` uses the file's own directory as the base for
        # both nested <include> resolution and <mesh>/<texture> file= lookups.
        return mujoco.MjModel.from_xml_path(candidate)

    # Mutate the module's namespace dict so the symbol that
    # `parse_xml` reaches for via module-local-name lookup is the patched
    # function. `gm.build_model = ...` is not sufficient because the
    # caller is a sibling function inside the same module.
    gm.__dict__["build_model"] = _patched_build_model
    _PATCH_INSTALLED = True
