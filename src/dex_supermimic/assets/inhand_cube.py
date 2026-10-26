# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common in-hand object config helpers."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent / "resources" / "inhand_cube"

INHAND_OBJECT_XML: Path = _ASSET_DIR / "xmls" / "cube.xml"
assert INHAND_OBJECT_XML.exists(), f"Missing in-hand object XML: {INHAND_OBJECT_XML}"

# Baseline cube — reflects values inside cube.xml. If cube.xml changes, update here.
BASELINE_EDGE_M = 0.054
BASELINE_MASS_KG = 0.120
CUBE_DENSITY = BASELINE_MASS_KG / (BASELINE_EDGE_M**3)  # ≈ 762.08 kg/m^3


def _get_spec(edge_m: float | None = None, *, viser: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(INHAND_OBJECT_XML))
    half = (BASELINE_EDGE_M if edge_m is None else edge_m) / 2.0
    if viser:
        _add_viser_cube_mesh(spec, half)
    if edge_m is None:
        return spec
    mass = CUBE_DENSITY * (edge_m**3)

    cube_geom = next((g for g in spec.geoms if g.name == "cube"), None)
    if cube_geom is None:
        raise ValueError(f"{INHAND_OBJECT_XML} must contain a geom named 'cube'")
    cube_geom.size = [half, half, half]
    cube_geom.mass = mass

    # Keep the visual ArUco box in lockstep with the collision box, otherwise a
    # non-baseline cube renders (and its goal ghost, derived from this geom) at
    # the wrong 54 mm size.
    cube_visual = next((g for g in spec.geoms if g.name == "cube_visual"), None)
    if cube_visual is None:
        raise ValueError(f"{INHAND_OBJECT_XML} must contain a geom named 'cube_visual'")
    cube_visual.size = [half, half, half]

    return spec


def _add_viser_cube_mesh(spec: mujoco.MjSpec, half: float) -> None:
    """Give mjviser a UV mesh: its primitive converter discards cube textures.

    Keep the collision box and all physical properties unchanged. The compiled
    cube texture is already a six-face strip, so it doubles as a 2D UV atlas.
    """
    model = spec.compile()
    texid = model.texture("arucocube").id
    width, height = int(model.tex_width[texid]), int(model.tex_height[texid])
    channels = int(model.tex_nchannel[texid])
    start = int(model.tex_adr[texid])
    spec.add_texture(
        name="viser_cube_atlas",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=width,
        height=height,
        nchannel=channels,
        data=model.tex_data[start : start + width * height * channels].tolist(),
    )
    material = spec.add_material(name="viser_cube")
    material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "viser_cube_atlas"

    # OpenGL cubemap directions in MuJoCo slot order: +X,-X,+Y,-Y,+Z,-Z.
    normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    rights = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]])
    ups = np.cross(normals, rights)
    vertices, faces, uvs = [], [], []
    for slot, (normal, right, up) in enumerate(zip(normals, rights, ups)):
        base = len(vertices)
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            vertices.append((normal + (2 * u - 1) * right + (2 * v - 1) * up) * half)
            # Match MuJoCo's box cubemap projection, including its texture
            # handedness (verified against the native renderer on all faces).
            tex_u, tex_v = u, 1 - v
            # Inset by half a pixel to avoid filtering across atlas faces.
            uvs.append(((0.5 + tex_u * (width - 1)) / width, (slot * width + 0.5 + tex_v * (width - 1)) / height))
        faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    spec.add_mesh(
        name="viser_cube_mesh",
        uservert=np.asarray(vertices).ravel().tolist(),
        userface=np.asarray(faces).ravel().tolist(),
        usertexcoord=np.asarray(uvs).ravel().tolist(),
        userfacetexcoord=np.asarray(faces).ravel().tolist(),
        smoothnormal=False,
    )
    visual = spec.geom("cube_visual")
    visual.type = mujoco.mjtGeom.mjGEOM_MESH
    visual.meshname = "viser_cube_mesh"
    visual.material = "viser_cube"


def get_inhand_object_cfg(edge_m: float | None = None) -> EntityCfg:
    from mjlab.entity import EntityCfg

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(-0.1404, 0.0042, 0.52),
            rot=(0.3894, -0.3796, -0.5980, -0.5888),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spec_fn=partial(_get_spec, edge_m),
    )
