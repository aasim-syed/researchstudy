#!/usr/bin/env python3
"""Build the benchmark model set in data/models/ from the raw downloads.

Sources (all fetched from public GitHub-hosted repositories):
  * alecjacobson/common-3d-test-models : bunny, fandisk, rocker-arm, armadillo, spot,
    beetle, nefertiti, igea
  * KhronosGroup/glTF-Sample-Assets    : SheenChair, ChairDamaskPurplegold,
    GlamVelvetSofa, SheenWoodLeatherSofa (furniture, multi-component, .glb)
  * isl-org/open3d_downloads           : SampleFountainRGBDImages (VCG-integrated
    scene scan, ~1M faces) and LoungeRGBDImages (room-scale RGB-D sequence, fused
    here with TSDF + marching cubes -> run scripts/fuse_lounge.py in a Python
    that has open3d).

Low-poly sources (<10K faces) are midpoint-subdivided (geometry unchanged) so every
model is inside the 10K-2M face range given in the brief. One synthetic robustness
case (fandisk-noisy) adds Gaussian noise, punches holes and injects a non-manifold fin.
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "models")
O3D = os.environ.get("O3D_ROOT", "/home/claude/o3d/extract")

MODELS = [
    # name, source file, category, notes
    ("bunny", "stanford-bunny.obj", "organic scan", "range scan, open boundaries (holes at base)"),
    ("fandisk", "fandisk.obj", "CAD", "sharp features, corners"),
    ("rocker_arm", "rocker-arm.obj", "CAD", "genus-1 mechanical part, sharp+smooth"),
    ("armadillo", "armadillo.obj", "organic", "high-frequency detail"),
    ("spot", "spot.obj", "organic", "smooth, has source UVs (attribute transfer test)"),
    ("beetle", "beetle.obj", "CAD/vehicle", "car body, sharp creases, non-watertight"),
    ("nefertiti", "nefertiti.obj", "photogrammetry scan", "dense scan"),
    ("igea", "igea.obj", "scan", "dense scan, 269K faces"),
    ("sheen_chair", "SheenChair.glb", "furniture", "multi-component chair, UVs"),
    ("damask_chair", "ChairDamaskPurplegold.glb", "furniture", "73 components, thin parts"),
    ("velvet_sofa", "GlamVelvetSofa.glb", "furniture", "sofa, open parts"),
    ("leather_sofa", "SheenWoodLeatherSofa.glb", "furniture", "sofa with wooden frame"),
    ("fountain_scan", os.path.join(O3D, "SampleFountainRGBDImages/scene/integrated.ply"),
     "scene scan", "RGB-D fused scene, ~1M faces, noisy, open, many components"),
]


def load(path):
    if not os.path.isabs(path):
        path = os.path.join(RAW, path)
    m = trimesh.load(path, force="mesh", process=False)
    return trimesh.Trimesh(m.vertices, m.faces, visual=m.visual, process=False)


def ensure_min_faces(m, min_faces=10000):
    while len(m.faces) < min_faces:
        uv = getattr(m.visual, "uv", None)
        if uv is not None and len(uv) == len(m.vertices):
            v, f, attr = trimesh.remesh.subdivide(m.vertices, m.faces, vertex_attributes={"uv": uv})
            m = trimesh.Trimesh(v, f, visual=trimesh.visual.TextureVisuals(uv=attr["uv"]), process=False)
        else:
            v, f = trimesh.remesh.subdivide(m.vertices, m.faces)
            m = trimesh.Trimesh(v, f, process=False)
    return m


def noisy_fandisk(m, seed=7):
    rng = np.random.default_rng(seed)
    v = m.vertices.copy()
    diag = np.linalg.norm(v.max(0) - v.min(0))
    v += rng.normal(scale=0.0015 * diag, size=v.shape)
    f = m.faces.copy()
    # punch 4 small holes
    c = trimesh.Trimesh(v, f, process=False).triangles_center
    keep = np.ones(len(f), bool)
    for idx in rng.choice(len(f), 4, replace=False):
        keep &= np.linalg.norm(c - c[idx], axis=1) > 0.012 * diag
    f = f[keep]
    # non-manifold fin: a third triangle on an existing interior edge
    a, b = f[100][0], f[100][1]
    tip = (v[a] + v[b]) / 2 + 0.01 * diag * np.array([0.3, 0.9, 0.1])
    v = np.vstack([v, tip])
    f = np.vstack([f, [a, b, len(v) - 1]])
    return trimesh.Trimesh(v, f, process=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name, src, cat, notes in MODELS:
        try:
            m = load(src)
        except Exception as e:  # noqa
            print(f"skip {name}: {e}")
            continue
        n0 = len(m.faces)
        m = ensure_min_faces(m)
        out = os.path.join(OUT, name + ".obj")
        m.export(out, include_normals=False)
        manifest.append(dict(name=name, file=f"models/{name}.obj", category=cat, notes=notes,
                             source=os.path.basename(src), source_faces=n0, faces=len(m.faces),
                             vertices=len(m.vertices)))
        print(name, n0, "->", len(m.faces))
        if name == "fandisk":
            nm = noisy_fandisk(m)
            nm.export(os.path.join(OUT, "fandisk_noisy.obj"))
            manifest.append(dict(name="fandisk_noisy", file="models/fandisk_noisy.obj",
                                 category="robustness", notes="fandisk + 0.15% noise, 4 holes, non-manifold fin",
                                 source="fandisk.obj", source_faces=n0, faces=len(nm.faces),
                                 vertices=len(nm.vertices)))
    lounge = os.path.join(OUT, "lounge_room.obj")
    if os.path.exists(lounge):
        m = trimesh.load(lounge, force="mesh", process=False)
        manifest.append(dict(name="lounge_room", file="models/lounge_room.obj", category="room-scale scan",
                             notes="Lounge RGB-D sequence fused with TSDF + marching cubes",
                             source="LoungeRGBDImages", source_faces=len(m.faces), faces=len(m.faces),
                             vertices=len(m.vertices)))
    with open(os.path.join(HERE, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(len(manifest), "models")


if __name__ == "__main__":
    sys.exit(main())
