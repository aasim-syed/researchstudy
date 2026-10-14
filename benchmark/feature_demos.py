#!/usr/bin/env python3
"""Figures for the optional user controls: density map, symmetry, guidance strokes, UV transfer,
pure-quad mode. Writes PNGs to docs/report/."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quadremesh import quadremesh  # noqa: E402
from quadremesh.viz import _faces_list, render, side_by_side, valences, view_matrix  # noqa: E402

OUT = os.path.join(ROOT, "docs/report")
M = os.path.join(ROOT, "data/models")


def grid(n=60, bump=0.15):
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    V = np.stack([x.ravel(), y.ravel(), bump * np.sin(3 * x.ravel()) * np.cos(2 * y.ravel())], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


def main():
    os.makedirs(OUT, exist_ok=True)
    # 1. adaptive density on the bunny (head region x6) and curvature-driven density
    base = quadremesh(os.path.join(M, "bunny.obj"), target_faces=3000, report=False)
    V = base.vertices
    head = V[np.argmax(V[:, 1])]
    dens = quadremesh(os.path.join(M, "bunny.obj"), target_faces=3000, report=False,
                      density_regions=[(head.tolist(), 0.35 * np.ptp(V[:, 1]), 6.0)])
    curv = quadremesh(os.path.join(M, "bunny.obj"), target_faces=3000, report=False, curvature_density=1.5)
    side_by_side([("uniform", base.vertices, base.faces), ("density region x6 (ears/head)", dens.vertices, dens.faces),
                  ("curvature-driven density", curv.vertices, curv.faces)],
                 os.path.join(OUT, "demo_density.png"), title="Adaptive density map", azim=-30, elev=10)
    # 2. symmetry
    a = quadremesh(os.path.join(M, "spot.obj"), target_faces=2500, report=False)
    s = quadremesh(os.path.join(M, "spot.obj"), target_faces=2500, report=False, symmetry_axis="x")
    side_by_side([("no symmetry hint", a.vertices, a.faces), ("symmetry_axis='x' (exact mirror)", s.vertices, s.faces)],
                 os.path.join(OUT, "demo_symmetry.png"), title="Symmetry hint", azim=0, elev=15)
    # 3. guidance strokes on a curved patch
    Vg, Fg = grid()
    t = np.linspace(0.05, 0.95, 40)
    stroke = np.stack([t, 0.2 + 0.6 * t, 0.15 * np.sin(3 * t) * np.cos(2 * (0.2 + 0.6 * t))], 1)
    plain = quadremesh((Vg, Fg), target_faces=600, report=False)
    guided = quadremesh((Vg, Fg), target_faces=600, report=False, strokes=[stroke])
    edges = quadremesh((Vg, Fg), target_faces=600, report=False, strokes=[stroke], strokes_as_edges=True)
    R = view_matrix(0, 90)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=140)
    for ax, (lab, r) in zip(axes, [("no stroke", plain), ("stroke guides edge flow", guided), ("stroke as edge loop", edges)]):
        render(ax, r.vertices, _faces_list(r.faces), R, lw=0.25)
        if lab != "no stroke":
            P = stroke @ R.T
            ax.plot(P[:, 0], P[:, 1], color="#e5534b", lw=1.6)
        ax.set_title(lab, fontsize=9)
    fig.suptitle("Guidance strokes", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demo_strokes.png"))
    plt.close(fig)
    # 4. UV transfer (per-corner UVs shown as a checker pattern)
    r = quadremesh(os.path.join(M, "spot.obj"), target_faces=3000, report=False)
    uv = r.uv.reshape(-1, 4, 2).mean(1)
    checker = ((np.floor(uv[:, 0] * 12) + np.floor(uv[:, 1] * 12)) % 2)[:, None]
    cols = np.where(checker > 0, np.array([[0.95, 0.75, 0.3]]), np.array([[0.25, 0.4, 0.7]]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=140)
    render(axes[0], r.vertices, _faces_list(r.faces), view_matrix(60, 10), face_colors=cols, lw=0.05)
    axes[0].set_title("transferred UVs (checker)", fontsize=9)
    pq = quadremesh(os.path.join(M, "spot.obj"), target_faces=3000, report=False, pure_quad=True)
    render(axes[1], pq.vertices, _faces_list(pq.faces), view_matrix(60, 10), lw=0.1)
    axes[1].set_title(f"pure_quad=True: {len(pq.faces)} faces, quad ratio {pq.quad_ratio:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "demo_uv_purequad.png"))
    plt.close(fig)
    print("demos written")


if __name__ == "__main__":
    main()
