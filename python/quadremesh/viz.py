"""Headless visualisation: field glyphs, singularity overlay, scaled-Jacobian heatmap.

Uses a tiny orthographic polygon renderer (painter's algorithm + Lambert shading) on top of
matplotlib's 2D PolyCollection, so it runs without OpenGL.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402


def view_matrix(azim=35.0, elev=20.0):
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    # y-up models: map (x, y, z) -> (x, -z, y) first
    up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    return Rx @ Rz @ up


def _faces_list(Q):
    out = []
    for q in Q:
        out.append(q[:3] if q[3] < 0 else q)
    return out


def scaled_jacobian(V, Q):
    sj = np.zeros(len(Q))
    for k, q in enumerate(Q):
        idx = q[:3] if q[3] < 0 else q
        P = V[idx]
        n = len(idx)
        nrm = np.zeros(3)
        for c in range(n):
            nrm += np.cross(P[(c + 1) % n] - P[c], P[(c - 1) % n] - P[c])
        ln = np.linalg.norm(nrm)
        if ln == 0:
            sj[k] = -1
            continue
        nrm /= ln
        m = 1.0
        for c in range(n):
            e1, e2 = P[(c + 1) % n] - P[c], P[(c - 1) % n] - P[c]
            den = np.linalg.norm(e1) * np.linalg.norm(e2)
            v = np.dot(np.cross(e1, e2), nrm) / den if den > 0 else -1
            if n == 3:
                v = min(1.0, v * 2 / np.sqrt(3))
            m = min(m, v)
        sj[k] = m
    return sj


def valences(V, Q):
    from collections import Counter
    ec = Counter()
    for q in Q:
        idx = list(q[:3] if q[3] < 0 else q)
        for c in range(len(idx)):
            a, b = idx[c], idx[(c + 1) % len(idx)]
            ec[(min(a, b), max(a, b))] += 1
    val = np.zeros(len(V), int)
    bnd = np.zeros(len(V), bool)
    for (a, b), c in ec.items():
        val[a] += 1
        val[b] += 1
        if c == 1:
            bnd[a] = bnd[b] = True
    return val, bnd


def render(ax, V, faces, R, face_colors=None, edges=True, base=(0.82, 0.84, 0.88), lw=0.15, light=(0.3, 0.4, 1.0)):
    P = V @ R.T
    polys = [P[f] for f in faces]
    depth = np.array([p[:, 2].mean() for p in polys])
    normals = np.array([np.cross(p[1] - p[0], p[2] - p[0]) for p in polys])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    L = np.array(light) / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(normals @ L)
    if face_colors is None:
        cols = np.array(base)[None] * shade[:, None]
    else:
        cols = np.asarray(face_colors)[:, :3] * (0.55 + 0.45 * shade[:, None])
    order = np.argsort(depth)
    pc = PolyCollection([polys[k][:, :2] for k in order], facecolors=np.clip(cols[order], 0, 1),
                        edgecolors=(0.15, 0.15, 0.2, 0.55) if edges else "none", linewidths=lw)
    ax.add_collection(pc)
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    return P, depth


def _visible_points(points_view, P, faces, depth_tol):
    """Crude visibility: keep points whose depth is within tol of the max depth in their pixel cell."""
    if len(points_view) == 0:
        return np.zeros(0, bool)
    g = 160
    xmin, ymin = P[:, :2].min(0)
    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) + 1e-12
    cell = lambda xy: np.clip(((xy - [xmin, ymin]) / span * g).astype(int), 0, g - 1)
    zbuf = np.full((g, g), -np.inf)
    cv = cell(P[:, :2])
    np.maximum.at(zbuf, (cv[:, 0], cv[:, 1]), P[:, 2])
    cp = cell(points_view[:, :2])
    return points_view[:, 2] >= zbuf[cp[:, 0], cp[:, 1]] - depth_tol


def plot_quality_panel(V, Q, out_png, title="", azim=35, elev=20, field=None, features=None):
    """3-panel figure: wireframe+singularities, scaled-Jacobian heatmap, (optional) field."""
    R = view_matrix(azim, elev)
    faces = _faces_list(Q)
    ncol = 3 if field is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.2), dpi=150)
    # 1. singularities
    P, _ = render(axes[0], V, faces, R)
    val, bnd = valences(V, Q)
    diag = np.linalg.norm(np.ptp(V, 0))
    vis = _visible_points(P, P, faces, 0.02 * diag)
    for cond, col, lab in [((val == 3) & ~bnd, "#1f6feb", "valence 3"), ((val == 5) & ~bnd, "#e5534b", "valence 5"),
                           ((val >= 6) & ~bnd, "#8957e5", "valence 6+")]:
        m = cond & vis
        axes[0].scatter(P[m, 0], P[m, 1], s=9, c=col, label=f"{lab} ({int(cond.sum())})", zorder=5, edgecolors="white",
                        linewidths=0.3)
    if features is not None and len(features):
        seg = np.stack([features[:, :3] @ R.T, features[:, 3:] @ R.T], 1)
        axes[0].add_collection(LineCollection(seg[:, :, :2], colors="#d29922", linewidths=0.8, zorder=4))
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    axes[0].set_title("edge flow + singularities", fontsize=9)
    # 2. scaled Jacobian heatmap
    sj = scaled_jacobian(V, Q)
    cmap = plt.get_cmap("RdYlGn")
    render(axes[1], V, faces, R, face_colors=cmap(np.clip((sj - 0.2) / 0.8, 0, 1)), edges=True, lw=0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.2, 1.0))
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.035, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    axes[1].set_title(f"scaled Jacobian (mean {sj.mean():.3f}, min {sj.min():.2f})", fontsize=9)
    # 3. field glyphs
    if field is not None:
        FV, FF, FQ, FN = field["V"], field["F"], field["Q"], field["N"]
        Pf, _ = render(axes[2], FV, list(FF), R, edges=False, base=(0.9, 0.9, 0.92))
        n = len(FV)
        rng = np.random.default_rng(0)
        k = min(n, 3500)
        idx = np.sort(rng.choice(n, k, replace=False))
        good = np.linalg.norm(FQ[idx], axis=1) > 0.5
        idx = idx[good]
        q1 = FQ[idx]
        q2 = np.cross(FN[idx], q1)
        Pi = FV[idx] @ R.T
        visf = _visible_points(Pi, Pf, None, 0.02 * diag)
        h = 0.012 * diag
        segs = []
        for d in (q1, q2):
            a = (FV[idx] - d * h) @ R.T
            b = (FV[idx] + d * h) @ R.T
            segs.append(np.stack([a[visf, :2], b[visf, :2]], 1))
        axes[2].add_collection(LineCollection(segs[0], colors="#1f6feb", linewidths=0.5))
        axes[2].add_collection(LineCollection(segs[1], colors="#e5534b", linewidths=0.5))
        axes[2].set_title("cross field (orientation)", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def side_by_side(meshes, out_png, title="", azim=35, elev=20):
    """meshes: list of (label, V, Q). Wireframe + singularities per method."""
    R = view_matrix(azim, elev)
    fig, axes = plt.subplots(1, len(meshes), figsize=(4.2 * len(meshes), 4.4), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (label, V, Q) in zip(axes, meshes):
        if V is None:
            ax.text(0.5, 0.5, f"{label}\nfailed", ha="center", va="center")
            ax.axis("off")
            continue
        P, _ = render(ax, V, _faces_list(Q), R, lw=0.12)
        val, bnd = valences(V, Q)
        diag = np.linalg.norm(np.ptp(V, 0))
        vis = _visible_points(P, P, None, 0.02 * diag)
        s3 = (val == 3) & ~bnd
        s5 = (val >= 5) & ~bnd
        ax.scatter(P[s3 & vis, 0], P[s3 & vis, 1], s=5, c="#1f6feb", zorder=5)
        ax.scatter(P[s5 & vis, 0], P[s5 & vis, 1], s=5, c="#e5534b", zorder=5)
        ax.set_title(f"{label}\n{len(Q)} faces, {int((s3 | s5).sum())} singularities", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png
