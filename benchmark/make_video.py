"""Render a short promo/demo video: input triangles vs structured quads, turntable per model.

Usage: python benchmark/make_video.py OUT.mp4 [--fps 30] [--frames 90] [--models fandisk spot ...]
Needs: the benchmark models in data/models, the built `build/quadremesh` CLI, `build/quadeval`, ffmpeg.
Every number shown on screen is read from the evaluator report JSON, none is typed by hand.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quadremesh import load_mesh  # noqa: E402
from quadremesh.viz import render, valences, view_matrix, _faces_list, _visible_points  # noqa: E402

BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"
W, H, DPI = 12.8, 7.2, 100  # 1280x720

# (model, target faces, feature angle)
HERO = {"fandisk": (5000, 35), "spot": (5000, 35), "sheen_chair": (5000, 35), "bunny": (5000, 35)}
MAX_COVERAGE = ["--coverage-mode", "--path-subdiv", "3", "--sing-budget-im", "--coverage-search"]


def read_quad_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            F.append(idx if len(idx) == 4 else idx + [-1] * (4 - len(idx)))
    return np.array(V), np.array(F, dtype=int)


def remesh(name, nfaces, angle, outdir):
    src = os.path.join(ROOT, "data", "models", name + ".obj")
    out = os.path.join(outdir, name + "_quad.obj")
    rep = os.path.join(outdir, name + "_report.json")
    if not os.path.exists(rep):
        subprocess.run([os.path.join(ROOT, "build", "quadremesh"), src, "-o", out, "-f", str(nfaces), "-a", str(angle)] + MAX_COVERAGE,
                       check=True)
        subprocess.run([os.path.join(ROOT, "build", "quadeval"), src, out, "--target", str(nfaces), "--feature-angle", str(angle),
                        "-o", rep], check=True)
    return src, out, json.load(open(rep))


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    return fig


def card(path, lines, sub=None):
    fig = new_fig()
    fig.text(0.5, 0.58, lines, ha="center", va="center", color=FG, fontsize=40, fontweight="bold")
    if sub:
        fig.text(0.5, 0.40, sub, ha="center", va="center", color=ACCENT, fontsize=20)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def frame(path, name, Vin, Fin, Vq, Fq, azim, stats):
    R = view_matrix(azim, 18)
    fig = new_fig()
    ax1 = fig.add_axes([0.02, 0.14, 0.47, 0.70])
    ax2 = fig.add_axes([0.51, 0.14, 0.47, 0.70])
    for a in (ax1, ax2):
        a.set_facecolor(BG)
    render(ax1, Vin, list(Fin), R, lw=0.1, base=(0.80, 0.82, 0.86))
    P, _ = render(ax2, Vq, _faces_list(Fq), R, lw=0.25, base=(0.80, 0.86, 0.95))
    val, bnd = valences(Vq, Fq)
    diag = np.linalg.norm(np.ptp(Vq, 0))
    vis = _visible_points(P, P, None, 0.02 * diag)
    s3, s5 = (val == 3) & ~bnd, (val >= 5) & ~bnd
    ax2.scatter(P[s3 & vis, 0], P[s3 & vis, 1], s=14, c="#58a6ff", zorder=5, edgecolors="white", linewidths=0.3)
    ax2.scatter(P[s5 & vis, 0], P[s5 & vis, 1], s=14, c="#f85149", zorder=5, edgecolors="white", linewidths=0.3)
    fig.text(0.25, 0.88, f"input: {len(Fin):,} triangles", ha="center", color="#8b949e", fontsize=18)
    fig.text(0.75, 0.88, f"ours: {len(Fq):,} faces (quad-dominant)", ha="center", color=FG, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.94, name.replace("_", " "), ha="center", color=ACCENT, fontsize=26, fontweight="bold")
    fig.text(0.5, 0.06, stats, ha="center", color=FG, fontsize=16)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=90, help="frames per model")
    ap.add_argument("--models", nargs="*", default=list(HERO))
    a = ap.parse_args()

    work = os.path.join(ROOT, "benchmark", "video_work")
    os.makedirs(work, exist_ok=True)
    fdir = os.path.join(work, "frames")
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    n = 0

    def save_hold(src, seconds):
        nonlocal n
        for _ in range(int(seconds * a.fps)):
            os.replace(src, src) if False else None
            dst = os.path.join(fdir, f"f{n:05d}.png")
            with open(src, "rb") as s, open(dst, "wb") as d:
                d.write(s.read())
            n += 1

    tmp = os.path.join(work, "card.png")
    card(tmp, "Triangles in.\nStructured quads out.", "feature-aligned quad remeshing, measured")
    save_hold(tmp, 2.5)

    summary = []
    for name in a.models:
        nfaces, angle = HERO[name]
        src, out, rep = remesh(name, nfaces, angle, work)
        Vin = load_mesh(src)["V"]
        Fin = load_mesh(src)["F"]
        Vq, Fq = read_quad_obj(out)
        sj = dig(rep, "quality", "scaled_jacobian", "mean")
        qr = dig(rep, "faces", "quad_ratio")
        cov = dig(rep, "features", "feature_coverage")
        parts = []
        if qr is not None:
            parts.append(f"{qr * 100:.1f}% quads")
        if sj is not None:
            parts.append(f"scaled Jacobian {sj:.3f}")
        if cov is not None:
            parts.append(f"feature coverage {cov * 100:.1f}%")
        stats = "   |   ".join(parts)
        summary.append((name, stats))
        for i in range(a.frames):
            p = os.path.join(fdir, f"f{n:05d}.png")
            frame(p, name, Vin, Fin, Vq, Fq, 20 + 360.0 * i / a.frames, stats)
            n += 1
        print(name, stats, flush=True)

    card(tmp, "Open source. Reproducible.\nEvery number measured.", f"{len(summary)} models shown, all metrics from the quadeval report")
    save_hold(tmp, 3.0)

    subprocess.run(["ffmpeg", "-y", "-framerate", str(a.fps), "-i", os.path.join(fdir, "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", a.out], check=True)
    print("wrote", a.out, n, "frames")


if __name__ == "__main__":
    main()
