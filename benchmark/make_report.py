#!/usr/bin/env python3
"""Build the visual report from benchmark results.

Outputs (in docs/report/):
  * <model>_ours_panel.png   - edge flow + singularity overlay, scaled-Jacobian heatmap, cross field
  * <model>_compare.png      - all methods side by side with singularities marked
  * charts.png               - per-method metric comparison
  * visual_report.html       - self-contained page (images embedded) with all tables
"""
import base64
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quadremesh import load_mesh, quadremesh  # noqa: E402
from quadremesh.viz import plot_quality_panel, side_by_side  # noqa: E402
from quadremesh import _quadremesh as core  # noqa: E402

RES = os.path.join(ROOT, "benchmark/results")
OUT = os.path.join(ROOT, "docs/report")
METHODS = ["ours", "instant_meshes", "quadriflow", "blender_quadriflow"]
EXTRA = ["ours_learned", "ours_featref", "ours_coverage"]
LABEL = {"ours": "quadremesh (ours)", "ours_learned": "quadremesh + learned field", "ours_featref": "quadremesh + feature refinement", "ours_coverage": "quadremesh --coverage-mode", "instant_meshes": "Instant Meshes", "quadriflow": "QuadriFlow",
         "blender_quadriflow": "Blender QuadriFlow"}
COLOR = {"ours": "#2f81f7", "ours_learned": "#1b4f9c", "ours_featref": "#6cb6ff", "ours_coverage": "#0a3069", "instant_meshes": "#e5534b", "quadriflow": "#3fb950", "blender_quadriflow": "#a371f7"}
VIEW = {"bunny": (-30, 10), "fandisk": (40, 25), "fandisk_noisy": (40, 25), "rocker_arm": (30, 35), "armadillo": (200, 10),
        "spot": (60, 10), "beetle": (30, 25), "nefertiti": (90, 5), "igea": (90, 5), "sheen_chair": (35, 20),
        "damask_chair": (35, 20), "velvet_sofa": (35, 20), "leather_sofa": (35, 20), "fountain_scan": (0, 70),
        "lounge_room": (30, 60)}


def b64(path, max_w=1500):
    """Embed as a downscaled JPEG so the self-contained report stays e-mailable."""
    import io

    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def charts(rows, path):
    models = sorted({r["model"] for r in rows}, key=lambda m: [r["model"] for r in rows].index(m))
    metrics = [("sing_per_1k", "singularities / 1k faces (lower is better)"), ("sj_mean", "mean scaled Jacobian"),
               ("dist_mean_pct", "mean distance error (% diag)"), ("feature_coverage", "feature coverage")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 3.0 * len(metrics)), dpi=130)
    x = np.arange(len(models))
    w = 0.2
    for ax, (key, title) in zip(axes, metrics):
        for k, m in enumerate(METHODS):
            vals = []
            for mo in models:
                r = next((r for r in rows if r["model"] == mo and r["method"] == m), None)
                vals.append(float(r[key]) if r and r.get("status") == "ok" and r.get(key) not in ("", None) else np.nan)
            ax.bar(x + (k - 1.5) * w, vals, w, label=LABEL[m], color=COLOR[m])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if key == "dist_mean_pct":
            ax.set_yscale("log")
    axes[0].legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(os.path.join(RES, "summary.json")))
    agg = json.load(open(os.path.join(RES, "aggregate.json")))
    manifest = json.load(open(os.path.join(ROOT, "data/manifest.json")))
    only = set(sys.argv[1:])
    imgs = {}
    for entry in manifest:
        name = entry["name"]
        if only and name not in only:
            continue
        az, el = VIEW.get(name, (35, 20))
        panel = os.path.join(OUT, f"{name}_ours_panel.png")
        comp = os.path.join(OUT, f"{name}_compare.png")
        if os.path.exists(panel) and os.path.exists(comp) and not os.environ.get("FORCE"):
            imgs[name] = (panel, comp)
            continue
        src = os.path.join(ROOT, "data", entry["file"])
        target = 15000 if entry["category"] in ("scene scan", "room-scale scan") else 5000
        # ours: rerun through the Python API to get the field for the panel (deterministic)
        r = quadremesh(src, target_faces=target, feature_angle=35.0, report=False)
        panel = os.path.join(OUT, f"{name}_ours_panel.png")
        field = r.field if len(r.field["V"]) <= 400000 else None
        plot_quality_panel(r.vertices, r.faces, panel, title=f"{name}: quadremesh, {len(r.faces)} faces",
                           field=field, features=r.feature_edges, azim=az, elev=el)
        meshes = []
        status = {r["method"]: r["status"] for r in rows if r["model"] == name}
        for m in METHODS:
            p = os.path.join(RES, "meshes", f"{name}__{m}.obj")
            if os.path.exists(p) and status.get(m) == "ok":
                V, Q = core.load_quad_obj(p)
                meshes.append((LABEL[m], V, Q))
            else:
                meshes.append((LABEL[m], None, None))
        comp = os.path.join(OUT, f"{name}_compare.png")
        side_by_side(meshes, comp, title=name, azim=az, elev=el)
        imgs[name] = (panel, comp)
        print("rendered", name, flush=True)
    charts(rows, os.path.join(OUT, "charts.png"))
    write_html(rows, agg, manifest, imgs)


def fmt(v, nd=3):
    if v in ("", None):
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def write_html(rows, agg, manifest, imgs):
    h = ["<!doctype html><html><head><meta charset='utf-8'><title>quadremesh visual report</title>",
         "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1f2328}"
         "table{border-collapse:collapse;font-size:12px;margin:12px 0}td,th{border:1px solid #d0d7de;padding:3px 6px;text-align:right}"
         "th{background:#f6f8fa}td:first-child,td:nth-child(2){text-align:left}img{max-width:100%;border:1px solid #eee;margin:6px 0}"
         ".ok{color:#1a7f37;font-weight:600}.bad{color:#cf222e;font-weight:600}h2{border-bottom:1px solid #d0d7de}</style></head><body>",
         "<h1>quadremesh - visual report</h1>",
         "<p>Benchmark of quadremesh against Instant Meshes, QuadriFlow and Blender QuadriFlow at equal target face count. "
         "All outputs are scored by the same evaluator (<code>quadeval</code>).</p>"]
    crit = agg.get("success_criteria", {})
    h.append("<h2>Success criteria (benchmark mean, quadremesh)</h2><table><tr><th>criterion</th><th>result</th></tr>")
    for k, v in crit.items():
        cls = "ok" if v is True else ("bad" if v is False else "")
        h.append(f"<tr><td>{k}</td><td class='{cls}'>{v}</td></tr>")
    h.append("</table><h2>Per-method means</h2><table><tr><th>method</th><th>ok/models</th><th>quad ratio</th><th>mean SJ</th>"
             "<th>dist %</th><th>feature cov.</th><th>singularities</th><th>sing./1k</th><th>runtime s</th></tr>")
    for m, a in agg.get("per_method", {}).items():
        h.append(f"<tr><td>{LABEL.get(m, m)}</td><td>{a.get('succeeded')}/{a.get('models')}</td><td>{fmt(a.get('quad_ratio'))}</td>"
                 f"<td>{fmt(a.get('sj_mean'))}</td><td>{fmt(a.get('dist_mean_pct'), 4)}</td><td>{fmt(a.get('feature_coverage'))}</td>"
                 f"<td>{fmt(a.get('singularities'), 1)}</td><td>{fmt(a.get('sing_per_1k'), 2)}</td><td>{fmt(a.get('runtime_s'), 1)}</td></tr>")
    h.append("</table>")
    if os.path.exists(os.path.join(OUT, "charts.png")):
        h.append(f"<img src='{b64(os.path.join(OUT, 'charts.png'))}'>")
    h.append("<h2>All runs</h2><table><tr><th>model</th><th>method</th><th>status</th><th>faces</th><th>quad</th><th>SJ mean</th>"
             "<th>SJ min</th><th>dist %</th><th>Hausdorff %</th><th>feat. cov.</th><th>bnd cov.</th><th>sing.</th><th>s</th></tr>")
    for r in rows:
        h.append(f"<tr><td>{r['model']}</td><td>{LABEL.get(r['method'], r['method'])}</td><td>{r['status']}</td>"
                 f"<td>{r.get('faces', '-')}</td><td>{fmt(r.get('quad_ratio'))}</td><td>{fmt(r.get('sj_mean'))}</td>"
                 f"<td>{fmt(r.get('sj_min'), 2)}</td><td>{fmt(r.get('dist_mean_pct'), 4)}</td><td>{fmt(r.get('hausdorff_pct'), 2)}</td>"
                 f"<td>{fmt(r.get('feature_coverage'))}</td><td>{fmt(r.get('boundary_coverage'))}</td>"
                 f"<td>{r.get('singularities', '-')}</td><td>{fmt(r.get('runtime_s'), 1)}</td></tr>")
    h.append("</table>")
    for entry in manifest:
        name = entry["name"]
        if name not in imgs:
            continue
        panel, comp = imgs[name]
        h.append(f"<h2>{name}</h2><p>{entry['category']} - {entry['notes']} - {entry['faces']} input faces</p>")
        h.append(f"<img src='{b64(panel)}'><img src='{b64(comp)}'>")
    demos = [("Adaptive density", "demo_density.png"), ("Symmetry hint", "demo_symmetry.png"),
             ("Guidance strokes", "demo_strokes.png"), ("UV transfer + pure-quad", "demo_uv_purequad.png")]
    h.append("<h2>User controls</h2>")
    for title, f in demos:
        pth = os.path.join(OUT, f)
        if os.path.exists(pth):
            h.append(f"<h3>{title}</h3><img src='{b64(pth)}'>")
    h.append("</body></html>")
    with open(os.path.join(OUT, "visual_report.html"), "w") as fh:
        fh.write("\n".join(h))
    print("wrote", os.path.join(OUT, "visual_report.html"))


if __name__ == "__main__":
    main()
