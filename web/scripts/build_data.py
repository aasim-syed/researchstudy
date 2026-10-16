"""Build web/public/data from the benchmark results (no hand-typed numbers).

Reads benchmark/results/summary.json, copies the quad meshes and turns the visual-report
images into web-sized WebP files (the per-model panel is split into its sub-plots).
Run from the repository root:  python web/scripts/build_data.py
"""
import json, shutil
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "data"
METHODS = ["ours", "ours_coverage", "instant_meshes", "quadriflow", "blender_quadriflow"]
KEYS = ["status", "faces", "quad_ratio", "sj_mean", "sj_min", "dist_mean_pct", "hausdorff_pct",
        "feature_coverage", "singularities", "sing_per_1k", "runtime_s"]
PANEL_NAMES = ["edges", "sj", "field"]


def webp(src: Path, dst: Path, width: int) -> None:
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "WEBP", quality=82, method=6)


def split_panel(src: Path):
    """Split an '<model>_ours_panel.png' into its sub-plots using the white gaps between them."""
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(int)
    body = a[int(a.shape[0] * 0.12):]
    ink = (np.abs(255 - body).sum(axis=2) > 30).sum(axis=0) > 2
    runs, start, gap = [], None, 0
    for x, v in enumerate(ink):
        if v:
            if start is None:
                start = x
            gap = 0
            end = x
        elif start is not None:
            gap += 1
            if gap > 15:
                runs.append((start, end)); start = None
    if start is not None:
        runs.append((start, end))
    merged = []
    for r in runs:  # a narrow run is a colour bar or legend: attach it to the plot before it
        if merged and r[1] - r[0] < 150:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] > 150]
    return im, runs


def main():
    rows = json.loads((ROOT / "benchmark/results/summary.json").read_text())
    models = {}
    for r in rows:
        if r["method"] not in METHODS:
            continue
        m = models.setdefault(r["model"], {"name": r["model"], "category": r["category"],
                                           "input_faces": r["input_faces"], "target": r["target"],
                                           "methods": {}})
        m["methods"][r["method"]] = {k: r.get(k) for k in KEYS}

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "meshes").mkdir(parents=True)
    rep = ROOT / "docs/report"
    for name, m in models.items():
        for meth in list(m["methods"]):
            src = ROOT / f"benchmark/results/meshes/{name}__{meth}.obj"
            if src.exists():
                shutil.copy(src, OUT / "meshes" / src.name)
                m["methods"][meth]["mesh"] = f"data/meshes/{src.name}"
            rsrc = ROOT / f"benchmark/results/reports/{name}__{meth}.json"
            if rsrc.exists():
                (OUT / "reports").mkdir(exist_ok=True)
                shutil.copy(rsrc, OUT / "reports" / rsrc.name)
                m["methods"][meth]["report"] = f"data/reports/{rsrc.name}"
        imgs = {}
        if (rep / f"{name}_compare.png").exists():
            webp(rep / f"{name}_compare.png", OUT / "img" / f"{name}_compare.webp", 2000)
            imgs["compare"] = f"data/img/{name}_compare.webp"
        p = rep / f"{name}_ours_panel.png"
        if p.exists():
            im, runs = split_panel(p)
            for label, (x0, x1) in zip(PANEL_NAMES, runs):
                crop = im.crop((max(0, x0 - 10), 0, min(im.width, x1 + 10), im.height))
                ink_rows = (np.abs(255 - np.asarray(crop).astype(int)).sum(axis=2) > 30).sum(axis=1) > 0
                y = 0
                while y < len(ink_rows) and not ink_rows[y]:
                    y += 1
                y1 = y
                while y1 < len(ink_rows) and ink_rows[y1]:
                    y1 += 1
                if y1 < 0.1 * len(ink_rows):  # the figure's main title, cut by the split: drop it
                    crop = crop.crop((0, y1 + 4, crop.width, crop.height))
                dst = OUT / "img" / f"{name}_{label}.webp"
                dst.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dst, "WEBP", quality=82, method=6)
                imgs[label] = f"data/img/{name}_{label}.webp"
            print(name, "panel parts:", len(runs))
        m["images"] = imgs

    demos = {}
    for d in ["demo_density", "demo_strokes", "demo_symmetry", "demo_uv_purequad", "charts"]:
        if (rep / f"{d}.png").exists():
            webp(rep / f"{d}.png", OUT / "img" / f"{d}.webp", 1800)
            demos[d] = f"data/img/{d}.webp"

    order = list(dict.fromkeys(r["model"] for r in rows))
    (OUT / "benchmark.json").write_text(json.dumps(
        {"source": "benchmark/results/summary.json", "models": [models[n] for n in order if n in models],
         "demos": demos}, indent=1))
    print("models:", len(models))


if __name__ == "__main__":
    main()
