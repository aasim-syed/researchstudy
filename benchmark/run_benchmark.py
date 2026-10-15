#!/usr/bin/env python3
"""Benchmark quadremesh against Instant Meshes, QuadriFlow and Blender QuadriFlow.

Every method gets the same input file and the same target face count; every output is scored
by the same evaluator (`quadeval`, the C++ metrics shared with the pipeline). Failures and
timeouts are recorded, not hidden.

usage:
  python benchmark/run_benchmark.py [--models a,b,...] [--methods ours,instant_meshes,...]
                                    [--out benchmark/results] [--timeout 1200]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.environ.get("QRM_BUILD", os.path.join(ROOT, "build"))
QUADREMESH = os.path.join(BUILD, "quadremesh")
QUADEVAL = os.path.join(BUILD, "quadeval")
QF_CLI = os.environ.get("QF_CLI", os.path.join(ROOT, "third_party/QuadriFlow/build/quadriflow"))
BLENDER_PY = os.environ.get("BLENDER_PY", "/home/claude/bpyenv/bin/python")
FEATURE_ANGLE = 35.0

OURS_LEARNED_SNIPPET = r"""
import sys
sys.path.insert(0, %r)
from quadremesh import quadremesh
src, dst, target, angle = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
quadremesh(src, target_faces=target, feature_angle=angle, learned_field=True, output=dst, report=False,
           transfer_uv=False)
""" % os.path.join(ROOT, "python")

IM_SNIPPET = r"""
import sys, numpy as np, trimesh, pynanoinstantmeshes as p
src, dst, target, angle = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
m = trimesh.load(src, force='mesh', process=False)
# Instant Meshes' pure-quad mode extracts a quad-dominant mesh and subdivides it once,
# so the requested count is target/4 to land on `target` output faces.
v, f = p.remesh(m.vertices.astype(np.float32), m.faces.astype(np.uint32), -1,
                face_count=max(1, target // 4), creaseAngle=angle, align_to_boundaries=True,
                deterministic=True)
with open(dst, 'w') as fh:
    fh.write(''.join('v %.9g %.9g %.9g\n' % tuple(x) for x in v))
    fh.write(''.join('f ' + ' '.join(str(int(i) + 1) for i in q) + '\n' for q in f))
"""


def target_for(entry):
    return 15000 if entry["category"] in ("scene scan", "room-scale scan") else 5000


def run(cmd, timeout, log):
    t = time.time()
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        ok = r.returncode == 0
        out = r.stdout[-4000:]
    except subprocess.TimeoutExpired:
        ok, out = False, "TIMEOUT"
    dt = time.time() - t
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n" + out)
    return ok, dt


def method_cmd(method, src, dst, target):
    if method == "ours":
        return [QUADREMESH, src, "-o", dst, "-f", str(target), "-a", str(FEATURE_ANGLE), "--no-uv"]
    if method == "ours_featref":
        return [QUADREMESH, src, "-o", dst, "-f", str(target), "-a", str(FEATURE_ANGLE), "--no-uv", "--feature-refine", "2"]
    if method == "ours_coverage":
        return [QUADREMESH, src, "-o", dst, "-f", str(target), "-a", str(FEATURE_ANGLE), "--no-uv", "--coverage-mode"]
    if method == "ours_learned":
        return [sys.executable, "-c", OURS_LEARNED_SNIPPET, src, dst, str(target), str(FEATURE_ANGLE)]
    if method == "instant_meshes":
        return [sys.executable, "-c", IM_SNIPPET, src, dst, str(target), str(FEATURE_ANGLE)]
    if method == "quadriflow":
        return [QF_CLI, "-i", src, "-o", dst, "-f", str(target), "-sharp", "-boundary"]
    if method == "blender_quadriflow":
        return [BLENDER_PY, os.path.join(ROOT, "benchmark/blender_quadriflow.py"), src, dst, str(target)]
    raise ValueError(method)


ALL_METHODS = "ours,ours_learned,ours_featref,ours_coverage,instant_meshes,quadriflow,blender_quadriflow"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--methods", default=ALL_METHODS)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark/results"))
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(ROOT, "data/manifest.json")))
    full_manifest = list(manifest)
    if args.models:
        keep = set(args.models.split(","))
        manifest = [m for m in manifest if m["name"] in keep]
    methods = args.methods.split(",")
    os.makedirs(os.path.join(args.out, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "reports"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    for entry in manifest:
        src = os.path.join(ROOT, "data", entry["file"])
        target = target_for(entry)
        for method in methods:
            tag = f"{entry['name']}__{method}"
            dst = os.path.join(args.out, "meshes", tag + ".obj")
            rep = os.path.join(args.out, "reports", tag + ".json")
            if args.skip_existing and os.path.exists(rep):
                continue
            if os.path.exists(dst):
                os.remove(dst)
            ok, dt = run(method_cmd(method, src, dst, target), args.timeout, os.path.join(args.out, "logs", tag + ".log"))
            if ok and os.path.exists(dst) and os.path.getsize(dst) > 0:
                ev = subprocess.run([QUADEVAL, src, dst, "--target", str(target), "--feature-angle", str(FEATURE_ANGLE),
                                     "--method", method, "--runtime", f"{dt:.3f}", "-o", rep])
                status = "ok" if ev.returncode == 0 else "eval_failed"
            else:
                status = "failed"
                json.dump({"method": method, "failed": True, "runtime_s": dt}, open(rep, "w"))
            print(f"{entry['name']:15s} {method:20s} {status:8s} {dt:7.1f}s", flush=True)
    # the summary always covers every model/method that has a report, so a partial re-run
    # (a subset of models or methods) does not drop the other results from the tables
    summarize(args.out, full_manifest, ALL_METHODS.split(","))


def summarize(out, manifest, methods):
    rows = []
    for entry in manifest:
        for method in methods:
            rep = os.path.join(out, "reports", f"{entry['name']}__{method}.json")
            if not os.path.exists(rep):
                continue
            r = json.load(open(rep))
            row = dict(model=entry["name"], category=entry["category"], input_faces=entry["faces"],
                       target=target_for(entry), method=method)
            if r.get("failed") or "quality" not in r:
                row.update(status="failed", runtime_s=r.get("runtime_s"))
            else:
                row.update(status="ok", faces=r["faces"]["total"], quad_ratio=r["faces"]["quad_ratio"],
                           sj_mean=r["quality"]["scaled_jacobian"]["mean"], sj_min=r["quality"]["scaled_jacobian"]["min"],
                           frac_sj_lt_05=r["quality"]["frac_sj_below_0.5"],
                           inverted=r["quality"]["inverted_or_degenerate"],
                           planarity_mean=r["quality"]["planarity"]["mean"],
                           dist_mean_pct=r["distance"]["mean_pct"], hausdorff_pct=r["distance"]["hausdorff_pct"],
                           feature_coverage=r["features"]["feature_coverage"],
                           boundary_coverage=r["features"]["boundary_coverage"],
                           singularities=r["topology"]["singularities"],
                           sing_per_1k=r["topology"]["singularities_per_1k_faces"],
                           nonmanifold_edges=r["topology"]["nonmanifold_edges"],
                           runtime_s=r.get("runtime_s"))
            rows.append(row)
    keys = ["model", "category", "input_faces", "target", "method", "status", "faces", "quad_ratio", "sj_mean", "sj_min",
            "frac_sj_lt_05", "inverted", "planarity_mean", "dist_mean_pct", "hausdorff_pct", "feature_coverage",
            "boundary_coverage", "singularities", "sing_per_1k", "nonmanifold_edges", "runtime_s"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
    # aggregate per method
    agg = {}
    for method in methods:
        rs = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        n_all = len([r for r in rows if r["method"] == method])
        if not rs:
            agg[method] = dict(models=n_all, succeeded=0)
            continue
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        agg[method] = dict(models=n_all, succeeded=len(rs), quad_ratio=mean("quad_ratio"), sj_mean=mean("sj_mean"),
                           dist_mean_pct=mean("dist_mean_pct"), feature_coverage=mean("feature_coverage"),
                           singularities=mean("singularities"), sing_per_1k=mean("sing_per_1k"),
                           runtime_s=mean("runtime_s"))
    # success criteria for ours + singularity comparison vs Instant Meshes, model by model
    crit = {}
    ours = {r["model"]: r for r in rows if r["method"] == "ours" and r["status"] == "ok"}
    im = {r["model"]: r for r in rows if r["method"] == "instant_meshes" and r["status"] == "ok"}
    if ours:
        a = agg["ours"]
        crit = {
            "quad_ratio>=0.98": a["quad_ratio"] >= 0.98,
            "mean_scaled_jacobian>=0.8": a["sj_mean"] >= 0.8,
            "mean_distance<0.5%": a["dist_mean_pct"] < 0.5,
            "feature_coverage>=0.9": a["feature_coverage"] >= 0.9,
        }
        both = [m for m in ours if m in im]
        if both:
            # compare per 1k faces to remove small face-count differences
            fewer = [m for m in both if ours[m]["sing_per_1k"] < im[m]["sing_per_1k"]]
            crit["fewer_singularities_than_instant_meshes"] = len(fewer) == len(both)
            crit["fewer_singularities_models"] = f"{len(fewer)}/{len(both)}"
            crit["singularities_ours_total"] = sum(ours[m]["singularities"] for m in both)
            crit["singularities_im_total"] = sum(im[m]["singularities"] for m in both)
    json.dump(dict(per_method=agg, success_criteria=crit), open(os.path.join(out, "aggregate.json"), "w"), indent=1)
    print(json.dumps(dict(per_method=agg, success_criteria=crit), indent=1))


if __name__ == "__main__":
    main()
