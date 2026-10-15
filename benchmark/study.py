#!/usr/bin/env python3
"""Reproducible improvement study.

Every experiment runs the quadremesh CLI with a given configuration on the benchmark models,
scores every output with `quadeval` (same evaluator as the main benchmark), and appends one
record to docs/study/experiments.jsonl. docs/study/STUDY.md is regenerated from that file only,
so every figure in it comes from a recorded run (with git commit, binary hash, command line).

usage:
  python benchmark/study.py run --name E1_default --desc "..." [--args "--feature-loops"]
                               [--env QRM_NO_FEATURE_PATHS=1] [--models a,b]
  python benchmark/study.py report
"""
import argparse
import datetime
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
STUDY = os.path.join(ROOT, "docs", "study")
LOG = os.path.join(STUDY, "experiments.jsonl")
BASE_REPORTS = os.path.join(ROOT, "benchmark", "results", "reports")  # baselines (IM etc.)


def target_for(entry):
    return 15000 if entry["category"] in ("scene scan", "room-scale scan") else 5000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:12]


def git_commit():
    try:
        c = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def run(args):
    manifest = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
    if args.models:
        keep = set(args.models.split(","))
        manifest = [m for m in manifest if m["name"] in keep]
    out = os.path.join(STUDY, "runs", args.name)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    for kv in args.env or []:
        k, v = kv.split("=", 1)
        env[k] = v
    extra = args.args.split() if args.args else []
    per_model = {}
    for e in manifest:
        src = os.path.join(ROOT, "data", e["file"])
        tgt = target_for(e)
        mesh = os.path.join(out, e["name"] + ".obj")
        rep = os.path.join(out, e["name"] + ".json")
        cmd = [os.path.join(BUILD, "quadremesh"), src, "-o", mesh, "-f", str(tgt), "-a", "35", "--no-uv"] + extra
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        dt = time.time() - t0
        if r.returncode != 0 or not os.path.exists(mesh):
            per_model[e["name"]] = {"failed": True, "runtime_s": dt}
            print(f"{e['name']:15s} FAILED", flush=True)
            continue
        subprocess.run([os.path.join(BUILD, "quadeval"), src, mesh, "--target", str(tgt), "--feature-angle", "35",
                        "--runtime", f"{dt:.3f}", "--method", args.name, "-o", rep], check=True)
        j = json.load(open(rep))
        rec = dict(faces=j["faces"]["total"], target=tgt, quad_ratio=j["faces"]["quad_ratio"],
                   sj_mean=j["quality"]["scaled_jacobian"]["mean"], dist_mean_pct=j["distance"]["mean_pct"],
                   feature_coverage=j["features"]["feature_coverage"],
                   feature_length_L=j["features"]["feature_length_pct_diag"],
                   singularities=j["topology"]["singularities"],
                   sing_per_1k=j["topology"]["singularities_per_1k_faces"], runtime_s=dt)
        per_model[e["name"]] = rec
        print(f"{e['name']:15s} F={rec['faces']:6d} q={rec['quad_ratio']:.3f} sj={rec['sj_mean']:.3f} "
              f"d={rec['dist_mean_pct']:.3f} feat={rec['feature_coverage']:.3f} sing={rec['singularities']} ({dt:.0f}s)",
              flush=True)
    ok = {k: v for k, v in per_model.items() if not v.get("failed")}
    agg = {}
    if ok:
        agg = dict(models=len(per_model), succeeded=len(ok),
                   quad_ratio=st.mean(v["quad_ratio"] for v in ok.values()),
                   sj_mean=st.mean(v["sj_mean"] for v in ok.values()),
                   dist_mean_pct=st.mean(v["dist_mean_pct"] for v in ok.values()),
                   feature_coverage=st.mean(v["feature_coverage"] for v in ok.values()),
                   sing_per_1k=st.mean(v["sing_per_1k"] for v in ok.values()),
                   face_ratio=st.mean(v["faces"] / v["target"] for v in ok.values()))
        # singularities vs Instant Meshes (same models, baseline reports from the main benchmark)
        wins, tot_o, tot_im = 0, 0, 0
        for k, v in ok.items():
            p = os.path.join(BASE_REPORTS, f"{k}__instant_meshes.json")
            if not os.path.exists(p):
                continue
            im = json.load(open(p))
            if "topology" not in im:
                continue
            wins += v["sing_per_1k"] < im["topology"]["singularities_per_1k_faces"]
            tot_o += v["singularities"]
            tot_im += im["topology"]["singularities"]
        agg.update(sing_wins_vs_im=wins, sing_total=tot_o, sing_total_im=tot_im)
    record = dict(name=args.name, desc=args.desc, args=" ".join(extra), env=args.env or [],
                  time=datetime.datetime.now().isoformat(timespec="seconds"), commit=git_commit(),
                  binary=sha(os.path.join(BUILD, "quadremesh")), models=list(per_model), aggregate=agg,
                  per_model=per_model)
    os.makedirs(STUDY, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    report()


def report():
    if not os.path.exists(LOG):
        return
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    h = ["# Feature-coverage improvement study", "",
         "Generated by `benchmark/study.py report` from `experiments.jsonl` - every number below is",
         "a recorded run of the quadremesh CLI scored by `quadeval` (same evaluator as the benchmark).",
         "Coverage definition (unchanged in every run): every 35° crease chain >= 3 L must have an",
         "output edge within 0.1 L and within 25°. Targets: 5 000 faces (objects), 15 000 (scenes).", "",
         "## Summary (benchmark means)", "",
         "| # | experiment | config | quads | SJ | dist % | **feature cov.** | sing/1k | sing. wins vs IM | sing. total (IM) | faces/target | commit |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(recs):
        a = r["aggregate"]
        if not a:
            continue
        cfg = (" ".join(r["env"]) + " " + r["args"]).strip() or "default"
        h.append(f"| {i} | {r['name']} | `{cfg}` | {a['quad_ratio']*100:.1f} % | {a['sj_mean']:.3f} | "
                 f"{a['dist_mean_pct']:.3f} | **{a['feature_coverage']*100:.1f} %** | {a['sing_per_1k']:.1f} | "
                 f"{a['sing_wins_vs_im']}/{a['succeeded']} | {a['sing_total']} ({a['sing_total_im']}) | "
                 f"{a['face_ratio']:.2f} | {r['commit']} |")
    h += ["", "## What each experiment changed", ""]
    for i, r in enumerate(recs):
        h.append(f"{i}. **{r['name']}** ({r['time']}, binary {r['binary']}): {r['desc']}")
    names = sorted({m for r in recs for m in r["per_model"]})
    h += ["", "## Feature coverage per model", "",
          "| model | " + " | ".join(str(i) for i in range(len(recs))) + " |",
          "|---|" + "---|" * len(recs)]
    for m in names:
        row = [m]
        for r in recs:
            v = r["per_model"].get(m)
            row.append("-" if not v else ("fail" if v.get("failed") else f"{v['feature_coverage']*100:.0f}"))
        h.append("| " + " | ".join(row) + " |")
    h += ["", "## Faces / singularities per model", "",
          "| model | " + " | ".join(str(i) for i in range(len(recs))) + " |",
          "|---|" + "---|" * len(recs)]
    for m in names:
        row = [m]
        for r in recs:
            v = r["per_model"].get(m)
            row.append("-" if not v else ("fail" if v.get("failed") else f"{v['faces']} / {v['singularities']}"))
        h.append("| " + " | ".join(row) + " |")
    with open(os.path.join(STUDY, "STUDY.md"), "w") as fh:
        fh.write("\n".join(h) + "\n")
    print("wrote", os.path.join(STUDY, "STUDY.md"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--desc", required=True)
    r.add_argument("--args", default="")
    r.add_argument("--env", action="append")
    r.add_argument("--models", default="")
    r.add_argument("--timeout", type=int, default=1500)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "report":
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
