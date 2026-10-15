#!/usr/bin/env python3
"""Reproducible study harness for the feature-coverage work.

Every number in docs/coverage_study.md is produced by this script from real runs:

  python benchmark/coverage_study.py run  NAME "CLI FLAGS" [--models a,b] [--jobs 2]
  python benchmark/coverage_study.py report            # regenerates docs/coverage_study.md

`run` remeshes each benchmark model with build/quadremesh + the given flags, scores the output
with build/quadeval (same evaluator as the main benchmark), and stores the per-model JSON
reports plus the exact command lines under benchmark/study/NAME/. `report` reads only those
files (and the Instant Meshes reports from benchmark/results/) and writes the tables.
"""
import concurrent.futures as cf
import datetime
import json
import os
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.environ.get("QRM_STUDY_BUILD", os.path.join(ROOT, "build"))
STUDY = os.path.join(ROOT, "benchmark", "study")
MAN = json.load(open(os.path.join(ROOT, "data", "manifest.json")))


def target_for(e):
    return 15000 if e["category"] in ("scene scan", "room-scale scan") else 5000


def run_one(name, flags, e):
    src = os.path.join(ROOT, "data", e["file"])
    out_dir = os.path.join(STUDY, name)
    mesh = os.path.join(out_dir, "meshes", e["name"] + ".obj")
    rep = os.path.join(out_dir, "reports", e["name"] + ".json")
    tgt = target_for(e)
    env = os.environ.copy()
    extra = []
    for tok in flags.split():
        if "=" in tok and not tok.startswith("-"):
            k, v = tok.split("=", 1)
            env[k] = v  # environment switch (e.g. QRM_NO_FEATURE_PATHS=1 to reproduce the pre-path baseline)
        else:
            extra.append(tok)
    cmd = [os.path.join(BUILD, "quadremesh"), src, "-o", mesh, "-f", str(tgt), "-a", "35", "--no-uv"] + extra
    t = time.time()
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        json.dump({"failed": True, "timeout_s": 1800, "cmd": cmd}, open(rep, "w"))
        return e["name"], False
    dt = time.time() - t
    if r.returncode != 0 or not os.path.exists(mesh):
        json.dump({"failed": True, "cmd": cmd, "log": r.stdout[-2000:]}, open(rep, "w"))
        return e["name"], False
    ev = [os.path.join(BUILD, "quadeval"), src, mesh, "--target", str(tgt), "--feature-angle", "35",
          "--runtime", f"{dt:.2f}", "--method", name, "-o", rep]
    subprocess.run(ev, check=True)
    d = json.load(open(rep))
    d["cmd"] = " ".join([f"{k}={env[k]}" for k in env if k.startswith("QRM_")] +
                        [os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd])
    json.dump(d, open(rep, "w"), indent=1)
    return e["name"], True


def cmd_run(name, flags, models=None, jobs=2):
    os.makedirs(os.path.join(STUDY, name, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(STUDY, name, "reports"), exist_ok=True)
    ents = [e for e in MAN if not models or e["name"] in models]
    meta = {"name": name, "flags": flags, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "git": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE,
                                  text=True).stdout.strip(),
            "models": [e["name"] for e in ents]}
    mp = os.path.join(STUDY, name, "meta.json")
    if os.path.exists(mp):  # re-run of a subset (e.g. scans after an out-of-memory kill): keep the record
        old = json.load(open(mp))
        if old.get("flags") != flags:
            raise SystemExit(f"run {name} exists with different flags")
        meta["models"] = sorted(set(old.get("models", [])) | set(meta["models"]))
        meta["reruns"] = old.get("reruns", []) + [{"date": meta["date"], "git": meta["git"], "models": [e["name"] for e in ents]}]
        meta["date"], meta["git"] = old["date"], old["git"]
    json.dump(meta, open(mp, "w"), indent=1)
    # 1M-face scans need ~2.5 GB each: run them one at a time after the objects
    small = [e for e in ents if e["category"] not in ("scene scan", "room-scale scan")]
    big = [e for e in ents if e["category"] in ("scene scan", "room-scale scan")]
    with cf.ThreadPoolExecutor(jobs) as ex:
        for nm, ok in ex.map(lambda e: run_one(name, flags, e), small):
            print(f"{name:24s} {nm:15s} {'ok' if ok else 'FAILED'}", flush=True)
    for e in big:
        nm, ok = run_one(name, flags, e)
        print(f"{name:24s} {nm:15s} {'ok' if ok else 'FAILED'}", flush=True)


def load(name):
    out = {}
    d = os.path.join(STUDY, name, "reports")
    for e in MAN:
        p = os.path.join(d, e["name"] + ".json")
        if os.path.exists(p):
            out[e["name"]] = json.load(open(p))
    return out


def im_reports():
    out = {}
    for e in MAN:
        p = os.path.join(ROOT, "benchmark", "results", "reports", f"{e['name']}__instant_meshes.json")
        if os.path.exists(p):
            out[e["name"]] = json.load(open(p))
    return out


def summary(rs, im):
    ok = {k: v for k, v in rs.items() if "quality" in v}
    if not ok:
        return None
    fewer = sum(1 for k, v in ok.items() if k in im and "topology" in im[k]
                and v["topology"]["singularities_per_1k_faces"] < im[k]["topology"]["singularities_per_1k_faces"])
    both = sum(1 for k in ok if k in im and "topology" in im[k])
    scans = {e["name"] for e in MAN if e["category"] in ("scene scan", "room-scale scan")}
    obj = [v for k, v in ok.items() if k not in scans]
    failed = sum(1 for v in rs.values() if "quality" not in v)
    return dict(
        n=f"{len(ok)}" + (f" (+{failed} failed)" if failed else ""),
        feat_obj=st.mean(v["features"]["feature_coverage"] for v in obj) if obj else float("nan"),
        quad=st.mean(v["faces"]["quad_ratio"] for v in ok.values()),
        sj=st.mean(v["quality"]["scaled_jacobian"]["mean"] for v in ok.values()),
        dist=st.mean(v["distance"]["mean_pct"] for v in ok.values()),
        feat=st.mean(v["features"]["feature_coverage"] for v in ok.values()),
        s1k=st.mean(v["topology"]["singularities_per_1k_faces"] for v in ok.values()),
        sing=sum(v["topology"]["singularities"] for v in ok.values()),
        faces_ratio=st.mean(v["faces"]["count_ratio"] for v in ok.values()),
        fewer=f"{fewer}/{both}",
    )


def cmd_report():
    im = im_reports()
    notes = {}
    npath = os.path.join(STUDY, "notes.json")
    if os.path.exists(npath):
        notes = json.load(open(npath))
    names = [n for n in notes.get("order", []) if os.path.isdir(os.path.join(STUDY, n))]
    L = ["# Feature-coverage study", "",
         "Generated by `python benchmark/coverage_study.py report` from the run directories in",
         "`benchmark/study/` (per-model quadeval JSON + exact command line in each report). No number",
         "below is typed by hand. Instant Meshes reference: `benchmark/results/reports/*__instant_meshes.json`.",
         "", "Criteria: quad ≥ 0.98, mean SJ ≥ 0.8, mean distance < 0.5 %, feature coverage ≥ 0.90,",
         "fewer singularities than Instant Meshes at equal face count.", "",
         "## Summary (mean over the models in each run)", "",
         "| iteration | flags | models | quad | SJ | dist % | **feature cov.** | feature cov. (13 non-scan models) | sing./1k | total sing. | faces / target | fewer sing. than IM |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ims = summary({k: v for k, v in im.items()}, im)
    for n in names:
        meta = json.load(open(os.path.join(STUDY, n, "meta.json")))
        s = summary(load(n), im)
        if not s:
            continue
        L.append(f"| {n} | `{meta['flags'] or '(default)'}` | {s['n']} | {s['quad']:.3f} | {s['sj']:.3f} | {s['dist']:.3f} | "
                 f"**{s['feat']:.3f}** | {s['feat_obj']:.3f} | {s['s1k']:.1f} | {s['sing']} | {s['faces_ratio']:.2f} | {s['fewer']} |")
    if ims:
        L.append(f"| *Instant Meshes* | – | {ims['n']} | {ims['quad']:.3f} | {ims['sj']:.3f} | {ims['dist']:.3f} | "
                 f"*{ims['feat']:.3f}* | {ims['feat_obj']:.3f} | {ims['s1k']:.1f} | {ims['sing']} | {ims['faces_ratio']:.2f} | – |")
    L += ["", "## Per-model feature coverage", "",
          "| model | " + " | ".join(names) + " | Instant Meshes |", "|---|" + "---|" * (len(names) + 1)]
    data = {n: load(n) for n in names}
    for e in MAN:
        row = [e["name"]]
        for n in names:
            v = data[n].get(e["name"])
            row.append(f"{v['features']['feature_coverage']:.2f}" if v and "features" in v else "–")
        v = im.get(e["name"])
        row.append(f"{v['features']['feature_coverage']:.2f}" if v and "features" in v else "–")
        L.append("| " + " | ".join(row) + " |")
    if notes.get("code_note"):
        L += ["", "Code provenance: " + notes["code_note"]]
    L += ["", "## Iterations", ""]
    for n in names:
        meta = json.load(open(os.path.join(STUDY, n, "meta.json")))
        L.append(f"### {n}")
        L.append(f"- flags: `{meta['flags'] or '(default)'}` · code: `{meta['git']}` · run: {meta['date']}")
        for line in notes.get(n, []):
            L.append(f"- {line}")
        L.append("")
    open(os.path.join(ROOT, "docs", "coverage_study.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20 + len(names)]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        models = None
        jobs = 2
        args = sys.argv[4:]
        for i, a in enumerate(args):
            if a == "--models":
                models = set(args[i + 1].split(","))
            if a == "--jobs":
                jobs = int(args[i + 1])
        cmd_run(sys.argv[2], sys.argv[3], models, jobs)
    elif sys.argv[1] == "report":
        cmd_report()
