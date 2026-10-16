#!/usr/bin/env python3
"""CUDA check for the optional PyTorch path (docs/cuda_checklist.md).

Prints one report to paste back for review:
  1. environment (driver, torch, CUDA, GPU)
  2. correctness: CUDA vs CPU results of optimize_field and predict_guide_field
  3. timing: median of N runs on each device, speed-up
  4. end to end: the CUDA-predicted guide field fed into the C++ remesher

Usage (repo root):  PYTHONPATH=python python scripts/cuda_check.py [--model data/models/fandisk.obj] [--runs 3]
"""
import argparse
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))


def line(k, v):
    print(f"  {k:<28} {v}")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(not available: {e.__class__.__name__})"


def cross_angle_deg(a, b):
    """Angle between two cross fields per vertex, folded to [0, 45] degrees (4-fold symmetry)."""
    na = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    nb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.abs((na * nb).sum(1)), 0, 1)))  # 0..90
    return np.minimum(ang, 90 - ang)


def timed(fn, runs, sync):
    fn()  # warm-up (CUDA context, kernels)
    sync()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        sync()
        ts.append(time.perf_counter() - t)
    return out, statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "data/models/fandisk.obj"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--faces", type=int, default=5000)
    ap.add_argument("--gpu-device", default="cuda", help=argparse.SUPPRESS)  # self-test only
    a = ap.parse_args()

    print("=== QUADREMESH CUDA CHECK ===")
    print("[1] environment")
    line("python", platform.python_version())
    line("platform", platform.platform())
    line("WSL", "yes" if "microsoft" in platform.release().lower() else "no")
    smi = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    line("nvidia-smi", smi or "(no output)")
    import torch
    line("torch", torch.__version__)
    line("torch built with CUDA", torch.version.cuda)
    cuda = torch.cuda.is_available()
    line("cuda available", cuda)
    if a.gpu_device != "cuda":
        print("  (self-test mode: comparing CPU with CPU, not a GPU result)")
    elif cuda:
        p = torch.cuda.get_device_properties(0)
        line("device", f"{p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB")
        line("arch list in torch", " ".join(torch.cuda.get_arch_list()))
    else:
        print("VERDICT: FAIL - torch cannot see a CUDA device; stop here and paste this output.")
        return 1

    from quadremesh import load_mesh, quadremesh
    from quadremesh.learned import predict_guide_field
    from quadremesh.torch_field import optimize_field

    m = load_mesh(a.model)
    V, F = np.asarray(m["V"]), np.asarray(m["F"])
    print(f"[2] correctness on {os.path.basename(a.model)} ({len(V)} vertices, {len(F)} triangles)")
    nosync = lambda: None  # noqa: E731
    cusync = torch.cuda.synchronize if a.gpu_device == "cuda" else nosync

    (d_cpu, w_cpu, _), t_opt_cpu = timed(lambda: optimize_field(V, F, device="cpu"), a.runs, nosync)
    (d_gpu, w_gpu, info), t_opt_gpu = timed(lambda: optimize_field(V, F, device=a.gpu_device), a.runs, cusync)
    ang = cross_angle_deg(d_cpu, d_gpu)
    ok_opt = info.get("device", "").startswith(a.gpu_device) and np.percentile(ang, 99) < 1.0
    line("optimize_field device", info.get("device"))
    line("field angle CPU vs CUDA", f"mean {ang.mean():.4f} deg, p99 {np.percentile(ang, 99):.4f}, max {ang.max():.4f}")
    line("weight max abs diff", f"{np.abs(w_cpu - w_gpu).max():.2e}")

    (g_cpu, c_cpu), t_net_cpu = timed(lambda: predict_guide_field(V, F, device="cpu"), a.runs, nosync)
    (g_gpu, c_gpu), t_net_gpu = timed(lambda: predict_guide_field(V, F, device=a.gpu_device), a.runs, cusync)
    ang2 = cross_angle_deg(g_cpu, g_gpu)
    ok_net = np.percentile(ang2, 99) < 1.0
    line("learned field CPU vs CUDA", f"mean {ang2.mean():.4f} deg, p99 {np.percentile(ang2, 99):.4f}, max {ang2.max():.4f}")
    line("confidence max abs diff", f"{np.abs(c_cpu - c_gpu).max():.2e}")

    print(f"[3] timing (median of {a.runs}, after warm-up)")
    line("optimize_field", f"CPU {t_opt_cpu:.3f} s | CUDA {t_opt_gpu:.3f} s | speed-up {t_opt_cpu / t_opt_gpu:.2f}x")
    line("predict_guide_field", f"CPU {t_net_cpu:.3f} s | CUDA {t_net_gpu:.3f} s | speed-up {t_net_cpu / t_net_gpu:.2f}x")
    if a.gpu_device == "cuda":
        line("peak GPU memory", f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

    print("[4] end to end (CUDA guide field -> C++ remesher)")
    t = time.perf_counter()
    r = quadremesh(a.model, target_faces=a.faces, feature_angle=35.0, guide_field=g_gpu, guide_weight=c_gpu, report=True)
    ok_e2e = len(r.faces) > 0
    line("output faces", f"{len(r.faces)} (target {a.faces}), quads {100 * r.quad_ratio:.1f}%")
    rep = r.report or {}
    try:
        line("mean scaled Jacobian", rep["quality"]["scaled_jacobian"]["mean"])
        line("mean distance (% diag)", rep["distance"]["mean_pct"])
        line("feature coverage", rep["features"]["feature_coverage"])
        line("singularities", rep["topology"]["singularities"])
    except (KeyError, TypeError):
        line("report", "(not available)")
    line("remesh time", f"{time.perf_counter() - t:.2f} s")

    ok = ok_opt and ok_net and ok_e2e
    print("VERDICT:", "PASS - CUDA path runs and matches CPU" if ok else
          f"CHECK - opt_match={ok_opt} net_match={ok_net} e2e={ok_e2e}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
