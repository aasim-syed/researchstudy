#!/usr/bin/env python3
"""Geometric limits of the feature-coverage metric, measured on the evaluator's own crease set.

For every benchmark model the evaluator (quadeval, QRM_EVAL_SEGMENTS) writes the 35-degree crease
segments it scores and the element size L = sqrt(area / target faces). From these polylines:

* chord bound(h): fraction of the crease length whose segment direction is within 25 deg of the
  chord through +-h/2 along the crease. A straight output edge of length ~h placed exactly on the
  crease cannot cover more (the coverage test needs <= 25 deg). Reported for h = L, L/2, L/4.
* straight pieces: the crease chains are split greedily into maximal pieces that one straight
  segment can cover (all segment directions within 25 deg of the piece chord, all points within
  0.1 L of it). The number of pieces needed for 90 % of the crease length is a lower bound on the
  number of output edges that must lie on creases; it is compared with the ~2 x target faces
  edges a quad mesh of that size has in total.

    python benchmark/crease_bounds.py            # writes docs/crease_bounds.md
"""
import collections
import json
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUADEVAL = os.environ.get("QUADEVAL", os.path.join(ROOT, "build", "quadeval"))
MAN = json.load(open(os.path.join(ROOT, "data", "manifest.json")))
COS25 = np.cos(np.radians(25))


def target_for(e):
    return 15000 if e["category"] in ("scene scan", "room-scale scan") else 5000


def crease_segments(e):
    src = os.path.join(ROOT, "data", e["file"])
    out = os.path.join(ROOT, "benchmark", "results", "meshes", f"{e['name']}__ours.obj")  # any output works
    with tempfile.TemporaryDirectory() as td:
        seg = os.path.join(td, "seg.txt")
        env = dict(os.environ, QRM_EVAL_SEGMENTS=seg)
        subprocess.run([QUADEVAL, src, out, "--target", str(target_for(e)), "--feature-angle", "35", "-o",
                        os.path.join(td, "r.json")], check=True, env=env, stdout=subprocess.DEVNULL)
        L = float(open(seg).readline().split()[2])
        S = np.loadtxt(seg, comments="#").reshape(-1, 6)
    return L, S


def chains(S):
    key = lambda p: tuple(np.round(p, 9))
    ids, P = {}, []

    def vid(p):
        k = key(p)
        if k not in ids:
            ids[k] = len(P)
            P.append(p)
        return ids[k]

    E = [(vid(r[:3]), vid(r[3:])) for r in S]
    P = np.array(P)
    adj = collections.defaultdict(list)
    for i, (a, b) in enumerate(E):
        adj[a].append(i)
        adj[b].append(i)
    used = [False] * len(E)
    out = []

    def walk(s, e):
        vs, v = [s], s
        while True:
            used[e] = True
            a, b = E[e]
            u = b if a == v else a
            vs.append(u)
            if len(adj[u]) != 2:
                break
            en = adj[u][0] if adj[u][1] == e else adj[u][1]
            if used[en]:
                break
            v, e = u, en
        return P[vs]

    for v in adj:
        if len(adj[v]) != 2:
            for e in adj[v]:
                if not used[e]:
                    out.append(walk(v, e))
    for e in range(len(E)):
        if not used[e]:
            out.append(walk(E[e][0], e))
    return out


def chord_bound(C, L, h):
    tot = ok = 0.0
    for X in C:
        seg = X[1:] - X[:-1]
        l = np.linalg.norm(seg, axis=1)
        s = np.r_[0, np.cumsum(l)]
        idx = np.arange(len(s))

        def at(t):
            f = np.interp(t, s, idx)
            i = min(int(f), len(X) - 2)
            return X[i] + (f - i) * (X[i + 1] - X[i])

        for k in range(len(seg)):
            if l[k] == 0:
                continue
            m = 0.5 * (s[k] + s[k + 1])
            ch = at(min(s[-1], m + h / 2)) - at(max(0.0, m - h / 2))
            tot += l[k]
            n = np.linalg.norm(ch)
            if n > 0 and abs(ch @ seg[k]) / (n * l[k]) >= COS25:
                ok += l[k]
    return ok / tot if tot else 1.0


def straight_pieces(C, L):
    pieces = []
    for X in C:
        i = 0
        while i < len(X) - 1:
            j = i + 1
            while j + 1 < len(X):
                Y = X[i:j + 2]
                ch = Y[-1] - Y[0]
                n = np.linalg.norm(ch)
                if n == 0:
                    break
                u = ch / n
                seg = Y[1:] - Y[:-1]
                sl = np.linalg.norm(seg, axis=1)
                if np.any(np.abs(seg @ u) < COS25 * np.maximum(sl, 1e-30)):
                    break
                lat = np.linalg.norm((Y - Y[0]) - np.outer((Y - Y[0]) @ u, u), axis=1)
                if lat.max() > 0.1 * L:
                    break
                j += 1
            pieces.append(np.linalg.norm(X[i + 1:j + 1] - X[i:j], axis=1).sum())
            i = j
    return np.array(pieces)


def main():
    rows = []
    for e in MAN:
        L, S = crease_segments(e)
        C = chains(S)
        total = sum(np.linalg.norm(X[1:] - X[:-1], axis=1).sum() for X in C)
        pcs = np.sort(straight_pieces(C, L))[::-1]
        need90 = int(np.searchsorted(np.cumsum(pcs), 0.9 * total) + 1) if total > 0 else 0
        rows.append(dict(name=e["name"], cat=e["category"], T=target_for(e), segs=len(S), len_L=total / L,
                         b1=chord_bound(C, L, L), b2=chord_bound(C, L, 0.5 * L), b4=chord_bound(C, L, 0.25 * L),
                         pieces=len(pcs), need90=need90))
        print(rows[-1], flush=True)
    out = ["# Geometric limits of the feature-coverage metric", "",
           "Generated by `python benchmark/crease_bounds.py` from the evaluator's own crease segments",
           "(35 deg, chains >= 3 L). See the script docstring for the definitions.", "",
           "| model | category | target faces | crease segments | crease length / L | chord bound, edges ~L | ~L/2 | ~L/4 | straight pieces | pieces for 90 % | edges in a mesh of that size (~2 x faces) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {r['cat']} | {r['T']} | {r['segs']} | {r['len_L']:.0f} | {r['b1']:.3f} | {r['b2']:.3f} | "
                   f"{r['b4']:.3f} | {r['pieces']} | {r['need90']} | {2 * r['T']} |")
    open(os.path.join(ROOT, "docs", "crease_bounds.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
