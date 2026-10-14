"""Learned cross-field prediction (optional PyTorch component).

A small frame-equivariant graph network predicts a 4-RoSy guidance field + confidence per
vertex. It is trained to recover the *clean-surface* field (feature directions, coarse-scale
principal curvature directions) from *noisy* inputs, i.e. it is a learned field denoiser for
scans. Its output is passed to the C++ core as a soft guide (`guide_field`, `guide_weight`),
where the angle-limited alignment pass guarantees it cannot add singularities.

Equivariance: inputs are invariant scalars plus candidate directions encoded as
z = r * exp(4i*phi) in each vertex's tangent frame; the network only (a) mixes candidates with
real, invariant gates and (b) passes messages with parallel transport exp(4i*rho_ij), so a
change of tangent frames rotates the output consistently.

Runs on CUDA when available (`device="auto"`), CPU otherwise.

  python -m quadremesh.learned train --data data/train --out python/quadremesh/models/field_v1.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .geometry_features import MeshGeometry, compute_geometry, target_field

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "models", "field_v1.pt")


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cmul(a, b):
    return torch.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], -1)


class FieldNet(nn.Module):
    def __init__(self, d_inv: int, n_cand: int, d_edge: int, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.n_cand = n_cand
        self.enc = nn.Sequential(nn.Linear(d_inv, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.gate = nn.Linear(hidden, n_cand)
        self.edge_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden + d_edge, hidden), nn.SiLU(), nn.Linear(hidden, 1)) for _ in range(layers)])
        self.self_w = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU()) for _ in range(layers)])
        self.conf = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, inv, cand, src, dst, rot, edge_feat):
        h = self.enc(inv)
        g = self.gate(h)                                   # real gates (invariant)
        z = (cand * g[..., None]).sum(1)                   # (n, 2)
        n = inv.shape[0]
        for k in range(len(self.edge_mlps)):
            a = self.edge_mlps[k](torch.cat([h[dst], h[src], edge_feat], -1))  # (E, 1), invariant
            msg = cmul(z[src], rot) * a                     # transport j -> i frame, real weight
            agg = torch.zeros_like(z).index_add_(0, dst, msg)
            deg = torch.zeros(n, 1, device=z.device).index_add_(0, dst, torch.ones_like(a))
            z = z * torch.sigmoid(self.self_w[k](h)) + agg / deg.clamp(min=1)
            h = h + self.upd[k](torch.cat([h, z.norm(dim=-1, keepdim=True)], -1))
        conf = torch.sigmoid(self.conf(torch.cat([h, z.norm(dim=-1, keepdim=True)], -1))).squeeze(-1)
        return z, conf


def to_tensors(geom: MeshGeometry, device):
    e = geom.edges
    src = np.concatenate([e[:, 1], e[:, 0]])
    dst = np.concatenate([e[:, 0], e[:, 1]])
    rho = np.concatenate([geom.rho, -geom.rho_rev]) if hasattr(geom, "rho_rev") else None
    if rho is None:
        # reverse transport: frame of e0 expressed in frame of e1 is (approximately) -rho
        rho = np.concatenate([geom.rho, -geom.rho])
    rot = np.stack([np.cos(4 * rho), np.sin(4 * rho)], 1)
    ef = np.concatenate([geom.edge_feat, geom.edge_feat])
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
    return dict(inv=t(geom.inv), cand=t(geom.cand), src=t(src, torch.long), dst=t(dst, torch.long), rot=t(rot),
                edge_feat=t(ef))


def load_model(checkpoint: Optional[str] = None, device="auto") -> Tuple[FieldNet, torch.device]:
    dev = pick_device(device)
    ck = torch.load(checkpoint or DEFAULT_CKPT, map_location=dev, weights_only=False)
    m = FieldNet(**ck["config"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, dev


@torch.no_grad()
def predict_guide_field(V, F, checkpoint: Optional[str] = None, device="auto", feature_angle=35.0):
    """Returns (directions (n,3) in input space, confidence (n,)) for the input vertices."""
    model, dev = load_model(checkpoint, device)
    geom = compute_geometry(V, F, feature_angle=feature_angle)
    z, conf = model(**to_tensors(geom, dev))
    z = z.cpu().numpy().astype(np.float64)
    theta = np.arctan2(z[:, 1], z[:, 0]) / 4.0
    d = np.cos(theta)[:, None] * geom.T1 + np.sin(theta)[:, None] * geom.T2
    return d, conf.cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------------------------- training
def _load(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def make_samples(paths, per_mesh=3, max_faces=60000, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for p in paths:
        V, F = _load(p)
        if len(F) > max_faces:  # keep training cheap: random face subset is not ok; decimate by skipping
            import trimesh
            m = trimesh.Trimesh(V, F, process=False)
            try:
                m = m.simplify_quadric_decimation(face_count=max_faces)
                V, F = np.asarray(m.vertices), np.asarray(m.faces)
            except Exception:
                pass
        clean = compute_geometry(V, F)
        zt, wt = target_field(clean)
        Vn = (V - clean.center) / clean.scale
        for k in range(per_mesh):
            sigma = [0.0, 0.0015, 0.003, 0.005][k % 4]
            noise = rng.normal(size=(len(V), 1)) * sigma * clean.N
            noisy = compute_geometry(Vn + noise, F)
            samples.append((os.path.basename(p), noisy, zt, wt))
        print(f"  {os.path.basename(p)}: {len(V)} verts", flush=True)
    return samples


def train(data_dir, out, epochs=60, lr=3e-3, device="auto", seed=0):
    torch.manual_seed(seed)
    dev = pick_device(device)
    paths = sorted(glob.glob(os.path.join(data_dir, "*.obj")))
    print(f"building {len(paths)} training meshes x noise levels on {dev}")
    samples = make_samples(paths, seed=seed)
    g0 = samples[0][1]
    cfg = dict(d_inv=g0.inv.shape[1], n_cand=g0.cand.shape[1], d_edge=g0.edge_feat.shape[1])
    model = FieldNet(**cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tens = [(name, to_tensors(g, dev), torch.as_tensor(zt, device=dev), torch.as_tensor(wt, device=dev))
            for name, g, zt, wt in samples]
    hist = []
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(len(tens))
        tot = 0.0
        for k in perm:
            _, x, zt, wt = tens[k]
            z, conf = model(**x)
            zn = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            align = (zn * zt).sum(-1)                   # cos(4 * angle error)
            loss_dir = (wt * (1 - align)).sum() / wt.sum().clamp(min=1e-6)
            # confidence should predict where the target is reliable AND we got it right
            tgt_conf = (wt * (align > 0.7).float()).detach()
            loss_conf = nn.functional.binary_cross_entropy(conf, tgt_conf)
            loss = loss_dir + 0.2 * loss_conf
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        hist.append(tot / len(tens))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d} loss {hist[-1]:.4f}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(config=cfg, state=model.state_dict(), history=hist), out)
    print("saved", out)
    return hist


@torch.no_grad()
def evaluate_alignment(paths, checkpoint=None, device="auto", sigma=0.003, seed=1):
    """Mean |angle error| (deg, 4-RoSy) on reliable vertices: learned vs raw curvature, noisy input."""
    model, dev = load_model(checkpoint, device)
    rng = np.random.default_rng(seed)
    rows = []
    for p in paths:
        V, F = _load(p)
        clean = compute_geometry(V, F)
        zt, wt = target_field(clean)
        Vn = (V - clean.center) / clean.scale
        noisy = compute_geometry(Vn + rng.normal(size=(len(V), 1)) * sigma * clean.N, F)
        z, _ = model(**to_tensors(noisy, dev))
        z = z.cpu().numpy()
        # raw baseline: finest-scale curvature candidate on the noisy mesh, feature dir where present
        raw = np.where(np.linalg.norm(noisy.cand[:, 3], axis=1, keepdims=True) > 0.5, noisy.cand[:, 3], noisy.cand[:, 0])
        def err(zz):
            a = np.arctan2(zz[:, 1], zz[:, 0]) - np.arctan2(zt[:, 1], zt[:, 0])
            a = np.abs((a + np.pi) % (2 * np.pi) - np.pi) / 4.0
            m = wt > 0.3
            return float(np.rad2deg(a[m]).mean()) if m.any() else float("nan")
        rows.append((os.path.basename(p), err(raw), err(z)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/train")
    t.add_argument("--out", default=DEFAULT_CKPT)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("eval")
    e.add_argument("meshes", nargs="+")
    e.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        t0 = time.time()
        train(a.data, a.out, epochs=a.epochs, device=a.device)
        print(f"training took {time.time() - t0:.0f}s")
    elif a.cmd == "eval":
        for name, raw, lrn in evaluate_alignment(a.meshes, a.ckpt):
            print(f"{name:25s} raw-curvature {raw:6.2f} deg   learned {lrn:6.2f} deg")


if __name__ == "__main__":
    main()
