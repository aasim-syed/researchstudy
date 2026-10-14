"""GPU-capable cross-field optimisation in PyTorch (the optional CUDA path).

Solves for a smooth 4-RoSy field z (complex, per vertex, in local tangent frames):

    min_z  sum_ij w_ij |z_i - R_ij z_j|^2 + lam * sum_i c_i |z_i - t_i|^2     s.t. z_k = t_k (hard)

with R_ij = exp(4i * rho_ij) the discrete connection, t the curvature/feature targets and hard
constraints on feature/boundary vertices. The normal equations are solved with conjugate
gradients using sparse matrix products, so the same code runs on CPU or CUDA
(`device="auto"` picks CUDA when present). The result can be fed to the C++ core as a guide
field (`quadremesh(..., guide_field=d, guide_weight=w)`).
"""
from __future__ import annotations

import numpy as np
import torch

from .geometry_features import compute_geometry
from .learned import pick_device


def _cg(apply_A, b, x0, iters=500, tol=1e-6):
    x = x0.clone()
    r = b - apply_A(x)
    p = r.clone()
    rs = (r * r).sum()
    b_norm = (b * b).sum().sqrt().clamp(min=1e-30)
    for _ in range(iters):
        Ap = apply_A(p)
        alpha = rs / (p * Ap).sum().clamp(min=1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() / b_norm < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def optimize_field(V, F, lam=0.5, feature_angle=35.0, device="auto", iters=800, dtype=torch.float64):
    """Returns (directions (n,3) in input space, weights (n,), info dict)."""
    dev = pick_device(device)
    g = compute_geometry(V, F, feature_angle=feature_angle)
    n = len(g.V)
    e = torch.as_tensor(g.edges, device=dev)
    i, j = e[:, 0], e[:, 1]
    rho = torch.as_tensor(g.rho, device=dev, dtype=dtype)
    c4, s4 = torch.cos(4 * rho), torch.sin(4 * rho)
    w = torch.ones(len(i), device=dev, dtype=dtype)
    # targets: feature candidate (hard) else coarse curvature (soft, weighted by reliability)
    feat = torch.as_tensor(g.cand[:, 3], device=dev, dtype=dtype)
    curv = torch.as_tensor(g.cand[:, 2], device=dev, dtype=dtype)
    hard = feat.norm(dim=1) > 0.5
    ci = curv.norm(dim=1)
    t = torch.where(hard[:, None], feat, curv / ci.clamp(min=1e-12)[:, None])
    free = ~hard

    def apply_A(z):  # z: (n, 2) real/imag; returns (L + lam*C) z on free vertices
        zr, zi = z[:, 0], z[:, 1]
        # R_ij z_j
        rzr = c4 * zr[j] - s4 * zi[j]
        rzi = s4 * zr[j] + c4 * zi[j]
        # conj(R_ij) z_i  (for the j-row)
        czr = c4 * zr[i] + s4 * zi[i]
        czi = -s4 * zr[i] + c4 * zi[i]
        out = torch.zeros_like(z)
        out[:, 0].index_add_(0, i, w * (zr[i] - rzr))
        out[:, 1].index_add_(0, i, w * (zi[i] - rzi))
        out[:, 0].index_add_(0, j, w * (zr[j] - czr))
        out[:, 1].index_add_(0, j, w * (zi[j] - czi))
        out = out + lam * ci[:, None] * z
        return out * free[:, None]

    z_hard = torch.where(hard[:, None], t, torch.zeros_like(t))
    b = (lam * ci[:, None] * t) * free[:, None] - apply_A(z_hard) * free[:, None]
    x0 = torch.where(free[:, None], t * 0.1, torch.zeros_like(t))
    t0 = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None
    z = _cg(lambda x: apply_A(x * free[:, None]), b, x0, iters=iters)
    z = z * free[:, None] + z_hard
    z = z.cpu().numpy()
    theta = np.arctan2(z[:, 1], z[:, 0]) / 4.0
    d = np.cos(theta)[:, None] * g.T1 + np.sin(theta)[:, None] * g.T2
    mag = np.linalg.norm(z, axis=1)
    weight = np.clip(mag / max(np.percentile(mag, 90), 1e-9), 0, 1)
    weight[hard.cpu().numpy()] = 1.0
    return d, weight, dict(device=str(dev), vertices=n, hard=int(hard.sum()))
