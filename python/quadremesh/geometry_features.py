"""Vectorised per-vertex geometry for the learned field (numpy / scipy).

All directional quantities are expressed in an arbitrary per-vertex tangent frame (t1, t2) and
encoded as 4-RoSy complex numbers z = r * exp(i * 4 * phi), so every model built on them is
equivariant to the choice of frame (rotating t1 by a multiplies every z by exp(-4ia)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class MeshGeometry:
    V: np.ndarray          # (n, 3) normalised to unit bbox diagonal
    F: np.ndarray          # (m, 3)
    N: np.ndarray          # (n, 3) vertex normals
    T1: np.ndarray         # (n, 3) tangent frame
    T2: np.ndarray
    edges: np.ndarray      # (e, 2) undirected
    rho: np.ndarray        # (e,) transport angle: frame of edges[:,1] expressed in frame of edges[:,0]
    inv: np.ndarray        # (n, d_inv) invariant features
    cand: np.ndarray       # (n, k, 2) candidate 4-RoSy directions (complex as 2 reals, magnitude = reliability)
    edge_feat: np.ndarray  # (e, d_e) invariant edge features
    scale: float           # original bbox diagonal
    center: np.ndarray


def _normalize(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _frames(N):
    a = np.where(np.abs(N[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = _normalize(a - N * np.sum(N * a, 1, keepdims=True))
    t2 = np.cross(N, t1)
    return t1, t2


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for j in range(3):
        np.add.at(N, F[:, j], fn)
    bad = np.linalg.norm(N, axis=1) < 1e-20
    N[bad] = [0, 0, 1]
    return _normalize(N), fn


def unique_edges(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(1)
    return np.unique(e, axis=0)


def edge_dihedral(V, F, fn, edges):
    """Dihedral angle (radians, 0 = flat) per undirected edge; boundary/non-manifold -> -1."""
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    he_s = np.sort(he, 1)
    key = he_s[:, 0].astype(np.int64) * (len(V) + 1) + he_s[:, 1]
    ekey = edges[:, 0].astype(np.int64) * (len(V) + 1) + edges[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    idx = np.searchsorted(ks, ekey)
    cnt = np.searchsorted(ks, ekey, side="right") - idx
    nfn = _normalize(fn)
    dih = -np.ones(len(edges))
    ok = cnt == 2
    f0 = fid[order[idx[ok]]]
    f1 = fid[order[idx[ok] + 1]]
    dih[ok] = np.arccos(np.clip(np.sum(nfn[f0] * nfn[f1], 1), -1, 1))
    return dih, cnt


def _transport_angles(N, T1, T2, edges):
    i, j = edges[:, 0], edges[:, 1]
    ni, nj = N[i], N[j]
    # rotate t1_j from n_j onto n_i (minimal rotation), measure its angle in frame i
    c = np.sum(ni * nj, 1, keepdims=True)
    axis = np.cross(nj, ni)
    t = T1[j]
    # Rodrigues with sin = |axis|, cos = c
    k = axis / np.maximum(1 + c, 1e-9)
    t_rot = t * c + np.cross(axis, t) + k * np.sum(axis * t, 1, keepdims=True)
    return np.arctan2(np.sum(t_rot * T2[i], 1), np.sum(t_rot * T1[i], 1))


def compute_geometry(V, F, target_edge=None, feature_angle=35.0):
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    bmin, bmax = V.min(0), V.max(0)
    center = 0.5 * (bmin + bmax)
    scale = float(np.linalg.norm(bmax - bmin)) or 1.0
    V = (V - center) / scale
    n = len(V)
    N, fn = vertex_normals(V, F)
    T1, T2 = _frames(N)
    E = unique_edges(F)
    i, j = E[:, 0], E[:, 1]
    elen = np.linalg.norm(V[i] - V[j], axis=1)
    h = float(elen.mean()) if len(elen) else 1.0
    L = target_edge if target_edge else max(h * 4, 0.01)
    # ---- per-vertex shape operator (edge-based least squares, both directions)
    ii = np.concatenate([i, j])
    jj = np.concatenate([j, i])
    e = V[jj] - V[ii]
    l2 = np.maximum(np.sum(e * e, 1), 1e-30)
    kappa = np.sum((N[jj] - N[ii]) * e, 1) / l2
    et = e - N[ii] * np.sum(N[ii] * e, 1, keepdims=True)
    lt = np.maximum(np.linalg.norm(et, axis=1), 1e-30)
    x = np.sum(et * T1[ii], 1) / lt
    y = np.sum(et * T2[ii], 1) / lt
    rows = np.stack([x * x, 2 * x * y, y * y], 1)
    AtA = np.zeros((n, 3, 3))
    Atb = np.zeros((n, 3))
    np.add.at(AtA, ii, rows[:, :, None] * rows[:, None, :])
    np.add.at(Atb, ii, rows * kappa[:, None])
    AtA += np.eye(3)[None] * 1e-9
    abc = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    # 3D tensors
    Tn = (abc[:, 0, None, None] * T1[:, :, None] * T1[:, None, :]
          + abc[:, 1, None, None] * (T1[:, :, None] * T2[:, None, :] + T2[:, :, None] * T1[:, None, :])
          + abc[:, 2, None, None] * T2[:, :, None] * T2[:, None, :])
    A = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n)).tocsr() + sp.eye(n)
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(1.0 / deg) @ A
    P = np.eye(3)[None] - N[:, :, None] * N[:, None, :]
    inv_feats, cands = [], []
    T = Tn.reshape(n, 9)
    done = 0
    for iters in (2, 8, 24):
        for _ in range(iters - done):
            T = W @ T
            Tm = T.reshape(n, 3, 3)
            T = np.einsum("nij,njk,nkl->nil", P, Tm, P).reshape(n, 9)
        done = iters
        Tm = T.reshape(n, 3, 3)
        a = np.einsum("ni,nij,nj->n", T1, Tm, T1)
        b = np.einsum("ni,nij,nj->n", T1, Tm, T2)
        c = np.einsum("ni,nij,nj->n", T2, Tm, T2)
        tr, det_d = 0.5 * (a + c), np.sqrt(0.25 * (a - c) ** 2 + b * b)
        k1, k2 = tr + det_d, tr - det_d
        diff = k1 - k2
        aniso = diff / (np.abs(k1) + np.abs(k2) + 1e-12)
        phi = 0.5 * np.arctan2(2 * b, a - c)           # max-curvature direction angle
        rel = np.clip(aniso * np.minimum(diff * L, 1.0), 0, 1)
        cands.append(np.stack([rel * np.cos(4 * phi), rel * np.sin(4 * phi)], 1))
        inv_feats += [np.log1p(np.abs(k1) * L), np.log1p(np.abs(k2) * L), aniso, np.sign(k1 * k2)]
    # ---- feature edges
    dih, cnt = edge_dihedral(V, F, fn, E)
    sharp = dih > np.deg2rad(feature_angle)
    bnd = cnt == 1
    fdir = np.zeros((n, 3))
    fcount = np.zeros(n)
    hard = sharp | bnd
    d = V[j[hard]] - V[i[hard]]
    for idx_side in (i[hard], j[hard]):
        dd = d.copy()
        # align sign to the accumulated direction (4-RoSy, so any sign works; keep magnitude)
        prev = fdir[idx_side]
        flip = np.sum(prev * dd, 1) < 0
        dd[flip] *= -1
        np.add.at(fdir, idx_side, dd)
        np.add.at(fcount, idx_side, 1)
    fd = fdir - N * np.sum(fdir * N, 1, keepdims=True)
    has = (fcount > 0) & (np.linalg.norm(fd, axis=1) > 1e-12)
    ang = np.arctan2(np.sum(fd * T2, 1), np.sum(fd * T1, 1))
    cands.append(np.stack([has * np.cos(4 * ang), has * np.sin(4 * ang)], 1))
    maxdih = np.zeros(n)
    np.maximum.at(maxdih, i, np.maximum(dih, 0))
    np.maximum.at(maxdih, j, np.maximum(dih, 0))
    isb = np.zeros(n)
    isb[i[bnd]] = 1
    isb[j[bnd]] = 1
    inv_feats += [maxdih, fcount.clip(0, 4) / 4.0, isb]
    inv = np.stack(inv_feats, 1).astype(np.float32)
    cand = np.stack(cands, 1).astype(np.float32)
    rho = _transport_angles(N, T1, T2, E)
    edge_feat = np.stack([np.maximum(dih, 0), bnd.astype(float), elen / L], 1).astype(np.float32)
    return MeshGeometry(V=V, F=F, N=N, T1=T1, T2=T2, edges=E, rho=rho, inv=inv, cand=cand, edge_feat=edge_feat,
                        scale=scale, center=center)


def target_field(geom: MeshGeometry):
    """Training target on a clean mesh: feature direction where present, else the coarse-scale
    principal curvature direction; weight = its reliability."""
    feat = geom.cand[:, 3]
    curv = geom.cand[:, 2]
    has_f = np.linalg.norm(feat, axis=1) > 0.5
    z = np.where(has_f[:, None], feat, curv)
    w = np.where(has_f, 1.0, np.linalg.norm(curv, axis=1))
    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return zn.astype(np.float32), w.astype(np.float32)
