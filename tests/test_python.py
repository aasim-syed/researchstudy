"""Python API tests: run with  PYTHONPATH=python python -m pytest tests"""
import os

import numpy as np
import pytest

from quadremesh import evaluate, quadremesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "data", "models")


def grid(n=40):
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    V = np.stack([x.ravel(), y.ravel(), 0.1 * np.sin(3 * x.ravel())], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


def test_tuple_input_and_report():
    V, F = grid()
    r = quadremesh((V, F), target_faces=300)
    assert r.quad_ratio > 0.98
    assert r.report["faces"]["total"] == len(r.faces)
    assert r.report["quality"]["scaled_jacobian"]["mean"] > 0.8


def test_deterministic():
    V, F = grid()
    a = quadremesh((V, F), target_faces=300, report=False)
    b = quadremesh((V, F), target_faces=300, report=False)
    assert np.array_equal(a.faces, b.faces) and np.array_equal(a.vertices, b.vertices)


def test_density_region_and_strokes():
    V, F = grid(50)
    base = quadremesh((V, F), target_faces=400, report=False)
    dens = quadremesh((V, F), target_faces=400, report=False, density_regions=[((0.2, 0.2, 0), 0.25, 6.0)])
    c = np.array([0.2, 0.2])
    near = lambda r: np.sum(np.linalg.norm(r.vertices[:, :2] - c, axis=1) < 0.25)
    assert near(dens) > 1.5 * near(base)
    stroke = np.array([[0.1, 0.5, 0.0], [0.9, 0.55, 0.0]])
    s = quadremesh((V, F), target_faces=400, report=False, strokes=[stroke])
    assert s.quad_ratio > 0.95


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, "spot.obj")), reason="benchmark models not built")
def test_uv_transfer_and_symmetry():
    r = quadremesh(os.path.join(MODELS, "spot.obj"), target_faces=1500)
    assert r.uv is not None and r.uv.shape == (4 * len(r.faces), 2)
    s = quadremesh(os.path.join(MODELS, "spot.obj"), target_faces=1500, symmetry_axis="x", report=False)
    xs = s.vertices[:, 0]
    c = 0.5 * (xs.min() + xs.max())
    # mirrored vertex set: every vertex has a mirror partner
    from scipy.spatial import cKDTree
    m = s.vertices.copy()
    m[:, 0] = 2 * c - m[:, 0]
    d, _ = cKDTree(s.vertices).query(m)
    assert np.max(d) < 1e-6 * np.ptp(s.vertices)


def test_evaluate_roundtrip(tmp_path):
    V, F = grid()
    r = quadremesh((V, F), target_faces=300, report=False, output=str(tmp_path / "q.obj"))
    rep = evaluate((V, F), str(tmp_path / "q.obj"), target_faces=300)
    assert rep["faces"]["total"] == len(r.faces)


def test_coverage_mode_kwarg():
    import json, os
    import quadremesh as q
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = [e["file"] for e in json.load(open(os.path.join(root, "data", "manifest.json"))) if e["name"] == "fandisk"][0]
    base = q.quadremesh(os.path.join(root, "data", f), 3000)
    cov = q.quadremesh(os.path.join(root, "data", f), 3000, coverage_mode=True)
    assert cov.report["features"]["feature_coverage"] >= base.report["features"]["feature_coverage"]
    assert cov.report["faces"]["quad_ratio"] >= 0.97
