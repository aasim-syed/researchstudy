"""Python API around the C++ core (pybind11 module `_quadremesh`)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from . import _quadremesh as _core

MeshLike = Union[str, os.PathLike, Tuple[np.ndarray, np.ndarray], Any]


@dataclass
class QuadRemeshResult:
    vertices: np.ndarray            # (n, 3)
    faces: np.ndarray               # (k, 4), -1 in column 3 marks a triangle
    uv: Optional[np.ndarray]        # (4k, 2) per-corner UVs or None
    report: Optional[Dict[str, Any]]
    timing: Dict[str, float]
    log: Sequence[str]
    field: Dict[str, np.ndarray] = field(default_factory=dict)  # V, F, Q (directions), N
    feature_edges: Optional[np.ndarray] = None

    @property
    def quad_ratio(self) -> float:
        if len(self.faces) == 0:
            return 0.0
        return float(np.mean(self.faces[:, 3] >= 0))

    def save(self, path: str) -> None:
        _core.save_quad_obj(str(path), self.vertices, self.faces, self.uv if self.uv is not None and len(self.uv) else None)


def load_mesh(src: MeshLike) -> Dict[str, np.ndarray]:
    """Load .obj/.ply/.glb/.gltf/.off, a (V, F) tuple, or any object with .vertices/.faces."""
    if isinstance(src, (str, os.PathLike)):
        return _core.load_mesh(str(src))
    if isinstance(src, (tuple, list)) and len(src) >= 2:
        return {"V": np.asarray(src[0], dtype=np.float64), "F": np.asarray(src[1], dtype=np.int32)}
    if hasattr(src, "vertices") and hasattr(src, "faces"):
        return {"V": np.asarray(src.vertices, dtype=np.float64), "F": np.asarray(src.faces, dtype=np.int32)}
    raise TypeError(f"cannot interpret mesh input of type {type(src)}")


def quadremesh(input: MeshLike, target_faces: int = 5000, feature_angle: float = 35.0, *,
               output: Optional[str] = None, report: Union[bool, str, None] = True,
               target_edge_length: Optional[float] = None, strokes: Optional[Sequence[np.ndarray]] = None,
               strokes_as_edges: bool = False, density: Optional[np.ndarray] = None,
               density_regions: Optional[Sequence[Tuple[Sequence[float], float, float]]] = None,
               curvature_density: float = 0.0, symmetry_axis: Union[str, int, None] = None,
               symmetry_offset: Optional[float] = None, guide_field: Optional[np.ndarray] = None,
               guide_weight: Optional[np.ndarray] = None, learned_field: Union[bool, str, None] = None,
               pure_quad: bool = False, seed: int = 0, verbose: bool = False,
               **options: Any) -> QuadRemeshResult:
    """Convert a triangle mesh to a structured quad (or quad-dominant) mesh.

    Parameters
    ----------
    input : path, (V, F) tuple, or mesh object with .vertices / .faces
    target_faces : desired quad count (ignored if target_edge_length is given)
    feature_angle : sharp-feature dihedral threshold in degrees (0 disables)
    output : optional .obj path to write
    report : True -> compute the quality report; a path -> also write it as JSON; False -> skip
    strokes : list of (k, 3) polylines that edge flow should follow
    density : per-input-vertex relative density (larger = smaller quads)
    density_regions : [(center_xyz, radius, factor), ...] spheres with denser/coarser quads
    symmetry_axis : 'x' | 'y' | 'z' -> mirror-symmetric result
    guide_field / guide_weight : external orientation field per input vertex (n, 3) / (n,)
    learned_field : True or a checkpoint path -> predict the guide field with the PyTorch
        model (`quadremesh.learned`), runs on CUDA when available
    pure_quad : guarantee 100% quads
    options : any other core parameter (curvature_weight, smooth_iterations, ...)
    """
    mesh = load_mesh(input)
    V, F = np.ascontiguousarray(mesh["V"], np.float64), np.ascontiguousarray(mesh["F"], np.int32)
    kw: Dict[str, Any] = dict(target_faces=int(target_faces), feature_angle=float(feature_angle), seed=int(seed),
                              verbose=bool(verbose), pure_quad=bool(pure_quad), strokes_as_edges=bool(strokes_as_edges),
                              curvature_density=float(curvature_density))
    if target_edge_length:
        kw["target_edge_length"] = float(target_edge_length)
    if strokes:
        kw["strokes"] = [np.asarray(s, dtype=np.float64).reshape(-1, 3) for s in strokes]
    if density is not None:
        kw["density"] = np.asarray(density, dtype=np.float64).reshape(-1)
    if density_regions:
        kw["density_regions"] = [(list(map(float, c)), float(r), float(f)) for c, r, f in density_regions]
    if symmetry_axis is not None:
        kw["symmetry_axis"] = symmetry_axis
    if symmetry_offset is not None:
        kw["symmetry_offset"] = float(symmetry_offset)
    if learned_field:
        from .learned import predict_guide_field
        ckpt = learned_field if isinstance(learned_field, str) else None
        guide_field, guide_weight = predict_guide_field(V, F, checkpoint=ckpt)
    if guide_field is not None:
        kw["guide_field"] = np.asarray(guide_field, dtype=np.float64).reshape(-1, 3)
        kw["guide_weight"] = (np.ones(len(V)) if guide_weight is None else np.asarray(guide_weight, np.float64).reshape(-1))
    kw.update(options)
    out = _core.remesh(V, F, mesh.get("UV"), mesh.get("FUV"), **kw)
    rep = None
    if report:
        js = _core.evaluate(V, F, out["V"], out["Q"], target_faces=-1 if target_edge_length else int(target_faces),
                            feature_angle=float(feature_angle) if feature_angle > 0 else 35.0,
                            runtime=float(out["timing"]["total"]), method="quadremesh")
        rep = json.loads(js)
        if isinstance(report, (str, os.PathLike)):
            with open(report, "w") as fh:
                json.dump(rep, fh, indent=1)
    res = QuadRemeshResult(vertices=out["V"], faces=out["Q"], uv=out["UV"] if len(out["UV"]) else None, report=rep,
                           timing=dict(out["timing"]), log=list(out["log"]),
                           field=dict(V=out["field_V"], F=out["field_F"], Q=out["field_Q"], N=out["field_N"]),
                           feature_edges=out["feature_edges"])
    if output:
        res.save(output)
    return res


def evaluate(input: MeshLike, quad: Union[str, Tuple[np.ndarray, np.ndarray]], target_faces: int = -1,
             feature_angle: float = 35.0, runtime: float = -1.0, method: str = "") -> Dict[str, Any]:
    """Method-agnostic quality report of a quad mesh (path or (V, Q)) against its source."""
    mesh = load_mesh(input)
    if isinstance(quad, (str, os.PathLike)):
        QV, Q = _core.load_quad_obj(str(quad))
    else:
        QV, Q = quad
    js = _core.evaluate(mesh["V"], mesh["F"], np.asarray(QV, np.float64), np.asarray(Q, np.int32),
                        target_faces=int(target_faces), feature_angle=float(feature_angle), runtime=float(runtime),
                        method=method)
    return json.loads(js)


def save_quad_obj(path: str, V: np.ndarray, Q: np.ndarray, UV: Optional[np.ndarray] = None) -> None:
    _core.save_quad_obj(str(path), V, Q, UV)
