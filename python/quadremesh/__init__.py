"""quadremesh: quality-driven triangle-to-structured-quad remeshing.

>>> from quadremesh import quadremesh
>>> res = quadremesh("chair.glb", target_faces=5000, feature_angle=35, output="chair_quad.obj",
...                  report="chair_report.json")
>>> res.quad_ratio, res.report["quality"]["scaled_jacobian"]["mean"]
"""
from .api import QuadRemeshResult, evaluate, load_mesh, quadremesh, save_quad_obj

__all__ = ["quadremesh", "evaluate", "load_mesh", "save_quad_obj", "QuadRemeshResult"]
__version__ = "1.0.0"
