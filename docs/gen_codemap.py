#!/usr/bin/env python3
"""Builds the code reference: for every pipeline stage the file, the function (signature copied
from the source), its input and output, the library it relies on, a short excerpt of the real code
and the figure that shows its effect.

Signatures and excerpts are read from the source files at generation time (located by an anchor
string), so they always match the code.

    python docs/gen_codemap.py      # writes docs/code_reference.md and the CODEMAP block of the write-up
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIBS = [
    ("Eigen 3", "header-only C++ linear algebra", "Every mesh is stored as Eigen matrices (V: n×3 vertex positions, F: m×3 triangles, Q: k×4 quads); vectors, normals and the small least-squares systems in the field and relaxation code."),
    ("libigl", "C++ geometry-processing library", "AABB trees for closest-point queries (distance error, coverage test, projection onto the input surface, crease snapping) and the PLY reader."),
    ("QuadriFlow (vendored, 11 patches)", "integer-grid quad solver, `third_party/qflow_patched`", "Orientation field, position field, min-cost-flow singularity optimisation and quad extraction - the core solver. Patches: determinism, constraint bugs, external sharp edges / normals / scale, exceptions instead of exit(), bounded loops (`PATCHES.md`)."),
    ("LEMON", "graph library (inside QuadriFlow)", "Network-simplex min-cost flow that QuadriFlow uses to move and cancel singularities globally."),
    ("Instant Meshes core", "`imcore`, field-aligned remesher", "Second extractor for parts QuadriFlow cannot solve (thin parts, the room scan), best-of-two on dense scans, and the internal reference for the singularity budget."),
    ("Intel TBB", "parallel runtime (inside Instant Meshes)", "Required by the Instant Meshes core; run single-threaded (nprocs = 1) for deterministic output."),
    ("Boost (headers)", "C++ utilities", "Required by LEMON's build."),
    ("cgltf", "single-header glTF / GLB loader", "Reads .gltf / .glb inputs (node transforms, UVs) in `src/io.cpp`."),
    ("pybind11", "C++ ↔ Python bindings", "Builds the `_quadremesh` Python module from `python/bindings.cpp`; every remesher parameter is a keyword argument."),
    ("NumPy", "Python arrays", "Vertex / face arrays passed to and from the C++ core."),
    ("PyTorch", "deep learning (optional component)", "`python/quadremesh/learned.py`: a frame-equivariant graph network that predicts a guide field per vertex; runs on CUDA when available."),
    ("matplotlib, Pillow", "plotting, images", "`python/quadremesh/viz.py` and `benchmark/make_report.py`: the edge-flow / singularity / scaled-Jacobian / cross-field renders and the charts of the visual report."),
    ("trimesh, pymeshlab", "Python mesh I/O and repair", "Preparing the benchmark inputs (`data/prepare_benchmark.py`, `scripts/prepare_lounge.py`) and loading meshes in the Python scripts."),
    ("pynanoinstantmeshes, Blender", "baselines", "The Instant Meshes and Blender QuadriFlow runs of the benchmark (`benchmark/run_benchmark.py`, `benchmark/blender_quadriflow.py`)."),
    ("markdown, WeasyPrint", "documents", "`docs/build_pdf.py` turns the write-up into the PDF."),
]

# (stage, file, function name, anchor for the signature, input, output, what it does, library,
#  excerpt anchor, excerpt lines, figure)
ENTRIES = [
    ("Load", "src/io.cpp", "load_mesh", "TriMesh load_mesh(const std::string& path)",
     "file path (.obj / .ply / .glb / .gltf / .off)", "TriMesh: V (n×3), F (m×3), optional UVs",
     "Reads any supported format into one triangle mesh; glTF node transforms are applied.", "cgltf, libigl",
     None, 0, None),
    ("Preprocess", "src/preprocess.cpp", "preprocess", "WorkMesh preprocess(const MatrixXd& V",
     "raw V, F (+ per-vertex attributes), Params", "WorkMesh: clean, manifold, consistently oriented triangles",
     "Welds duplicate vertices, removes degenerate faces, repairs non-manifold edges and vertices, orients faces, fills small holes, drops debris.", "Eigen",
     None, 0, None),
    ("Creases", "src/features.cpp", "detect_features", "FeatureSet detect_features(",
     "V, F, angle threshold (35°), minimum chain length (3 L), support radius (L)", "FeatureSet: sharp half-edges, crease segments, corner vertices",
     "Dihedral-angle test plus 1-ring and patch-scale tests, so scan noise is not a crease; chains shorter than 3 L are pruned.", "Eigen",
     None, 0, "fig-crease"),
    ("Field constraints", "src/field.cpp", "build_constraints", "FieldConstraints build_constraints(",
     "V, F, normals, crease flags, curvature, guide field, strokes", "per-vertex direction + weight (hard on creases / boundaries, soft elsewhere)",
     "Turns creases, boundaries, curvature directions and the optional learned field into constraints for the cross field.", "Eigen",
     None, 0, "fig-field"),
    ("Adaptive size", "src/field.cpp", "density_to_scale", "VectorXd density_to_scale(",
     "V, F, per-vertex density, max ratio (8)", "per-vertex scale target with the same expected face count",
     "Converts the crease-distance density into the scale the field solve aims for (S = d^-1/2, normalised).", "Eigen",
     "    // S = d^-1/2, normalised so that sum A/S^2 == sum A  (same expected quad count)", 5, "fig-density"),
    ("Integer grid", "src/qflow_driver.cpp", "run_quadriflow", "QFResult run_quadriflow(const QFJob& job)",
     "one connected component: V, F, constraints, scale, face budget, seed", "QFResult: quad vertices and faces, field for visualisation",
     "Runs the patched QuadriFlow: orientation field, scale, position field, min-cost-flow singularity optimisation, extraction.", "QuadriFlow, LEMON",
     None, 0, "fig-grid"),
    ("Fallback extractor", "src/im_driver.cpp", "run_instant_meshes", "IMResult run_instant_meshes(",
     "V, F (flat arrays), face count, crease angle", "IMResult: pure quads",
     "Instant Meshes extraction for parts QuadriFlow cannot handle, and the reference for the singularity budget.", "Instant Meshes core, TBB",
     None, 0, None),
    ("Whole pipeline", "src/remesh.cpp", "quadremesh", "Result quadremesh(const TriMesh& input, const Params& p)",
     "input TriMesh + Params (target faces, feature angle, options)", "Result: quad mesh, field, crease segments, log, timings",
     "Runs every stage per component, the coverage stage, the quality guard, face-count control and (optionally) the coverage search.", "all of the above",
     None, 0, "fig-pipeline"),
    ("Crease paths + relaxation", "src/postprocess.cpp", "relax_and_project", "void relax_and_project(QuadMesh& m",
     "quad mesh, reference surface with creases, L, iterations", "same mesh, vertices moved: edge paths snapped onto creases, smoothed, reprojected",
     "Directed-edge Dijkstra finds an edge path along every crease and snaps it on; then tangential relaxation and reprojection onto the input surface.", "libigl (AABB)",
     "double c = len * (1.0 + 6.0 * dm / L) + len * 1.5 * (1.0 - align);", 2, None),
    ("Near-miss snapping", "src/crease_cut.cpp", "snap_near_creases", "int snap_near_creases(QuadMesh& m",
     "quad mesh, crease polylines, L, max distance (0.5 L), max angle (75°)", "number of edges snapped; mesh vertices moved onto creases",
     "Moves both ends of an edge that runs beside an uncovered crease onto it, if quality and existing coverage hold.", "libigl (AABB), Eigen",
     "        Vector3d qa = closest_on_chain(S[i].c, pa), qb = closest_on_chain(S[i].c, pb);", 1, "fig-snap"),
    ("Path-edge subdivision", "src/crease_cut.cpp", "subdivide_path_edges", "int subdivide_path_edges(QuadMesh& m",
     "quad mesh, crease edge paths, crease polylines, L, pieces (3), max edges", "number of edges split; mesh with new crease vertices, still all quads",
     "Splits path edges whose chord misses a curved crease into three at crease points; each neighbouring quad becomes two quads.", "Eigen",
     "            // a quad that became a hexagon (a n1 n2 b c d) is split right here, with the diagonal", 2, "fig-subdiv"),
    ("Coverage check", "src/crease_cut.cpp", "uncovered_crease_runs", "std::vector<std::vector<Vector3d>> uncovered_crease_runs(",
     "quad mesh, crease polylines, L, tolerances (0.1 L, 25°)", "the crease pieces not yet covered (+ covered / total length)",
     "The evaluator's coverage test, used inside the pipeline to decide what to fix.", "libigl (AABB)",
     None, 0, "fig-cov"),
    ("Scaled Jacobian", "src/metrics.cpp", "scaled_jacobian", "double scaled_jacobian(const MatrixXd& V, int a, int b, int c, int d)",
     "vertex positions, the 4 corner indices of a quad", "number in [-1, 1]: worst corner's signed sine",
     "Shape quality of one quad; used by the evaluator, the quality guards and the snapping checks.", "Eigen",
     "    int q[4] = {a, b, c, d};", 18, "fig-sj"),
    ("Evaluator", "src/metrics.cpp", "evaluate", "std::string evaluate(const TriMesh& input, const QuadMesh& out",
     "input triangles, output quads, EvalParams (angle 35°, target faces)", "JSON report: quad ratio, SJ, distance, feature coverage, singularities, topology",
     "The one scoring function used by the CLI report, quadeval, the benchmark and the coverage search.", "libigl (AABB), Eigen",
     "        J.num(\"feature_coverage\", ft > 0 ? fc / ft : 1.0);", 1, None),
    ("UV transfer", "src/transfer.cpp", "transfer_uvs", "void transfer_uvs(const TriMesh& src, QuadMesh& out)",
     "input mesh with UVs, output quads", "output quads with per-corner UVs",
     "Each quad samples the UV chart under its centroid, so faces never straddle a seam.", "libigl (AABB)",
     None, 0, "fig-uv"),
    ("Python API", "python/quadremesh/api.py", "quadremesh", "def quadremesh(input: MeshLike",
     "path / (V, F) / mesh object, target_faces, feature_angle, options", "QuadRemeshResult: V, Q, UV, quality report",
     "Python front end; keyword options map one-to-one to the C++ Params.", "pybind11, NumPy",
     None, 0, None),
    ("Learned field", "python/quadremesh/learned.py", "predict_guide_field", "def predict_guide_field(",
     "V, F, checkpoint, device", "per-vertex guide direction + confidence",
     "Optional PyTorch component: predicts a guide field that the remesher uses as soft guidance; CUDA when available.", "PyTorch",
     None, 0, None),
]

FIGS = {
    "fig-crease": ("report/fandisk_ours_panel.png", "creases (orange lines) and singularities on fandisk"),
    "fig-field": ("report/fandisk_ours_panel.png", "right panel: the cross field"),
    "fig-density": ("report/demo_density.png", "adaptive density"),
    "fig-grid": ("report/fandisk_ours_panel.png", "left panel: the extracted quads"),
    "fig-pipeline": ("report/charts.png", "per-method metrics across the benchmark"),
    "fig-snap": ("report/sheen_chair_compare.png", "edge flow along the chair's piping"),
    "fig-subdiv": ("report/rocker_arm_compare.png", "curved creases on the rocker arm"),
    "fig-cov": ("report/fandisk_compare.png", "fandisk, all methods"),
    "fig-sj": ("report/fandisk_ours_panel.png", "middle panel: scaled-Jacobian heatmap"),
    "fig-uv": ("report/demo_uv_purequad.png", "UV transfer and pure-quad output"),
}


def read(path):
    return open(os.path.join(ROOT, path)).read().splitlines()


def signature(path, anchor):
    lines = read(path)
    for i, l in enumerate(lines):
        if anchor in l:
            sig = [l.rstrip()]
            j = i
            while not re.search(r"[{;:]\s*$", sig[-1]) and j + 1 < len(lines) and len(sig) < 6:
                j += 1
                sig.append(lines[j].rstrip())
            s = "\n".join(sig).rstrip("{ ").rstrip()
            return s
    raise SystemExit(f"anchor not found in {path}: {anchor}")


def excerpt(path, anchor, n):
    lines = read(path)
    for i, l in enumerate(lines):
        if anchor in l:
            return "\n".join(x.rstrip() for x in lines[i:i + n]), i + 1
    raise SystemExit(f"excerpt anchor not found in {path}: {anchor}")


def build():
    out = ["## 9. Code reference", "",
           "Generated by `docs/gen_codemap.py` from the source: signatures and excerpts are copied from the files named.", "",
           "### Libraries and what each one does here", "", "| library | what it is | how it is used |", "|---|---|---|"]
    for n, w, h in LIBS:
        out.append(f"| {n} | {w} | {h} |")
    out += ["", "### Pipeline functions", ""]
    for stage, path, name, sig_anchor, inp, outp, what, lib, ex_anchor, ex_n, fig in ENTRIES:
        lang = "python" if path.endswith(".py") else "cpp"
        sig = signature(path, sig_anchor)
        out += [f"#### {stage}: `{name}`", "", f"- **File:** `{path}`", f"- **Input:** {inp}", f"- **Output:** {outp}",
                f"- **What it does:** {what}", f"- **Library:** {lib}", "", f"```{lang}", sig, "```"]
        if ex_anchor:
            ex, ln = excerpt(path, ex_anchor, ex_n)
            out += ["", f"Excerpt (`{path}`, line {ln}):", "", f"```{lang}", ex, "```"]
        if fig:
            p, cap = FIGS[fig]
            out += ["", f"![{cap}]({p})", "", f"*Figure: `docs/{p}` - {cap}.*"]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    text = build()
    open(os.path.join(ROOT, "docs", "code_reference.md"), "w").write("# quadremesh: code reference\n\n" + text.split("\n", 1)[1] + "\n")
    wp = os.path.join(ROOT, "docs", "technical_writeup.md")
    s = open(wp).read()
    pat = re.compile(r"(<!-- CODEMAP:BEGIN -->\n).*?(<!-- CODEMAP:END -->)", re.S)
    if pat.search(s):
        s = pat.sub(lambda m: m.group(1) + text + "\n" + m.group(2), s)
        open(wp, "w").write(s)
        print("write-up CODEMAP updated")
    print(f"{len(ENTRIES)} functions, {len(LIBS)} libraries")
