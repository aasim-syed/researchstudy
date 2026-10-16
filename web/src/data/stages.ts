export type Visual = "input" | "edges" | "field" | "sj" | "compare" | "density" | "uv" | "viewer" | "metrics";
export type Group = "io" | "core" | "third" | "eval";

export interface Stage {
  id: string;
  n: number;
  title: string;
  short: string;
  group: Group;
  icon: string;
  file: string;
  signature: string;
  input: string;
  output: string;
  what: string;
  features: string[];
  visual: Visual;
  visualCaption: string;
  extra?: { visual: Visual; caption: string };
}

/** Every function name and file below exists in the repository (src/, apps/, python/). */
export const STAGES: Stage[] = [
  {
    id: "input", n: 1, title: "Load mesh", short: "Input", group: "io", icon: "FileInput",
    file: "src/io.cpp", signature: "TriMesh load_mesh(const std::string& path)",
    input: "path to .obj / .ply / .glb / .gltf / .off", output: "TriMesh: V (n×3 positions), F (m×3 triangles), UVs",
    what: "The CLI (apps/quadremesh_cli.cpp) or the Python call quadremesh(path, target_faces) reads the triangle mesh. Everything after this point runs in C++.",
    features: ["libigl readers + cgltf for glTF", "same entry for CLI and pybind11 API", "target face count and crease angle as options"],
    visual: "input", visualCaption: "The selected benchmark model as it enters the pipeline.",
  },
  {
    id: "preprocess", n: 2, title: "Clean & repair", short: "Preprocess", group: "core", icon: "Wrench",
    file: "src/preprocess.cpp", signature: "WorkMesh preprocess(V, F, A, const Params&, PrepStats&)",
    input: "raw TriMesh", output: "clean, manifold, outward-oriented triangle mesh",
    what: "Real-world meshes (scans, glTF assets) are broken in small ways that make a field solver fail. This stage makes the input safe to solve on.",
    features: ["weld duplicate vertices", "drop degenerate / duplicate faces", "repair non-manifold edges (repair_manifold)", "fill small holes", "remove tiny components, orient outward"],
    visual: "input", visualCaption: "Input statistics for the selected model.",
  },
  {
    id: "features", n: 3, title: "Detect creases", short: "Creases", group: "core", icon: "Spline",
    file: "src/features.cpp", signature: "FeatureSet detect_features(V, F, angle_deg, min_len, support)",
    input: "clean triangle mesh, crease angle 35°", output: "crease edges grouped into chains",
    what: "A crease is an edge where the two faces meet at a dihedral angle above 35°. Short or isolated noisy edges are rejected; chains shorter than 3 L (L = target edge length) are dropped.",
    features: ["dihedral angle > 35°", "noise rejection by neighbour support", "chains ≥ 3 L kept", "these chains are what feature coverage measures"],
    visual: "edges", visualCaption: "Orange lines: the crease chains the quads must follow (edge-flow view of our output).",
  },
  {
    id: "field", n: 4, title: "Cross field + adaptive size", short: "Field", group: "core", icon: "Compass",
    file: "src/field.cpp", signature: "build_constraints(V, F, N, …)  ·  density_to_scale(V, F, density, max_ratio)",
    input: "creases, normals, optional guidance", output: "per-vertex cross-field constraints + target scale",
    what: "A 4-direction (4-RoSy) cross field tells every point which way the quad edges should run; it is locked to the creases. A density field makes quads smaller near creases while keeping the total face count.",
    features: ["4-RoSy field locked to creases", "density d = 1 + 8·exp(−(dist / 1.5 L)²)", "scale S = d^(−1/2), normalised to the face budget", "optional: strokes, symmetry, PyTorch learned field"],
    visual: "field", visualCaption: "Cross field: red/blue ticks are the two quad directions at each point.",
    extra: { visual: "density", caption: "Adaptive size: smaller quads near creases at the same face count." },
  },
  {
    id: "grid", n: 5, title: "Integer-grid solve", short: "Quad grid", group: "third", icon: "Grid3x3",
    file: "src/qflow_driver.cpp · src/im_driver.cpp", signature: "QFResult run_quadriflow(const QFJob&)  ·  run_instant_meshes(…)",
    input: "field + scale per connected part", output: "raw all-quad mesh",
    what: "QuadriFlow (vendored, 11 small patches) turns the field into an integer grid using min-cost flow (LEMON), which keeps singularities few. Parts it cannot solve fall back to Instant Meshes.",
    features: ["third-party: QuadriFlow + LEMON", "min-cost flow → few singularities", "Instant Meshes fallback per part", "solved per connected component"],
    visual: "compare", visualCaption: "What integer-grid solvers produce: ours vs Instant Meshes, QuadriFlow and Blender at equal face count.",
  },
  {
    id: "post", n: 6, title: "Crease paths + relax", short: "Post-process", group: "core", icon: "Route",
    file: "src/postprocess.cpp", signature: "trace_feature_paths(m, …)  ·  relax_and_project(m, ref, L, iterations, st)",
    input: "raw quads + crease chains", output: "quads with an edge path on every crease, smoothed on the surface",
    what: "For every crease, Dijkstra finds the best chain of quad edges near it and snaps those vertices onto the crease. Then vertices are relaxed tangentially and projected back to the original surface.",
    features: ["Dijkstra edge-path recovery", "path vertices snapped to the crease", "tangential relaxation", "projection keeps distance error ~0.09 %"],
    visual: "edges", visualCaption: "Quad edges running along the creases; dots are singularities (blue = valence 3, red = valence 5).",
  },
  {
    id: "coverage", n: 7, title: "Coverage stage", short: "Coverage", group: "core", icon: "Target",
    file: "src/crease_cut.cpp", signature: "snap_near_creases(m, chains, L, max_dist, …)  ·  subdivide_path_edges(m, paths, chains, L, pieces, …)",
    input: "post-processed quads", output: "more crease segments covered by an edge",
    what: "Creases that still have no edge close enough get local fixes. Every operation is kept only if it covers more than it costs, within a singularity budget of 85 % of Instant Meshes' rate.",
    features: ["near-miss snapping (net-gain rule)", "path-edge subdivision into 3 (hexagon split, no valence-2)", "singularity budget from an internal IM solve", "max-coverage setting: --coverage-mode --path-subdiv 3"],
    visual: "edges", visualCaption: "Crease paths after the coverage stage (orange).",
  },
  {
    id: "guards", n: 8, title: "Quality guards", short: "Guards", group: "core", icon: "ShieldCheck",
    file: "src/remesh.cpp", signature: "quadremesh_matched(input, p)  ·  quadremesh_matched_once(input, p)",
    input: "candidate quad mesh", output: "a mesh that meets the SJ and face-count limits",
    what: "Adaptive size can squeeze quads. If mean scaled Jacobian drops below 0.81 the mesh is re-solved with half the density, then none. If faces exceed 1.10 × target it is re-solved with a smaller budget.",
    features: ["mean SJ ≥ 0.81 or re-solve", "faces ≤ 1.10 × target or re-solve", "coverage still measured at the requested size"],
    visual: "sj", visualCaption: "Scaled Jacobian per quad: green = square corners (1.0), red = squashed.",
  },
  {
    id: "search", n: 9, title: "Coverage search", short: "Search", group: "core", icon: "Search",
    file: "src/remesh.cpp", signature: "Result quadremesh(const TriMesh& input, const Params& p)",
    input: "4 candidate settings", output: "the best candidate that meets every criterion",
    what: "Optional (max-coverage). Four solves with different density settings are scored by the same evaluator; the one with the highest coverage that still passes every criterion is kept, otherwise the defaults.",
    features: ["4 candidates, scored by evaluate()", "keeps only fully valid candidates", "falls back to defaults", "flag: --coverage-search"],
    visual: "metrics", visualCaption: "Default vs max-coverage on the selected model, scored by the evaluator.",
  },
  {
    id: "uv", n: 10, title: "UV transfer", short: "UVs", group: "core", icon: "Layers",
    file: "src/transfer.cpp", signature: "void transfer_uvs(const TriMesh& src, QuadMesh& out)",
    input: "quad mesh + textured input", output: "quad mesh with UVs",
    what: "Texture coordinates are carried over from the input so the remeshed asset keeps its texture.",
    features: ["closest point + barycentric lookup (libigl AABB)", "keeps textures usable", "pure-quad UV demo in the report"],
    visual: "uv", visualCaption: "UVs carried from the triangle input to the quad output.",
  },
  {
    id: "output", n: 11, title: "Quad mesh + report", short: "Output", group: "io", icon: "Box",
    file: "src/io.cpp", signature: "void save_quad_obj(const std::string& path, const QuadMesh& m)",
    input: "final quad mesh", output: ".obj (quads + UVs) and a JSON quality report",
    what: "The result is written as an OBJ with quad faces. Rotate it below, compare with the baselines, and download any of them.",
    features: ["OBJ with quad faces", "JSON report via --report", "downloadable below for every method"],
    visual: "viewer", visualCaption: "Interactive: drag to orbit, scroll to zoom. Dots are singularities.",
  },
  {
    id: "evaluate", n: 12, title: "Evaluate", short: "Evaluate", group: "eval", icon: "Gauge",
    file: "src/metrics.cpp · apps/quadeval_cli.cpp", signature: "std::string evaluate(input, out, const EvalParams&, runtime, method)",
    input: "input mesh + any quad mesh", output: "JSON: quads, SJ, distance, coverage, singularities",
    what: "One evaluator scores every method the same way: our CLI's --report, quadeval, the benchmark and the coverage search all call it.",
    features: ["quad ratio", "scaled Jacobian", "two-sided distance (% of bbox diagonal)", "feature coverage (0.1 L, 25°)", "singularities vs Instant Meshes"],
    visual: "compare", visualCaption: "All four methods at equal face count, singularities marked.",
  },
];

export const GROUP_STYLE: Record<Group, { label: string; color: string }> = {
  io: { label: "data in / out", color: "#34d399" },
  core: { label: "our C++ core", color: "#38bdf8" },
  third: { label: "third-party solver", color: "#a78bfa" },
  eval: { label: "evaluation", color: "#fbbf24" },
};
