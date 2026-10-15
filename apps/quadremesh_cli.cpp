// quadremesh CLI
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "quadremesh/quadremesh.h"

using namespace qrm;

static void usage() {
    std::printf(
        "quadremesh %s - triangle to structured quad remeshing\n\n"
        "usage: quadremesh INPUT(.obj|.ply|.glb|.gltf|.off) -o OUTPUT.obj [options]\n\n"
        "size:\n"
        "  -f, --faces N            target quad count (default 5000)\n"
        "  -e, --edge-length X      target edge length in input units (overrides --faces)\n"
        "features / structure:\n"
        "  -a, --feature-angle DEG  sharp-feature dihedral threshold (default 35, 0 = off)\n"
        "  --min-feature-length K   drop feature chains shorter than K x edge length (3.0)\n"
        "  --no-boundary            do not constrain open boundaries\n"
        "  --feature-refine N       refine N times around features without an edge loop (0)\n"
        "  --feature-loops          insert edge loops (strip splits) along creases without an edge path\n"
        "  --no-face-match          do not re-solve to hit the target face count after insertions\n"
        "  --curvature-weight W     principal-curvature alignment strength 0..1 (0.35)\n"
        "  --strokes FILE           guidance polylines: 'x y z' per line, blank line between strokes\n"
        "  --strokes-as-edges       also force edge loops onto the strokes\n"
        "  --guide FILE             external/learned field: 'dx dy dz w' per input vertex\n"
        "  --guide-strength W       (0.6)\n"
        "adaptive density:\n"
        "  --density FILE           one relative density value per input vertex\n"
        "  --density-region X Y Z R F   multiply density by F inside sphere (repeatable)\n"
        "  --curvature-density K    curvature-driven refinement strength (0 = off)\n"
        "symmetry:\n"
        "  --symmetry x|y|z         mirror-symmetric output across this axis plane\n"
        "  --symmetry-offset C      plane coordinate (default: bounding-box centre)\n"
        "output:\n"
        "  --pure-quad              guarantee 100%% quads\n"
        "  --smooth N               relaxation iterations (8)\n"
        "  --no-uv                  skip UV transfer\n"
        "  --report FILE.json       write quality report\n"
        "  --field FILE.txt         dump field (V, N, Q per line) for visualisation\n"
        "  --seed S                 (0)\n"
        "  -v, --verbose\n",
        version());
}

static std::vector<std::vector<Vector3d>> read_strokes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read strokes " + path);
    std::vector<std::vector<Vector3d>> s(1);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream is(line);
        double x, y, z;
        if (is >> x >> y >> z) s.back().push_back(Vector3d(x, y, z));
        else if (!s.back().empty()) s.emplace_back();
    }
    if (s.back().empty()) s.pop_back();
    return s;
}

static MatrixXd read_matrix(const std::string& path, int cols) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    MatrixXd M(v.size() / cols, cols);
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < cols; ++j) M(i, j) = v[i * cols + j];
    return M;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string in, out, report, field_dump, guide_path;
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-o" || a == "--output") out = next();
        else if (a == "-f" || a == "--faces") p.target_faces = std::stoi(next());
        else if (a == "-e" || a == "--edge-length") p.target_edge_length = std::stod(next());
        else if (a == "-a" || a == "--feature-angle") p.feature_angle = std::stod(next());
        else if (a == "--min-feature-length") p.min_feature_length = std::stod(next());
        else if (a == "--no-boundary") p.preserve_boundary = false;
        else if (a == "--feature-refine") p.feature_refine_rounds = std::stoi(next());
        else if (a == "--feature-loops") p.feature_loops = true;
        else if (a == "--partial-loops") p.feature_loops = p.feature_loop_partial = true;
        else if (a == "--loop-budget") p.feature_loop_budget = std::stod(next());
        else if (a == "--curved-splits") p.curved_feature_splits = true;
        else if (a == "--tri-budget") p.triangle_budget = std::stod(next());
        else if (a == "--refine-max-len") p.feature_refine_max_len = std::stod(next());
        else if (a == "--sing-budget") p.singularity_budget = std::stod(next());
        else if (a == "--trace-all") p.trace_all_features = true;
        else if (a == "--field-feature-budget") p.feature_field_budget = std::stod(next());
        else if (a == "--no-face-match") p.match_face_count = false;
        else if (a == "--curvature-weight") p.curvature_weight = std::stod(next());
        else if (a == "--strokes") p.strokes = read_strokes(next());
        else if (a == "--strokes-as-edges") p.strokes_as_edges = true;
        else if (a == "--guide") guide_path = next();
        else if (a == "--guide-strength") p.guide_strength = std::stod(next());
        else if (a == "--density") p.density = read_matrix(next(), 1).col(0);
        else if (a == "--density-region") {
            DensityRegion r;
            r.center.x() = std::stod(next()), r.center.y() = std::stod(next()), r.center.z() = std::stod(next());
            r.radius = std::stod(next()), r.factor = std::stod(next());
            p.density_regions.push_back(r);
        } else if (a == "--curvature-density") p.curvature_density = std::stod(next());
        else if (a == "--symmetry") {
            std::string ax = next();
            p.symmetry_axis = ax == "x" ? 0 : ax == "y" ? 1 : ax == "z" ? 2 : -1;
        } else if (a == "--symmetry-offset") p.symmetry_offset = std::stod(next());
        else if (a == "--pure-quad") p.pure_quad = true;
        else if (a == "--smooth") p.smooth_iterations = std::stoi(next());
        else if (a == "--no-uv") p.transfer_uv = false;
        else if (a == "--report") report = next();
        else if (a == "--field") field_dump = next();
        else if (a == "--seed") p.seed = (unsigned)std::stoul(next());
        else if (a == "-v" || a == "--verbose") p.verbose = 1;
        else if (a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
        else in = a;
    }
    if (in.empty() || out.empty()) { usage(); return 1; }
    try {
        TriMesh m = load_mesh(in);
        if (!guide_path.empty()) {
            MatrixXd G = read_matrix(guide_path, 4);
            if (G.rows() != m.V.rows()) throw std::runtime_error("guide field size does not match input vertices");
            p.guide_field = G.leftCols(3);
            p.guide_weight = G.col(3);
        }
        Result r = quadremesh(m, p);
        save_quad_obj(out, r.mesh);
        if (!field_dump.empty()) {
            std::ofstream fo(field_dump);
            for (int i = 0; i < r.field_V.rows(); ++i)
                fo << r.field_V(i, 0) << " " << r.field_V(i, 1) << " " << r.field_V(i, 2) << " " << r.field_N(i, 0) << " "
                   << r.field_N(i, 1) << " " << r.field_N(i, 2) << " " << r.field_Q(i, 0) << " " << r.field_Q(i, 1) << " "
                   << r.field_Q(i, 2) << "\n";
        }
        if (!report.empty()) {
            EvalParams ep;
            ep.feature_angle = p.feature_angle > 0 ? p.feature_angle : 35.0;
            ep.min_feature_length = p.min_feature_length;
            ep.target_faces = p.target_edge_length > 0 ? -1 : p.target_faces;
            std::string js = evaluate(m, r.mesh, ep, r.timing.total, "quadremesh");
            std::ofstream fo(report);
            fo << js;
        }
        std::printf("quadremesh: %d faces -> %d quads/tris (%.2fs) -> %s\n", (int)m.F.rows(), (int)r.mesh.Q.rows(),
                    r.timing.total, out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
