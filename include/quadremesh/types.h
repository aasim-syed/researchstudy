// quadremesh - quality-driven triangle-to-structured-quad remeshing
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace qrm {

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::VectorXi;

// Triangle mesh: V (n x 3), F (m x 3). Optional per-corner UVs (UV + FUV, libigl style).
struct TriMesh {
    MatrixXd V;
    MatrixXi F;
    MatrixXd UV;   // (k x 2), optional
    MatrixXi FUV;  // (m x 3) indices into UV, optional
    bool has_uv() const { return UV.rows() > 0 && FUV.rows() == F.rows(); }
};

// Quad-dominant mesh: Q (k x 4). A row with Q(i,3) == -1 is a triangle.
struct QuadMesh {
    MatrixXd V;
    MatrixXi Q;
    MatrixXd UV;   // optional, per corner: (k*4 x 2) row 4*i+j is corner j of face i
    bool has_uv() const { return UV.rows() == 4 * Q.rows() && Q.rows() > 0; }
};

// A density "sphere" region: inside radius the target density is multiplied by factor.
struct DensityRegion {
    Vector3d center = Vector3d::Zero();
    double radius = 0.0;
    double factor = 1.0;  // >1 -> more (smaller) quads
};

struct Params {
    // ---- size control -------------------------------------------------------------
    int target_faces = 5000;           // target quad count (ignored if target_edge_length > 0)
    double target_edge_length = -1.0;  // absolute target edge length in input units

    // ---- features -----------------------------------------------------------------
    double feature_angle = 35.0;        // dihedral threshold in degrees (<= 0 disables)
    double min_feature_length = 3.0;    // drop feature chains shorter than this x edge length
    bool preserve_boundary = true;
    bool match_face_count = true;
    bool trace_all_features = true;     // edge-path recovery over all creases (not only the field subset)
    double feature_field_budget = 0.3;  // crease length the field may constrain (x faces x L), non-scan inputs    // edge-path recovery on the full crease set (not the field's reduced set)       // re-solve once if post-solve insertions overshoot the target
    bool feature_loops = false;         // opt-in: insert edge loops along creases without an edge path
    double feature_loop_budget = 0.5;
    bool feature_loop_partial = false;
    double triangle_budget = 0.015;     // max triangles added by the coverage stage (x target faces)
    double singularity_budget = 10.0;   // max irregular vertices added by the coverage stage (per 1k target faces)
    bool curved_feature_splits = false; // split path edges that leave a curved crease (quad + triangle)  // split only the crease run (+ short extension), not whole strips   // max faces added by loop insertion (x target faces)
    double feature_density = 8.0;        // >0: smaller elements near creases inside the field solve
    double feature_density_width = 1.5;  // (density 1 + K exp(-(d / (W L))^2), same expected face count)
    bool crease_cuts = false;
    bool coverage_search = false;
    int coverage_search_candidates = 4;  // 4 or 8 solves
    int coverage_target_faces = 0;       // internal: face count the coverage tolerance refers to (0 = target_faces)
    bool sing_budget_im = false;         // coverage stage may use singularities up to 85 % of an Instant
                                         // Meshes solve of the same input at the same face count
    int path_subdivision = 0;            // 3 / 5: split path edges that miss their crease into this many pieces
    double snap_near = 0.5;              // >0: snap edges within this many L onto uncovered creases
    std::vector<Vector3d> density_focus;  // input-space points that get extra density (error-driven)
    double density_focus_strength = 8.0;
    double density_focus_width = 1.0;        // try several feature-size settings, keep the best (see remesh.cpp)            // embed still-uncovered creases by cutting the mesh along them
    int crease_cut_rounds = 2;
    int feature_refine_rounds = 0;
    double feature_refine_max_len = 0;  // > 0: refine only around untraced chains up to this length (x L)      // opt-in: local refinement around features without an edge path
                                        // (raises coverage, costs faces/singularities/quad ratio)

    // ---- field design -------------------------------------------------------------
    double curvature_weight = 0.35;     // strength of principal-curvature alignment [0,1]
    double curvature_min_anisotropy = 0.25;
    // Guidance strokes (polylines in input space). Edge flow follows them; with
    // strokes_as_edges they also become hard feature edges (edge loops on the stroke).
    std::vector<std::vector<Vector3d>> strokes;
    bool strokes_as_edges = false;
    // External / learned orientation field per *input* vertex (n x 3) + confidence (n).
    MatrixXd guide_field;
    VectorXd guide_weight;
    double guide_strength = 0.6;

    // ---- adaptive density ---------------------------------------------------------
    VectorXd density;                    // per input vertex, relative (>0); empty = uniform
    std::vector<DensityRegion> density_regions;
    double curvature_density = 0.0;      // 0 = off; >0 adds curvature-driven refinement
    double max_density_ratio = 8.0;      // clamp of density variation (max/min)

    // ---- symmetry -----------------------------------------------------------------
    int symmetry_axis = -1;              // 0/1/2 = mirror plane normal x/y/z, -1 off
    double symmetry_offset = std::numeric_limits<double>::quiet_NaN();  // NaN = bbox centre

    // ---- robustness / preprocessing -----------------------------------------------
    double max_hole_perimeter = 0.03;    // fill holes with perimeter < this x bbox diagonal
    double min_component_area = 2e-4;    // drop components smaller than this x total area
    int min_component_faces = 12;        // minimum quad budget per kept component

    // ---- output -------------------------------------------------------------------
    bool pure_quad = false;              // guarantee 100% quads (global split, budget/4)
    int smooth_iterations = 8;           // tangential relaxation + reprojection passes
    bool transfer_uv = true;
    unsigned int seed = 0;
    int verbose = 0;
};

struct Timing {
    double preprocess = 0, features = 0, field = 0, extraction = 0, postprocess = 0, total = 0;
};

struct Result {
    QuadMesh mesh;
    // diagnostics for visualisation (in input space, on the preprocessed mesh)
    MatrixXd field_V;         // vertices of the mesh the field lives on
    MatrixXi field_F;
    MatrixXd field_Q;         // orientation field (one representative direction / vertex)
    MatrixXd field_N;
    MatrixXd feature_edges;   // (k x 6) segments p0,p1 of detected features (input space)
    std::vector<std::string> log;
    Timing timing;
    int components_in = 0, components_out = 0, components_fallback = 0, components_im = 0, components_approx = 0;
};

}  // namespace qrm
