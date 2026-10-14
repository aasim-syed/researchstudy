#pragma once
#include <vector>

#include "quadremesh/types.h"

namespace qrm {

struct CurvatureInfo {
    MatrixXd N;     // vertex normals
    MatrixXd D1;    // max-curvature direction (tangent), 4-RoSy representative
    VectorXd k1, k2;
    VectorXd conf;  // [0,1] reliability of D1 as an alignment target
};

// Edge-based shape-operator fit per vertex + tangent-space tensor diffusion over a radius
// matching the target quad size (noise-robust, linear time).
CurvatureInfo principal_directions(const MatrixXd& V, const MatrixXi& F, double radius, double min_anisotropy);

// Orientation constraints for the field solver: direction (n x 3), weight (n), position
// constraint flag (n). Features/boundary: hard direction + position. Strokes: hard direction.
// Curvature / external guide: soft direction.
struct FieldConstraints {
    MatrixXd dir;
    VectorXd w;
    std::vector<char> pos;
};

FieldConstraints build_constraints(const MatrixXd& V, const MatrixXi& F, const MatrixXd& N,
                                   const std::vector<int>& he_sharp, bool boundary_hard,
                                   const CurvatureInfo* curv, double curvature_weight,
                                   const MatrixXd* guide, const VectorXd* guide_w, double guide_strength,
                                   const std::vector<std::vector<Vector3d>>& strokes, double stroke_radius,
                                   double soft_fade = 0.0);

// Per-vertex target scale multiplier from relative density (keeps total quad count).
VectorXd density_to_scale(const MatrixXd& V, const MatrixXi& F, const VectorXd& density, double max_ratio);

}  // namespace qrm
