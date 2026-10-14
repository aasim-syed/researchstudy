#pragma once
// Thin driver around the vendored, patched QuadriFlow integer-grid stage.
#include <array>
#include <vector>

#include "quadremesh/field.h"
#include "quadremesh/types.h"

namespace qrm {

struct QFJob {
    MatrixXd V;                 // (n x 3) refined component mesh
    MatrixXi F;                 // (m x 3)
    std::vector<int> sharp_he;  // feature half-edges
    FieldConstraints cons;      // per-vertex orientation / position constraints
    VectorXd scale;             // optional per-vertex target scale multiplier (density)
    MatrixXd normals;           // optional smoothed vertex normals (n x 3)
    int faces = 1000;
    bool preserve_boundary = true;
    bool min_cost_flow = false;
    double soft_max_angle = 30.0;  // soft targets further than this from the smooth field are ignored
    int soft_iterations = 8;
    unsigned int seed = 0;
};

struct QFResult {
    bool ok = false;
    MatrixXd V;                            // output vertices (input space of the job)
    std::vector<std::array<int, 4>> Q;     // output quads (may contain repeated vertices)
    MatrixXd fieldQ, fieldN;               // orientation field on the job mesh (n x 3) if sizes match
    int orientation_singularities = 0;
};

QFResult run_quadriflow(const QFJob& job);

}  // namespace qrm
