#pragma once
#include <string>
#include <vector>

#include "quadremesh/types.h"

namespace qrm {

// Scaled Jacobian of a quad (min over the 4 corners, w.r.t. the quad's average normal).
// Range [-1, 1], 1 = square corners. For triangles (d < 0) the corner minimum of
// sin(angle) * 2/sqrt(3) is used (1 = equilateral).
double scaled_jacobian(const MatrixXd& V, int a, int b, int c, int d);
// Max distance of the corners from the best-fit plane, divided by the mean edge length.
double planarity_error(const MatrixXd& V, int a, int b, int c, int d);

struct EvalParams {
    double feature_angle = 35.0;      // same detector as the pipeline
    double min_feature_length = 3.0;  // x element size L
    double coverage_tol = 0.1;        // x L, distance tolerance for feature coverage
    double coverage_angle = 25.0;     // degrees, edge direction tolerance
    int samples = 200000;             // surface samples per direction for distance error
    int target_faces = -1;            // for size accuracy; also defines L if > 0
    unsigned int seed = 1;
};

// Full quality report as a JSON string (stable key order). Distances are reported as % of
// the input bounding-box diagonal.
std::string evaluate(const TriMesh& input, const QuadMesh& output, const EvalParams& ep,
                     double runtime_seconds = -1.0, const std::string& method = "");

}  // namespace qrm
