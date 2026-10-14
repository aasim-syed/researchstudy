#pragma once
#include "quadremesh/types.h"

namespace qrm {

// Per-corner UV transfer from the source triangle mesh. Each output face picks the UV chart
// (island) of the source triangle under its centroid and every corner is evaluated inside
// that chart, so faces never straddle a UV seam.
void transfer_uvs(const TriMesh& src, QuadMesh& out);

// Generic per-vertex attribute transfer (closest point + barycentric interpolation).
MatrixXd transfer_vertex_attributes(const MatrixXd& V, const MatrixXi& F, const MatrixXd& A, const MatrixXd& P);

}  // namespace qrm
