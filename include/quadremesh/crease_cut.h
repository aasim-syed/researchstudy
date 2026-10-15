#pragma once
#include <vector>

#include "quadremesh/types.h"

namespace qrm {

// Coverage test with the evaluator's definition (a crease point is covered when an output edge
// lies within tol*L and within max_angle degrees of the crease direction). Returns the maximal
// uncovered pieces of the given crease chains (as sub-polylines, padded by pad*L on both sides)
// that are at least min_run*L long. covered/total receive crease lengths.
std::vector<std::vector<Vector3d>> uncovered_crease_runs(const QuadMesh& m,
                                                         const std::vector<std::vector<Vector3d>>& chains, double L,
                                                         double min_run, double pad, double tol, double max_angle,
                                                         double* covered = nullptr, double* total = nullptr);

// Crease embedding by cutting: the crease polyline is traced across the faces of the quad mesh
// (closest-point walk), new vertices are inserted where it crosses edges (or existing vertices are
// moved onto it when it passes within 20 % of an edge's end), every crossed face is split along the
// crease (with extra vertices on the crease where it bends inside a face), and the resulting
// polygons are re-split into quads plus at most one triangle each. At the ends of the run the cut
// is continued straight to the boundary of the face it ends in. After the cut the crease is a chain
// of mesh edges by construction. Returns the number of faces added, or -1 if the run could not be
// embedded cleanly (the mesh is then unchanged).
int embed_crease_run(QuadMesh& m, const std::vector<Vector3d>& run, double L, double min_quality = 0.05,
                     std::vector<Vector3d>* crease_vertices = nullptr, double snap = 0.2);

}  // namespace qrm
