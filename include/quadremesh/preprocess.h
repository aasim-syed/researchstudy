#pragma once
#include <string>
#include <vector>

#include "quadremesh/types.h"

namespace qrm {

struct PrepStats {
    int welded = 0, degenerate = 0, duplicate = 0, nonmanifold_edges = 0, vertex_splits = 0;
    int orientation_flips = 0, holes_filled = 0, holes_kept = 0, components_dropped = 0;
};

// Working mesh with per-vertex attributes that are carried through every topological edit
// (averaged on splits, copied on duplication).
struct WorkMesh {
    MatrixXd V;
    MatrixXi F;
    MatrixXd A;                 // (n x k) attributes, k may be 0
    std::vector<int> sharp;     // per half-edge (3f+j) feature flag; empty = none
    std::vector<int> vtag;      // per-vertex tag (bit 1 = on symmetry plane)
};

// Weld, remove degenerate/duplicate faces, resolve non-manifold edges and vertices,
// orient consistently (outward for closed parts), fill small holes, drop tiny components.
WorkMesh preprocess(const MatrixXd& V, const MatrixXi& F, const MatrixXd& A, const Params& p,
                    PrepStats& stats);

// Split every edge longer than max_len (longest-edge bisection), propagating attributes,
// feature flags and vertex tags. Deterministic.
void refine_long_edges(WorkMesh& m, double max_len);

// Keep the half-space side >= 0 of the plane (axis, offset); triangles straddling the plane
// are split and new vertices are tagged as lying on the plane.
void clip_to_halfspace(WorkMesh& m, int axis, double offset);

// Remove unreferenced vertices (attributes follow).
void compact_vertices(WorkMesh& m);

}  // namespace qrm
