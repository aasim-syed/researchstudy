#pragma once
#include <vector>

#include "quadremesh/types.h"

namespace qrm {

struct FeatureSet {
    std::vector<int> he_sharp;     // per half-edge (3f+j) of the mesh it was detected on
    std::vector<int> vdeg;         // number of incident feature edges per vertex
    std::vector<int> corners;      // vertices that must stay fixed (junctions, endpoints, kinks)
    MatrixXd segments;             // (k x 6) feature segments (p0, p1)
    MatrixXd boundary_segments;    // (k x 6) open-boundary segments
    int raw_edges = 0, kept_edges = 0, chains = 0;
};

// Dihedral-angle features, robust to noise: an edge is a candidate only if both its own
// dihedral and the dihedral of the averaged 1-ring normals on either side exceed the
// threshold; candidates are chained, and chains shorter than min_len (absolute) are pruned
// iteratively (dangling first), keeping short connectors between junctions.
// support: radius of the side patches for the scale test (<= 0 disables).
FeatureSet detect_features(const MatrixXd& V, const MatrixXi& F, double angle_deg, double min_len, double support = -1);

// Map guidance strokes (polylines) onto mesh edge paths (Dijkstra between the vertices
// nearest to consecutive samples) and mark them as features.
void add_stroke_features(const MatrixXd& V, const MatrixXi& F, const std::vector<std::vector<Vector3d>>& strokes,
                         FeatureSet& fs);

// Recompute vdeg / corners / segments from he_sharp (after edits).
void finalize_features(const MatrixXd& V, const MatrixXi& F, FeatureSet& fs, double corner_angle_deg = 50.0);

}  // namespace qrm
