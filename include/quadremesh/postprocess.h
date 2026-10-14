#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "quadremesh/types.h"

namespace qrm {

// Reference geometry the output is projected onto during relaxation.
struct SurfaceRef {
    struct Impl;
    std::shared_ptr<Impl> impl;
    // V/F: surface; FE: feature edges (k x 2 into V); BE: boundary edges; corners: vertex ids
    SurfaceRef(const MatrixXd& V, const MatrixXi& F, const MatrixXi& FE, const MatrixXi& BE,
               const std::vector<int>& corners, const std::vector<int>& plane_vertices,
               int sym_axis, double sym_offset);
    Vector3d project_surface(const Vector3d& p, Vector3d* normal = nullptr) const;
    double dist2_feature(const Vector3d& p, Vector3d* c) const;
    double dist2_boundary(const Vector3d& p, Vector3d* c, bool* on_plane = nullptr) const;
    double dist2_corner(const Vector3d& p, Vector3d* c) const;
    MatrixXd corner_points() const;
    bool has_features() const;
    bool has_boundary() const;
    int sym_axis() const;
    double sym_offset() const;
};

struct PostStats {
    int degenerate_removed = 0, duplicates_removed = 0, nonmanifold_removed = 0, tris_merged = 0;
    int tris_remaining = 0, smoothing_rejected = 0, feature_vertices = 0, corner_vertices = 0;
};

void cleanup(QuadMesh& m, PostStats& st);
void merge_triangle_pairs(QuadMesh& m, PostStats& st, double min_sj = 0.3);
void split_to_pure_quads(QuadMesh& m);
// Adaptive density: refine faces larger than the local target size (1->4 quad / 1->3 tri splits)
// with conforming hanging-node templates on the transition ring (1 split edge -> quad+tri,
// 2 adjacent -> 2 quads, 2 opposite -> 2 quads; >= 3 -> refine). Up to max_levels passes.
// target(P) returns the target edge length at each query point (rows of P).
int refine_to_size(QuadMesh& m, const std::function<VectorXd(const MatrixXd&)>& target, int max_levels);
void relax_and_project(QuadMesh& m, const SurfaceRef& ref, double L, int iterations, PostStats& st);
// Mirror across the plane and weld seam vertices (those within tol of the plane).
void mirror_and_weld(QuadMesh& m, int axis, double offset, double tol);

}  // namespace qrm
