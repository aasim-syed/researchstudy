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
    // Feature curves as ordered polylines, split at corners / junctions; closed loops are split
    // in two. Chains shorter than min_len are skipped.
    std::vector<std::vector<Vector3d>> feature_chains(double min_len) const;
    bool has_features() const;
    bool has_boundary() const;
    int sym_axis() const;
    double sym_offset() const;
};

struct PostStats {
    int degenerate_removed = 0, duplicates_removed = 0, nonmanifold_removed = 0, tris_merged = 0;
    int tris_remaining = 0, smoothing_rejected = 0, feature_vertices = 0, corner_vertices = 0, feature_paths = 0;
    int feature_chains = 0;
    double feature_len_traced = 0, feature_len_total = 0;
    std::vector<std::vector<Vector3d>> untraced;  // feature chains without a recovered edge path
    std::vector<std::vector<int>> paths;          // recovered edge paths (output vertex ids)
    std::vector<std::vector<Vector3d>> path_chains;  // the feature chain each path follows
    int curved_splits = 0;
    std::vector<Vector3d> pinned;  // positions of vertices placed on creases by crease cuts (kept fixed)
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
// Feature loop insertion: for each chain without an edge path, find runs of quads it crosses
// rung-to-rung (lengthwise along a quad strip) and split that whole strip lengthwise, with the cut
// through the chain inside the run and on the strip midline elsewhere. Pure quads in, pure quads
// out; no new irregular vertices. Returns faces added (stops at max_added).
// partial_ext < 0: split whole strips; >= 0: split only the crease's run (+ up to partial_ext
// faces to reach the strip's end), leaving one triangle per open end.
int insert_feature_loops(QuadMesh& m, const std::vector<std::vector<Vector3d>>& chains, double L, int max_added,
                         int partial_ext = -1, int max_tris = -1);
// Mirror across the plane and weld seam vertices (those within tol of the plane).
// Curved creases: split every recovered path edge whose chord leaves the crease by more than
// tol*L (or turns > max_angle deg against it) at the crease point; each adjacent quad becomes
// quad + triangle. Up to `rounds` passes. Returns the number of splits.
int split_curved_paths(QuadMesh& m, PostStats& st, double L, double tol, double max_angle, int rounds,
                       int max_tris = -1);
void mirror_and_weld(QuadMesh& m, int axis, double offset, double tol);

}  // namespace qrm
