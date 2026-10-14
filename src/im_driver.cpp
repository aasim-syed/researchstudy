// Headless, in-memory Instant Meshes driver (adapted from third_party/instant-meshes/src/batch.cpp).
#include "im_driver.h"

#include <iostream>
#include <set>
#include <sstream>

#include "adjacency.h"
#include "dedge.h"
#include "extract.h"
#include "field.h"
#include "hierarchy.h"
#include "meshstats.h"
#include "normal.h"
#include "subdivide.h"

// global expected by the Instant Meshes sources (worker threads); 1 = deterministic, serial
int nprocs = 1;

namespace qrm {

namespace {
struct CoutSilencer {  // Instant Meshes logs progress to std::cout
    std::streambuf* old;
    std::ostringstream sink;
    CoutSilencer() : old(std::cout.rdbuf(sink.rdbuf())) {}
    ~CoutSilencer() { std::cout.rdbuf(old); }
};
}  // namespace

IMResult run_instant_meshes(const std::vector<double>& Vin, const std::vector<int>& Fin, int faces,
                            double crease_angle_deg, bool align_boundaries) {
    IMResult res;
    CoutSilencer quiet;
    const int rosy = 4, posy = 4;
    const bool deterministic = true, extrinsic = true;
    MatrixXu F(3, Fin.size() / 3);
    MatrixXf V(3, Vin.size() / 3), N;
    for (size_t i = 0; i < Fin.size(); ++i) F(i % 3, i / 3) = (uint32_t)Fin[i];
    for (size_t i = 0; i < Vin.size(); ++i) V(i % 3, i / 3) = (Float)Vin[i];
    VectorXf A;
    std::set<uint32_t> crease_in, crease_out;
    AdjacencyMatrix adj = nullptr;
    MeshStats stats = compute_mesh_stats(F, V, deterministic);
    // pure-quad output = quad-dominant extraction subdivided once -> request faces / 4
    int face_count = std::max(1, faces / 4);
    Float face_area = stats.mSurfaceArea / face_count;
    Float scale = std::sqrt(face_area);
    MultiResolutionHierarchy mRes;
    {
        VectorXu V2E, E2E;
        VectorXb boundary, nonManifold;
        if (stats.mMaximumEdgeLength * 2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2) {
            build_dedge(F, V, V2E, E2E, boundary, nonManifold);
            subdivide(F, V, V2E, E2E, boundary, nonManifold, std::min(scale / 2, (Float)stats.mAverageEdgeLength * 2),
                      deterministic);
        }
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);
        adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);
        if (crease_angle_deg > 0)
            generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold, (Float)crease_angle_deg, N, crease_in);
        else
            generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);
        compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);
        mRes.setE2E(std::move(E2E));
    }
    mRes.setAdj(std::move(adj));
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));
    mRes.setA(std::move(A));
    mRes.setN(std::move(N));
    mRes.setScale(scale);
    mRes.build(deterministic);
    mRes.resetSolution();
    if (align_boundaries) {
        mRes.clearConstraints();
        for (uint32_t i = 0; i < 3 * mRes.F().cols(); ++i) {
            if (mRes.E2E()[i] == INVALID) {
                uint32_t i0 = mRes.F()(i % 3, i / 3), i1 = mRes.F()((i + 1) % 3, i / 3);
                Vector3f p0 = mRes.V().col(i0), p1 = mRes.V().col(i1), edge = p1 - p0;
                if (edge.squaredNorm() > 0) {
                    edge.normalize();
                    mRes.CO().col(i0) = p0;
                    mRes.CO().col(i1) = p1;
                    mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
                    mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] = mRes.COw()[i1] = 1.0f;
                }
            }
        }
        mRes.propagateConstraints(rosy, posy);
    }
    Optimizer optimizer(mRes, false);
    optimizer.setRoSy(rosy);
    optimizer.setPoSy(posy);
    optimizer.setExtrinsic(extrinsic);
    optimizer.optimizeOrientations(-1);
    optimizer.notify();
    optimizer.wait();
    optimizer.optimizePositions(-1);
    optimizer.notify();
    optimizer.wait();
    optimizer.shutdown();
    MatrixXf O_extr, N_extr, Nf_extr;
    std::vector<std::vector<TaggedLink>> adj_extr;
    extract_graph(mRes, extrinsic, rosy, posy, adj_extr, O_extr, N_extr, crease_in, crease_out, deterministic);
    MatrixXu F_extr;
    extract_faces(adj_extr, O_extr, N_extr, Nf_extr, F_extr, posy, mRes.scale(), crease_out, true, true, nullptr, 0);
    res.V.resize(3 * O_extr.cols());
    for (int i = 0; i < O_extr.cols(); ++i)
        for (int k = 0; k < 3; ++k) res.V[3 * i + k] = O_extr(k, i);
    for (int i = 0; i < F_extr.cols(); ++i)
        res.Q.push_back({(int)F_extr(0, i), (int)F_extr(1, i), (int)F_extr(2, i), (int)F_extr(3, i)});
    res.ok = !res.Q.empty();
    return res;
}

}  // namespace qrm
