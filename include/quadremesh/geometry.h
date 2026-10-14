#pragma once
// Small topology / geometry helpers shared by the pipeline stages.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace qrm {

struct UnionFind {
    std::vector<int> p;
    explicit UnionFind(int n = 0) : p(n) { std::iota(p.begin(), p.end(), 0); }
    int find(int x) {
        while (p[x] != x) x = p[x] = p[p[x]];
        return x;
    }
    bool unite(int a, int b) {
        a = find(a), b = find(b);
        if (a == b) return false;
        if (a < b) std::swap(a, b);
        p[a] = b;  // smaller index wins -> deterministic representatives
        return true;
    }
};

// Deterministic, platform-independent RNG (PCG32).
struct Pcg32 {
    uint64_t state = 0x853c49e6748fea9bULL, inc = 0xda3e39cb94b95bdbULL;
    explicit Pcg32(uint64_t seed = 1) {
        state = 0U;
        inc = (seed << 1u) | 1u;
        next();
        state += 0x853c49e6748fea9bULL + seed;
        next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    double uniform() { return (next() >> 5) * (1.0 / 134217728.0); }  // [0,1)
};

// Undirected edge list with face incidences, built by sorting (deterministic).
struct EdgeTopology {
    // For each half-edge h = 3*f + j  (edge F(f,j) -> F(f,(j+1)%3)): undirected edge id.
    std::vector<int> he2e;
    // Undirected edges: endpoints (a < b) and incident half-edges.
    std::vector<std::pair<int, int>> ev;
    std::vector<std::vector<int>> ehe;
    void build(const Eigen::MatrixXi& F);
    int twin(int h) const;  // the other half-edge of a manifold edge, -1 otherwise
    bool boundary(int e) const { return ehe[e].size() == 1; }
};

inline Eigen::Vector3d face_normal(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int f) {
    Eigen::Vector3d a = V.row(F(f, 0)), b = V.row(F(f, 1)), c = V.row(F(f, 2));
    Eigen::Vector3d n = (b - a).cross(c - a);
    double l = n.norm();
    return l > 0 ? Eigen::Vector3d(n / l) : Eigen::Vector3d(0, 0, 0);
}

inline double face_area(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int f) {
    Eigen::Vector3d a = V.row(F(f, 0)), b = V.row(F(f, 1)), c = V.row(F(f, 2));
    return 0.5 * (b - a).cross(c - a).norm();
}

double surface_area(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
double bbox_diagonal(const Eigen::MatrixXd& V);
// Area-weighted vertex normals.
Eigen::MatrixXd vertex_normals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
// Connected components of faces (through shared vertices). Returns component id per face.
int face_components(const Eigen::MatrixXi& F, int nv, std::vector<int>& comp);
// Closest point on segment.
inline double point_segment_dist2(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                                  Eigen::Vector3d* closest = nullptr) {
    Eigen::Vector3d ab = b - a;
    double l2 = ab.squaredNorm();
    double t = l2 > 0 ? std::max(0.0, std::min(1.0, (p - a).dot(ab) / l2)) : 0.0;
    Eigen::Vector3d c = a + t * ab;
    if (closest) *closest = c;
    return (p - c).squaredNorm();
}

}  // namespace qrm
