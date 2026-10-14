#include "quadremesh/transfer.h"

#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>

#include <map>
#include <memory>

#include "quadremesh/geometry.h"

namespace qrm {

namespace {
Eigen::RowVector3d bary(const MatrixXd& V, const MatrixXi& F, int f, const Eigen::RowVector3d& p) {
    Eigen::RowVector3d a = V.row(F(f, 0)), b = V.row(F(f, 1)), c = V.row(F(f, 2));
    Eigen::RowVector3d v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1), d20 = v2.dot(v0), d21 = v2.dot(v1);
    double den = d00 * d11 - d01 * d01;
    if (std::abs(den) < 1e-300) return Eigen::RowVector3d(1.0 / 3, 1.0 / 3, 1.0 / 3);
    double v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den;
    return Eigen::RowVector3d(1 - v - w, v, w);
}
}  // namespace

MatrixXd transfer_vertex_attributes(const MatrixXd& V, const MatrixXi& F, const MatrixXd& A, const MatrixXd& P) {
    igl::AABB<MatrixXd, 3> tree;
    tree.init(V, F);
    VectorXd D;
    VectorXi I;
    MatrixXd C;
    tree.squared_distance(V, F, P, D, I, C);
    MatrixXd out(P.rows(), A.cols());
    for (int i = 0; i < P.rows(); ++i) {
        Eigen::RowVector3d b = bary(V, F, I[i], C.row(i));
        out.row(i) = b[0] * A.row(F(I[i], 0)) + b[1] * A.row(F(I[i], 1)) + b[2] * A.row(F(I[i], 2));
    }
    return out;
}

void transfer_uvs(const TriMesh& src, QuadMesh& out) {
    if (!src.has_uv() || out.Q.rows() == 0) return;
    // UV islands: faces connected through shared UV vertex indices
    UnionFind uf((int)src.UV.rows());
    for (int f = 0; f < src.FUV.rows(); ++f) {
        uf.unite(src.FUV(f, 0), src.FUV(f, 1));
        uf.unite(src.FUV(f, 0), src.FUV(f, 2));
    }
    std::vector<int> island(src.F.rows());
    std::map<int, std::vector<int>> island_faces;
    for (int f = 0; f < src.F.rows(); ++f) {
        island[f] = uf.find(src.FUV(f, 0));
        island_faces[island[f]].push_back(f);
    }
    igl::AABB<MatrixXd, 3> tree;
    tree.init(src.V, src.F);
    std::map<int, std::pair<MatrixXi, std::shared_ptr<igl::AABB<MatrixXd, 3>>>> itree;
    auto island_tree = [&](int id) -> std::pair<MatrixXi, std::shared_ptr<igl::AABB<MatrixXd, 3>>>& {
        auto it = itree.find(id);
        if (it != itree.end()) return it->second;
        const auto& fl = island_faces[id];
        MatrixXi Fi(fl.size(), 3);
        for (size_t k = 0; k < fl.size(); ++k) Fi.row(k) = src.F.row(fl[k]);
        auto t = std::make_shared<igl::AABB<MatrixXd, 3>>();
        t->init(src.V, Fi);
        return itree[id] = {Fi, t};
    };
    const int nq = (int)out.Q.rows();
    out.UV.resize(4 * nq, 2);
    out.UV.setZero();
    for (int q = 0; q < nq; ++q) {
        int n = out.Q(q, 3) < 0 ? 3 : 4;
        Eigen::RowVector3d c = Eigen::RowVector3d::Zero();
        for (int j = 0; j < n; ++j) c += out.V.row(out.Q(q, j));
        c /= n;
        int fi;
        Eigen::RowVector3d cc;
        tree.squared_distance(src.V, src.F, c, fi, cc);
        int isl = island[fi];
        auto& it = island_tree(isl);
        const auto& fl = island_faces[isl];
        for (int j = 0; j < n; ++j) {
            Eigen::RowVector3d p = out.V.row(out.Q(q, j));
            int k;
            Eigen::RowVector3d cp;
            it.second->squared_distance(src.V, it.first, p, k, cp);
            int f = fl[k];
            Eigen::RowVector3d b = bary(src.V, src.F, f, cp);
            out.UV.row(4 * q + j) = b[0] * src.UV.row(src.FUV(f, 0)) + b[1] * src.UV.row(src.FUV(f, 1)) + b[2] * src.UV.row(src.FUV(f, 2));
        }
    }
}

}  // namespace qrm
