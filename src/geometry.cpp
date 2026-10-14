#include "quadremesh/geometry.h"

#include <algorithm>
#include <array>

namespace qrm {

void EdgeTopology::build(const Eigen::MatrixXi& F) {
    const int m = (int)F.rows();
    std::vector<std::array<int, 3>> rec;  // (a, b, he)
    rec.reserve(3 * m);
    for (int f = 0; f < m; ++f)
        for (int j = 0; j < 3; ++j) {
            int a = F(f, j), b = F(f, (j + 1) % 3);
            rec.push_back({std::min(a, b), std::max(a, b), 3 * f + j});
        }
    std::sort(rec.begin(), rec.end());
    he2e.assign(3 * m, -1);
    ev.clear();
    ehe.clear();
    for (size_t i = 0; i < rec.size();) {
        size_t k = i;
        int e = (int)ev.size();
        ev.push_back({rec[i][0], rec[i][1]});
        ehe.emplace_back();
        while (k < rec.size() && rec[k][0] == rec[i][0] && rec[k][1] == rec[i][1]) {
            he2e[rec[k][2]] = e;
            ehe[e].push_back(rec[k][2]);
            ++k;
        }
        i = k;
    }
}

int EdgeTopology::twin(int h) const {
    const auto& v = ehe[he2e[h]];
    if (v.size() != 2) return -1;
    return v[0] == h ? v[1] : v[0];
}

double surface_area(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    double a = 0;
    for (int f = 0; f < F.rows(); ++f) a += face_area(V, F, f);
    return a;
}

double bbox_diagonal(const Eigen::MatrixXd& V) {
    if (V.rows() == 0) return 0;
    return (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
}

Eigen::MatrixXd vertex_normals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(V.rows(), 3);
    for (int f = 0; f < F.rows(); ++f) {
        Eigen::Vector3d a = V.row(F(f, 0)), b = V.row(F(f, 1)), c = V.row(F(f, 2));
        Eigen::Vector3d n = (b - a).cross(c - a);  // area weighted
        for (int j = 0; j < 3; ++j) N.row(F(f, j)) += n.transpose();
    }
    for (int i = 0; i < N.rows(); ++i) {
        double l = N.row(i).norm();
        if (l > 0) N.row(i) /= l;
        else N.row(i) << 0, 0, 1;
    }
    return N;
}

int face_components(const Eigen::MatrixXi& F, int nv, std::vector<int>& comp) {
    UnionFind uf(nv);
    for (int f = 0; f < F.rows(); ++f) {
        uf.unite(F(f, 0), F(f, 1));
        uf.unite(F(f, 0), F(f, 2));
    }
    std::vector<int> label(nv, -1);
    comp.assign(F.rows(), -1);
    int nc = 0;
    for (int f = 0; f < F.rows(); ++f) {
        int r = uf.find(F(f, 0));
        if (label[r] < 0) label[r] = nc++;
        comp[f] = label[r];
    }
    return nc;
}

}  // namespace qrm
