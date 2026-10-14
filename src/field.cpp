#include "quadremesh/field.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <queue>

#include "quadremesh/geometry.h"

namespace qrm {

namespace {
inline void tangent_frame(const Vector3d& n, Vector3d& t1, Vector3d& t2) {
    Vector3d a = std::abs(n.x()) < 0.9 ? Vector3d(1, 0, 0) : Vector3d(0, 1, 0);
    t1 = (a - n * n.dot(a)).normalized();
    t2 = n.cross(t1);
}
}  // namespace

CurvatureInfo principal_directions(const MatrixXd& V, const MatrixXi& F, double radius, double min_aniso) {
    CurvatureInfo ci;
    const int nv = (int)V.rows();
    ci.N = vertex_normals(V, F);
    EdgeTopology topo;
    topo.build(F);
    std::vector<std::vector<int>> nbr(nv);
    double hsum = 0;
    for (auto& e : topo.ev) {
        nbr[e.first].push_back(e.second);
        nbr[e.second].push_back(e.first);
        hsum += (V.row(e.first) - V.row(e.second)).norm();
    }
    const double h = topo.ev.empty() ? 1.0 : hsum / topo.ev.size();
    // ---- per-vertex fit of II in a local frame: kappa(d) = d^T II d
    std::vector<Eigen::Matrix3d> T(nv, Eigen::Matrix3d::Zero());
    for (int i = 0; i < nv; ++i) {
        Vector3d n = ci.N.row(i), t1, t2;
        tangent_frame(n, t1, t2);
        Eigen::Matrix3d AtA = Eigen::Matrix3d::Identity() * 1e-9;
        Vector3d Atb = Vector3d::Zero();
        for (int j : nbr[i]) {
            Vector3d e = V.row(j) - V.row(i);
            double l2 = e.squaredNorm();
            if (l2 <= 0) continue;
            Vector3d dn = ci.N.row(j) - ci.N.row(i);
            double kappa = dn.dot(e) / l2;
            Vector3d et = e - n * n.dot(e);
            double lt = et.norm();
            if (lt <= 0) continue;
            double x = et.dot(t1) / lt, y = et.dot(t2) / lt;
            Vector3d row(x * x, 2 * x * y, y * y);
            AtA += row * row.transpose();
            Atb += row * kappa;
        }
        Vector3d s = AtA.ldlt().solve(Atb);
        T[i] = s[0] * t1 * t1.transpose() + s[1] * (t1 * t2.transpose() + t2 * t1.transpose()) + s[2] * t2 * t2.transpose();
    }
    // ---- tensor diffusion (radius ~ sqrt(2 * iters) * h)
    int iters = (int)std::ceil(0.5 * (radius / h) * (radius / h));
    iters = std::max(2, std::min(iters, 40));
    std::vector<Eigen::Matrix3d> T2(nv);
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < nv; ++i) {
            Eigen::Matrix3d acc = T[i];
            for (int j : nbr[i]) acc += T[j];
            acc /= (1.0 + nbr[i].size());
            Vector3d n = ci.N.row(i);
            Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - n * n.transpose();
            T2[i] = P * acc * P;
        }
        T.swap(T2);
    }
    ci.D1.resize(nv, 3);
    ci.k1.resize(nv);
    ci.k2.resize(nv);
    ci.conf.resize(nv);
    for (int i = 0; i < nv; ++i) {
        Vector3d n = ci.N.row(i), t1, t2;
        tangent_frame(n, t1, t2);
        Eigen::Matrix2d S;
        S << t1.dot(T[i] * t1), t1.dot(T[i] * t2), t2.dot(T[i] * t1), t2.dot(T[i] * t2);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(S);
        double a = es.eigenvalues()[0], b = es.eigenvalues()[1];
        Eigen::Vector2d v = std::abs(b) >= std::abs(a) ? es.eigenvectors().col(1) : es.eigenvectors().col(0);
        ci.k1[i] = std::abs(b) >= std::abs(a) ? b : a;
        ci.k2[i] = std::abs(b) >= std::abs(a) ? a : b;
        ci.D1.row(i) = (v[0] * t1 + v[1] * t2).normalized().transpose();
        double diff = std::abs(a - b);
        double aniso = diff / (std::abs(a) + std::abs(b) + 1e-12);
        // reliability: anisotropic AND curved at the scale of the target element
        double strength = std::min(1.0, diff * radius);
        double c = std::max(0.0, (aniso - min_aniso) / std::max(1e-6, 1.0 - min_aniso));
        ci.conf[i] = std::min(1.0, c * strength * 2.0);
    }
    return ci;
}

FieldConstraints build_constraints(const MatrixXd& V, const MatrixXi& F, const MatrixXd& N,
                                   const std::vector<int>& he_sharp, bool boundary_hard,
                                   const CurvatureInfo* curv, double curvature_weight, const MatrixXd* guide,
                                   const VectorXd* guide_w, double guide_strength,
                                   const std::vector<std::vector<Vector3d>>& strokes, double stroke_radius,
                                   double soft_fade) {
    const int nv = (int)V.rows();
    FieldConstraints fc;
    fc.dir = MatrixXd::Zero(nv, 3);
    fc.w = VectorXd::Zero(nv);
    fc.pos.assign(nv, 0);
    auto tangent = [&](int i, Vector3d d) {
        Vector3d n = N.row(i);
        d -= n * n.dot(d);
        double l = d.norm();
        return l > 1e-12 ? Vector3d(d / l) : Vector3d(0, 0, 0);
    };
    // soft: curvature
    if (curv && curvature_weight > 0) {
        for (int i = 0; i < nv; ++i) {
            double w = curvature_weight * curv->conf[i];
            if (w <= 1e-3) continue;
            fc.dir.row(i) = tangent(i, curv->D1.row(i)).transpose();
            fc.w[i] = w;
        }
    }
    // soft: external / learned guide (takes precedence where more confident)
    if (guide && guide->rows() == nv && guide_strength > 0) {
        for (int i = 0; i < nv; ++i) {
            double w = guide_strength * (guide_w && guide_w->size() == nv ? (*guide_w)[i] : 1.0);
            if (w <= fc.w[i] || w <= 1e-3) continue;
            Vector3d d = tangent(i, guide->row(i));
            if (d.squaredNorm() == 0) continue;
            fc.dir.row(i) = d.transpose();
            fc.w[i] = std::min(w, 0.95);
        }
    }
    // hard: strokes (direction only) - every vertex within stroke_radius of the polyline
    if (!strokes.empty()) {
        for (int i = 0; i < nv; ++i) {
            Vector3d p = V.row(i);
            double best = stroke_radius * stroke_radius;
            Vector3d bd(0, 0, 0);
            for (const auto& s : strokes)
                for (size_t k = 0; k + 1 < s.size(); ++k) {
                    double d2 = point_segment_dist2(p, s[k], s[k + 1]);
                    if (d2 < best) { best = d2; bd = s[k + 1] - s[k]; }
                }
            if (bd.squaredNorm() > 0) {
                Vector3d d = tangent(i, bd);
                if (d.squaredNorm() > 0) {
                    fc.dir.row(i) = d.transpose();
                    fc.w[i] = 1.0;
                }
            }
        }
    }
    // hard: feature + boundary edges (direction + position)
    EdgeTopology topo;
    topo.build(F);
    // soft constraints fade out near hard ones (within soft_fade): the hard feature directions
    // should propagate smoothly instead of competing with a nearby curvature direction.
    if (soft_fade > 0) {
        std::vector<std::vector<std::pair<int, double>>> adj(nv);
        for (auto& e : topo.ev) {
            double l = (V.row(e.first) - V.row(e.second)).norm();
            adj[e.first].push_back({e.second, l});
            adj[e.second].push_back({e.first, l});
        }
        std::vector<double> dist(nv, 1e300);
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
        for (size_t e = 0; e < topo.ev.size(); ++e) {
            bool bnd = topo.ehe[e].size() == 1, sh = false;
            for (int h : topo.ehe[e]) sh |= (!he_sharp.empty() && he_sharp[h]);
            if (sh || (bnd && boundary_hard))
                for (int v : {topo.ev[e].first, topo.ev[e].second})
                    if (dist[v] > 0) { dist[v] = 0; pq.push({0.0, v}); }
        }
        while (!pq.empty()) {
            auto [d, v] = pq.top();
            pq.pop();
            if (d > dist[v] || d > soft_fade) continue;
            for (auto [u, l] : adj[v])
                if (d + l < dist[u]) { dist[u] = d + l; pq.push({d + l, u}); }
        }
        for (int i = 0; i < nv; ++i)
            if (fc.w[i] < 1.0 && dist[i] < soft_fade) fc.w[i] *= dist[i] / soft_fade;
    }
    for (size_t e = 0; e < topo.ev.size(); ++e) {
        bool bnd = topo.ehe[e].size() == 1;
        bool sh = false;
        for (int h : topo.ehe[e]) sh |= (!he_sharp.empty() && he_sharp[h]);
        if (!(sh || (bnd && boundary_hard))) continue;
        int a = topo.ev[e].first, b = topo.ev[e].second;
        Vector3d d = V.row(b) - V.row(a);
        if (d.squaredNorm() == 0) continue;
        for (int v : {a, b}) {
            fc.dir.row(v) = tangent(v, d).transpose();
            if (fc.dir.row(v).squaredNorm() == 0) fc.dir.row(v) = d.normalized().transpose();
            fc.w[v] = 1.0;
            fc.pos[v] = 1;
        }
    }
    return fc;
}

VectorXd density_to_scale(const MatrixXd& V, const MatrixXi& F, const VectorXd& density, double max_ratio) {
    const int nv = (int)V.rows();
    VectorXd A = VectorXd::Zero(nv);
    for (int f = 0; f < F.rows(); ++f) {
        double a = face_area(V, F, f) / 3.0;
        for (int j = 0; j < 3; ++j) A[F(f, j)] += a;
    }
    VectorXd d = density.cwiseMax(1e-9);
    // clamp ratio around the area-weighted geometric mean
    double lg = 0, at = 0;
    for (int i = 0; i < nv; ++i) { lg += A[i] * std::log(d[i]); at += A[i]; }
    double gm = std::exp(lg / std::max(at, 1e-30));
    double r = std::sqrt(std::max(max_ratio, 1.0));
    for (int i = 0; i < nv; ++i) d[i] = std::min(std::max(d[i], gm / r), gm * r);
    // S = d^-1/2, normalised so that sum A/S^2 == sum A  (same expected quad count)
    VectorXd S(nv);
    double num = 0;
    for (int i = 0; i < nv; ++i) { S[i] = 1.0 / std::sqrt(d[i]); num += A[i] / (S[i] * S[i]); }
    double k = std::sqrt(num / std::max(at, 1e-30));
    S *= k;
    return S;
}

}  // namespace qrm
