#include "decimate.h"

#include <Eigen/Dense>
#include <algorithm>
#include <queue>
#include <set>
#include <vector>

#include "quadremesh/geometry.h"

namespace qrm {

namespace {
using Mat4 = Eigen::Matrix4d;

struct Cand {
    double cost;
    int a, b, stamp_a, stamp_b;
    Vector3d p;
    bool operator>(const Cand& o) const { return cost > o.cost || (cost == o.cost && (a > o.a || (a == o.a && b > o.b))); }
};
}  // namespace

void decimate_qem(MatrixXd& V, MatrixXi& F, int target_faces) {
    const int nv = (int)V.rows();
    int nf = (int)F.rows();
    if (nf <= target_faces || nf < 4) return;
    std::vector<Vector3d> P(nv);
    for (int i = 0; i < nv; ++i) P[i] = V.row(i).transpose();
    std::vector<std::array<int, 3>> T(nf);
    for (int f = 0; f < nf; ++f) T[f] = {F(f, 0), F(f, 1), F(f, 2)};
    std::vector<char> fdead(nf, 0), vdead(nv, 0);
    std::vector<std::vector<int>> vf(nv);
    for (int f = 0; f < nf; ++f)
        for (int j = 0; j < 3; ++j) vf[T[f][j]].push_back(f);
    std::vector<Mat4> Qm(nv, Mat4::Zero());
    auto plane_q = [](const Vector3d& n, const Vector3d& p, double w) {
        Eigen::Vector4d pl(n.x(), n.y(), n.z(), -n.dot(p));
        return Mat4(w * pl * pl.transpose());
    };
    // face quadrics (area weighted) + boundary constraint planes
    EdgeTopology topo;
    topo.build(F);
    for (int f = 0; f < nf; ++f) {
        Vector3d a = P[T[f][0]], b = P[T[f][1]], c = P[T[f][2]];
        Vector3d n = (b - a).cross(c - a);
        double ar = n.norm();
        if (ar <= 0) continue;
        n /= ar;
        Mat4 q = plane_q(n, a, ar);
        for (int j = 0; j < 3; ++j) Qm[T[f][j]] += q;
    }
    for (size_t e = 0; e < topo.ev.size(); ++e) {
        if (topo.ehe[e].size() != 1) continue;
        int h = topo.ehe[e][0], f = h / 3;
        Vector3d a = P[topo.ev[e].first], b = P[topo.ev[e].second];
        Vector3d fnorm = face_normal(V, F, f);
        Vector3d n = (b - a).cross(fnorm);
        double l = n.norm();
        if (l <= 0) continue;
        n /= l;
        Mat4 q = plane_q(n, a, 100.0 * (b - a).squaredNorm());
        Qm[topo.ev[e].first] += q;
        Qm[topo.ev[e].second] += q;
    }
    std::vector<int> stamp(nv, 0);
    auto evaluate = [&](int a, int b, Cand& c) {
        Mat4 q = Qm[a] + Qm[b];
        Eigen::Matrix4d A = q;
        A.row(3) << 0, 0, 0, 1;
        Vector3d p;
        if (std::abs(A.determinant()) > 1e-12) {
            Eigen::Vector4d x = A.inverse() * Eigen::Vector4d(0, 0, 0, 1);
            p = x.head<3>();
            // keep the optimum near the edge
            Vector3d m = 0.5 * (P[a] + P[b]);
            if ((p - m).norm() > 2.0 * (P[a] - P[b]).norm()) p = m;
        } else {
            p = 0.5 * (P[a] + P[b]);
        }
        Eigen::Vector4d ph(p.x(), p.y(), p.z(), 1);
        c.cost = std::max(0.0, ph.dot(q * ph));
        c.a = a, c.b = b, c.p = p, c.stamp_a = stamp[a], c.stamp_b = stamp[b];
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> heap;
    for (auto& e : topo.ev) {
        Cand c;
        evaluate(e.first, e.second, c);
        heap.push(c);
    }
    auto neighbours = [&](int v, std::set<int>& out) {
        for (int f : vf[v])
            if (!fdead[f])
                for (int k = 0; k < 3; ++k)
                    if (T[f][k] != v) out.insert(T[f][k]);
    };
    std::vector<char> on_boundary(nv, 0);
    for (size_t e = 0; e < topo.ev.size(); ++e)
        if (topo.ehe[e].size() == 1) on_boundary[topo.ev[e].first] = on_boundary[topo.ev[e].second] = 1;
    int alive = nf;
    while (alive > target_faces && !heap.empty()) {
        Cand c = heap.top();
        heap.pop();
        int a = c.a, b = c.b;
        if (vdead[a] || vdead[b] || c.stamp_a != stamp[a] || c.stamp_b != stamp[b]) continue;
        // link condition: common neighbours must be exactly the opposite vertices of shared faces
        std::set<int> na, nb;
        neighbours(a, na);
        neighbours(b, nb);
        if (!na.count(b)) continue;
        int shared_faces = 0;
        for (int f : vf[a])
            if (!fdead[f] && (T[f][0] == b || T[f][1] == b || T[f][2] == b)) shared_faces++;
        int common = 0;
        for (int x : na)
            if (nb.count(x)) common++;
        if (common != shared_faces) continue;
        if (on_boundary[a] && on_boundary[b] && shared_faces != 1) continue;
        // normal flip check
        bool flip = false;
        for (int v : {a, b}) {
            for (int f : vf[v]) {
                if (fdead[f]) continue;
                bool has_both = (T[f][0] == b || T[f][1] == b || T[f][2] == b) && (T[f][0] == a || T[f][1] == a || T[f][2] == a);
                if (has_both) continue;
                Vector3d p0 = P[T[f][0]], p1 = P[T[f][1]], p2 = P[T[f][2]];
                Vector3d n0 = (p1 - p0).cross(p2 - p0);
                Vector3d q[3] = {p0, p1, p2};
                for (int k = 0; k < 3; ++k)
                    if (T[f][k] == v) q[k] = c.p;
                Vector3d n1 = (q[1] - q[0]).cross(q[2] - q[0]);
                if (n0.dot(n1) <= 0.2 * n0.norm() * n1.norm()) { flip = true; break; }
            }
            if (flip) break;
        }
        if (flip) continue;
        // collapse b -> a
        P[a] = c.p;
        Qm[a] += Qm[b];
        on_boundary[a] = on_boundary[a] || on_boundary[b];
        vdead[b] = 1;
        for (int f : vf[b]) {
            if (fdead[f]) continue;
            bool has_a = T[f][0] == a || T[f][1] == a || T[f][2] == a;
            if (has_a) { fdead[f] = 1; alive--; continue; }
            for (int k = 0; k < 3; ++k)
                if (T[f][k] == b) T[f][k] = a;
            vf[a].push_back(f);
        }
        stamp[a]++;
        std::set<int> nn;
        neighbours(a, nn);
        for (int x : nn) {
            Cand c2;
            evaluate(a, x, c2);
            heap.push(c2);
        }
    }
    // compact
    std::vector<int> map(nv, -1);
    int k = 0;
    for (int f = 0; f < nf; ++f)
        if (!fdead[f])
            for (int j = 0; j < 3; ++j)
                if (map[T[f][j]] < 0) map[T[f][j]] = k++;
    MatrixXd V2(k, 3);
    for (int i = 0; i < nv; ++i)
        if (map[i] >= 0) V2.row(map[i]) = P[i].transpose();
    MatrixXi F2(alive, 3);
    int r = 0;
    for (int f = 0; f < nf; ++f)
        if (!fdead[f]) F2.row(r++) << map[T[f][0]], map[T[f][1]], map[T[f][2]];
    V = V2;
    F = F2.topRows(r);
}

}  // namespace qrm
