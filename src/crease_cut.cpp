#include "quadremesh/crease_cut.h"

#include <igl/AABB.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace qrm {

namespace {

using Poly = std::vector<int>;

double pt_seg_d2(const Vector3d& p, const Vector3d& a, const Vector3d& b) {
    Vector3d ab = b - a;
    double l2 = ab.squaredNorm();
    double t = l2 > 0 ? std::clamp((p - a).dot(ab) / l2, 0.0, 1.0) : 0.0;
    return (a + t * ab - p).squaredNorm();
}

// closest points between segments p0p1 (param s) and q0q1 (param t)
void seg_seg(const Vector3d& p0, const Vector3d& p1, const Vector3d& q0, const Vector3d& q1, double& s, double& t) {
    Vector3d d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
    if (a <= 1e-30 && e <= 1e-30) { s = t = 0; return; }
    if (a <= 1e-30) { s = 0; t = std::clamp(f / e, 0.0, 1.0); return; }
    double c = d1.dot(r);
    if (e <= 1e-30) { t = 0; s = std::clamp(-c / a, 0.0, 1.0); return; }
    double b = d1.dot(d2), den = a * e - b * b;
    s = den > 1e-30 ? std::clamp((b * f - c * e) / den, 0.0, 1.0) : 0.0;
    t = (b * s + f) / e;
    if (t < 0) { t = 0; s = std::clamp(-c / a, 0.0, 1.0); }
    else if (t > 1) { t = 1; s = std::clamp((b - c) / a, 0.0, 1.0); }
}

std::vector<Poly> polys_of(const QuadMesh& m) {
    std::vector<Poly> P(m.Q.rows());
    for (int i = 0; i < m.Q.rows(); ++i)
        for (int j = 0; j < 4; ++j)
            if (m.Q(i, j) >= 0) P[i].push_back(m.Q(i, j));
    return P;
}

// unique undirected edges of the mesh
void mesh_edges(const QuadMesh& m, MatrixXi& E) {
    std::set<std::pair<int, int>> es;
    for (int i = 0; i < m.Q.rows(); ++i) {
        int n = m.Q(i, 3) < 0 ? 3 : 4;
        for (int j = 0; j < n; ++j) {
            int a = m.Q(i, j), b = m.Q(i, (j + 1) % n);
            es.insert({std::min(a, b), std::max(a, b)});
        }
    }
    E.resize(es.size(), 2);
    int k = 0;
    for (auto& e : es) E.row(k++) << e.first, e.second;
}

// resample a polyline: points + unit direction of the segment they lie on + arclength
struct Sample {
    Vector3d p, d;
    double s;
};
std::vector<Sample> resample(const std::vector<Vector3d>& c, double h) {
    std::vector<Sample> out;
    double acc = 0;
    for (size_t k = 0; k + 1 < c.size(); ++k) {
        Vector3d ab = c[k + 1] - c[k];
        double l = ab.norm();
        if (l <= 0) continue;
        int n = std::max(1, (int)std::ceil(l / h));
        for (int i = 0; i < n; ++i) out.push_back({c[k] + ab * ((i + 0.5) / n), ab / l, acc + l * (i + 0.5) / n});
        acc += l;
    }
    return out;
}

std::vector<Vector3d> sub_polyline(const std::vector<Vector3d>& c, double s0, double s1) {
    std::vector<Vector3d> out;
    double acc = 0;
    for (size_t k = 0; k + 1 < c.size(); ++k) {
        double l = (c[k + 1] - c[k]).norm();
        double a = acc, b = acc + l;
        if (b >= s0 && a <= s1 && l > 0) {
            double t0 = std::clamp((s0 - a) / l, 0.0, 1.0), t1 = std::clamp((s1 - a) / l, 0.0, 1.0);
            Vector3d p0 = c[k] + t0 * (c[k + 1] - c[k]), p1 = c[k] + t1 * (c[k + 1] - c[k]);
            if (out.empty() || (out.back() - p0).norm() > 1e-12) out.push_back(p0);
            if ((out.back() - p1).norm() > 1e-12) out.push_back(p1);
        }
        acc = b;
    }
    return out;
}

Vector3d newell(const std::vector<Vector3d>& P) {
    Vector3d n = Vector3d::Zero();
    for (size_t i = 0; i < P.size(); ++i) {
        const Vector3d& a = P[i];
        const Vector3d& b = P[(i + 1) % P.size()];
        n.x() += (a.y() - b.y()) * (a.z() + b.z());
        n.y() += (a.z() - b.z()) * (a.x() + b.x());
        n.z() += (a.x() - b.x()) * (a.y() + b.y());
    }
    double l = n.norm();
    return l > 0 ? Vector3d(n / l) : Vector3d(0, 0, 1);
}

// minimum corner scaled Jacobian of a polygon (corners in order) w.r.t. normal n
double min_corner_sj(const std::vector<Vector3d>& P, const Vector3d& n) {
    double q = 1;
    for (size_t i = 0; i < P.size(); ++i) {
        const Vector3d& c = P[i];
        Vector3d e1 = P[(i + 1) % P.size()] - c, e2 = P[(i + P.size() - 1) % P.size()] - c;
        double l = e1.norm() * e2.norm();
        if (l <= 1e-30) return -1;
        q = std::min(q, e1.cross(e2).dot(n) / l);
    }
    return q;
}

// split a polygon into quads (+ one triangle if odd) by cutting off the best quad each time
void tessellate(const Poly& poly, const std::vector<Vector3d>& V, std::vector<Poly>& out) {
    Poly p = poly;
    while (p.size() > 4) {
        std::vector<Vector3d> P;
        for (int v : p) P.push_back(V[v]);
        Vector3d n = newell(P);
        const int k = (int)p.size();
        int best = 0;
        double bq = -1e9;
        for (int i = 0; i < k; ++i) {
            std::vector<Vector3d> qd = {P[i], P[(i + 1) % k], P[(i + 2) % k], P[(i + 3) % k]};
            std::vector<Vector3d> rest;
            for (int j = 3; j <= k; ++j) rest.push_back(P[(i + j) % k]);
            double q = std::min(min_corner_sj(qd, n), rest.size() >= 3 && rest.size() <= 4 ? min_corner_sj(rest, n) : 1.0);
            if (q > bq) bq = q, best = i;
        }
        out.push_back({p[best], p[(best + 1) % k], p[(best + 2) % k], p[(best + 3) % k]});
        Poly r;
        for (int j = 3; j <= k; ++j) r.push_back(p[(best + j) % k]);
        p = r;
    }
    out.push_back(p);
}

struct Node {
    int type = 0;  // 0 existing vertex, 1 split of edge (a, b)
    int v = -1, a = -1, b = -1;
    double t = 0;     // parameter on (a, b)
    Vector3d pos;     // crease point
    double sigma = 0;  // sample parameter (ordering)
};

}  // namespace

std::vector<std::vector<Vector3d>> uncovered_crease_runs(const QuadMesh& m,
                                                         const std::vector<std::vector<Vector3d>>& chains, double L,
                                                         double min_run, double pad, double tol, double max_angle,
                                                         double* covered, double* total) {
    std::vector<std::vector<Vector3d>> runs;
    MatrixXi E;
    mesh_edges(m, E);
    if (covered) *covered = 0;
    if (total) *total = 0;
    if (E.rows() == 0) return runs;
    std::vector<std::vector<int>> vinc(m.V.rows());
    for (int i = 0; i < E.rows(); ++i) vinc[E(i, 0)].push_back(i), vinc[E(i, 1)].push_back(i);
    igl::AABB<MatrixXd, 3> tree;
    tree.init(m.V, E);
    const double tl = tol * L, cos_a = std::cos(max_angle * M_PI / 180.0);
    for (auto& c : chains) {
        auto S = resample(c, 0.1 * L);
        if (S.empty()) continue;
        MatrixXd P(S.size(), 3);
        for (size_t i = 0; i < S.size(); ++i) P.row(i) = S[i].p.transpose();
        VectorXd D;
        VectorXi I;
        MatrixXd C;
        tree.squared_distance(m.V, E, P, D, I, C);
        std::vector<char> ok(S.size(), 0);
        for (size_t i = 0; i < S.size(); ++i) {
            if (D[i] > tl * tl) continue;
            auto aligned = [&](int e) {
                Vector3d d = m.V.row(E(e, 1)) - m.V.row(E(e, 0));
                double l = d.norm();
                if (l <= 0 || std::abs(d.dot(S[i].d) / l) < cos_a) return false;
                return pt_seg_d2(S[i].p, m.V.row(E(e, 0)).transpose(), m.V.row(E(e, 1)).transpose()) <= tl * tl;
            };
            int e0 = I[i];
            bool good = aligned(e0);
            for (int v : {E(e0, 0), E(e0, 1)})
                for (int e : vinc[v])
                    if (!good && aligned(e)) good = true;
            ok[i] = good;
        }
        double len = 0;
        for (size_t k = 0; k + 1 < c.size(); ++k) len += (c[k + 1] - c[k]).norm();
        const double step = len / S.size();
        for (size_t i = 0; i < S.size(); ++i) {
            if (total) *total += step;
            if (ok[i] && covered) *covered += step;
        }
        for (size_t i = 0; i < S.size();) {
            if (ok[i]) { ++i; continue; }
            size_t j = i;
            while (j < S.size() && !ok[j]) ++j;
            double s0 = S[i].s - 0.5 * step, s1 = S[j - 1].s + 0.5 * step;
            if (s1 - s0 >= min_run * L) {
                auto r = sub_polyline(c, std::max(0.0, s0 - pad * L), std::min(len, s1 + pad * L));
                if (r.size() >= 2) runs.push_back(r);
            }
            i = j;
        }
    }
    return runs;
}

int embed_crease_run(QuadMesh& m, const std::vector<Vector3d>& run, double L, double min_quality,
                     std::vector<Vector3d>* crease_vertices, double snap) {
    if (run.size() < 2 || m.Q.rows() == 0) return -1;
    std::vector<Poly> polys = polys_of(m);
    // triangle soup for closest-face queries
    std::vector<std::array<int, 3>> tris;
    std::vector<int> tri_face;
    for (int f = 0; f < (int)polys.size(); ++f) {
        const Poly& p = polys[f];
        for (size_t j = 1; j + 1 < p.size(); ++j) {
            tris.push_back({p[0], p[j], p[j + 1]});
            tri_face.push_back(f);
        }
    }
    MatrixXi TF(tris.size(), 3);
    for (size_t i = 0; i < tris.size(); ++i) TF.row(i) << tris[i][0], tris[i][1], tris[i][2];
    igl::AABB<MatrixXd, 3> tree;
    tree.init(m.V, TF);
    auto closest = [&](const Vector3d& p, Vector3d& c, double& d2) {
        MatrixXd P = p.transpose();
        VectorXd D;
        VectorXi I;
        MatrixXd C;
        tree.squared_distance(m.V, TF, P, D, I, C);
        c = C.row(0).transpose();
        d2 = D[0];
        return tri_face[I[0]];
    };

    // samples along the run, with straight extensions so the cut can end on a face boundary
    const double h = 0.04 * L;
    auto S = resample(run, h);
    if (S.size() < 2) return -1;
    double rlen = 0;
    for (size_t k = 0; k + 1 < run.size(); ++k) rlen += (run[k + 1] - run[k]).norm();
    // samples along the run only; the run's ends are anchored inside their faces below
    std::vector<Vector3d> pts = {run.front()};
    for (auto& x : S) pts.push_back(x.p);
    pts.push_back(run.back());
    const int is = 0, ie = (int)pts.size() - 1;
    const int n = (int)pts.size();
    std::vector<int> face(n);
    std::vector<Vector3d> proj(n);
    for (int i = 0; i < n; ++i) {
        double d2;
        face[i] = closest(pts[i], proj[i], d2);
        if (d2 > (0.8 * L) * (0.8 * L)) return -2;  // mesh far from the crease
    }
    auto tangent_at = [&](bool start) {
        const Vector3d& a = start ? pts.front() : pts.back();
        for (int k = 1; k < n; ++k) {
            const Vector3d& b = start ? pts[k] : pts[n - 1 - k];
            if ((b - a).norm() > 0.15 * L || k + 1 == n) return Vector3d((start ? a - b : b - a).normalized());
        }
        return Vector3d(1, 0, 0);
    };

    // vertex -> faces
    std::vector<std::vector<int>> vf(m.V.rows());
    for (int f = 0; f < (int)polys.size(); ++f)
        for (int v : polys[f]) vf[v].push_back(f);
    auto shared_edge = [&](int f, int g, int& a, int& b) {
        const Poly& p = polys[f];
        for (size_t j = 0; j < p.size(); ++j) {
            int x = p[j], y = p[(j + 1) % p.size()];
            const Poly& q = polys[g];
            for (size_t k = 0; k < q.size(); ++k)
                if (q[k] == y && q[(k + 1) % q.size()] == x) { a = x, b = y; return true; }
        }
        return false;
    };
    auto shared_vertex = [&](int f, int g) {
        for (int v : polys[f])
            if (std::find(polys[g].begin(), polys[g].end(), v) != polys[g].end()) return v;
        return -1;
    };
    std::vector<Node> nodes;
    bool fail = false;
    // transition between consecutive samples in different faces (recursive refinement)
    std::function<void(const Vector3d&, const Vector3d&, int, const Vector3d&, const Vector3d&, int, double, double, int)>
        trans = [&](const Vector3d& pa, const Vector3d& ca, int fa, const Vector3d& pb, const Vector3d& cb, int fb,
                    double sa, double sb, int depth) {
            if (fa == fb || fail) return;
            int a, b;
            if (shared_edge(fa, fb, a, b)) {
                double s, t;
                seg_seg(ca, cb, m.V.row(a).transpose(), m.V.row(b).transpose(), s, t);
                Node nd;
                nd.pos = pa + s * (pb - pa);
                nd.sigma = sa + s * (sb - sa);
                if (t < snap) nd.type = 0, nd.v = a;
                else if (t > 1 - snap) nd.type = 0, nd.v = b;
                else nd.type = 1, nd.a = a, nd.b = b, nd.t = t;
                nodes.push_back(nd);
                return;
            }
            if (depth >= 7) {
                int v = shared_vertex(fa, fb);
                if (v < 0) { fail = true; return; }
                Node nd;
                nd.type = 0, nd.v = v, nd.pos = 0.5 * (pa + pb), nd.sigma = 0.5 * (sa + sb);
                nodes.push_back(nd);
                return;
            }
            Vector3d pm = 0.5 * (pa + pb), cm;
            double d2;
            int fm = closest(pm, cm, d2);
            double sm = 0.5 * (sa + sb);
            trans(pa, ca, fa, pm, cm, fm, sa, sm, depth + 1);
            trans(pm, cm, fm, pb, cb, fb, sm, sb, depth + 1);
        };
    for (int i = 0; i + 1 < n && !fail; ++i)
        if (face[i] != face[i + 1]) trans(pts[i], proj[i], face[i], pts[i + 1], proj[i + 1], face[i + 1], i, i + 1, 0);
    if (fail) return -3;
    // dedupe repeated vertex nodes; where the crease comes back to an element it already used
    // (it wiggles around a vertex or across an edge), erase the loop in between
    std::vector<Node> M;
    auto key = [](const Node& nd) {
        return nd.type == 0 ? std::make_pair(nd.v, -1) : std::make_pair(std::min(nd.a, nd.b), std::max(nd.a, nd.b));
    };
    for (auto& nd : nodes) {
        if (!M.empty() && nd.type == 0 && M.back().type == 0 && M.back().v == nd.v) continue;
        int hit = -1;
        for (int k = 0; k < (int)M.size(); ++k)
            if (key(M[k]) == key(nd)) hit = k;
        if (hit >= 0) { M.resize(hit + 1); continue; }
        M.push_back(nd);
    }
    // anchors: where the run starts / ends inside a face, the cut continues from the run's end
    // point to the vertex of that face that lies best along the crease direction (type 2: an
    // existing vertex that is not moved). The run end point becomes an interior vertex.
    bool head = M.empty() || (M.front().pos - pts.front()).norm() > 0.1 * L;
    bool tail = M.empty() || (M.back().pos - pts.back()).norm() > 0.1 * L;
    auto anchor = [&](int f, const Vector3d& p, const Vector3d& d, int avoid) {
        int best = -1;
        double bc = -0.3;
        for (int u : polys[f]) {
            if (u == avoid) continue;
            Vector3d e = m.V.row(u).transpose() - p;
            if (e.norm() <= 1e-12) continue;
            double c = e.normalized().dot(d);
            if (c > bc) bc = c, best = u;
        }
        return best;
    };
    if (head) {
        Node nd;
        nd.type = 2;
        nd.v = anchor(face[0], pts.front(), tangent_at(true), !M.empty() && M.front().type == 0 ? M.front().v : -1);
        if (nd.v < 0) return -9;
        nd.pos = m.V.row(nd.v).transpose();
        nd.sigma = 0;
        M.insert(M.begin(), nd);
    }
    if (tail) {
        Node nd;
        nd.type = 2;
        nd.v = anchor(face[n - 1], pts.back(), tangent_at(false), M.back().type != 1 ? M.back().v : -1);
        if (nd.v < 0) return -9;
        nd.pos = m.V.row(nd.v).transpose();
        nd.sigma = n - 1;
        M.push_back(nd);
    }
    {
        std::set<int> vs;
        for (auto& nd : M)
            if (nd.type != 1 && !vs.insert(nd.v).second) return -5;
    }
    if (M.size() < 2) return -4;

    // vertex nodes: the crease point nearest to the vertex (it is moved there)
    auto crease_point = [&](const Vector3d& x, double sig) {
        Vector3d best = x;
        double bd = 1e300;
        int lo = std::max(0, (int)sig - 30), hi = std::min(n - 1, (int)sig + 30);
        for (int i = lo; i < hi; ++i) {
            Vector3d ab = pts[i + 1] - pts[i];
            double l2 = ab.squaredNorm();
            double t = l2 > 0 ? std::clamp((x - pts[i]).dot(ab) / l2, 0.0, 1.0) : 0.0;
            Vector3d c = pts[i] + t * ab;
            if ((c - x).squaredNorm() < bd) bd = (c - x).squaredNorm(), best = c;
        }
        return best;
    };
    std::vector<Vector3d> V(m.V.rows());
    for (int i = 0; i < m.V.rows(); ++i) V[i] = m.V.row(i).transpose();
    for (auto& nd : M)
        if (nd.type == 0) {
            Vector3d c = crease_point(V[nd.v], nd.sigma);
            if ((c - V[nd.v]).norm() > 0.8 * L) return -6;
            nd.pos = c;
        }

    // interior vertices where the crease bends inside a face (chord sag / turn)
    std::vector<std::vector<Vector3d>> interior(M.size() - 1);
    std::function<void(Vector3d, Vector3d, int, int, int, std::vector<Vector3d>&)> bend =
        [&](Vector3d A, Vector3d B, int i0, int i1, int depth, std::vector<Vector3d>& out) {
            if (depth > 2 || i1 - i0 < 3) return;
            Vector3d ch = B - A;
            double cl = ch.norm();
            if (cl < 0.2 * L) return;
            Vector3d cd = ch / cl;
            int worst = -1;
            double wd = 0;
            bool turn = false;
            for (int i = std::max(i0, is); i <= std::min(i1, ie); ++i) {
                double d = std::sqrt(pt_seg_d2(pts[i], A, B));
                if (d > wd) wd = d, worst = i;
                if (i + 1 <= ie) {
                    Vector3d sd = pts[i + 1] - pts[i];
                    if (sd.norm() > 0 && std::abs(sd.normalized().dot(cd)) < std::cos(20.0 * M_PI / 180.0)) turn = true;
                }
            }
            if (worst < 0 || (wd < 0.06 * L && !turn) || wd < 0.01 * L) return;
            bend(A, pts[worst], i0, worst, depth + 1, out);
            out.push_back(pts[worst]);
            bend(pts[worst], B, worst, i1, depth + 1, out);
        };
    for (size_t k = 0; k + 1 < M.size(); ++k) {
        // an anchor's segment runs anchor -> run end point (interior vertex) -> along the crease
        Vector3d A = M[k].type == 2 ? pts.front() : M[k].pos;
        Vector3d B = M[k + 1].type == 2 ? pts.back() : M[k + 1].pos;
        if (M[k].type == 2) interior[k].push_back(pts.front());
        bend(A, B, (int)std::floor(M[k].sigma), (int)std::ceil(M[k + 1].sigma), 0, interior[k]);
        if (M[k + 1].type == 2) interior[k].push_back(pts.back());
    }

    if (std::getenv("QRM_DEBUG_CUT") && std::string(std::getenv("QRM_DEBUG_CUT")) == "3") {
        std::fprintf(stderr, "run %.3fL nodes:", rlen / L);
        for (size_t k = 0; k < M.size(); ++k) {
            double dv = M[k].type == 1 ? 0 : (m.V.row(M[k].v).transpose() - M[k].pos).norm() / L;
            std::fprintf(stderr, " [t%d s%.0f mv%.2f int%zu]", M[k].type, M[k].sigma, dv, k + 1 < M.size() ? interior[k].size() : 0);
        }
        std::fprintf(stderr, "\n");
    }
    // ---- apply: edge splits, vertex moves, polygon cuts
    std::vector<Vector3d> pnorm(polys.size());  // reference normal per face (before any change)
    std::vector<double> pq(polys.size());        // and its quality
    for (size_t f = 0; f < polys.size(); ++f) {
        std::vector<Vector3d> P;
        for (int v : polys[f]) P.push_back(V[v]);
        pnorm[f] = newell(P);
        pq[f] = min_corner_sj(P, pnorm[f]);
    }
    const int nf_before = (int)polys.size();
    std::set<int> touched;
    std::vector<int> nid(M.size());
    for (size_t k = 0; k < M.size(); ++k) {
        auto& nd = M[k];
        if (nd.type == 2) {
            nid[k] = nd.v;
            for (int f : vf[nd.v]) touched.insert(f);
            continue;
        }
        if (nd.type == 0) {
            V[nd.v] = nd.pos;
            nid[k] = nd.v;
            for (int f : vf[nd.v]) touched.insert(f);
            continue;
        }
        int w = (int)V.size();
        V.push_back(nd.pos);
        nid[k] = w;
        for (int f : vf[nd.a]) {
            Poly& p = polys[f];
            for (size_t j = 0; j < p.size(); ++j) {
                int x = p[j], y = p[(j + 1) % p.size()];
                if ((x == nd.a && y == nd.b) || (x == nd.b && y == nd.a)) {
                    p.insert(p.begin() + j + 1, w);
                    touched.insert(f);
                    break;
                }
            }
        }
    }
    auto find_poly = [&](int A, int B, int& f, int& ia, int& ib) {
        for (int g : touched) {
            const Poly& p = polys[g];
            auto a = std::find(p.begin(), p.end(), A), b = std::find(p.begin(), p.end(), B);
            if (a != p.end() && b != p.end()) { f = g, ia = (int)(a - p.begin()), ib = (int)(b - p.begin()); return true; }
        }
        return false;
    };
    for (size_t k = 0; k + 1 < M.size(); ++k) {
        int A = nid[k], B = nid[k + 1];
        int f, ia, ib;
        if (!find_poly(A, B, f, ia, ib)) return -7;
        Poly p = polys[f];
        const int ps = (int)p.size();
        bool adjacent = (ia + 1) % ps == ib || (ib + 1) % ps == ia;
        if (adjacent && interior[k].empty()) continue;  // already an edge
        std::vector<int> I;
        for (auto& x : interior[k]) { I.push_back((int)V.size()); V.push_back(x); }
        Poly p1, p2;
        for (int j = ia;; j = (j + 1) % ps) { p1.push_back(p[j]); if (j == ib) break; }
        for (auto it = I.rbegin(); it != I.rend(); ++it) p1.push_back(*it);
        for (int j = ib;; j = (j + 1) % ps) { p2.push_back(p[j]); if (j == ia) break; }
        for (int x : I) p2.push_back(x);
        if (p1.size() < 3 || p2.size() < 3) return -1;
        polys[f] = p1;
        polys.push_back(p2);
        pnorm.push_back(pnorm[f]);
        pq.push_back(pq[f]);
        touched.insert((int)polys.size() - 1);
    }
    // re-split touched polygons into quads (+ one triangle)
    std::vector<Poly> out;
    std::vector<Vector3d> onorm;
    std::vector<double> oq;
    std::vector<char> is_new;
    for (int f = 0; f < (int)polys.size(); ++f)
        if (!touched.count(f)) {
            out.push_back(polys[f]);
            onorm.push_back(f < (int)pnorm.size() ? pnorm[f] : Vector3d(0, 0, 1));
            oq.push_back(f < (int)pq.size() ? pq[f] : 1.0);
            is_new.push_back(0);
        }
    for (int f : touched) {
        std::vector<Poly> pieces;
        if (polys[f].size() <= 4) pieces.push_back(polys[f]);
        else tessellate(polys[f], V, pieces);
        for (auto& q : pieces) {
            if (q.size() < 3) return -1;
            out.push_back(q);
            onorm.push_back(pnorm[f]);
            oq.push_back(pq[f]);
            is_new.push_back(1);
        }
    }
    // local untangling: the vertices of the new pieces that are not on the crease (and not
    // anchors) are moved towards the average of their neighbours in the tangent plane, so the
    // pieces cut off by the crease become convex. Crease vertices stay where they are.
    {
        std::set<int> fixed_v;
        for (size_t k = 0; k < M.size(); ++k)
            if (M[k].type != 2) fixed_v.insert(nid[k]);  // anchors are ordinary vertices: they may move
        for (int x = (int)m.V.rows(); x < (int)V.size(); ++x) fixed_v.insert(x);  // new crease vertices
        std::set<int> freev;
        for (size_t f = 0; f < out.size(); ++f)
            if (is_new[f])
                for (int v : out[f])
                    if (!fixed_v.count(v)) freev.insert(v);
        std::map<int, std::set<int>> nb;
        std::map<int, Vector3d> vn;
        std::vector<size_t> ring;  // faces incident to free vertices (checked below as well)
        for (size_t f = 0; f < out.size(); ++f) {
            bool hit = false;
            for (size_t j = 0; j < out[f].size(); ++j) {
                int v = out[f][j];
                if (!freev.count(v)) continue;
                hit = true;
                nb[v].insert(out[f][(j + 1) % out[f].size()]);
                nb[v].insert(out[f][(j + out[f].size() - 1) % out[f].size()]);
                vn[v] += onorm[f];
            }
            if (hit && !is_new[f]) ring.push_back(f);
        }
        for (int it = 0; it < 10; ++it) {
            std::map<int, Vector3d> np;
            for (int v : freev) {
                Vector3d c = Vector3d::Zero();
                for (int u : nb[v]) c += V[u];
                c /= std::max<size_t>(1, nb[v].size());
                Vector3d nrm = vn[v].normalized();
                Vector3d d = c - V[v];
                d -= d.dot(nrm) * nrm;
                np[v] = V[v] + 0.5 * d;
            }
            for (auto& kv : np) V[kv.first] = kv.second;
        }
        for (size_t f = 0; f < out.size(); ++f) {
            if (!is_new[f] && std::find(ring.begin(), ring.end(), f) == ring.end()) continue;
            std::vector<Vector3d> P;
            for (int v : out[f]) P.push_back(V[v]);
            // a face may not end worse than min_quality, unless it already was
            if (min_corner_sj(P, onorm[f]) < std::min(min_quality, oq[f] - 0.05)) {
                if (std::getenv("QRM_DEBUG_CUT") && std::string(std::getenv("QRM_DEBUG_CUT")) == "4")
                    std::fprintf(stderr, "q-fail: new=%d q0=%.2f piece n=%zu q=%.2f nodes=%zu\n", is_new[f], oq[f], out[f].size(),
                                 min_corner_sj(P, onorm[f]), M.size());
                return -8;
            }
        }
    }
    m.V.resize(V.size(), 3);
    for (size_t i = 0; i < V.size(); ++i) m.V.row(i) = V[i].transpose();
    m.Q.resize(out.size(), 4);
    for (size_t f = 0; f < out.size(); ++f) {
        for (int j = 0; j < 4; ++j) m.Q(f, j) = j < (int)out[f].size() ? out[f][j] : -1;
    }
    m.UV.resize(0, 2);
    if (crease_vertices) {
        for (size_t k = 0; k < M.size(); ++k)
            if (M[k].type != 2) crease_vertices->push_back(V[nid[k]]);
        for (auto& I : interior)
            for (auto& x : I) crease_vertices->push_back(x);
    }
    return (int)out.size() - nf_before;
}

}  // namespace qrm
