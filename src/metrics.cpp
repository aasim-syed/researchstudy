#include "quadremesh/metrics.h"

#include <igl/AABB.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

#include "quadremesh/features.h"
#include "quadremesh/geometry.h"
#include "quadremesh/preprocess.h"

namespace qrm {

double scaled_jacobian(const MatrixXd& V, int a, int b, int c, int d) {
    if (d < 0) {
        int t[3] = {a, b, c};
        double m = 1e9;
        for (int k = 0; k < 3; ++k) {
            Vector3d p = V.row(t[k]), e1 = V.row(t[(k + 1) % 3]) - V.row(t[k]), e2 = V.row(t[(k + 2) % 3]) - V.row(t[k]);
            double den = e1.norm() * e2.norm();
            if (den <= 0) return 0;
            m = std::min(m, e1.cross(e2).norm() / den * 2.0 / std::sqrt(3.0));
        }
        return std::min(1.0, m);
    }
    int q[4] = {a, b, c, d};
    Vector3d n = Vector3d::Zero();
    for (int k = 0; k < 4; ++k) {
        Vector3d e1 = V.row(q[(k + 1) % 4]) - V.row(q[k]), e2 = V.row(q[(k + 3) % 4]) - V.row(q[k]);
        n += e1.cross(e2);
    }
    double ln = n.norm();
    if (ln <= 0) return -1;
    n /= ln;
    double m = 1e9;
    for (int k = 0; k < 4; ++k) {
        Vector3d e1 = V.row(q[(k + 1) % 4]) - V.row(q[k]), e2 = V.row(q[(k + 3) % 4]) - V.row(q[k]);
        double den = e1.norm() * e2.norm();
        if (den <= 0) return -1;
        m = std::min(m, e1.cross(e2).dot(n) / den);
    }
    return m;
}

double planarity_error(const MatrixXd& V, int a, int b, int c, int d) {
    if (d < 0) return 0;
    Vector3d p[4] = {V.row(a), V.row(b), V.row(c), V.row(d)};
    Vector3d n = (p[2] - p[0]).cross(p[3] - p[1]);
    double ln = n.norm();
    double el = 0;
    for (int k = 0; k < 4; ++k) el += (p[(k + 1) % 4] - p[k]).norm();
    el /= 4;
    if (ln <= 0 || el <= 0) return 0;
    n /= ln;
    Vector3d c0 = (p[0] + p[1] + p[2] + p[3]) / 4;
    double m = 0;
    for (auto& q : p) m = std::max(m, std::abs((q - c0).dot(n)));
    return m / el;
}

namespace {

struct Stats {
    double mean = 0, min = 0, max = 0, p05 = 0, p50 = 0, p95 = 0, stdev = 0;
};
Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0, sq = 0;
    for (double x : v) sum += x, sq += x * x;
    s.mean = sum / v.size();
    s.stdev = std::sqrt(std::max(0.0, sq / v.size() - s.mean * s.mean));
    s.min = v.front();
    s.max = v.back();
    auto q = [&](double t) { return v[std::min(v.size() - 1, (size_t)(t * (v.size() - 1) + 0.5))]; };
    s.p05 = q(0.05), s.p50 = q(0.5), s.p95 = q(0.95);
    return s;
}

struct Json {
    std::ostringstream os;
    bool first = true;
    int depth = 0;
    Json() { os << std::setprecision(6); }
    void sep() {
        if (!first) os << ",";
        os << "\n" << std::string(2 * depth, ' ');
        first = false;
    }
    void open(const std::string& k) { sep(); os << "\"" << k << "\": {"; depth++; first = true; }
    void close() { depth--; os << "\n" << std::string(2 * depth, ' ') << "}"; first = false; }
    void num(const std::string& k, double v) {
        sep();
        if (std::isfinite(v)) os << "\"" << k << "\": " << v;
        else os << "\"" << k << "\": null";
    }
    void integer(const std::string& k, long long v) { sep(); os << "\"" << k << "\": " << v; }
    void str(const std::string& k, const std::string& v) { sep(); os << "\"" << k << "\": \"" << v << "\""; }
    void boolean(const std::string& k, bool v) { sep(); os << "\"" << k << "\": " << (v ? "true" : "false"); }
    void stat(const std::string& k, const Stats& s) {
        open(k);
        num("mean", s.mean), num("min", s.min), num("p05", s.p05), num("median", s.p50), num("p95", s.p95), num("max", s.max), num("std", s.stdev);
        close();
    }
};

// area-weighted surface samples of a triangle mesh
MatrixXd sample_surface(const MatrixXd& V, const MatrixXi& F, int n, Pcg32& rng) {
    std::vector<double> cdf(F.rows());
    double acc = 0;
    for (int f = 0; f < F.rows(); ++f) cdf[f] = (acc += face_area(V, F, f));
    MatrixXd P(n, 3);
    for (int i = 0; i < n; ++i) {
        double r = rng.uniform() * acc;
        int f = (int)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
        f = std::min(f, (int)F.rows() - 1);
        double u = rng.uniform(), v = rng.uniform();
        if (u + v > 1) u = 1 - u, v = 1 - v;
        P.row(i) = (1 - u - v) * V.row(F(f, 0)) + u * V.row(F(f, 1)) + v * V.row(F(f, 2));
    }
    return P;
}

}  // namespace

std::string evaluate(const TriMesh& input, const QuadMesh& out, const EvalParams& ep, double runtime, const std::string& method) {
    Json J;
    J.os << "{";
    J.str("method", method);
    const double diag = std::max(bbox_diagonal(input.V), 1e-12);
    const int nq = (int)out.Q.rows();

    // ---------------------------------------------------------- face statistics
    int nquad = 0, ntri = 0;
    for (int i = 0; i < nq; ++i) (out.Q(i, 3) < 0 ? ntri : nquad)++;
    J.open("faces");
    J.integer("total", nq), J.integer("quads", nquad), J.integer("triangles", ntri), J.integer("vertices", out.V.rows());
    J.num("quad_ratio", nq ? (double)nquad / nq : 0.0);
    if (ep.target_faces > 0) J.integer("target", ep.target_faces), J.num("count_ratio", (double)nq / ep.target_faces);
    J.close();
    if (nq == 0) {
        J.os << "\n}";
        return J.os.str() + "\n";
    }

    // ---------------------------------------------------------- clean input reference
    Params pp;
    pp.max_hole_perimeter = 0;
    pp.min_component_area = 0;
    PrepStats ps;
    WorkMesh clean = preprocess(input.V, input.F, MatrixXd(input.V.rows(), 0), pp, ps);
    const double area_in = surface_area(clean.V, clean.F);
    const double L = std::sqrt(area_in / (ep.target_faces > 0 ? ep.target_faces : nq));

    // output as triangles (split quads along the shorter diagonal)
    MatrixXi OT(nquad * 2 + ntri, 3);
    {
        int k = 0;
        for (int i = 0; i < nq; ++i) {
            int a = out.Q(i, 0), b = out.Q(i, 1), c = out.Q(i, 2), d = out.Q(i, 3);
            if (d < 0) { OT.row(k++) << a, b, c; continue; }
            if ((out.V.row(a) - out.V.row(c)).squaredNorm() <= (out.V.row(b) - out.V.row(d)).squaredNorm()) {
                OT.row(k++) << a, b, c;
                OT.row(k++) << a, c, d;
            } else {
                OT.row(k++) << a, b, d;
                OT.row(k++) << b, c, d;
            }
        }
    }

    // ---------------------------------------------------------- element quality
    std::vector<double> sj, sj_quad, plan, aspect, elen;
    int sj_below_05 = 0, inverted = 0;
    for (int i = 0; i < nq; ++i) {
        int a = out.Q(i, 0), b = out.Q(i, 1), c = out.Q(i, 2), d = out.Q(i, 3);
        double s = scaled_jacobian(out.V, a, b, c, d);
        sj.push_back(s);
        if (s < 0.5) sj_below_05++;
        if (s <= 0) inverted++;
        int n = d < 0 ? 3 : 4;
        int q[4] = {a, b, c, d};
        double mn = 1e300, mx = 0;
        for (int k = 0; k < n; ++k) {
            double l = (out.V.row(q[(k + 1) % n]) - out.V.row(q[k])).norm();
            mn = std::min(mn, l), mx = std::max(mx, l);
            elen.push_back(l);
        }
        if (d >= 0) {
            sj_quad.push_back(s);
            plan.push_back(planarity_error(out.V, a, b, c, d));
            aspect.push_back(mn > 0 ? mx / mn : 1e9);
        }
    }
    J.open("quality");
    J.stat("scaled_jacobian", stats(sj));
    J.stat("scaled_jacobian_quads", stats(sj_quad));
    J.num("frac_sj_below_0.5", (double)sj_below_05 / nq);
    J.integer("inverted_or_degenerate", inverted);
    J.stat("planarity", stats(plan));
    J.stat("aspect_ratio", stats(aspect));
    Stats es = stats(elen);
    J.stat("edge_length_rel", [&] { Stats r = es; double m = es.mean > 0 ? es.mean : 1; r.mean /= m; r.min /= m; r.max /= m; r.p05 /= m; r.p50 /= m; r.p95 /= m; r.stdev /= m; return r; }());
    J.num("edge_length_mean_pct_diag", es.mean / diag * 100);
    J.close();

    // ---------------------------------------------------------- topology / singularities
    std::map<std::pair<int, int>, int> ecount;
    std::vector<int> valence(out.V.rows(), 0);
    for (int i = 0; i < nq; ++i) {
        int n = out.Q(i, 3) < 0 ? 3 : 4;
        for (int k = 0; k < n; ++k) {
            int a = out.Q(i, k), b = out.Q(i, (k + 1) % n);
            ecount[{std::min(a, b), std::max(a, b)}]++;
        }
    }
    std::vector<char> onb(out.V.rows(), 0), used(out.V.rows(), 0);
    int nm_edges = 0, bnd_edges = 0;
    for (auto& kv : ecount) {
        valence[kv.first.first]++, valence[kv.first.second]++;
        used[kv.first.first] = used[kv.first.second] = 1;
        if (kv.second == 1) { onb[kv.first.first] = onb[kv.first.second] = 1; bnd_edges++; }
        if (kv.second > 2) nm_edges++;
    }
    int sing_int = 0, sing_bnd = 0, nint = 0, nbndv = 0;
    std::map<int, int> vhist;
    for (int v = 0; v < out.V.rows(); ++v) {
        if (!used[v]) continue;
        if (onb[v]) {
            nbndv++;
            if (valence[v] != 3 && valence[v] != 2) sing_bnd++;
        } else {
            nint++;
            vhist[valence[v]]++;
            if (valence[v] != 4) sing_int++;
        }
    }
    std::vector<int> comp;
    int ncomp = face_components(OT, (int)out.V.rows(), comp);
    J.open("topology");
    J.integer("singularities", sing_int);
    J.integer("boundary_irregular", sing_bnd);
    J.num("singularities_per_1k_faces", 1000.0 * sing_int / nq);
    J.integer("interior_vertices", nint), J.integer("boundary_vertices", nbndv);
    J.integer("boundary_edges", bnd_edges), J.integer("nonmanifold_edges", nm_edges), J.integer("components", ncomp);
    J.open("valence_histogram");
    for (auto& kv : vhist) J.integer(std::to_string(kv.first), kv.second);
    J.close();
    J.close();

    // ---------------------------------------------------------- geometric fidelity
    {
        Pcg32 rng(ep.seed);
        igl::AABB<MatrixXd, 3> tin, tout;
        tin.init(clean.V, clean.F);
        tout.init(out.V, OT);
        MatrixXd Po = sample_surface(out.V, OT, ep.samples, rng);
        MatrixXd Pi = sample_surface(clean.V, clean.F, ep.samples, rng);
        VectorXd d_oi, d_io;
        VectorXi I;
        MatrixXd C;
        tin.squared_distance(clean.V, clean.F, Po, d_oi, I, C);
        tout.squared_distance(out.V, OT, Pi, d_io, I, C);
        d_oi = d_oi.cwiseSqrt();
        d_io = d_io.cwiseSqrt();
        double m_oi = d_oi.mean(), m_io = d_io.mean();
        double rms = std::sqrt(0.5 * (d_oi.squaredNorm() / d_oi.size() + d_io.squaredNorm() / d_io.size()));
        J.open("distance");
        J.num("mean_pct", 50.0 * (m_oi + m_io) / diag);
        J.num("out_to_in_mean_pct", 100.0 * m_oi / diag);
        J.num("in_to_out_mean_pct", 100.0 * m_io / diag);
        J.num("rms_pct", 100.0 * rms / diag);
        J.num("hausdorff_pct", 100.0 * std::max(d_oi.maxCoeff(), d_io.maxCoeff()) / diag);
        J.num("area_ratio", surface_area(out.V, OT) / std::max(area_in, 1e-30));
        J.close();
    }

    // ---------------------------------------------------------- feature coverage
    {
        FeatureSet fs = detect_features(clean.V, clean.F, ep.feature_angle, ep.min_feature_length * L, 0.3 * L);
        // output edge segments
        std::vector<std::pair<int, int>> oe;
        for (auto& kv : ecount) oe.push_back(kv.first);
        MatrixXi E(oe.size(), 2);
        std::vector<std::vector<int>> vinc(out.V.rows());
        for (size_t i = 0; i < oe.size(); ++i) {
            E(i, 0) = oe[i].first, E(i, 1) = oe[i].second;
            vinc[oe[i].first].push_back((int)i), vinc[oe[i].second].push_back((int)i);
        }
        igl::AABB<MatrixXd, 3> te;
        te.init(out.V, E);
        const double tol = ep.coverage_tol * L;
        const double cos_a = std::cos(ep.coverage_angle * M_PI / 180.0);
        auto coverage = [&](const MatrixXd& S, double& covered_len, double& total_len) {
            covered_len = total_len = 0;
            if (S.rows() == 0) return;
            std::vector<Vector3d> pts, dirs;
            std::vector<double> wts;
            for (int s = 0; s < S.rows(); ++s) {
                Vector3d a = S.row(s).head<3>(), b = S.row(s).tail<3>();
                double l = (b - a).norm();
                if (l <= 0) continue;
                int k = std::max(1, (int)std::ceil(l / (0.1 * L)));
                for (int i = 0; i < k; ++i) {
                    pts.push_back(a + (b - a) * ((i + 0.5) / k));
                    dirs.push_back((b - a) / l);
                    wts.push_back(l / k);
                }
            }
            MatrixXd P(pts.size(), 3);
            for (size_t i = 0; i < pts.size(); ++i) P.row(i) = pts[i].transpose();
            VectorXd D;
            VectorXi I;
            MatrixXd C;
            te.squared_distance(out.V, E, P, D, I, C);
            for (size_t i = 0; i < pts.size(); ++i) {
                total_len += wts[i];
                if (D[i] > tol * tol) continue;
                bool ok = false;
                auto aligned = [&](int e) {
                    Vector3d d = out.V.row(E(e, 1)) - out.V.row(E(e, 0));
                    double l = d.norm();
                    if (l <= 0) return false;
                    if (std::abs(d.dot(dirs[i]) / l) < cos_a) return false;
                    return point_segment_dist2(pts[i], out.V.row(E(e, 0)), out.V.row(E(e, 1))) <= tol * tol;
                };
                int e0 = I[i];
                if (aligned(e0)) ok = true;
                for (int v : {E(e0, 0), E(e0, 1)})
                    for (int e : vinc[v])
                        if (!ok && aligned(e)) ok = true;
                if (ok) covered_len += wts[i];
            }
        };
        double fc, ft, bc, bt;
        coverage(fs.segments, fc, ft);
        coverage(fs.boundary_segments, bc, bt);
        J.open("features");
        J.num("feature_coverage", ft > 0 ? fc / ft : 1.0);
        J.num("boundary_coverage", bt > 0 ? bc / bt : 1.0);
        J.num("combined_coverage", (ft + bt) > 0 ? (fc + bc) / (ft + bt) : 1.0);
        J.num("feature_length_pct_diag", 100.0 * ft / diag);
        J.num("boundary_length_pct_diag", 100.0 * bt / diag);
        J.integer("feature_segments", fs.segments.rows());
        J.num("tolerance_L", ep.coverage_tol);
        J.num("feature_angle", ep.feature_angle);
        J.close();
    }
    if (runtime >= 0) J.num("runtime_s", runtime);
    J.integer("input_faces", input.F.rows());
    J.os << "\n}";
    return J.os.str() + "\n";
}

}  // namespace qrm
