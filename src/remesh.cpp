#include <chrono>
#include <cstdlib>
#include <cmath>
#include <map>
#include <sstream>

#include <igl/AABB.h>

#include <algorithm>

#include "decimate.h"
#include "im_driver.h"
#include "qflow_driver.h"
#include "quadremesh/features.h"
#include "quadremesh/field.h"
#include "quadremesh/geometry.h"
#include "quadremesh/postprocess.h"
#include "quadremesh/crease_cut.h"
#include "quadremesh/metrics.h"
#include "quadremesh/preprocess.h"
#include "quadremesh/quadremesh.h"
#include "quadremesh/transfer.h"

namespace qrm {

const char* version() { return "1.0.0"; }

namespace {
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Logger {
    std::vector<std::string>& out;
    int verbose;
    template <class... T>
    void operator()(T&&... args) {
        std::ostringstream os;
        (os << ... << args);
        out.push_back(os.str());
        if (verbose) std::fprintf(stderr, "[quadremesh] %s\n", os.str().c_str());
    }
};

// Fallback for components the integer solver cannot handle: split every triangle into 3
// quads (always conforming, exact geometry). Used only for tiny parts.
void tri_to_quads(const MatrixXd& V, const MatrixXi& F, MatrixXd& OV, std::vector<std::array<int, 4>>& OQ) {
    std::map<std::pair<int, int>, int> mid;
    std::vector<Vector3d> pts;
    for (int i = 0; i < V.rows(); ++i) pts.push_back(V.row(i));
    auto m = [&](int a, int b) {
        auto k = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = mid.find(k);
        if (it != mid.end()) return it->second;
        pts.push_back(0.5 * (pts[a] + pts[b]));
        return mid[k] = (int)pts.size() - 1;
    };
    for (int f = 0; f < F.rows(); ++f) {
        int a = F(f, 0), b = F(f, 1), c = F(f, 2);
        int ab = m(a, b), bc = m(b, c), ca = m(c, a);
        pts.push_back((pts[a] + pts[b] + pts[c]) / 3.0);
        int g = (int)pts.size() - 1;
        OQ.push_back({a, ab, g, ca});
        OQ.push_back({b, bc, g, ab});
        OQ.push_back({c, ca, g, bc});
    }
    OV.resize(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) OV.row(i) = pts[i].transpose();
}
// Distance (mean, 95th percentile) from samples of the triangle mesh (V,F) to the quads.
void coverage_distance(const MatrixXd& V, const MatrixXi& F, const MatrixXd& QV,
                       const std::vector<std::array<int, 4>>& Q, double& mean, double& p95) {
    MatrixXi T(2 * Q.size(), 3);
    for (size_t i = 0; i < Q.size(); ++i) {
        T.row(2 * i) << Q[i][0], Q[i][1], Q[i][2];
        T.row(2 * i + 1) << Q[i][0], Q[i][2], Q[i][3];
    }
    igl::AABB<MatrixXd, 3> tree;
    tree.init(QV, T);
    int n = std::min<int>((int)F.rows(), 4000);
    MatrixXd P(n, 3);
    for (int k = 0; k < n; ++k) {
        int f = (int)((long long)k * F.rows() / n);
        P.row(k) = (V.row(F(f, 0)) + V.row(F(f, 1)) + V.row(F(f, 2))) / 3.0;
    }
    VectorXd D;
    VectorXi I;
    MatrixXd C;
    tree.squared_distance(QV, T, P, D, I, C);
    std::vector<double> d(n);
    for (int k = 0; k < n; ++k) d[k] = std::sqrt(D[k]);
    mean = 0;
    for (double x : d) mean += x;
    mean /= std::max(1, n);
    std::sort(d.begin(), d.end());
    p95 = d.empty() ? 0 : d[std::min<size_t>(d.size() - 1, (size_t)(0.95 * d.size()))];
}
// Interior irregular vertices of a raw quad list (valence != 4, boundary excluded).
int count_singularities(int nv, const std::vector<std::array<int, 4>>& Q) {
    std::map<std::pair<int, int>, int> ec;
    for (auto& q : Q) {
        int n = q[3] < 0 || q[3] == q[2] || q[3] == q[0] ? 3 : 4;
        for (int k = 0; k < n; ++k) {
            int a = q[k], b = q[(k + 1) % n];
            if (a == b) continue;
            ec[{std::min(a, b), std::max(a, b)}]++;
        }
    }
    std::vector<int> val(nv, 0);
    std::vector<char> bnd(nv, 0);
    for (auto& kv : ec) {
        val[kv.first.first]++, val[kv.first.second]++;
        if (kv.second == 1) bnd[kv.first.first] = bnd[kv.first.second] = 1;
    }
    int s = 0;
    for (int v = 0; v < nv; ++v)
        if (val[v] > 0 && !bnd[v] && val[v] != 4) s++;
    return s;
}
}  // namespace

static Result quadremesh_once(const TriMesh& input, const Params& p) {
    Result R;
    Logger log{R.log, p.verbose};
    auto t0 = Clock::now();
    if (input.V.rows() < 3 || input.F.rows() < 1) throw std::runtime_error("empty input mesh");

    // ---------------------------------------------------------------- normalise
    const Vector3d bmin = input.V.colwise().minCoeff(), bmax = input.V.colwise().maxCoeff();
    const Vector3d center = 0.5 * (bmin + bmax);
    const double diag = std::max((bmax - bmin).norm(), 1e-12);
    const double s = 1.0 / diag;
    MatrixXd Vn = (input.V.rowwise() - center.transpose()) * s;

    // per-vertex attributes carried through preprocessing: [density | guide xyz, guide w]
    const bool has_density = p.density.size() == input.V.rows();
    const bool has_guide = p.guide_field.rows() == input.V.rows() && p.guide_field.cols() == 3;
    int na = (has_density ? 1 : 0) + (has_guide ? 4 : 0);
    MatrixXd A(input.V.rows(), na);
    {
        int c = 0;
        if (has_density) A.col(c++) = p.density;
        if (has_guide) {
            A.block(0, c, input.V.rows(), 3) = p.guide_field;
            A.col(c + 3) = p.guide_weight.size() == input.V.rows() ? p.guide_weight : VectorXd::Ones(input.V.rows());
        }
    }

    // ---------------------------------------------------------------- preprocess
    PrepStats ps;
    WorkMesh wm = preprocess(Vn, input.F, A, p, ps);
    log("preprocess: welded=", ps.welded, " degenerate=", ps.degenerate, " duplicate=", ps.duplicate,
        " nonmanifold_edges=", ps.nonmanifold_edges, " vertex_splits=", ps.vertex_splits, " flips=",
        ps.orientation_flips, " holes_filled=", ps.holes_filled, " holes_kept=", ps.holes_kept,
        " components_dropped=", ps.components_dropped, " -> V=", wm.V.rows(), " F=", wm.F.rows());
    if (wm.F.rows() == 0) throw std::runtime_error("nothing left after preprocessing");
    const double full_area = surface_area(wm.V, wm.F);

    // ---------------------------------------------------------------- symmetry (remesh one half)
    const bool sym = p.symmetry_axis >= 0 && p.symmetry_axis < 3;
    double sym_off = 0;
    if (sym) {
        sym_off = std::isnan(p.symmetry_offset) ? 0.0 : (p.symmetry_offset - center[p.symmetry_axis]) * s;
        clip_to_halfspace(wm, p.symmetry_axis, sym_off);
        log("symmetry: axis=", p.symmetry_axis, " plane=", sym_off, " half F=", wm.F.rows());
    }
    auto t1 = Clock::now();

    // ---------------------------------------------------------------- target size
    double L;  // target edge length of the final mesh (normalised units)
    if (p.target_edge_length > 0) L = p.target_edge_length * s;
    else L = std::sqrt(full_area / std::max(1, p.target_faces));
    const double Lqf = p.pure_quad ? 2.0 * L : L;  // field resolution
    const double work_area = surface_area(wm.V, wm.F);
    const int qf_total = std::max(4, (int)std::lround(work_area / (Lqf * Lqf)));
    log("target: L=", L, " field faces=", qf_total, p.pure_quad ? " (pure-quad split x4)" : "");

    // ---------------------------------------------------------------- features
    const bool dense_input = input.F.rows() > 30LL * std::max(1, p.target_faces);
    FeatureSet fs = detect_features(wm.V, wm.F, p.feature_angle, p.min_feature_length * Lqf, 1.0 * Lqf);
    // the unreduced crease set: used for edge-path recovery (vertex moves only), even where the
    // field below only gets a reduced set as hard constraints
    const FeatureSet fs_full = fs;
    // Feature budget: a quad mesh with N faces can carry roughly O(sqrt(N)) edge loops; when the
    // detected feature network is far denser than the resolution (typical for raw scans), keep
    // only the strongest/longest creases (raise angle and minimum chain length) - otherwise the
    // integer solve is over-constrained, slow and produces poor elements.
    {
        auto feature_len = [&](const FeatureSet& f) {
            double l = 0;
            for (int i = 0; i < f.segments.rows(); ++i) l += (f.segments.row(i).head<3>() - f.segments.row(i).tail<3>()).norm();
            return l;
        };
        // dense scans (input much finer than the target) get a stricter budget than CAD-like inputs
        const double budget = (dense_input ? 0.06 : p.feature_field_budget) * qf_total * Lqf;
        double ang = p.feature_angle, ml = p.min_feature_length;
        for (int it = 0; it < 4 && p.feature_angle > 0 && feature_len(fs) > budget; ++it) {
            ang = std::min(ang + 10.0, 80.0);
            ml *= 2.0;
            fs = detect_features(wm.V, wm.F, ang, ml * Lqf, 1.0 * Lqf);
            log("features: network denser than resolution -> angle ", ang, ", min length ", ml, " L");
        }
        if (p.feature_angle > 0 && feature_len(fs) > budget) {
            fs = detect_features(wm.V, wm.F, 0.0, 0.0);
            log("features: still denser than resolution (raw scan) -> hard feature constraints disabled");
        }
    }
    if (p.strokes_as_edges && !p.strokes.empty()) {
        std::vector<std::vector<Vector3d>> st;
        for (auto& stroke : p.strokes) {
            std::vector<Vector3d> q;
            for (auto& x : stroke) q.push_back((x - center) * s);
            st.push_back(q);
        }
        add_stroke_features(wm.V, wm.F, st, fs);
    }
    wm.sharp = fs.he_sharp;
    log("features: candidates=", fs.raw_edges, " kept=", fs.kept_edges, " corners=", fs.corners.size());
    // reference geometry for projection: preprocessed surface with its features
    MatrixXd refV = wm.V;
    MatrixXi refF = wm.F;
    std::vector<int> ref_corners = fs.corners, plane_verts;
    for (int v = 0; v < (int)wm.vtag.size(); ++v)
        if (wm.vtag[v] & 1) plane_verts.push_back(v);
    MatrixXi refFE, refBE;
    {
        EdgeTopology topo;
        topo.build(wm.F);
        std::vector<std::pair<int, int>> fe, be;
        for (size_t e = 0; e < topo.ev.size(); ++e) {
            if (topo.ehe[e].size() == 1) be.push_back(topo.ev[e]);
            else if ((p.trace_all_features ? fs_full : fs).he_sharp[topo.ehe[e][0]]) fe.push_back(topo.ev[e]);
        }
        refFE.resize(fe.size(), 2);
        for (size_t i = 0; i < fe.size(); ++i) refFE.row(i) << fe[i].first, fe[i].second;
        if (const char* dp = std::getenv("QRM_DUMP_FE")) {  // diagnostics: crease edges in input units
            if (std::FILE* fp = std::fopen(dp, "w")) {
                for (auto& e : fe) {
                    Vector3d a = wm.V.row(e.first).transpose() / s + center, b = wm.V.row(e.second).transpose() / s + center;
                    std::fprintf(fp, "%.9g %.9g %.9g %.9g %.9g %.9g\n", a.x(), a.y(), a.z(), b.x(), b.y(), b.z());
                }
                std::fclose(fp);
            }
        }
        refBE.resize(be.size(), 2);
        for (size_t i = 0; i < be.size(); ++i) refBE.row(i) << be[i].first, be[i].second;
        // boundary corners (sharp turns of open boundaries) are fixed too
        std::vector<std::vector<int>> bn(wm.V.rows());
        for (auto& e : be) bn[e.first].push_back(e.second), bn[e.second].push_back(e.first);
        if (p.preserve_boundary)
            for (int v = 0; v < wm.V.rows(); ++v)
                if (bn[v].size() == 2 && !(wm.vtag[v] & 1)) {
                    Vector3d a = (wm.V.row(bn[v][0]) - wm.V.row(v)).normalized(), b = (wm.V.row(bn[v][1]) - wm.V.row(v)).normalized();
                    if (a.dot(b) > std::cos(130.0 * M_PI / 180.0)) ref_corners.push_back(v);
                }
    }
    R.feature_edges.resize(refFE.rows(), 6);
    for (int i = 0; i < refFE.rows(); ++i) {
        R.feature_edges.block(i, 0, 1, 3) = refV.row(refFE(i, 0)) / s + center.transpose();
        R.feature_edges.block(i, 3, 1, 3) = refV.row(refFE(i, 1)) / s + center.transpose();
    }

    // ---------------------------------------------------------------- refine for the solver
    MatrixXd prepV;  // pre-refinement mesh (refinement keeps vertex ids and appends new ones)
    MatrixXi prepF;
    {
        double avg = 0;
        EdgeTopology topo;
        topo.build(wm.F);
        for (auto& e : topo.ev) avg += (wm.V.row(e.first) - wm.V.row(e.second)).norm();
        avg /= std::max<size_t>(1, topo.ev.size());
        double target_len = std::min(Lqf / 2, avg * 2);
        prepV = wm.V;
        prepF = wm.F;
        refine_long_edges(wm, target_len);
        log("refine: max edge ", target_len, " -> V=", wm.V.rows(), " F=", wm.F.rows());
    }
    auto t2 = Clock::now();

    // ---------------------------------------------------------------- field design inputs
    MatrixXd N = vertex_normals(wm.V, wm.F);
    // Normals for the field are smoothed at the scale of the target element (not across
    // feature edges): fine-scale scan noise otherwise turns into field singularities.
    MatrixXd Nfield = N;
    {
        EdgeTopology topo;
        topo.build(wm.F);
        double h = 0;
        for (auto& e : topo.ev) h += (wm.V.row(e.first) - wm.V.row(e.second)).norm();
        h /= std::max<size_t>(1, topo.ev.size());
        int iters = (int)std::lround(std::pow(0.5 * Lqf / std::max(h, 1e-12), 2));
        iters = std::max(0, std::min(iters, 30));
        if (std::getenv("QRM_NO_NSMOOTH") || !dense_input) iters = 0;  // scans only: on clean CAD meshes the solver normals are better
        if (iters > 0) {
            std::vector<std::vector<int>> nb(wm.V.rows());
            for (size_t e = 0; e < topo.ev.size(); ++e) {
                bool sharp = topo.ehe[e].size() != 2;
                for (int hh : topo.ehe[e]) sharp |= (!wm.sharp.empty() && wm.sharp[hh]);
                if (sharp) continue;
                nb[topo.ev[e].first].push_back(topo.ev[e].second);
                nb[topo.ev[e].second].push_back(topo.ev[e].first);
            }
            MatrixXd T = Nfield;
            for (int it = 0; it < iters; ++it) {
                for (int i = 0; i < wm.V.rows(); ++i) {
                    Vector3d acc = Nfield.row(i).transpose();
                    for (int j : nb[i]) acc += Nfield.row(j).transpose();
                    double l = acc.norm();
                    if (l > 0) T.row(i) = (acc / l).transpose();
                    else T.row(i) = Nfield.row(i);
                }
                Nfield.swap(T);
            }
            log("field normals smoothed: ", iters, " iterations");
        }
    }
    CurvatureInfo curv;
    bool use_curv = p.curvature_weight > 0 || p.curvature_density > 0;
    if (use_curv) curv = principal_directions(wm.V, wm.F, 0.75 * Lqf, p.curvature_min_anisotropy);
    MatrixXd guide;
    VectorXd guide_w;
    int col = has_density ? 1 : 0;
    if (has_guide) {
        guide = wm.A.block(0, col, wm.V.rows(), 3);
        guide_w = wm.A.col(col + 3);
    }
    std::vector<std::vector<Vector3d>> strokes_n;
    for (auto& stroke : p.strokes) {
        std::vector<Vector3d> q;
        for (auto& x : stroke) q.push_back((x - center) * s);
        strokes_n.push_back(q);
    }
    bool boundary_hard = p.preserve_boundary || sym;
    FieldConstraints cons = build_constraints(wm.V, wm.F, N, wm.sharp, boundary_hard, use_curv ? &curv : nullptr,
                                              p.curvature_weight, has_guide ? &guide : nullptr,
                                              has_guide ? &guide_w : nullptr, p.guide_strength, strokes_n, 0.6 * Lqf,
                                              2.0 * Lqf);
    // density -> scale
    VectorXd scale;
    bool adaptive = has_density || !p.density_regions.empty() || p.curvature_density > 0;
    if (adaptive) {
        VectorXd d = has_density ? VectorXd(wm.A.col(0)) : VectorXd::Ones(wm.V.rows());
        for (int i = 0; i < wm.V.rows(); ++i) {
            Vector3d x = wm.V.row(i).transpose() / s + center;
            for (auto& r : p.density_regions)
                if ((x - r.center).norm() <= r.radius) d[i] *= r.factor;
            if (p.curvature_density > 0) {
                double k = (std::abs(curv.k1[i]) + std::abs(curv.k2[i])) * Lqf;
                d[i] *= 1.0 + p.curvature_density * std::min(k, 4.0);
            }
        }
        scale = density_to_scale(wm.V, wm.F, d, p.max_density_ratio);
    }
    // Density is realised by solving the field at the coarse end of the size range and then
    // refining faces that are larger than the local target (a singularity-free field cannot
    // carry a non-harmonic density variation, so the integer solve stays uniform).
    double coarse_factor = 1.0;
    if (adaptive) {
        std::vector<std::pair<double, double>> sa;
        VectorXd Av = VectorXd::Zero(wm.V.rows());
        for (int f = 0; f < wm.F.rows(); ++f)
            for (int j = 0; j < 3; ++j) Av[wm.F(f, j)] += face_area(wm.V, wm.F, f) / 3;
        double at = 0;
        for (int i = 0; i < wm.V.rows(); ++i) { sa.push_back({scale[i], Av[i]}); at += Av[i]; }
        std::sort(sa.begin(), sa.end());
        double acc = 0;
        for (auto& x : sa) {
            acc += x.second;
            if (acc >= 0.85 * at) { coarse_factor = std::max(1.0, x.first); break; }
        }
        log("density: scale range ", scale.minCoeff(), "..", scale.maxCoeff(), " coarse factor ", coarse_factor);
    }

    // ---------------------------------------------------------------- per component solve
    std::vector<int> comp;
    int nc = face_components(wm.F, (int)wm.V.rows(), comp);
    R.components_in = nc;
    std::vector<double> carea(nc, 0);
    for (int f = 0; f < wm.F.rows(); ++f) carea[comp[f]] += face_area(wm.V, wm.F, f);
    std::vector<std::vector<int>> cfaces(nc);
    for (int f = 0; f < wm.F.rows(); ++f) cfaces[comp[f]].push_back(f);

    std::vector<Vector3d> outV;
    std::vector<std::array<int, 4>> outQ;
    R.field_V = wm.V;
    R.field_F = wm.F;
    R.field_Q = MatrixXd::Zero(wm.V.rows(), 3);
    R.field_N = N;
    int total_sing = 0;
    double t_extract = 0;
    int skipped_small = 0;
    for (int c = 0; c < nc; ++c) {
        const auto& fl = cfaces[c];
        // parts smaller than a few target quads cannot be represented at this resolution
        // (dense scans: small disconnected fragments are reconstruction debris; each closed
        // fragment would need >= 8 singularities, so fragments below ~40 quads are dropped)
        if (carea[c] < (dense_input ? 40.0 : 4.0) * Lqf * Lqf) { skipped_small++; continue; }
        std::vector<int> vmap(wm.V.rows(), -1), vlist;
        MatrixXi Fc(fl.size(), 3);
        for (size_t k = 0; k < fl.size(); ++k)
            for (int j = 0; j < 3; ++j) {
                int v = wm.F(fl[k], j);
                if (vmap[v] < 0) { vmap[v] = (int)vlist.size(); vlist.push_back(v); }
                Fc(k, j) = vmap[v];
            }
        QFJob job;
        job.V.resize(vlist.size(), 3);
        for (size_t i = 0; i < vlist.size(); ++i) job.V.row(i) = wm.V.row(vlist[i]);
        job.F = Fc;
        job.sharp_he.resize(3 * fl.size());
        for (size_t k = 0; k < fl.size(); ++k)
            for (int j = 0; j < 3; ++j) job.sharp_he[3 * k + j] = wm.sharp[3 * fl[k] + j];
        job.normals.resize(vlist.size(), 3);
        for (size_t i = 0; i < vlist.size(); ++i) job.normals.row(i) = Nfield.row(vlist[i]);
        job.cons.dir.resize(vlist.size(), 3);
        job.cons.w.resize(vlist.size());
        job.cons.pos.resize(vlist.size());
        for (size_t i = 0; i < vlist.size(); ++i) {
            job.cons.dir.row(i) = cons.dir.row(vlist[i]);
            job.cons.w[i] = cons.w[vlist[i]];
            job.cons.pos[i] = cons.pos[vlist[i]];
        }
        int budget = (int)std::lround(qf_total * carea[c] / work_area / (coarse_factor * coarse_factor));
        budget = std::max(budget, p.min_component_faces);
        job.preserve_boundary = boundary_hard;
        job.seed = p.seed;
        // feature-adaptive size inside the field solve: smaller elements near creases, larger
        // elsewhere, same expected face count (density -> conformal scale target of the solve)
        if (p.feature_density > 0 || std::getenv("QRM_FEAT_SCALE")) {
            const double k = std::getenv("QRM_FEAT_SCALE") ? std::atof(std::getenv("QRM_FEAT_SCALE")) : p.feature_density;
            const double wdt = std::getenv("QRM_FEAT_WIDTH") ? std::atof(std::getenv("QRM_FEAT_WIDTH")) : p.feature_density_width;
            std::vector<std::pair<int, int>> se;
            for (int e = 0; e < refFE.rows(); ++e) se.push_back({refFE(e, 0), refFE(e, 1)});
            if (!se.empty() && k > 0) {
                MatrixXd SV(2 * se.size(), 3);
                MatrixXi SE(se.size(), 2);
                for (size_t i = 0; i < se.size(); ++i) {
                    SV.row(2 * i) = refV.row(se[i].first);
                    SV.row(2 * i + 1) = refV.row(se[i].second);
                    SE.row(i) << (int)(2 * i), (int)(2 * i + 1);
                }
                igl::AABB<MatrixXd, 3> st;
                st.init(SV, SE);
                VectorXd D;
                VectorXi I;
                MatrixXd C;
                st.squared_distance(SV, SE, job.V, D, I, C);
                VectorXd d(job.V.rows());
                for (int i = 0; i < job.V.rows(); ++i) d[i] = 1.0 + k * std::exp(-D[i] / (wdt * wdt * Lqf * Lqf));
                job.scale = density_to_scale(job.V, job.F, d, p.max_density_ratio);
                log("feature scale: ", se.size(), " crease edges, min dist ", std::sqrt(D.minCoeff()) / Lqf, " L, density ",
                    d.minCoeff(), "..", d.maxCoeff(), ", scale ", job.scale.minCoeff(), "..", job.scale.maxCoeff());
            }
        }
        if (const char* dump = std::getenv("QRM_DUMP_COMPONENTS")) {
            TriMesh dm;
            dm.V = job.V;
            dm.F = job.F;
            save_tri_obj(std::string(dump) + "/comp_" + std::to_string(c) + ".obj", dm);
        }
        QFResult qr;
        bool good = false;
        QFResult best_rej;               // best rejected candidate (lowest coverage p95)
        double best_rej_p95 = 1e300, best_rej_mean = 1e300;
        // keep the FIRST (= lowest budget) candidate within the loose tolerance, else the best
        auto keep_best = [&](const QFResult& r, double mean, double p95) {
            if (r.Q.empty()) return;
            bool loose_ok = p95 < 3.0 * Lqf && mean < 0.6 * Lqf;
            bool have_loose = best_rej_p95 < 3.0 * Lqf && best_rej_mean < 0.6 * Lqf;
            if (have_loose) return;
            if (loose_ok || p95 < best_rej_p95) { best_rej = r; best_rej_p95 = p95; best_rej_mean = mean; }
        };
        // Instant Meshes extractor (robust on scans / thin parts; more singularities than the
        // integer-grid solve, so it is the second choice)
        auto try_im = [&](int im_faces) -> bool {
            if (job.F.rows() == 0) return false;
            std::vector<double> iv(3 * job.V.rows());
            std::vector<int> ifc(3 * job.F.rows());
            for (int i = 0; i < job.V.rows(); ++i)
                for (int k = 0; k < 3; ++k) iv[3 * i + k] = job.V(i, k);
            for (int f = 0; f < job.F.rows(); ++f)
                for (int k = 0; k < 3; ++k) ifc[3 * f + k] = job.F(f, k);
            auto te0 = Clock::now();
            IMResult ir = run_instant_meshes(iv, ifc, im_faces, p.feature_angle, boundary_hard);
            t_extract += secs(te0, Clock::now());
            if (!ir.ok) {
                log("component ", c, ": Instant Meshes extractor produced no faces (target ", im_faces, ")");
                return false;
            }
            QFResult alt;
            alt.V.resize(ir.V.size() / 3, 3);
            for (int i = 0; i < alt.V.rows(); ++i)
                for (int k = 0; k < 3; ++k) alt.V(i, k) = ir.V[3 * i + k];
            alt.Q = ir.Q;
            double cm = 0, c95 = 0;
            coverage_distance(job.V, job.F, alt.V, alt.Q, cm, c95);
            keep_best(alt, cm, c95);
            if (cm < 0.25 * Lqf && c95 < 1.0 * Lqf) {
                qr = alt;
                qr.ok = true;
                R.components_im++;
                log("component ", c, ": solved by the Instant Meshes extractor (", qr.Q.size(), " quads)");
                return true;
            }
            log("component ", c, ": Instant Meshes extractor rejected (cover mean/p95=", cm, "/", c95, ")");
            return false;
        };
        // Solve; if the extracted quads do not cover the component (thin parts, narrow strips
        // between features), retry at a finer field resolution before falling back.
        double qa = 0;
        // thin tubes / plates need several quads across: grow the budget geometrically
        static const int growth[5] = {1, 3, 9, 0, 0};  // cap the overshoot at 9x the nominal budget
        // attempts 0-4: full constraints, growing budget; 5-6: no feature constraints (tiny or
        // degenerate parts where the hard constraints are unsatisfiable)
        const FieldConstraints full_cons = job.cons;
        const MatrixXd comp_V0 = job.V;
        const MatrixXi comp_F0 = job.F;
        const std::vector<int> full_sharp = job.sharp_he;
        for (int attempt = 0; attempt < 8 && !good; ++attempt) {
            if (attempt < 5) {
                job.faces = budget * growth[attempt];
                if (job.faces == 0) continue;
                if (attempt > 0 && job.faces > (int)job.F.rows() / 2) continue;
            } else if (attempt == 7) {
                // last solver attempt: finer input tessellation (long edges relative to the
                // lattice make the integer stage's edge subdivision fail on some scans)
                WorkMesh jm;
                jm.V = job.V;
                jm.F = job.F;
                jm.A.resize(job.V.rows(), 0);
                refine_long_edges(jm, 0.2 * Lqf);
                job.V = jm.V;
                job.F = jm.F;
                job.faces = std::max(budget, 24);
                job.sharp_he.assign(3 * job.F.rows(), 0);
                job.cons.dir = MatrixXd::Zero(job.V.rows(), 3);
                job.cons.w = VectorXd::Zero(job.V.rows());
                job.cons.pos.assign(job.V.rows(), 0);
            } else {
                job.faces = std::max(budget, 24) * (attempt == 5 ? 1 : 4);
                job.sharp_he.assign(job.sharp_he.size(), 0);
                job.cons.w.setZero();
                std::fill(job.cons.pos.begin(), job.cons.pos.end(), 0);
            }
            auto te0 = Clock::now();
            try {
                qr = run_quadriflow(job);
            } catch (const std::exception& e) {
                log("component ", c, ": solver exception ", e.what());
                qr.ok = false;
                // an internal inconsistency comes from the hard constraints, not the resolution:
                // go straight to the unconstrained attempts
                if (attempt < 4) attempt = 4;
                if (job.F.rows() > 200000) attempt = 8;  // large scan component: go to the second extractor
            }
            t_extract += secs(te0, Clock::now());
            good = qr.ok && (int)qr.Q.size() >= std::min(4, budget);
            qa = 0;
            double cov_mean = 0, cov_p95 = 0;
            if (good) {
                for (auto& q : qr.Q) {
                    Vector3d a = qr.V.row(q[0]), b = qr.V.row(q[1]), cc = qr.V.row(q[2]), d = qr.V.row(q[3]);
                    qa += 0.5 * ((b - a).cross(cc - a).norm() + (cc - a).cross(d - a).norm());
                }
                // coverage: how far is the component surface from the extracted quads?
                // (thin shells legitimately collapse to one sheet, so area alone is not enough)
                coverage_distance(job.V, job.F, qr.V, qr.Q, cov_mean, cov_p95);
                good = qa < 1.6 * carea[c] && cov_mean < 0.25 * Lqf && cov_p95 < 1.0 * Lqf;
                if (!good && qa < 1.6 * carea[c]) keep_best(qr, cov_mean, cov_p95);
            }
            if (!good)
                log("component ", c, ": attempt ", attempt, " rejected (faces=", job.faces, " ok=", qr.ok, " quads=",
                    qr.Q.size(), " area ratio=", qa / carea[c], " cover mean/p95=", cov_mean, "/", cov_p95, ")");
            // thin parts: the IM extractor at the nominal budget keeps the face count on target,
            // growing the integer-grid budget overshoots it
            if (!good && attempt == 0 && qr.ok && try_im(std::max(budget, 16))) good = true;
        }
        // Best of both extractors on large, dense components (scans): the integer-grid result is
        // replaced by the Instant Meshes one only if it has clearly fewer singularities.
        if (good && dense_input && job.F.rows() > 50000) {
            QFResult keep = qr;
            int s_qf = count_singularities((int)qr.V.rows(), qr.Q);
            MatrixXd sv = job.V;
            MatrixXi sf = job.F;
            job.V = comp_V0;
            job.F = comp_F0;
            int im_before = R.components_im;
            if (try_im(budget)) {
                int s_im = count_singularities((int)qr.V.rows(), qr.Q);
                double r_qf = s_qf / std::max(1.0, (double)keep.Q.size()), r_im = s_im / std::max(1.0, (double)qr.Q.size());
                if (r_im < 0.85 * r_qf) {
                    log("component ", c, ": Instant Meshes result kept (", s_im, " vs ", s_qf, " singularities)");
                } else {
                    qr = keep;
                    R.components_im = im_before;
                    log("component ", c, ": integer-grid result kept (", s_qf, " vs ", s_im, " singularities)");
                }
            } else {
                qr = keep;
            }
            job.V = sv;
            job.F = sf;
        }
        // second extractor: Instant Meshes, at growing budgets (on the original component mesh)
        if (!good) {
            job.V = comp_V0;
            job.F = comp_F0;
        }
        for (int im_try = 0; im_try < 3 && !good; ++im_try) {
            int im_faces = std::max(budget, 16) * (im_try == 0 ? 1 : (im_try == 1 ? 4 : 16));
            if (im_try > 0 && im_faces > (int)job.F.rows() / 2) break;
            good = try_im(im_faces);
        }
        // nothing passed: rather than an exact but very dense fallback, accept the best
        // approximate extraction when it stays within a few target edge lengths (thin tubes /
        // plates thinner than one element: below the requested resolution in one dimension)
        if (!good && best_rej_p95 < 3.0 * Lqf && best_rej_mean < 0.6 * Lqf) {
            qr = best_rej;
            qr.ok = true;
            good = true;
            R.components_approx++;
            log("component ", c, ": accepted best approximate extraction (cover mean/p95=", best_rej_mean, "/", best_rej_p95,
                ", ", qr.Q.size(), " quads) - part thinner than the target element size");
        }
        int base = (int)outV.size();
        if (good) {
            for (int i = 0; i < qr.V.rows(); ++i) outV.push_back(qr.V.row(i));
            for (auto q : qr.Q) outQ.push_back({q[0] + base, q[1] + base, q[2] + base, q[3] + base});
            total_sing += qr.orientation_singularities;
            if (qr.fieldQ.rows() == (int)vlist.size())
                for (size_t i = 0; i < vlist.size(); ++i) R.field_Q.row(vlist[i]) = qr.fieldQ.row(i);
            R.components_out++;
        } else {
            MatrixXd OV;
            std::vector<std::array<int, 4>> OQ;
            // last resort on the *unrefined* faces of this component (3 quads per triangle)
            std::vector<int> pmap(prepV.rows(), -1);
            std::vector<std::array<int, 3>> pf;
            MatrixXd PVc;
            std::vector<int> pv;
            for (int f = 0; f < prepF.rows(); ++f) {
                if (vmap[prepF(f, 0)] < 0) continue;  // vertex of another component
                std::array<int, 3> t;
                for (int j = 0; j < 3; ++j) {
                    int v = prepF(f, j);
                    if (pmap[v] < 0) { pmap[v] = (int)pv.size(); pv.push_back(v); }
                    t[j] = pmap[v];
                }
                pf.push_back(t);
            }
            MatrixXd PV(pv.size(), 3);
            for (size_t i = 0; i < pv.size(); ++i) PV.row(i) = prepV.row(pv[i]);
            MatrixXi PF(pf.size(), 3);
            for (size_t i = 0; i < pf.size(); ++i) PF.row(i) << pf[i][0], pf[i][1], pf[i][2];
            // pair triangles into quads (best shape first), then split every face into quads:
            // conforming, exact geometry, much better shaped than splitting each triangle
            {
                const MatrixXd& BV = (PF.rows() > 0 && PF.rows() < job.F.rows()) ? PV : job.V;
                const MatrixXi& BF = (PF.rows() > 0 && PF.rows() < job.F.rows()) ? PF : job.F;
                // decimate first so the fallback still follows the face budget
                // (pairs -> 4 quads, so ~budget/2 triangles give ~budget quads)
                MatrixXd DV = BV;
                MatrixXi DF = BF;
                decimate_qem(DV, DF, std::max(8, budget / 2));
                QuadMesh fm;
                fm.V = DV;
                fm.Q.resize(DF.rows(), 4);
                for (int f = 0; f < DF.rows(); ++f) fm.Q.row(f) << DF(f, 0), DF(f, 1), DF(f, 2), -1;
                PostStats tmp;
                merge_triangle_pairs(fm, tmp, 0.2);
                split_to_pure_quads(fm);
                OV = fm.V;
                OQ.clear();
                for (int f = 0; f < fm.Q.rows(); ++f) OQ.push_back({fm.Q(f, 0), fm.Q(f, 1), fm.Q(f, 2), fm.Q(f, 3)});
            }
            for (int i = 0; i < OV.rows(); ++i) outV.push_back(OV.row(i));
            for (auto q : OQ) outQ.push_back({q[0] + base, q[1] + base, q[2] + base, q[3] + base});
            R.components_fallback++;
            log("component ", c, ": fallback tri->quad split (", fl.size(), " tris, budget ", budget, ")");
        }
    }
    log("solver: components=", nc, " below-resolution parts skipped=", skipped_small, " instant-meshes=", R.components_im, " approximate=", R.components_approx, " fallback=", R.components_fallback, " field singularities=", total_sing,
        " raw quads=", outQ.size());
    auto t3 = Clock::now();

    // ---------------------------------------------------------------- post-process
    QuadMesh qm;
    qm.V.resize(outV.size(), 3);
    for (size_t i = 0; i < outV.size(); ++i) qm.V.row(i) = outV[i].transpose();
    qm.Q.resize(outQ.size(), 4);
    for (size_t i = 0; i < outQ.size(); ++i)
        for (int j = 0; j < 4; ++j) qm.Q(i, j) = outQ[i][j];
    PostStats pst;
    auto sing_now = [&]() {
        std::vector<std::array<int, 4>> qq(qm.Q.rows());
        for (int i = 0; i < qm.Q.rows(); ++i) qq[i] = {qm.Q(i, 0), qm.Q(i, 1), qm.Q(i, 2), qm.Q(i, 3)};
        return count_singularities((int)qm.V.rows(), qq);
    };
    log("singularities: raw ", sing_now());
    cleanup(qm, pst);
    log("singularities: after cleanup ", sing_now());
    merge_triangle_pairs(qm, pst);
    cleanup(qm, pst);
    log("singularities: after triangle merge ", sing_now());
    if (adaptive) {
        MatrixXd SV = wm.V;
        MatrixXi SF = wm.F;
        MatrixXd SA = scale;
        auto target = [&](const MatrixXd& P) -> VectorXd {
            return (transfer_vertex_attributes(SV, SF, SA, P).col(0) * Lqf).eval();
        };
        int nref = refine_to_size(qm, target, 3);
        merge_triangle_pairs(qm, pst);
        cleanup(qm, pst);
        log("density refinement: ", nref, " faces split -> ", qm.Q.rows(), " faces");
    }
    if (p.pure_quad) {
        split_to_pure_quads(qm);
        log("pure-quad split -> ", qm.Q.rows(), " quads");
    }
    SurfaceRef ref(refV, refF, refFE, refBE, ref_corners, plane_verts, sym ? p.symmetry_axis : -1, sym_off);
    if (!std::getenv("QRM_NOPOST")) relax_and_project(qm, ref, L, p.smooth_iterations, pst);
    log("feature paths: ", pst.feature_paths, "/", pst.feature_chains, " chains, length ", pst.feature_len_traced, "/",
        pst.feature_len_total);
    // ---- budgeted feature-coverage stage ----------------------------------------------------
    // Every coverage operation (edge-loop insertion, local refinement, curved-crease splits) is
    // applied tentatively and kept only if the mesh stays inside two budgets: triangles (quad
    // ratio) and irregular vertices (singularities). Both are topological, so checks are cheap.
    if (!std::getenv("QRM_NOPOST") && (p.feature_loops || p.feature_refine_rounds > 0 || p.curved_feature_splits || p.crease_cuts)) {
        const int T = std::max(1, p.target_faces);
        auto ntris = [&]() { int t = 0; for (int i = 0; i < qm.Q.rows(); ++i) t += qm.Q(i, 3) < 0; return t; };
        auto nsing = [&]() {
            std::vector<std::array<int, 4>> qq(qm.Q.rows());
            for (int i = 0; i < qm.Q.rows(); ++i) qq[i] = {qm.Q(i, 0), qm.Q(i, 1), qm.Q(i, 2), qm.Q(i, 3)};
            return count_singularities((int)qm.V.rows(), qq);
        };
        const int tri_cap = ntris() + (int)(p.triangle_budget * T);
        const int sing_cap = nsing() + (int)(p.singularity_budget * T / 1000.0);
        int accepted = 0, rejected = 0;
        auto within = [&]() { return ntris() <= tri_cap && nsing() <= sing_cap; };
        auto chain_len = [](const std::vector<Vector3d>& c) {
            double l = 0;
            for (size_t k = 0; k + 1 < c.size(); ++k) l += (c[k + 1] - c[k]).norm();
            return l;
        };
        // 1. edge-loop insertion for creases without a path (longest first)
        if (p.feature_loops && !pst.untraced.empty()) {
            auto chains = pst.untraced;
            std::sort(chains.begin(), chains.end(), [&](const std::vector<Vector3d>& x, const std::vector<Vector3d>& y) {
                return chain_len(x) > chain_len(y);
            });
            int budget_add = (int)(p.feature_loop_budget * T), added = 0;
            for (auto& ch : chains) {
                if (p.feature_refine_rounds > 0 && p.feature_refine_max_len > 0 && chain_len(ch) <= p.feature_refine_max_len * L)
                    continue;  // short creases are left to local refinement
                QuadMesh backup = qm;
                int nadd = insert_feature_loops(qm, {ch}, L, budget_add - added, p.feature_loop_partial ? 8 : -1, -1);
                if (!nadd) continue;
                merge_triangle_pairs(qm, pst);
                cleanup(qm, pst);
                if (within()) { added += nadd; accepted++; }
                else { qm = backup; rejected++; }
            }
            relax_and_project(qm, ref, L, p.smooth_iterations, pst);
            log("feature loops: +", added, " faces -> ", qm.Q.rows(), "; paths ", pst.feature_paths, "/", pst.feature_chains,
                ", length ", pst.feature_len_traced, "/", pst.feature_len_total);
        }
        // 2. local refinement around short creases without a path (shortest first)
        for (int round = 0; round < p.feature_refine_rounds && !pst.untraced.empty() && !p.pure_quad; ++round) {
            auto chains = pst.untraced;
            std::sort(chains.begin(), chains.end(), [&](const std::vector<Vector3d>& x, const std::vector<Vector3d>& y) {
                return chain_len(x) < chain_len(y);
            });
            const double Lr = L / (1 << round);
            int nref_total = 0;
            for (auto& ch : chains) {
                if (p.feature_refine_max_len > 0 && chain_len(ch) > p.feature_refine_max_len * L) continue;
                MatrixXd SV(2 * (ch.size() - 1), 3);
                MatrixXi SE(ch.size() - 1, 2);
                for (size_t k = 0; k + 1 < ch.size(); ++k) {
                    SV.row(2 * k) = ch[k].transpose();
                    SV.row(2 * k + 1) = ch[k + 1].transpose();
                    SE.row(k) << (int)(2 * k), (int)(2 * k + 1);
                }
                if (SE.rows() == 0) continue;
                igl::AABB<MatrixXd, 3> stree;
                stree.init(SV, SE);
                auto target = [&](const MatrixXd& P) -> VectorXd {
                    VectorXd D, t(P.rows());
                    VectorXi I;
                    MatrixXd C;
                    stree.squared_distance(SV, SE, P, D, I, C);
                    for (int i = 0; i < P.rows(); ++i) t[i] = D[i] < (1.2 * Lr) * (1.2 * Lr) ? 0.5 * Lr : 1e30;
                    return t;
                };
                QuadMesh backup = qm;
                int nref = refine_to_size(qm, target, 1);
                if (!nref) continue;
                merge_triangle_pairs(qm, pst);
                cleanup(qm, pst);
                if (within()) { nref_total += nref; accepted++; }
                else { qm = backup; rejected++; }
            }
            if (!nref_total) break;
            relax_and_project(qm, ref, L, p.smooth_iterations, pst);
            log("feature refinement ", round + 1, ": ", nref_total, " faces split -> ", qm.Q.rows(), " faces; paths ",
                pst.feature_paths, "/", pst.feature_chains, ", length ", pst.feature_len_traced, "/", pst.feature_len_total);
        }
        // 3. curved creases: split path edges whose chord leaves the crease (budget halved on overrun)
        if (p.curved_feature_splits && !pst.paths.empty()) {
            int tri_left = std::max(0, tri_cap - ntris());
            for (int attempt = 0; attempt < 5 && tri_left >= 2; ++attempt) {
                QuadMesh backup = qm;
                PostStats bst = pst;
                int ns = split_curved_paths(qm, pst, L, 0.09, 23.0, 2, tri_left);
                cleanup(qm, pst);
                if (within()) { log("curved feature splits: ", ns, " -> ", qm.Q.rows(), " faces"); accepted++; break; }
                qm = backup;
                pst = bst;
                tri_left /= 2;
                rejected++;
            }
        }
        // 4. crease cuts: embed the remaining uncovered crease pieces as mesh edges (longest first)
        if (p.crease_cuts) {
            const auto all_chains = ref.feature_chains(0.0);
            for (int round = 0; round < p.crease_cut_rounds; ++round) {
                double cov = 0, tot = 0;
                auto runs = uncovered_crease_runs(qm, all_chains, L, 0.15, 0.1, 0.1, 25.0, &cov, &tot);
                std::sort(runs.begin(), runs.end(), [&](const std::vector<Vector3d>& x, const std::vector<Vector3d>& y) {
                    return chain_len(x) > chain_len(y);
                });
                int ok = 0, bad = 0, rej = 0;
                std::map<int, int> why;
                for (auto& r : runs) {
                    QuadMesh backup = qm;
                    std::vector<Vector3d> cv;
                    int rc = embed_crease_run(qm, r, L, 0.05, &cv, 0.2);
                    if (rc == -8) { cv.clear(); rc = embed_crease_run(qm, r, L, 0.05, &cv, 0.34); }
                    if (rc < 0) { bad++; why[rc]++; continue; }
                    if (std::getenv("QRM_DEBUG_CUT") && std::string(std::getenv("QRM_DEBUG_CUT")) == "2") {
                        double c0 = 0, t0 = 0, c1 = 0, t1 = 0;
                        uncovered_crease_runs(backup, {r}, L, 1e9, 0, 0.1, 25.0, &c0, &t0);
                        uncovered_crease_runs(qm, {r}, L, 1e9, 0, 0.1, 25.0, &c1, &t1);
                        log("  cut: run ", chain_len(r) / L, " L, faces ", backup.Q.rows(), " -> ", qm.Q.rows(), ", run coverage ",
                            c0 / std::max(t0, 1e-30), " -> ", c1 / std::max(t1, 1e-30));
                    }
                    cleanup(qm, pst);
                    if (within()) { ok++; accepted++; pst.pinned.insert(pst.pinned.end(), cv.begin(), cv.end()); }
                    else { qm = backup; rej++; rejected++; }
                }
                if (std::getenv("QRM_DEBUG_CUT"))
                    for (auto& w : why) log("  cut failure code ", w.first, ": ", w.second);
                if (std::getenv("QRM_DEBUG_CUT")) {
                    double c1 = 0, t1 = 0;
                    uncovered_crease_runs(qm, all_chains, L, 1e9, 0, 0.1, 25.0, &c1, &t1);
                    log("  coverage right after cuts (before relaxation): ", c1 / std::max(t1, 1e-30));
                }
                if (ok) relax_and_project(qm, ref, L, p.smooth_iterations, pst);
                double cov2 = 0, tot2 = 0;
                uncovered_crease_runs(qm, all_chains, L, 1e9, 0, 0.1, 25.0, &cov2, &tot2);
                log("crease cuts ", round + 1, ": ", runs.size(), " uncovered runs, ", ok, " cut, ", bad, " not embeddable, ", rej,
                    " over budget; coverage ", cov / std::max(tot, 1e-30), " -> ", cov2 / std::max(tot2, 1e-30), ", ", qm.Q.rows(), " faces");
                if (!ok) break;
            }
        }
        log("coverage stage: ", accepted, " operations kept, ", rejected, " rejected by budget (triangles ", ntris(), "/",
            tri_cap, ", singularities ", nsing(), "/", sing_cap, ")");
    }
    if (sym) {
        mirror_and_weld(qm, p.symmetry_axis, sym_off, 1e-9);
        cleanup(qm, pst);
    }
    log("post: degenerate=", pst.degenerate_removed, " duplicates=", pst.duplicates_removed, " nonmanifold=",
        pst.nonmanifold_removed, " tris_merged=", pst.tris_merged, " tris_left=", pst.tris_remaining,
        " feature_v=", pst.feature_vertices, " corner_v=", pst.corner_vertices, " feature_paths=", pst.feature_paths, " rejected_moves=",
        pst.smoothing_rejected);

    // ---------------------------------------------------------------- back to input space + UVs
    qm.V = (qm.V / s).rowwise() + center.transpose();
    if (p.transfer_uv && input.has_uv()) transfer_uvs(input, qm);
    R.field_V = (R.field_V / s).rowwise() + center.transpose();
    R.mesh = qm;
    auto t4 = Clock::now();
    R.timing.preprocess = secs(t0, t1);
    R.timing.features = secs(t1, t2);
    R.timing.field = secs(t2, t3) - t_extract;
    R.timing.extraction = t_extract;
    R.timing.postprocess = secs(t3, t4);
    R.timing.total = secs(t0, t4);
    log("done: ", qm.V.rows(), " vertices, ", qm.Q.rows(), " faces in ", R.timing.total, " s");
    return R;
}

}  // namespace qrm

namespace qrm {

// Face-count control: loop insertion / feature refinement add faces after the solve, so the
// solve budget is re-scaled once (twice at most) to land on the requested face count.
static Result quadremesh_matched(const TriMesh& input, const Params& p) {
    Result r = quadremesh_once(input, p);
    if (p.target_edge_length > 0 || p.target_faces <= 0 || !p.match_face_count) return r;
    Params q = p;
    double solve_target = p.target_faces;
    for (int it = 0; it < 2; ++it) {
        double ratio = (double)r.mesh.Q.rows() / p.target_faces;
        if (ratio < 1.10) break;
        solve_target = std::max(0.25 * p.target_faces, solve_target / ratio);
        q.target_faces = (int)std::lround(solve_target);
        q.feature_loop_budget = p.feature_loop_budget * p.target_faces / std::max(1.0, solve_target);
        Result r2 = quadremesh_once(input, q);
        r2.log.insert(r2.log.begin(), "face-count control: solve budget " + std::to_string(q.target_faces) + " (output was " +
                                          std::to_string(r.mesh.Q.rows()) + " faces)");
        r = std::move(r2);
    }
    return r;
}


// value of "key" (optionally the first "sub" key after it) in the evaluator's JSON report
static double json_num(const std::string& js, const std::string& key, const std::string& sub = "") {
    size_t i = js.find("\"" + key + "\"");
    if (i == std::string::npos) return std::nan("");
    if (!sub.empty()) {
        i = js.find("\"" + sub + "\"", i);
        if (i == std::string::npos) return std::nan("");
        i += sub.size() + 2;
    } else {
        i += key.size() + 2;
    }
    i = js.find(':', i);
    return std::strtod(js.c_str() + i + 1, nullptr);
}

// Coverage search: the field solve is sensitive to where the element size is reduced near
// creases, so several settings are solved and each result is scored with the evaluator itself
// (the same code as quadeval: feature coverage, quad ratio, scaled Jacobian, distance,
// singularities). The result with the highest feature coverage that keeps quad ratio >= 98 %,
// mean SJ >= 0.8, mean distance < 0.5 % and at most `singularity_budget` more singularities per
// 1k faces than the first candidate is returned. Runtime grows with the number of candidates.
Result quadremesh(const TriMesh& input, const Params& p) {
    if (!p.coverage_search) return quadremesh_matched(input, p);
    Params base = p;
    base.coverage_search = false;
    std::vector<Params> cands;
    cands.push_back(base);
    for (double w : {1.5, 2.5}) {
        Params q = base;
        q.feature_density = 8.0;
        q.feature_density_width = w;
        cands.push_back(q);
    }
    {
        Params q = base;
        q.feature_density = 8.0;
        q.feature_density_width = 2.0;
        q.seed = base.seed + 1;
        cands.push_back(q);
    }
    EvalParams ep;
    ep.feature_angle = p.feature_angle > 0 ? p.feature_angle : 35.0;
    ep.min_feature_length = 3.0;
    ep.target_faces = p.target_faces;
    ep.samples = 50000;
    Result best;
    double best_cov = -1, s_cap = 1e30;
    std::vector<std::string> notes;
    for (size_t k = 0; k < cands.size(); ++k) {
        Result r;
        try {
            r = quadremesh_matched(input, cands[k]);
        } catch (const std::exception& e) {
            notes.push_back("coverage search: candidate " + std::to_string(k) + " failed: " + e.what());
            continue;
        }
        std::string js = evaluate(input, r.mesh, ep);
        double cov = json_num(js, "feature_coverage"), quad = json_num(js, "quad_ratio"),
               sj = json_num(js, "scaled_jacobian", "mean"), dist = json_num(js, "distance", "mean_pct"),
               s1k = json_num(js, "singularities_per_1k_faces");
        if (k == 0) s_cap = s1k + p.singularity_budget;
        bool ok = quad >= 0.98 && sj >= 0.8 && dist < 0.5 && s1k <= s_cap;
        char buf[256];
        std::snprintf(buf, sizeof buf, "coverage search: candidate %zu (feature density %.0f, width %.1f, seed %u): coverage %.3f quad %.3f SJ %.3f dist %.3f%% sing/1k %.1f%s",
                      k, cands[k].feature_density, cands[k].feature_density_width, cands[k].seed, cov, quad, sj, dist, s1k, ok ? "" : " (fails a criterion)");
        notes.push_back(buf);
        if (p.verbose) std::fprintf(stderr, "[quadremesh] %s\n", buf);
        if ((ok && cov > best_cov) || best_cov < 0) {
            if (!ok && best_cov >= 0) continue;
            best = std::move(r);
            best_cov = ok ? cov : -0.5;
        }
    }
    best.log.insert(best.log.begin(), notes.begin(), notes.end());
    return best;
}

}  // namespace qrm
