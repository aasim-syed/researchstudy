#include <chrono>
#include <cstdlib>
#include <cmath>
#include <map>
#include <sstream>

#include <igl/AABB.h>

#include <algorithm>

#include "qflow_driver.h"
#include "quadremesh/features.h"
#include "quadremesh/field.h"
#include "quadremesh/geometry.h"
#include "quadremesh/postprocess.h"
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
}  // namespace

Result quadremesh(const TriMesh& input, const Params& p) {
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
    FeatureSet fs = detect_features(wm.V, wm.F, p.feature_angle, p.min_feature_length * Lqf, 0.3 * Lqf);
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
            else if (fs.he_sharp[topo.ehe[e][0]]) fe.push_back(topo.ev[e]);
        }
        refFE.resize(fe.size(), 2);
        for (size_t i = 0; i < fe.size(); ++i) refFE.row(i) << fe[i].first, fe[i].second;
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
    {
        double avg = 0;
        EdgeTopology topo;
        topo.build(wm.F);
        for (auto& e : topo.ev) avg += (wm.V.row(e.first) - wm.V.row(e.second)).norm();
        avg /= std::max<size_t>(1, topo.ev.size());
        double target_len = std::min(Lqf / 2, avg * 2);
        refine_long_edges(wm, target_len);
        log("refine: max edge ", target_len, " -> V=", wm.V.rows(), " F=", wm.F.rows());
    }
    auto t2 = Clock::now();

    // ---------------------------------------------------------------- field design inputs
    MatrixXd N = vertex_normals(wm.V, wm.F);
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
    for (int c = 0; c < nc; ++c) {
        const auto& fl = cfaces[c];
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
        if (const char* dump = std::getenv("QRM_DUMP_COMPONENTS")) {
            TriMesh dm;
            dm.V = job.V;
            dm.F = job.F;
            save_tri_obj(std::string(dump) + "/comp_" + std::to_string(c) + ".obj", dm);
        }
        // Solve; if the extracted quads do not cover the component (thin parts, narrow strips
        // between features), retry at a finer field resolution before falling back.
        QFResult qr;
        double qa = 0;
        bool good = false;
        // thin tubes / plates need several quads across: grow the budget geometrically
        static const int growth[5] = {1, 3, 9, 27, 81};
        for (int attempt = 0; attempt < 5 && !good; ++attempt) {
            job.faces = budget * growth[attempt];
            if (attempt > 0 && job.faces > (int)job.F.rows() / 2) break;
            auto te0 = Clock::now();
            try {
                qr = run_quadriflow(job);
            } catch (const std::exception& e) {
                log("component ", c, ": solver exception ", e.what());
                qr.ok = false;
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
                good = qa < 1.6 * carea[c] && cov_mean < 0.15 * Lqf && cov_p95 < 0.6 * Lqf;
            }
            if (!good)
                log("component ", c, ": attempt ", attempt, " rejected (faces=", job.faces, " ok=", qr.ok, " quads=",
                    qr.Q.size(), " area ratio=", qa / carea[c], " cover mean/p95=", cov_mean, "/", cov_p95, ")");
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
            tri_to_quads(job.V, job.F, OV, OQ);
            for (int i = 0; i < OV.rows(); ++i) outV.push_back(OV.row(i));
            for (auto q : OQ) outQ.push_back({q[0] + base, q[1] + base, q[2] + base, q[3] + base});
            R.components_fallback++;
            log("component ", c, ": fallback tri->quad split (", fl.size(), " tris, budget ", budget, ")");
        }
    }
    log("solver: components=", nc, " fallback=", R.components_fallback, " field singularities=", total_sing,
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
    cleanup(qm, pst);
    merge_triangle_pairs(qm, pst);
    cleanup(qm, pst);
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
    if (sym) {
        mirror_and_weld(qm, p.symmetry_axis, sym_off, 1e-9);
        cleanup(qm, pst);
    }
    log("post: degenerate=", pst.degenerate_removed, " duplicates=", pst.duplicates_removed, " nonmanifold=",
        pst.nonmanifold_removed, " tris_merged=", pst.tris_merged, " tris_left=", pst.tris_remaining,
        " feature_v=", pst.feature_vertices, " corner_v=", pst.corner_vertices, " rejected_moves=",
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
