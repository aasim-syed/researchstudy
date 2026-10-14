#include "quadremesh/postprocess.h"

#include <igl/AABB.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>

#include "quadremesh/geometry.h"
#include "quadremesh/metrics.h"

namespace qrm {

struct SurfaceRef::Impl {
    MatrixXd V;
    MatrixXi F, FE, BE, CE;
    MatrixXd N;  // face normals
    igl::AABB<MatrixXd, 3> tf, te, tb, tc;
    std::vector<char> be_plane;
    int axis = -1;
    double offset = 0;
};

SurfaceRef::SurfaceRef(const MatrixXd& V, const MatrixXi& F, const MatrixXi& FE, const MatrixXi& BE,
                       const std::vector<int>& corners, const std::vector<int>& plane_vertices, int sym_axis,
                       double sym_offset)
    : impl(std::make_shared<Impl>()) {
    impl->V = V;
    impl->F = F;
    impl->FE = FE;
    impl->BE = BE;
    impl->axis = sym_axis;
    impl->offset = sym_offset;
    impl->N.resize(F.rows(), 3);
    for (int f = 0; f < F.rows(); ++f) impl->N.row(f) = face_normal(V, F, f).transpose();
    impl->tf.init(impl->V, impl->F);
    if (FE.rows()) impl->te.init(impl->V, impl->FE);
    if (BE.rows()) {
        impl->tb.init(impl->V, impl->BE);
        std::vector<char> pv(V.rows(), 0);
        for (int v : plane_vertices) pv[v] = 1;
        impl->be_plane.resize(BE.rows());
        for (int i = 0; i < BE.rows(); ++i) impl->be_plane[i] = pv[BE(i, 0)] && pv[BE(i, 1)];
    }
    impl->CE.resize(corners.size(), 1);
    for (size_t i = 0; i < corners.size(); ++i) impl->CE(i, 0) = corners[i];
    if (!corners.empty()) impl->tc.init(impl->V, impl->CE);
}

Vector3d SurfaceRef::project_surface(const Vector3d& p, Vector3d* normal) const {
    int i;
    Eigen::RowVector3d c;
    impl->tf.squared_distance(impl->V, impl->F, p.transpose(), i, c);
    if (normal) *normal = impl->N.row(i).transpose();
    return c.transpose();
}
double SurfaceRef::dist2_feature(const Vector3d& p, Vector3d* c) const {
    if (!impl->FE.rows()) return 1e300;
    int i;
    Eigen::RowVector3d cc;
    double d = impl->te.squared_distance(impl->V, impl->FE, p.transpose(), i, cc);
    if (c) *c = cc.transpose();
    return d;
}
double SurfaceRef::dist2_boundary(const Vector3d& p, Vector3d* c, bool* on_plane) const {
    if (!impl->BE.rows()) return 1e300;
    int i;
    Eigen::RowVector3d cc;
    double d = impl->tb.squared_distance(impl->V, impl->BE, p.transpose(), i, cc);
    if (c) *c = cc.transpose();
    if (on_plane) *on_plane = impl->be_plane[i];
    return d;
}
double SurfaceRef::dist2_corner(const Vector3d& p, Vector3d* c) const {
    if (!impl->CE.rows()) return 1e300;
    int i;
    Eigen::RowVector3d cc;
    double d = impl->tc.squared_distance(impl->V, impl->CE, p.transpose(), i, cc);
    if (c) *c = cc.transpose();
    return d;
}
MatrixXd SurfaceRef::corner_points() const {
    MatrixXd P(impl->CE.rows(), 3);
    for (int i = 0; i < impl->CE.rows(); ++i) P.row(i) = impl->V.row(impl->CE(i, 0));
    return P;
}
bool SurfaceRef::has_features() const { return impl->FE.rows() > 0; }
bool SurfaceRef::has_boundary() const { return impl->BE.rows() > 0; }
int SurfaceRef::sym_axis() const { return impl->axis; }
double SurfaceRef::sym_offset() const { return impl->offset; }

// ------------------------------------------------------------------------------------------
namespace {
using Face = std::array<int, 4>;  // [3] == -1 -> triangle

std::vector<Face> to_faces(const QuadMesh& m) {
    std::vector<Face> f(m.Q.rows());
    for (int i = 0; i < m.Q.rows(); ++i) f[i] = {m.Q(i, 0), m.Q(i, 1), m.Q(i, 2), m.Q(i, 3)};
    return f;
}
void from_faces(QuadMesh& m, const std::vector<Face>& f) {
    m.Q.resize(f.size(), 4);
    for (size_t i = 0; i < f.size(); ++i)
        for (int j = 0; j < 4; ++j) m.Q(i, j) = f[i][j];
    m.UV.resize(0, 2);
}
inline int fsize(const Face& f) { return f[3] < 0 ? 3 : 4; }

void remove_unreferenced(QuadMesh& m) {
    std::vector<int> map(m.V.rows(), -1);
    int n = 0;
    std::vector<char> used(m.V.rows(), 0);
    for (int i = 0; i < m.Q.rows(); ++i)
        for (int j = 0; j < 4; ++j)
            if (m.Q(i, j) >= 0) used[m.Q(i, j)] = 1;
    for (int v = 0; v < m.V.rows(); ++v)
        if (used[v]) map[v] = n++;
    MatrixXd V2(n, 3);
    for (int v = 0; v < m.V.rows(); ++v)
        if (used[v]) V2.row(map[v]) = m.V.row(v);
    for (int i = 0; i < m.Q.rows(); ++i)
        for (int j = 0; j < 4; ++j)
            if (m.Q(i, j) >= 0) m.Q(i, j) = map[m.Q(i, j)];
    m.V = V2;
}
}  // namespace

void cleanup(QuadMesh& m, PostStats& st) {
    std::vector<Face> in = to_faces(m), out;
    // 1. degenerate: remove repeated consecutive vertices
    for (auto f : in) {
        std::vector<int> v;
        for (int j = 0; j < fsize(f); ++j) {
            int x = f[j];
            if (x < 0) continue;
            if (std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
        }
        if (v.size() < 3) { st.degenerate_removed++; continue; }
        Face g = {v[0], v[1], v[2], v.size() == 4 ? v[3] : -1};
        if (v.size() != (size_t)fsize(f)) st.degenerate_removed++;
        // zero-area check
        double sj = scaled_jacobian(m.V, g[0], g[1], g[2], g[3]);
        if (sj == 0 && g[3] < 0) { st.degenerate_removed++; continue; }
        out.push_back(g);
    }
    // 2. duplicates (same vertex set)
    {
        std::map<std::vector<int>, int> seen;
        std::vector<Face> o2;
        for (auto& f : out) {
            std::vector<int> k(f.begin(), f.begin() + fsize(f));
            std::sort(k.begin(), k.end());
            if (seen.count(k)) { st.duplicates_removed++; continue; }
            seen[k] = 1;
            o2.push_back(f);
        }
        out.swap(o2);
    }
    // 3. non-manifold edges: keep the two best-shaped faces per edge
    for (int round = 0; round < 3; ++round) {
        std::map<std::pair<int, int>, std::vector<int>> ef;
        for (size_t i = 0; i < out.size(); ++i)
            for (int j = 0; j < fsize(out[i]); ++j) {
                int a = out[i][j], b = out[i][(j + 1) % fsize(out[i])];
                ef[{std::min(a, b), std::max(a, b)}].push_back((int)i);
            }
        std::vector<char> drop(out.size(), 0);
        bool any = false;
        for (auto& kv : ef) {
            if (kv.second.size() <= 2) continue;
            std::vector<std::pair<double, int>> q;
            for (int f : kv.second) {
                if (drop[f]) continue;
                q.push_back({-scaled_jacobian(m.V, out[f][0], out[f][1], out[f][2], out[f][3]), f});
            }
            std::sort(q.begin(), q.end());
            for (size_t k = 2; k < q.size(); ++k) { drop[q[k].second] = 1; any = true; st.nonmanifold_removed++; }
        }
        if (!any) break;
        std::vector<Face> o2;
        for (size_t i = 0; i < out.size(); ++i)
            if (!drop[i]) o2.push_back(out[i]);
        out.swap(o2);
    }
    // 4. orientation consistency (BFS over shared edges)
    {
        std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> ef;  // edge -> (face, dir)
        for (size_t i = 0; i < out.size(); ++i)
            for (int j = 0; j < fsize(out[i]); ++j) {
                int a = out[i][j], b = out[i][(j + 1) % fsize(out[i])];
                ef[{std::min(a, b), std::max(a, b)}].push_back({(int)i, a < b ? 1 : -1});
            }
        std::vector<std::vector<std::pair<int, int>>> adj(out.size());  // (nbr, same_dir)
        for (auto& kv : ef)
            if (kv.second.size() == 2) {
                int same = kv.second[0].second == kv.second[1].second;
                adj[kv.second[0].first].push_back({kv.second[1].first, same});
                adj[kv.second[1].first].push_back({kv.second[0].first, same});
            }
        std::vector<int> flip(out.size(), -1);
        for (size_t s = 0; s < out.size(); ++s) {
            if (flip[s] >= 0) continue;
            flip[s] = 0;
            std::vector<int> stack = {(int)s};
            while (!stack.empty()) {
                int f = stack.back();
                stack.pop_back();
                for (auto [g, same] : adj[f])
                    if (flip[g] < 0) { flip[g] = flip[f] ^ same; stack.push_back(g); }
            }
        }
        for (size_t i = 0; i < out.size(); ++i)
            if (flip[i] == 1) {
                if (fsize(out[i]) == 4) std::swap(out[i][1], out[i][3]);
                else std::swap(out[i][1], out[i][2]);
            }
    }
    from_faces(m, out);
    remove_unreferenced(m);
    st.tris_remaining = 0;
    for (auto& f : out) st.tris_remaining += f[3] < 0;
}

void merge_triangle_pairs(QuadMesh& m, PostStats& st, double min_sj) {
    std::vector<Face> f = to_faces(m);
    std::map<std::pair<int, int>, std::vector<int>> ef;
    for (size_t i = 0; i < f.size(); ++i)
        if (f[i][3] < 0)
            for (int j = 0; j < 3; ++j) {
                int a = f[i][j], b = f[i][(j + 1) % 3];
                ef[{std::min(a, b), std::max(a, b)}].push_back((int)i);
            }
    std::vector<std::tuple<double, int, int, Face>> cand;
    for (auto& kv : ef) {
        if (kv.second.size() != 2) continue;
        int t0 = kv.second[0], t1 = kv.second[1];
        int a = kv.first.first, b = kv.first.second;
        auto other = [&](int t) {
            for (int j = 0; j < 3; ++j)
                if (f[t][j] != a && f[t][j] != b) return j;
            return -1;
        };
        int j0 = other(t0), j1 = other(t1);
        if (j0 < 0 || j1 < 0) continue;
        // quad: opposite0, next..., following t0 orientation
        int c = f[t0][j0];
        int n1 = f[t0][(j0 + 1) % 3], n2 = f[t0][(j0 + 2) % 3];
        Face q = {c, n1, f[t1][j1], n2};
        double s = scaled_jacobian(m.V, q[0], q[1], q[2], q[3]);
        if (s >= min_sj) cand.push_back({-s, t0, t1, q});
    }
    std::sort(cand.begin(), cand.end(), [](auto& x, auto& y) { return std::get<0>(x) < std::get<0>(y) || (std::get<0>(x) == std::get<0>(y) && std::get<1>(x) < std::get<1>(y)); });
    std::vector<char> used(f.size(), 0);
    std::vector<Face> out;
    for (auto& c : cand) {
        int t0 = std::get<1>(c), t1 = std::get<2>(c);
        if (used[t0] || used[t1]) continue;
        used[t0] = used[t1] = 1;
        out.push_back(std::get<3>(c));
        st.tris_merged++;
    }
    for (size_t i = 0; i < f.size(); ++i)
        if (!used[i]) out.push_back(f[i]);
    from_faces(m, out);
    st.tris_remaining = 0;
    for (auto& x : out) st.tris_remaining += x[3] < 0;
}

void split_to_pure_quads(QuadMesh& m) {
    std::vector<Face> f = to_faces(m);
    std::map<std::pair<int, int>, int> mid;
    std::vector<Vector3d> nv;
    int n0 = (int)m.V.rows();
    auto midpoint = [&](int a, int b) {
        auto k = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = mid.find(k);
        if (it != mid.end()) return it->second;
        int id = n0 + (int)nv.size();
        nv.push_back(0.5 * (m.V.row(a) + m.V.row(b)).transpose());
        mid[k] = id;
        return id;
    };
    std::vector<Face> out;
    for (auto& q : f) {
        int n = fsize(q);
        Vector3d c = Vector3d::Zero();
        for (int j = 0; j < n; ++j) c += m.V.row(q[j]).transpose();
        c /= n;
        int ci = n0 + (int)nv.size();
        nv.push_back(c);
        for (int j = 0; j < n; ++j) {
            int a = q[j], mp = midpoint(q[j], q[(j + 1) % n]), mm = midpoint(q[(j + n - 1) % n], q[j]);
            out.push_back({a, mp, ci, mm});
        }
    }
    MatrixXd V2(n0 + nv.size(), 3);
    V2.topRows(n0) = m.V;
    for (size_t i = 0; i < nv.size(); ++i) V2.row(n0 + i) = nv[i].transpose();
    m.V = V2;
    from_faces(m, out);
}

int refine_to_size(QuadMesh& m, const std::function<VectorXd(const MatrixXd&)>& target, int max_levels) {
    int total = 0;
    for (int level = 0; level < max_levels; ++level) {
        std::vector<Face> f = to_faces(m);
        const int nf = (int)f.size();
        MatrixXd C(nf, 3);
        VectorXd h(nf);
        for (int i = 0; i < nf; ++i) {
            int n = fsize(f[i]);
            Vector3d c = Vector3d::Zero();
            for (int k = 0; k < n; ++k) c += m.V.row(f[i][k]).transpose();
            C.row(i) = (c / n).transpose();
            double a = 0;
            for (int k = 1; k + 1 < n; ++k) {
                Vector3d p0 = m.V.row(f[i][0]), p1 = m.V.row(f[i][k]), p2 = m.V.row(f[i][k + 1]);
                a += 0.5 * (p1 - p0).cross(p2 - p0).norm();
            }
            h[i] = std::sqrt(std::max(a, 0.0));
        }
        VectorXd t = target(C);
        std::vector<char> mark(nf, 0);
        int nmark = 0;
        for (int i = 0; i < nf; ++i)
            if (h[i] > 1.45 * t[i]) mark[i] = 1, nmark++;
        if (!nmark) break;
        // edge -> faces
        std::map<std::pair<int, int>, std::vector<int>> ef;
        for (int i = 0; i < nf; ++i)
            for (int k = 0; k < fsize(f[i]); ++k) {
                int a = f[i][k], b = f[i][(k + 1) % fsize(f[i])];
                ef[{std::min(a, b), std::max(a, b)}].push_back(i);
            }
        auto edge_split = [&](int i, int k) {  // is edge k of face i split by a marked neighbour?
            int a = f[i][k], b = f[i][(k + 1) % fsize(f[i])];
            for (int g : ef[{std::min(a, b), std::max(a, b)}])
                if (g != i && mark[g]) return true;
            return false;
        };
        // closure: only the three hanging-node templates are allowed on unmarked faces
        for (bool changed = true; changed;) {
            changed = false;
            for (int i = 0; i < nf; ++i) {
                if (mark[i]) continue;
                int ns = 0;
                for (int k = 0; k < fsize(f[i]); ++k) ns += edge_split(i, k);
                if ((fsize(f[i]) == 4 && ns >= 3) || (fsize(f[i]) == 3 && ns >= 1)) { mark[i] = 1; changed = true; }
            }
        }
        // split
        std::map<std::pair<int, int>, int> mid;
        std::vector<Vector3d> nv;
        const int n0 = (int)m.V.rows();
        auto midpoint = [&](int a, int b) {
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            int id = n0 + (int)nv.size();
            nv.push_back(0.5 * (m.V.row(a) + m.V.row(b)).transpose());
            mid[key] = id;
            return id;
        };
        std::vector<Face> out;
        for (int i = 0; i < nf; ++i) {
            if (!mark[i]) continue;
            int n = fsize(f[i]);
            int ci = n0 + (int)nv.size();
            nv.push_back(C.row(i).transpose());
            for (int k = 0; k < n; ++k)
                out.push_back({f[i][k], midpoint(f[i][k], f[i][(k + 1) % n]), ci, midpoint(f[i][(k + n - 1) % n], f[i][k])});
        }
        for (int i = 0; i < nf; ++i) {
            if (mark[i]) continue;
            const Face& q = f[i];
            int n = fsize(q);
            std::vector<int> sk;
            for (int k = 0; k < n; ++k)
                if (edge_split(i, k)) sk.push_back(k);
            if (sk.empty()) { out.push_back(q); continue; }
            if (sk.size() == 1) {
                int k = sk[0];
                int a = q[k], b = q[(k + 1) % 4], c = q[(k + 2) % 4], d = q[(k + 3) % 4], mm = midpoint(a, b);
                out.push_back({mm, b, c, d});
                out.push_back({a, mm, d, -1});
            } else if ((sk[1] - sk[0]) == 2) {  // opposite edges
                int k = sk[0];
                int a = q[k], b = q[(k + 1) % 4], c = q[(k + 2) % 4], d = q[(k + 3) % 4];
                int m1 = midpoint(a, b), m2 = midpoint(c, d);
                out.push_back({a, m1, m2, d});
                out.push_back({m1, b, c, m2});
            } else {  // adjacent edges k, k+1 (cyclic)
                int k = (sk[0] == 0 && sk[1] == 3) ? 3 : sk[0];
                int a = q[k], b = q[(k + 1) % 4], c = q[(k + 2) % 4], d = q[(k + 3) % 4];
                int m1 = midpoint(a, b), m2 = midpoint(b, c);
                out.push_back({m1, b, m2, c});
                out.push_back({a, m1, c, d});
            }
        }
        MatrixXd V2(n0 + nv.size(), 3);
        V2.topRows(n0) = m.V;
        for (size_t k = 0; k < nv.size(); ++k) V2.row(n0 + k) = nv[k].transpose();
        m.V = V2;
        from_faces(m, out);
        total += nmark;
    }
    return total;
}

void relax_and_project(QuadMesh& m, const SurfaceRef& ref, double L, int iterations, PostStats& st) {
    const int nv = (int)m.V.rows();
    std::vector<Face> f = to_faces(m);
    // adjacency
    std::map<std::pair<int, int>, int> ecount;
    std::vector<std::vector<int>> vf(nv);
    for (size_t i = 0; i < f.size(); ++i)
        for (int j = 0; j < fsize(f[i]); ++j) {
            int a = f[i][j], b = f[i][(j + 1) % fsize(f[i])];
            ecount[{std::min(a, b), std::max(a, b)}]++;
            vf[a].push_back((int)i);
        }
    std::vector<std::vector<int>> nbr(nv);
    std::vector<char> bnd(nv, 0);
    std::vector<std::vector<int>> bnbr(nv);
    for (auto& kv : ecount) {
        int a = kv.first.first, b = kv.first.second;
        nbr[a].push_back(b), nbr[b].push_back(a);
        if (kv.second == 1) {
            bnd[a] = bnd[b] = 1;
            bnbr[a].push_back(b), bnbr[b].push_back(a);
        }
    }
    auto local_min_sj = [&](int v) {
        double mn = 1;
        for (int fi : vf[v]) mn = std::min(mn, scaled_jacobian(m.V, f[fi][0], f[fi][1], f[fi][2], f[fi][3]));
        return mn;
    };
    // a snap/move is acceptable if it does not create (more) badly shaped faces
    auto try_move = [&](int v, const Vector3d& np, double before) {
        Vector3d old = m.V.row(v);
        m.V.row(v) = np.transpose();
        double after = local_min_sj(v);
        if (after >= before - 1e-9 || after > 0.3) return true;
        m.V.row(v) = old.transpose();
        return false;
    };
    // classify: 0 interior, 1 feature, 2 boundary, 3 plane-boundary, 4 corner (fixed)
    std::vector<char> type(nv, 0);
    const double tol_corner = 0.3 * L, tol_feat = 0.35 * L;
    const int axis = ref.sym_axis();
    const double off = ref.sym_offset();
    for (int v = 0; v < nv; ++v) {
        Vector3d p = m.V.row(v), c, cc;
        double before = local_min_sj(v);
        if (bnd[v]) {
            bool plane = false;
            ref.dist2_boundary(p, &c, &plane);
            if (plane && axis >= 0) {
                type[v] = 3;
                m.V(v, axis) = off;
            } else if (ref.dist2_corner(p, &cc) < tol_corner * tol_corner && try_move(v, cc, before)) {
                type[v] = 4;
            } else {
                type[v] = 2;
                try_move(v, c, before);
            }
            continue;
        }
        if (ref.has_features()) {
            double df = ref.dist2_feature(p, &c);
            if (df < tol_feat * tol_feat) {
                if (ref.dist2_corner(p, &cc) < tol_corner * tol_corner && try_move(v, cc, before)) {
                    type[v] = 4;
                    st.corner_vertices++;
                    continue;
                }
                if (try_move(v, c, before)) {
                    type[v] = 1;
                    st.feature_vertices++;
                    continue;
                }
            }
        }
        try_move(v, ref.project_surface(p), before);
    }
    // corner recovery: every input corner gets the nearest boundary/feature output vertex
    // (within 0.75 L) snapped onto it, even if extraction chamfered the corner
    {
        MatrixXd CP = ref.corner_points();
        for (int k = 0; k < CP.rows(); ++k) {
            Vector3d c = CP.row(k).transpose();
            int best = -1;
            double bd = (0.75 * L) * (0.75 * L);
            for (int v = 0; v < nv; ++v) {
                if (type[v] != 1 && type[v] != 2 && type[v] != 4) continue;
                double d = (m.V.row(v).transpose() - c).squaredNorm();
                if (d < bd) { bd = d; best = v; }
            }
            if (best < 0 || type[best] == 4) continue;
            double before = local_min_sj(best);
            Vector3d old = m.V.row(best);
            m.V.row(best) = c.transpose();
            if (local_min_sj(best) > std::min(before, 0.0) - 1e-9) {
                type[best] = 4;
                st.corner_vertices++;
            } else {
                m.V.row(best) = old.transpose();
            }
        }
    }
    // a feature vertex slides along the curve only between two nearby feature neighbours
    std::vector<std::vector<int>> fnbr(nv);
    for (int v = 0; v < nv; ++v)
        if (type[v] == 1)
            for (int u : nbr[v])
                if ((type[u] == 1 || type[u] == 4) && (m.V.row(u) - m.V.row(v)).norm() < 2.5 * L) fnbr[v].push_back(u);

    // untangling: move vertices of inverted / degenerate faces to the projected neighbour
    // average until their worst incident face improves
    auto untangle = [&](int rounds) {
        for (int r = 0; r < rounds; ++r) {
            int moved = 0;
            for (size_t fi = 0; fi < f.size(); ++fi) {
                if (scaled_jacobian(m.V, f[fi][0], f[fi][1], f[fi][2], f[fi][3]) > 0.15) continue;
                for (int j = 0; j < fsize(f[fi]); ++j) {
                    int v = f[fi][j];
                    if (type[v] == 4 || nbr[v].empty()) continue;
                    double before = local_min_sj(v);
                    Vector3d old = m.V.row(v);
                    // candidate targets: 1-ring average and incident-face centroid average,
                    // constrained to the vertex's curve / surface
                    Vector3d avg = Vector3d::Zero(), fc = Vector3d::Zero();
                    for (int u : nbr[v]) avg += m.V.row(u).transpose();
                    avg /= nbr[v].size();
                    for (int g : vf[v]) {
                        Vector3d c = Vector3d::Zero();
                        for (int k = 0; k < fsize(f[g]); ++k) c += m.V.row(f[g][k]).transpose();
                        fc += c / fsize(f[g]);
                    }
                    fc /= std::max<size_t>(1, vf[v].size());
                    double best = before;
                    Vector3d bestp = old;
                    for (const Vector3d& t : {avg, fc, Vector3d(0.5 * (avg + old)), Vector3d(0.5 * (fc + old))}) {
                        Vector3d np, c;
                        if (type[v] == 1) { ref.dist2_feature(t, &c); np = c; }
                        else if (type[v] == 2) { ref.dist2_boundary(t, &c); np = c; }
                        else if (type[v] == 3) { np = ref.project_surface(t); np[axis] = off; }
                        else np = ref.project_surface(t);
                        m.V.row(v) = np.transpose();
                        double q = local_min_sj(v);
                        if (q > best + 1e-9) { best = q; bestp = np; }
                    }
                    m.V.row(v) = bestp.transpose();
                    if (best > before + 1e-9) moved++;
                }
            }
            if (!moved) break;
        }
    };
    untangle(20);
    for (int it = 0; it < iterations; ++it) {
        for (int v = 0; v < nv; ++v) {
            if (type[v] == 4) continue;
            const std::vector<int>* N = &nbr[v];
            if (type[v] == 1) {
                if (fnbr[v].size() != 2) continue;
                N = &fnbr[v];
            } else if (type[v] == 2 || type[v] == 3) {
                if (bnbr[v].size() != 2) continue;
                N = &bnbr[v];
            }
            if (N->empty()) continue;
            Vector3d p = m.V.row(v), avg = Vector3d::Zero();
            for (int u : *N) avg += m.V.row(u).transpose();
            avg /= N->size();
            Vector3d np, nrm, c;
            if (type[v] == 0) {
                ref.project_surface(p, &nrm);
                Vector3d d = avg - p;
                d -= nrm * nrm.dot(d);
                np = ref.project_surface(p + 0.6 * d);
            } else if (type[v] == 1) {
                ref.dist2_feature(p + 0.6 * (avg - p), &c);
                np = c;
            } else if (type[v] == 2) {
                ref.dist2_boundary(p + 0.6 * (avg - p), &c);
                np = c;
            } else {
                np = ref.project_surface(p + 0.6 * (avg - p));
                np[axis] = off;
            }
            double before = local_min_sj(v);
            m.V.row(v) = np.transpose();
            double after = local_min_sj(v);
            if (after < before - 1e-9 && after < 0.6) {
                m.V.row(v) = p.transpose();
                st.smoothing_rejected++;
            }
        }
    }
    untangle(20);
}

void mirror_and_weld(QuadMesh& m, int axis, double offset, double tol) {
    const int n = (int)m.V.rows();
    std::vector<int> mapv(n);
    std::vector<Vector3d> extra;
    for (int v = 0; v < n; ++v) {
        if (std::abs(m.V(v, axis) - offset) <= tol) {
            m.V(v, axis) = offset;
            mapv[v] = v;
        } else {
            Vector3d p = m.V.row(v);
            p[axis] = 2 * offset - p[axis];
            mapv[v] = n + (int)extra.size();
            extra.push_back(p);
        }
    }
    MatrixXd V2(n + extra.size(), 3);
    V2.topRows(n) = m.V;
    for (size_t i = 0; i < extra.size(); ++i) V2.row(n + i) = extra[i].transpose();
    std::vector<Face> f = to_faces(m), out = f;
    for (auto& q : f) {
        Face r = q;
        if (fsize(q) == 4) r = {mapv[q[0]], mapv[q[3]], mapv[q[2]], mapv[q[1]]};
        else r = {mapv[q[0]], mapv[q[2]], mapv[q[1]], -1};
        out.push_back(r);
    }
    m.V = V2;
    from_faces(m, out);
}

}  // namespace qrm
