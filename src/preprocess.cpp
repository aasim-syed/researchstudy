#include "quadremesh/preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

#include "quadremesh/geometry.h"

namespace qrm {

namespace {

struct KeyHash {
    size_t operator()(const std::array<int64_t, 3>& k) const {
        return (size_t)(k[0] * 73856093LL ^ k[1] * 19349663LL ^ k[2] * 83492791LL);
    }
};

// Weld vertices closer than eps (grid rounding). Returns new index per old vertex.
std::vector<int> weld(const MatrixXd& V, double eps, int& nunique) {
    std::unordered_map<std::array<int64_t, 3>, int, KeyHash> map;
    map.reserve(V.rows() * 2);
    std::vector<int> remap(V.rows());
    nunique = 0;
    for (int i = 0; i < V.rows(); ++i) {
        std::array<int64_t, 3> k = {(int64_t)std::llround(V(i, 0) / eps), (int64_t)std::llround(V(i, 1) / eps),
                                    (int64_t)std::llround(V(i, 2) / eps)};
        auto it = map.find(k);
        if (it == map.end()) {
            map.emplace(k, nunique);
            remap[i] = nunique++;
        } else {
            remap[i] = it->second;
        }
    }
    return remap;
}

void rebuild(WorkMesh& m, const std::vector<std::array<int, 3>>& faces, const std::vector<std::array<int, 3>>* sharp3 = nullptr) {
    m.F.resize(faces.size(), 3);
    for (size_t f = 0; f < faces.size(); ++f)
        for (int j = 0; j < 3; ++j) m.F(f, j) = faces[f][j];
    if (sharp3) {
        m.sharp.assign(3 * faces.size(), 0);
        for (size_t f = 0; f < faces.size(); ++f)
            for (int j = 0; j < 3; ++j) m.sharp[3 * f + j] = (*sharp3)[f][j];
    }
}

}  // namespace

void compact_vertices(WorkMesh& m) {
    std::vector<int> map(m.V.rows(), -1);
    int n = 0;
    for (int f = 0; f < m.F.rows(); ++f)
        for (int j = 0; j < 3; ++j)
            if (map[m.F(f, j)] < 0) map[m.F(f, j)] = n++;
    // keep original ordering for determinism
    std::vector<int> order;
    order.reserve(n);
    {
        std::vector<int> used(m.V.rows(), 0);
        for (int f = 0; f < m.F.rows(); ++f)
            for (int j = 0; j < 3; ++j) used[m.F(f, j)] = 1;
        n = 0;
        for (int i = 0; i < (int)used.size(); ++i)
            if (used[i]) { map[i] = n++; order.push_back(i); }
    }
    MatrixXd V2(n, 3), A2(n, m.A.cols());
    std::vector<int> tag2(n, 0);
    for (int i = 0; i < n; ++i) {
        V2.row(i) = m.V.row(order[i]);
        if (m.A.cols()) A2.row(i) = m.A.row(order[i]);
        if (!m.vtag.empty()) tag2[i] = m.vtag[order[i]];
    }
    for (int f = 0; f < m.F.rows(); ++f)
        for (int j = 0; j < 3; ++j) m.F(f, j) = map[m.F(f, j)];
    m.V = V2;
    m.A = A2;
    if (!m.vtag.empty()) m.vtag = tag2;
}

void repair_manifold(WorkMesh& m, PrepStats& st) {
    // ---------------------------------------------------------------- 3. non-manifold edges + orientation
    // Decide which face pairs are "glued" across each edge.
    const int nf = (int)m.F.rows();
    std::vector<Vector3d> fn(nf);
    for (int f = 0; f < nf; ++f) fn[f] = face_normal(m.V, m.F, f);
    EdgeTopology topo;
    topo.build(m.F);
    std::vector<int> glue(3 * nf, -1);  // half-edge -> glued half-edge
    for (size_t e = 0; e < topo.ev.size(); ++e) {
        const auto& hs = topo.ehe[e];
        if (hs.size() < 2) continue;
        if (hs.size() == 2) {
            glue[hs[0]] = hs[1];
            glue[hs[1]] = hs[0];
            continue;
        }
        st.nonmanifold_edges++;
        // choose the pair with the most consistent normals; prefer opposite edge directions
        double best = -1e9;
        int bi = -1, bj = -1;
        for (size_t i = 0; i < hs.size(); ++i)
            for (size_t j = i + 1; j < hs.size(); ++j) {
                int fi = hs[i] / 3, fj = hs[j] / 3;
                int ai = m.F(fi, hs[i] % 3), aj = m.F(fj, hs[j] % 3);
                double s = fn[fi].dot(ai != aj ? fn[fj] : Vector3d(-fn[fj]));
                if (ai != aj) s += 0.5;  // orientation-consistent pairs preferred
                if (s > best) { best = s; bi = hs[i]; bj = hs[j]; }
            }
        glue[bi] = bj;
        glue[bj] = bi;
    }
    // BFS orientation over glued pairs; a pair that cannot be made consistent is unglued.
    std::vector<int> flip(nf, -1);
    for (int s = 0; s < nf; ++s) {
        if (flip[s] >= 0) continue;
        flip[s] = 0;
        std::queue<int> q;
        q.push(s);
        while (!q.empty()) {
            int f = q.front();
            q.pop();
            for (int j = 0; j < 3; ++j) {
                int h = 3 * f + j, g = glue[h];
                if (g < 0) continue;
                int fg = g / 3;
                // consistent if the shared edge is traversed in opposite directions
                bool opposite = m.F(f, j) != m.F(fg, g % 3);
                int want = flip[f] ^ (opposite ? 0 : 1);
                if (flip[fg] < 0) {
                    flip[fg] = want;
                    q.push(fg);
                } else if (flip[fg] != want) {
                    glue[h] = glue[g] = -1;  // Moebius-like conflict: cut here
                }
            }
        }
    }
    for (int f = 0; f < nf; ++f)
        if (flip[f] == 1) {
            st.orientation_flips++;
            std::swap(m.F(f, 1), m.F(f, 2));
        }
    // after flipping, recompute glue in new half-edge indexing: rebuild by matching endpoints
    {
        std::vector<int> g2(3 * nf, -1);
        for (int f = 0; f < nf; ++f)
            for (int j = 0; j < 3; ++j) {
                int h = 3 * f + j;
                // original half-edge index of this edge before flip
                int jo = j;
                if (flip[f] == 1) {
                    // F was (a,b,c) -> (a,c,b): new edge j maps to old edge
                    static const int map_new_to_old[3] = {2, 1, 0};
                    jo = map_new_to_old[j];
                }
                int ho = 3 * f + jo;
                int go = glue[ho];
                if (go < 0) continue;
                int fg = go / 3, jg = go % 3;
                int jgn = jg;
                if (flip[fg] == 1) {
                    static const int map_old_to_new[3] = {2, 1, 0};
                    jgn = map_old_to_new[jg];
                }
                g2[h] = 3 * fg + jgn;
            }
        glue.swap(g2);
    }
    // ---------------------------------------------------------------- 4. split vertices into fans
    {
        // union corners (f, j) that share a vertex and are connected through glued edges
        UnionFind uf(3 * nf);
        for (int f = 0; f < nf; ++f)
            for (int j = 0; j < 3; ++j) {
                int h = 3 * f + j, g = glue[h];
                if (g < 0) continue;
                int fg = g / 3, jg = g % 3;
                // edge h: F(f,j)->F(f,j+1); twin g: F(fg,jg)->F(fg,jg+1) == reversed
                int a = m.F(f, j), b = m.F(f, (j + 1) % 3);
                for (int k = 0; k < 3; ++k) {
                    if (m.F(fg, k) == a) uf.unite(3 * f + j, 3 * fg + k);
                    if (m.F(fg, k) == b) uf.unite(3 * f + (j + 1) % 3, 3 * fg + k);
                }
            }
        std::vector<int> first_class(m.V.rows(), -1);
        std::map<int, int> class_vertex;
        std::vector<Vector3d> newV;
        std::vector<int> newsrc;
        MatrixXi F2 = m.F;
        int nv = (int)m.V.rows();
        std::vector<int> cls_to_v(3 * nf, -1);
        for (int f = 0; f < nf; ++f)
            for (int j = 0; j < 3; ++j) {
                int c = uf.find(3 * f + j), v = m.F(f, j);
                if (cls_to_v[c] >= 0) { F2(f, j) = cls_to_v[c]; continue; }
                if (first_class[v] < 0) {
                    first_class[v] = c;
                    cls_to_v[c] = v;
                } else {
                    cls_to_v[c] = nv + (int)newsrc.size();
                    newsrc.push_back(v);
                    st.vertex_splits++;
                }
                F2(f, j) = cls_to_v[c];
            }
        if (!newsrc.empty()) {
            MatrixXd V2(nv + newsrc.size(), 3), A2(nv + newsrc.size(), m.A.cols());
            V2.topRows(nv) = m.V;
            if (m.A.cols()) A2.topRows(nv) = m.A;
            for (size_t k = 0; k < newsrc.size(); ++k) {
                V2.row(nv + k) = m.V.row(newsrc[k]);
                if (m.A.cols()) A2.row(nv + k) = m.A.row(newsrc[k]);
            }
            m.V = V2;
            m.A = A2;
        }
        m.F = F2;
    }

}

WorkMesh preprocess(const MatrixXd& V, const MatrixXi& F, const MatrixXd& A, const Params& p, PrepStats& st) {
    WorkMesh m;
    const double diag = std::max(bbox_diagonal(V), 1e-12);
    // ---------------------------------------------------------------- 1. weld
    int nu = 0;
    std::vector<int> remap = weld(V, 1e-7 * diag, nu);
    st.welded = (int)V.rows() - nu;
    m.V.resize(nu, 3);
    m.A.resize(nu, A.cols());
    std::vector<int> cnt(nu, 0);
    m.A.setZero();
    for (int i = 0; i < V.rows(); ++i) {
        if (cnt[remap[i]] == 0) m.V.row(remap[i]) = V.row(i);
        if (A.cols()) m.A.row(remap[i]) += A.row(i);
        cnt[remap[i]]++;
    }
    if (A.cols())
        for (int i = 0; i < nu; ++i) m.A.row(i) /= std::max(cnt[i], 1);

    // ---------------------------------------------------------------- 2. degenerate + duplicates
    std::vector<std::array<int, 3>> faces;
    faces.reserve(F.rows());
    const double area_eps = 1e-14 * diag * diag;
    for (int f = 0; f < F.rows(); ++f) {
        std::array<int, 3> t = {remap[F(f, 0)], remap[F(f, 1)], remap[F(f, 2)]};
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) { st.degenerate++; continue; }
        Vector3d a = m.V.row(t[0]), b = m.V.row(t[1]), c = m.V.row(t[2]);
        if ((b - a).cross(c - a).norm() * 0.5 < area_eps) { st.degenerate++; continue; }
        faces.push_back(t);
    }
    {
        std::vector<std::pair<std::array<int, 3>, int>> keys;
        keys.reserve(faces.size());
        for (int f = 0; f < (int)faces.size(); ++f) {
            auto k = faces[f];
            std::sort(k.begin(), k.end());
            keys.push_back({k, f});
        }
        std::sort(keys.begin(), keys.end());
        std::vector<char> keep(faces.size(), 1);
        for (size_t i = 1; i < keys.size(); ++i)
            if (keys[i].first == keys[i - 1].first) { keep[keys[i].second] = 0; st.duplicate++; }
        std::vector<std::array<int, 3>> f2;
        for (size_t f = 0; f < faces.size(); ++f)
            if (keep[f]) f2.push_back(faces[f]);
        faces.swap(f2);
    }
    rebuild(m, faces);
    compact_vertices(m);

    repair_manifold(m, st);

    // ---------------------------------------------------------------- 5. fill small holes
    {
        EdgeTopology t2;
        t2.build(m.F);
        // boundary half-edges: next vertex map
        std::unordered_map<int, int> nextv;
        std::set<int> ambiguous;
        std::vector<int> bstart;
        for (size_t e = 0; e < t2.ev.size(); ++e)
            if (t2.ehe[e].size() == 1) {
                int h = t2.ehe[e][0];
                int a = m.F(h / 3, h % 3), b = m.F(h / 3, (h % 3 + 1) % 3);
                // hole loop runs opposite to the face orientation
                if (nextv.count(b)) ambiguous.insert(b);
                nextv[b] = a;
                bstart.push_back(b);
            }
        std::sort(bstart.begin(), bstart.end());
        std::unordered_map<int, char> visited;
        std::vector<std::array<int, 3>> addF;
        std::vector<Vector3d> addV;
        std::vector<Eigen::RowVectorXd> addA;
        const double maxper = p.max_hole_perimeter * diag;
        int nv = (int)m.V.rows();
        for (int s : bstart) {
            if (visited.count(s)) continue;
            std::vector<int> loop;
            int v = s;
            bool ok = true;
            while (!visited.count(v)) {
                visited[v] = 1;
                if (ambiguous.count(v)) { ok = false; break; }
                loop.push_back(v);
                auto it = nextv.find(v);
                if (it == nextv.end()) { ok = false; break; }
                v = it->second;
                if ((int)loop.size() > 100000) { ok = false; break; }
            }
            if (!ok || v != s || loop.size() < 3) continue;
            double per = 0;
            for (size_t k = 0; k < loop.size(); ++k)
                per += (m.V.row(loop[k]) - m.V.row(loop[(k + 1) % loop.size()])).norm();
            if (per > maxper) { st.holes_kept++; continue; }
            st.holes_filled++;
            if (loop.size() == 3) {
                addF.push_back({loop[0], loop[1], loop[2]});
                continue;
            }
            Vector3d c = Vector3d::Zero();
            Eigen::RowVectorXd ca = Eigen::RowVectorXd::Zero(m.A.cols());
            for (int u : loop) {
                c += m.V.row(u).transpose();
                if (m.A.cols()) ca += m.A.row(u);
            }
            c /= loop.size();
            if (m.A.cols()) ca /= loop.size();
            int ci = nv + (int)addV.size();
            addV.push_back(c);
            addA.push_back(ca);
            for (size_t k = 0; k < loop.size(); ++k) addF.push_back({loop[k], loop[(k + 1) % loop.size()], ci});
        }
        if (!addF.empty()) {
            MatrixXd V2(nv + addV.size(), 3), A2(nv + addV.size(), m.A.cols());
            V2.topRows(nv) = m.V;
            if (m.A.cols()) A2.topRows(nv) = m.A;
            for (size_t k = 0; k < addV.size(); ++k) {
                V2.row(nv + k) = addV[k].transpose();
                if (m.A.cols()) A2.row(nv + k) = addA[k];
            }
            MatrixXi F2(m.F.rows() + addF.size(), 3);
            F2.topRows(m.F.rows()) = m.F;
            for (size_t k = 0; k < addF.size(); ++k)
                for (int j = 0; j < 3; ++j) F2(m.F.rows() + k, j) = addF[k][j];
            m.V = V2;
            m.A = A2;
            m.F = F2;
        }
    }

    // hole filling can touch non-manifold boundary vertices: repair again
    repair_manifold(m, st);

    // ---------------------------------------------------------------- 6. tiny components, outward orientation
    {
        std::vector<int> comp;
        int nc = face_components(m.F, (int)m.V.rows(), comp);
        std::vector<double> area(nc, 0), vol(nc, 0);
        std::vector<int> nbnd(nc, 0);
        EdgeTopology t3;
        t3.build(m.F);
        for (int f = 0; f < m.F.rows(); ++f) {
            area[comp[f]] += face_area(m.V, m.F, f);
            Vector3d a = m.V.row(m.F(f, 0)), b = m.V.row(m.F(f, 1)), c = m.V.row(m.F(f, 2));
            vol[comp[f]] += a.dot(b.cross(c)) / 6.0;
            for (int j = 0; j < 3; ++j)
                if (t3.ehe[t3.he2e[3 * f + j]].size() == 1) nbnd[comp[f]]++;
        }
        double total = 0;
        for (double a : area) total += a;
        std::vector<std::array<int, 3>> f2;
        for (int f = 0; f < m.F.rows(); ++f) {
            int c = comp[f];
            if (area[c] < p.min_component_area * total) continue;
            std::array<int, 3> t = {m.F(f, 0), m.F(f, 1), m.F(f, 2)};
            if (nbnd[c] == 0 && vol[c] < 0) std::swap(t[1], t[2]);  // closed & inside-out
            f2.push_back(t);
        }
        for (int c = 0; c < nc; ++c)
            if (area[c] < p.min_component_area * total) st.components_dropped++;
        rebuild(m, f2);
        compact_vertices(m);
    }
    m.sharp.clear();
    m.vtag.assign(m.V.rows(), 0);
    return m;
}

// ------------------------------------------------------------------------------------------
void refine_long_edges(WorkMesh& m, double max_len) {
    const double L2 = max_len * max_len;
    bool has_sharp = !m.sharp.empty();
    if (m.vtag.size() != (size_t)m.V.rows()) m.vtag.assign(m.V.rows(), 0);
    for (int pass = 0; pass < 60; ++pass) {
        const int nf = (int)m.F.rows();
        EdgeTopology topo;
        topo.build(m.F);
        // pick for each face its longest edge if too long; split edges chosen by all their faces
        std::vector<char> split_edge(topo.ev.size(), 0);
        std::vector<char> face_busy(nf, 0);
        // process edges by decreasing length (deterministic tie-break on id)
        std::vector<std::pair<double, int>> cand;
        for (size_t e = 0; e < topo.ev.size(); ++e) {
            double l2 = (m.V.row(topo.ev[e].first) - m.V.row(topo.ev[e].second)).squaredNorm();
            if (l2 > L2) cand.push_back({-l2, (int)e});
        }
        if (cand.empty()) break;
        std::sort(cand.begin(), cand.end());
        for (auto& c : cand) {
            int e = c.second;
            bool free = true;
            for (int h : topo.ehe[e]) free &= !face_busy[h / 3];
            if (!free) continue;
            split_edge[e] = 1;
            for (int h : topo.ehe[e]) face_busy[h / 3] = 1;
        }
        // create midpoints
        std::vector<int> mid(topo.ev.size(), -1);
        int nv = (int)m.V.rows();
        std::vector<int> newv_src_a, newv_src_b;
        for (size_t e = 0; e < topo.ev.size(); ++e)
            if (split_edge[e]) {
                mid[e] = nv + (int)newv_src_a.size();
                newv_src_a.push_back(topo.ev[e].first);
                newv_src_b.push_back(topo.ev[e].second);
            }
        MatrixXd V2(nv + newv_src_a.size(), 3), A2(nv + newv_src_a.size(), m.A.cols());
        V2.topRows(nv) = m.V;
        if (m.A.cols()) A2.topRows(nv) = m.A;
        m.vtag.resize(nv + newv_src_a.size(), 0);
        for (size_t k = 0; k < newv_src_a.size(); ++k) {
            V2.row(nv + k) = 0.5 * (m.V.row(newv_src_a[k]) + m.V.row(newv_src_b[k]));
            if (m.A.cols()) A2.row(nv + k) = 0.5 * (m.A.row(newv_src_a[k]) + m.A.row(newv_src_b[k]));
            m.vtag[nv + k] = m.vtag[newv_src_a[k]] & m.vtag[newv_src_b[k]];
        }
        std::vector<std::array<int, 3>> faces;
        std::vector<std::array<int, 3>> sh;
        faces.reserve(nf + newv_src_a.size() * 2);
        for (int f = 0; f < nf; ++f) {
            int js = -1;
            for (int j = 0; j < 3; ++j)
                if (split_edge[topo.he2e[3 * f + j]]) js = j;
            std::array<int, 3> s0 = {0, 0, 0};
            if (has_sharp) s0 = {m.sharp[3 * f], m.sharp[3 * f + 1], m.sharp[3 * f + 2]};
            if (js < 0) {
                faces.push_back({m.F(f, 0), m.F(f, 1), m.F(f, 2)});
                sh.push_back(s0);
                continue;
            }
            int a = m.F(f, js), b = m.F(f, (js + 1) % 3), c = m.F(f, (js + 2) % 3);
            int mm = mid[topo.he2e[3 * f + js]];
            int sab = s0[js], sbc = s0[(js + 1) % 3], sca = s0[(js + 2) % 3];
            faces.push_back({a, mm, c});
            sh.push_back({sab, 0, sca});
            faces.push_back({mm, b, c});
            sh.push_back({sab, sbc, 0});
        }
        m.V = V2;
        m.A = A2;
        rebuild(m, faces, has_sharp ? &sh : nullptr);
    }
}

// ------------------------------------------------------------------------------------------
void clip_to_halfspace(WorkMesh& m, int axis, double offset) {
    const double diag = bbox_diagonal(m.V);
    const double eps = 1e-6 * diag;
    if (m.vtag.size() != (size_t)m.V.rows()) m.vtag.assign(m.V.rows(), 0);
    // snap near-plane vertices
    for (int i = 0; i < m.V.rows(); ++i)
        if (std::abs(m.V(i, axis) - offset) < eps) {
            m.V(i, axis) = offset;
            m.vtag[i] |= 1;
        }
    auto sd = [&](int v) { return m.V(v, axis) - offset; };
    bool has_sharp = !m.sharp.empty();
    std::map<std::pair<int, int>, int> cut;  // edge -> new vertex on plane
    std::vector<Vector3d> newV;
    std::vector<Eigen::RowVectorXd> newA;
    int nv = (int)m.V.rows();
    auto cutv = [&](int a, int b) {
        auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = cut.find(key);
        if (it != cut.end()) return it->second;
        double da = sd(a), db = sd(b);
        double t = da / (da - db);
        Vector3d p = (1 - t) * m.V.row(a).transpose() + t * m.V.row(b).transpose();
        p[axis] = offset;
        int id = nv + (int)newV.size();
        newV.push_back(p);
        if (m.A.cols()) newA.push_back((1 - t) * m.A.row(a) + t * m.A.row(b));
        else newA.push_back(Eigen::RowVectorXd());
        cut[key] = id;
        return id;
    };
    std::vector<std::array<int, 3>> faces, sh;
    for (int f = 0; f < m.F.rows(); ++f) {
        int v[3] = {m.F(f, 0), m.F(f, 1), m.F(f, 2)};
        double d[3] = {sd(v[0]), sd(v[1]), sd(v[2])};
        std::array<int, 3> s0 = {0, 0, 0};
        if (has_sharp) s0 = {m.sharp[3 * f], m.sharp[3 * f + 1], m.sharp[3 * f + 2]};
        int npos = 0, nneg = 0;
        for (int j = 0; j < 3; ++j) {
            if (d[j] > 0) npos++;
            else if (d[j] < 0) nneg++;
        }
        if (nneg == 0) { faces.push_back({v[0], v[1], v[2]}); sh.push_back(s0); continue; }
        if (npos == 0) continue;
        // general case: walk the polygon, keep the positive part (Sutherland-Hodgman)
        std::vector<int> poly;
        std::vector<int> polys;  // sharp flag of edge starting at poly[k]
        for (int j = 0; j < 3; ++j) {
            int a = v[j], b = v[(j + 1) % 3];
            double da = d[j], db = d[(j + 1) % 3];
            if (da >= 0) { poly.push_back(a); polys.push_back(s0[j]); }
            if ((da > 0 && db < 0) || (da < 0 && db > 0)) {
                int c = cutv(a, b);
                poly.push_back(c);
                polys.push_back(da > 0 ? 0 : s0[j]);  // edge from cut point along plane is new
                if (da > 0) polys[polys.size() - 2] = s0[j];
            }
        }
        for (size_t k = 1; k + 1 < poly.size(); ++k) {
            faces.push_back({poly[0], poly[k], poly[k + 1]});
            sh.push_back({k == 1 ? polys[0] : 0, polys[k], k + 2 == poly.size() ? polys[k + 1] : 0});
        }
    }
    MatrixXd V2(nv + newV.size(), 3), A2(nv + newV.size(), m.A.cols());
    V2.topRows(nv) = m.V;
    if (m.A.cols()) A2.topRows(nv) = m.A;
    m.vtag.resize(nv + newV.size(), 1);
    for (size_t k = 0; k < newV.size(); ++k) {
        V2.row(nv + k) = newV[k].transpose();
        if (m.A.cols()) A2.row(nv + k) = newA[k];
        m.vtag[nv + k] = 1;
    }
    m.V = V2;
    m.A = A2;
    rebuild(m, faces, has_sharp ? &sh : nullptr);
    compact_vertices(m);
}

}  // namespace qrm
