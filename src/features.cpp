#include "quadremesh/features.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <set>

#include "quadremesh/geometry.h"

namespace qrm {

namespace {

struct FeatureGraph {
    std::vector<std::vector<std::pair<int, int>>> adj;  // vertex -> (nbr, edge id)
};

}  // namespace

void finalize_features(const MatrixXd& V, const MatrixXi& F, FeatureSet& fs, double corner_angle_deg) {
    EdgeTopology topo;
    topo.build(F);
    const int nv = (int)V.rows();
    fs.vdeg.assign(nv, 0);
    std::vector<std::vector<int>> vnbr(nv);
    std::vector<std::vector<double>> seg;
    std::vector<std::vector<double>> bseg;
    for (size_t e = 0; e < topo.ev.size(); ++e) {
        bool s = false;
        for (int h : topo.ehe[e]) s |= fs.he_sharp[h] != 0;
        int a = topo.ev[e].first, b = topo.ev[e].second;
        if (topo.ehe[e].size() == 1) {
            bseg.push_back({V(a, 0), V(a, 1), V(a, 2), V(b, 0), V(b, 1), V(b, 2)});
            continue;
        }
        if (!s) continue;
        for (int h : topo.ehe[e]) fs.he_sharp[h] = 1;
        fs.vdeg[a]++, fs.vdeg[b]++;
        vnbr[a].push_back(b), vnbr[b].push_back(a);
        seg.push_back({V(a, 0), V(a, 1), V(a, 2), V(b, 0), V(b, 1), V(b, 2)});
    }
    fs.segments.resize(seg.size(), 6);
    for (size_t i = 0; i < seg.size(); ++i)
        for (int k = 0; k < 6; ++k) fs.segments(i, k) = seg[i][k];
    fs.boundary_segments.resize(bseg.size(), 6);
    for (size_t i = 0; i < bseg.size(); ++i)
        for (int k = 0; k < 6; ++k) fs.boundary_segments(i, k) = bseg[i][k];
    fs.kept_edges = (int)seg.size();
    // corners
    const double cos_kink = std::cos((180.0 - corner_angle_deg) * M_PI / 180.0);
    fs.corners.clear();
    for (int v = 0; v < nv; ++v) {
        if (fs.vdeg[v] == 0) continue;
        if (fs.vdeg[v] != 2) { fs.corners.push_back(v); continue; }
        Vector3d d0 = (V.row(vnbr[v][0]) - V.row(v)).normalized(), d1 = (V.row(vnbr[v][1]) - V.row(v)).normalized();
        if (d0.dot(d1) > cos_kink) fs.corners.push_back(v);  // turning angle larger than threshold
    }
}

FeatureSet detect_features(const MatrixXd& V, const MatrixXi& F, double angle_deg, double min_len, double support) {
    FeatureSet fs;
    const int nf = (int)F.rows(), nv = (int)V.rows();
    fs.he_sharp.assign(3 * nf, 0);
    if (angle_deg <= 0) {
        finalize_features(V, F, fs);
        return fs;
    }
    EdgeTopology topo;
    topo.build(F);
    std::vector<Vector3d> fn(nf);
    for (int f = 0; f < nf; ++f) fn[f] = face_normal(V, F, f);
    const double cos_t = std::cos(angle_deg * M_PI / 180.0);
    const double cos_t2 = std::cos(0.6 * angle_deg * M_PI / 180.0);
    // averaged normal of a face with its neighbours, not crossing edge `skip`
    auto side_normal = [&](int f, int skip_e) {
        Vector3d n = fn[f] * face_area(V, F, f);
        for (int j = 0; j < 3; ++j) {
            int e = topo.he2e[3 * f + j];
            if (e == skip_e) continue;
            for (int h : topo.ehe[e])
                if (h / 3 != f && topo.ehe[e].size() == 2) {
                    // do not average across edges that are themselves strongly bent
                    if (fn[h / 3].dot(fn[f]) > cos_t2) n += fn[h / 3] * face_area(V, F, h / 3);
                }
        }
        double l = n.norm();
        return l > 0 ? Vector3d(n / l) : fn[f];
    };
    std::vector<char> cand(topo.ev.size(), 0);
    for (size_t e = 0; e < topo.ev.size(); ++e) {
        if (topo.ehe[e].size() != 2) continue;
        int f0 = topo.ehe[e][0] / 3, f1 = topo.ehe[e][1] / 3;
        if (fn[f0].dot(fn[f1]) >= cos_t) continue;
        if (side_normal(f0, (int)e).dot(side_normal(f1, (int)e)) >= cos_t2) continue;
        cand[e] = 1;
    }
    // scale test: a crease must persist when each side is averaged over a patch of radius
    // `support` (noise and small bumps do not). Patches grow by BFS without crossing candidates.
    if (support > 0) {
        std::vector<int> stamp(nf, -1);
        std::vector<char> keep(topo.ev.size(), 0);
        const double cos_t3 = std::cos(0.7 * angle_deg * M_PI / 180.0);
        for (size_t e = 0; e < topo.ev.size(); ++e) {
            if (!cand[e]) continue;
            Vector3d mid = 0.5 * (V.row(topo.ev[e].first) + V.row(topo.ev[e].second)).transpose();
            Vector3d side[2];
            for (int s = 0; s < 2; ++s) {
                int f0 = topo.ehe[e][s] / 3;
                int tag = (int)(2 * e + s);
                std::vector<int> queue = {f0};
                stamp[f0] = tag;
                Vector3d acc = Vector3d::Zero();
                for (size_t qi = 0; qi < queue.size() && qi < 300; ++qi) {
                    int f = queue[qi];
                    acc += fn[f] * face_area(V, F, f);
                    for (int j = 0; j < 3; ++j) {
                        int ee = topo.he2e[3 * f + j];
                        if (cand[ee] || topo.ehe[ee].size() != 2) continue;
                        int g = topo.ehe[ee][0] / 3 == f ? topo.ehe[ee][1] / 3 : topo.ehe[ee][0] / 3;
                        if (stamp[g] == tag) continue;
                        Vector3d cg = (V.row(F(g, 0)) + V.row(F(g, 1)) + V.row(F(g, 2))).transpose() / 3.0;
                        if ((cg - mid).norm() > support) continue;
                        stamp[g] = tag;
                        queue.push_back(g);
                    }
                }
                side[s] = acc.normalized();
            }
            if (side[0].dot(side[1]) < cos_t3) keep[e] = 1;
        }
        for (size_t e = 0; e < topo.ev.size(); ++e) cand[e] = keep[e];
    }
    for (size_t e = 0; e < topo.ev.size(); ++e) fs.raw_edges += cand[e];
    // chain pruning
    std::vector<double> elen(topo.ev.size());
    for (size_t e = 0; e < topo.ev.size(); ++e) elen[e] = (V.row(topo.ev[e].first) - V.row(topo.ev[e].second)).norm();
    for (int round = 0; round < 20; ++round) {
        std::vector<std::vector<std::pair<int, int>>> adj(nv);
        for (size_t e = 0; e < topo.ev.size(); ++e)
            if (cand[e]) {
                adj[topo.ev[e].first].push_back({topo.ev[e].second, (int)e});
                adj[topo.ev[e].second].push_back({topo.ev[e].first, (int)e});
            }
        std::vector<char> seen(topo.ev.size(), 0);
        bool changed = false;
        int nchains = 0;
        auto trace = [&](int start, int e0, std::vector<int>& chain, int& endv) {
            int v = start, e = e0;
            while (true) {
                seen[e] = 1;
                chain.push_back(e);
                int u = topo.ev[e].first == v ? topo.ev[e].second : topo.ev[e].first;
                if (adj[u].size() != 2) { endv = u; return; }
                int en = adj[u][0].second == e ? adj[u][1].second : adj[u][0].second;
                if (seen[en]) { endv = u; return; }
                v = u, e = en;
            }
        };
        // open chains from nodes
        for (int v = 0; v < nv; ++v) {
            if (adj[v].empty() || adj[v].size() == 2) continue;
            for (auto& pr : adj[v]) {
                if (seen[pr.second]) continue;
                std::vector<int> chain;
                int endv = -1;
                trace(v, pr.second, chain, endv);
                nchains++;
                double len = 0;
                for (int e : chain) len += elen[e];
                bool dangling = adj[v].size() == 1 || adj[endv].size() == 1;
                bool junction_link = adj[v].size() >= 3 && adj[endv].size() >= 3;
                bool drop = len < min_len && (dangling || !junction_link);
                if (drop) {
                    for (int e : chain) cand[e] = 0;
                    changed = true;
                }
            }
        }
        // closed loops
        for (size_t e = 0; e < topo.ev.size(); ++e) {
            if (!cand[e] || seen[e]) continue;
            std::vector<int> chain;
            int endv = -1;
            trace(topo.ev[e].first, (int)e, chain, endv);
            nchains++;
            double len = 0;
            for (int c : chain) len += elen[c];
            if (len < 2.0 * min_len) {
                for (int c : chain) cand[c] = 0;
                changed = true;
            }
        }
        fs.chains = nchains;
        if (!changed) break;
    }
    for (size_t e = 0; e < topo.ev.size(); ++e)
        if (cand[e])
            for (int h : topo.ehe[e]) fs.he_sharp[h] = 1;
    finalize_features(V, F, fs);
    return fs;
}

void add_stroke_features(const MatrixXd& V, const MatrixXi& F, const std::vector<std::vector<Vector3d>>& strokes,
                         FeatureSet& fs) {
    if (strokes.empty()) return;
    EdgeTopology topo;
    topo.build(F);
    const int nv = (int)V.rows();
    std::vector<std::vector<std::pair<int, int>>> adj(nv);
    for (size_t e = 0; e < topo.ev.size(); ++e) {
        adj[topo.ev[e].first].push_back({topo.ev[e].second, (int)e});
        adj[topo.ev[e].second].push_back({topo.ev[e].first, (int)e});
    }
    auto nearest = [&](const Vector3d& p) {
        int best = 0;
        double bd = 1e300;
        for (int v = 0; v < nv; ++v) {
            double d = (V.row(v).transpose() - p).squaredNorm();
            if (d < bd) { bd = d; best = v; }
        }
        return best;
    };
    for (const auto& s : strokes) {
        if (s.size() < 2) continue;
        std::vector<int> pts;
        for (const auto& p : s) {
            int v = nearest(p);
            if (pts.empty() || pts.back() != v) pts.push_back(v);
        }
        for (size_t k = 0; k + 1 < pts.size(); ++k) {
            // Dijkstra with edge cost = length * (1 + deviation from the straight segment)
            int src = pts[k], dst = pts[k + 1];
            Vector3d A = V.row(src), B = V.row(dst);
            std::vector<double> dist(nv, 1e300);
            std::vector<int> prev_e(nv, -1);
            std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
            dist[src] = 0;
            pq.push({0, src});
            while (!pq.empty()) {
                auto [d, v] = pq.top();
                pq.pop();
                if (d > dist[v]) continue;
                if (v == dst) break;
                for (auto [u, e] : adj[v]) {
                    double l = (V.row(u) - V.row(v)).norm();
                    double dev = std::sqrt(point_segment_dist2(V.row(u).transpose(), A, B));
                    double nd = d + l + 2.0 * dev;
                    if (nd < dist[u]) {
                        dist[u] = nd;
                        prev_e[u] = e;
                        pq.push({nd, u});
                    }
                }
            }
            int v = dst;
            while (v != src && prev_e[v] >= 0) {
                int e = prev_e[v];
                for (int h : topo.ehe[e]) fs.he_sharp[h] = 1;
                v = topo.ev[e].first == v ? topo.ev[e].second : topo.ev[e].first;
            }
        }
    }
    finalize_features(V, F, fs);
}

}  // namespace qrm
