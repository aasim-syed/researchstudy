#include "qflow_driver.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "config.hpp"
#include "field-math.hpp"
#include "optimizer.hpp"
#include "parametrizer.hpp"

namespace qflow {
int g_verbose = 0;
}

namespace qrm {

QFResult run_quadriflow(const QFJob& job) {
    QFResult res;
    const int n = (int)job.V.rows();
    if (n < 3 || job.F.rows() < 1) return res;
    auto P = std::make_unique<qflow::Parametrizer>();
    P->V = job.V.transpose();
    P->F = job.F.transpose();
    P->NormalizeMesh();
    P->flag_preserve_boundary = job.preserve_boundary ? 1 : 0;
    P->flag_preserve_sharp = 0;
    P->use_ext_sharp = 1;
    P->ext_sharp_edges = job.sharp_he;
    if (P->ext_sharp_edges.size() != (size_t)(3 * job.F.rows())) P->ext_sharp_edges.assign(3 * job.F.rows(), 0);
    P->skip_subdivide = 1;
    P->flag_minimum_cost_flow = job.min_cost_flow ? 1 : 0;
    const bool adaptive = job.scale.size() == n;
    P->flag_adaptive_scale = adaptive ? 1 : 0;
    P->hierarchy.rng_seed = (int)job.seed;

    P->Initialize(job.faces);
    qflow::Hierarchy& H = P->hierarchy;
    const bool same = H.mV[0].cols() == n;

    // ---- constraints (orientation / position) on the finest level
    // Hard constraints (features, boundary, strokes; w >= 1) go into the multiresolution solve.
    // Soft ones (curvature / learned guide) are applied afterwards by an angle-limited
    // alignment pass on the finest level, so they bend edge flow locally without being able
    // to create new singularities.
    H.clearConstraints();
    std::vector<int> soft_ids;
    if (same && job.cons.w.size() == n) {
        for (int i = 0; i < n; ++i) {
            double w = job.cons.w[i];
            if (w <= 0) continue;
            Eigen::Vector3d d = job.cons.dir.row(i).transpose();
            Eigen::Vector3d nn = H.mN[0].col(i);
            d -= nn * nn.dot(d);
            if (d.squaredNorm() < 1e-20) continue;
            d.normalize();
            if (w < 1.0) { soft_ids.push_back(i); continue; }
            H.mCQ[0].col(i) = d;
            H.mCQw[0][i] = 1.0;
            if (job.cons.pos[i]) {
                H.mCO[0].col(i) = H.mV[0].col(i);
                H.mCOw[0][i] = 1.0;
            }
        }
    }
    H.propagateConstraints();
    qflow::Optimizer::optimize_orientations(H);

    if (!soft_ids.empty()) {
        auto& Q = H.mQ[0];
        const auto& N = H.mN[0];
        auto& adj = H.mAdj[0];
        std::vector<Eigen::Vector3d> target(n, Eigen::Vector3d::Zero());
        std::vector<double> tw(n, 0.0);
        for (int i : soft_ids) {
            Eigen::Vector3d d = job.cons.dir.row(i).transpose(), nn = N.col(i);
            target[i] = (d - nn * nn.dot(d)).normalized();
            tw[i] = job.cons.w[i];
        }
        const double cos_lim = std::cos(job.soft_max_angle * M_PI / 180.0);
        for (int it = 0; it < job.soft_iterations; ++it) {
            for (auto& phase : H.mPhases[0])
                for (int i : phase) {
                    const Eigen::Vector3d n_i = N.col(i);
                    // smoothness (same update as the solver)
                    double weight_sum = 0;
                    Eigen::Vector3d sum = Q.col(i);
                    for (auto& link : adj[i]) {
                        if (link.weight == 0) continue;
                        auto value = qflow::compat_orientation_extrinsic_4(sum, n_i, Q.col(link.id), N.col(link.id));
                        sum = value.first * weight_sum + value.second * link.weight;
                        sum -= n_i * n_i.dot(sum);
                        weight_sum += link.weight;
                        double norm = sum.norm();
                        if (norm > RCPOVERFLOW) sum /= norm;
                    }
                    // hard constraint keeps priority
                    if (H.mCQw[0][i] >= 1.0) {
                        auto v = qflow::compat_orientation_extrinsic_4(sum, n_i, H.mCQ[0].col(i), n_i);
                        sum = v.second;
                    } else if (tw[i] > 0) {
                        auto v = qflow::compat_orientation_extrinsic_4(sum, n_i, target[i], n_i);
                        double c = std::abs(v.first.normalized().dot(v.second.normalized()));
                        if (c > cos_lim) {
                            sum = v.first * (1 - tw[i]) + v.second * tw[i];
                            sum -= n_i * n_i.dot(sum);
                        }
                    }
                    double norm = sum.norm();
                    if (weight_sum > 0 && norm > RCPOVERFLOW) Q.col(i) = sum / norm;
                }
        }
    }
    P->ComputeOrientationSingularities();
    res.orientation_singularities = (int)P->singularities.size();
    if (same) {  // capture now: ComputeIndexMap later subdivides the hierarchy mesh
        res.fieldQ = H.mQ[0].transpose();
        res.fieldN = H.mN[0].transpose();
    }

    if (adaptive) {
        // slope (conformal scale gradient) estimation traces geodesics and can fail near
        // boundaries; the scale solve works without it (K = 0 -> plain smooth scale field)
        try {
            P->EstimateSlope();
        } catch (const std::exception&) {
            for (auto& K : H.mK) K.setZero();
        }
        if (same) H.mExtS = job.scale;
    }
    qflow::Optimizer::optimize_scale(H, P->rho, P->flag_adaptive_scale);
    if (adaptive && std::getenv("QRM_DEBUG_SCALE")) {
        std::fprintf(stderr, "scale target min %g max %g | solved S min %g max %g | mScale %g\n", job.scale.minCoeff(),
                     job.scale.maxCoeff(), H.mS[0].minCoeff(), H.mS[0].maxCoeff(), H.mScale);
    }
    if (adaptive) {
        // upstream only solves the scale on the finest level; the coarse levels of the position
        // solve then run at uniform scale and erase the density map. Restrict S upwards.
        for (size_t l = 0; l + 1 < H.mS.size(); ++l) {
            const auto& up = H.mToUpper[l];
            for (int i = 0; i < up.cols(); ++i) {
                Eigen::Vector2d acc = H.mS[l].col(up(0, i));
                int cnt = 1;
                if (up(1, i) >= 0) { acc += H.mS[l].col(up(1, i)); cnt++; }
                H.mS[l + 1].col(i) = acc / cnt;
            }
        }
    }
    P->flag_adaptive_scale = 1;
    qflow::Optimizer::optimize_positions(H, P->flag_adaptive_scale);
    P->ComputePositionSingularities();
    P->ComputeIndexMap(adaptive ? 1 : 0);

    res.V.resize(P->O_compact.size(), 3);
    for (size_t i = 0; i < P->O_compact.size(); ++i) {
        Eigen::Vector3d t = P->O_compact[i] * P->normalize_scale + P->normalize_offset;
        res.V.row(i) = t.transpose();
    }
    res.Q.reserve(P->F_compact.size());
    for (auto& q : P->F_compact) res.Q.push_back({q[0], q[1], q[2], q[3]});
    res.ok = !res.Q.empty();
    return res;
}

}  // namespace qrm
