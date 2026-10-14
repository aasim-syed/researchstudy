#pragma once
// Fallback extractor: the Instant Meshes pipeline (Jakob et al. 2015, BSD licence), vendored
// in third_party/instant-meshes. Used for components the integer-grid solver cannot handle.
#include <array>
#include <vector>

namespace qrm {

struct IMResult {
    bool ok = false;
    std::vector<double> V;               // 3 * n
    std::vector<std::array<int, 4>> Q;   // pure quads
};

// faces: target quad count of the final (pure-quad) mesh.
IMResult run_instant_meshes(const std::vector<double>& V, const std::vector<int>& F, int faces,
                            double crease_angle_deg, bool align_boundaries);

}  // namespace qrm
