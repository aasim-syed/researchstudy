# Vendored QuadriFlow (patched)

Upstream: https://github.com/hjwdzh/QuadriFlow (MIT-style license, see LICENSE.txt).
Used by quadremesh as the integer-grid / min-cost-flow stage. The pristine upstream copy in
`third_party/QuadriFlow` is built separately as the *baseline* CLI.

Patches applied (all marked `quadremesh patch` or visible in `git diff` against upstream):

1. `hierarchy.cpp` - field initialisation uses a seeded PCG32 instead of `srand/rand`
   (bit-for-bit deterministic across libc implementations).
2. `hierarchy.cpp` - `propagateConstraints` used the constraint *direction* as the normal when
   merging two constrained children; fixed to use the level normals.
3. `hierarchy.hpp` / `optimizer.cpp` - `mExtS`: an external per-vertex target scale (our
   adaptive density map). Upstream clamps the adaptive scale to [0.75, 1].
4. `parametrizer.hpp` / `parametrizer-mesh.cpp` - `ext_sharp_edges` / `use_ext_sharp`: feature
   half-edges come from our detector (user angle threshold, chained + length-filtered, guidance
   strokes) instead of the hard-coded 60 degree test; `skip_subdivide` because our
   preprocessing already refines long edges while propagating feature flags.
5. `qflow_quiet.hpp` (force-included) + `dedge.cpp` - console output gated by `qflow::g_verbose`.

Orientation constraints (feature/boundary hard constraints, curvature/learned soft
constraints, guidance strokes) are injected through the existing `mCQ/mCQw/mCO/mCOw` API.
6. `hierarchy.cpp` - `propagateConstraints` summed child weights, so coarse-level weights exceeded 1
   and the blend `(1-w)*field + w*constraint` extrapolated (unstable with soft constraints).
   Weights are now `min(1, max(w0, w1))`.
7. `qflow_quiet.hpp` also maps QuadriFlow's internal `exit()` calls (consistency asserts in
   subdivision / flip fixing / integer solve) to C++ exceptions; the driver catches them and
   retries the component at another resolution or falls back.
8. `parametrizer-mesh.cpp` - `FixValence` loops are bounded (it could spin forever on some
   non-manifold boundary configurations of large scans).
9. `parametrizer-flip.cpp` - `FixHoles` boundary walks are bounded (infinite loop on non-manifold quad fans).
10. `subdivide.cpp` - unbounded rotation search in `subdivide_edgeDiff` (spins forever on inconsistent data) now raises.
11. `Parametrizer::ext_normals` - optional externally smoothed normals for the field (noisy scans).
