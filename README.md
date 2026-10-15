# quadremesh

Quality-driven triangle → structured quad remeshing. A C++17 core (Eigen, libigl, a patched
QuadriFlow integer-grid solver), pybind11 Python bindings, a CLI, an evaluation tool, an optional
PyTorch learned-field component and a reproducible benchmark against Instant Meshes, QuadriFlow
and Blender QuadriFlow.

```
quadremesh chair.glb -o chair_quad.obj --faces 5000 --feature-angle 35 --report chair.json
```

```python
from quadremesh import quadremesh
res = quadremesh("chair.glb", target_faces=5000, feature_angle=35,
                 output="chair_quad.obj", report="chair.json")
print(res.quad_ratio, res.report["quality"]["scaled_jacobian"]["mean"])
```

## Pipeline

| stage | what it does | where |
|---|---|---|
| 1. preprocess | weld, drop degenerate/duplicate faces, resolve non-manifold edges & vertices (best-pair gluing + fan splitting), consistent/outward orientation, fill small holes, drop tiny fragments | `src/preprocess.cpp` |
| 2. features | dihedral test + 1-ring and patch-scale tests (noise-robust), chaining, length pruning, corners; guidance strokes → edge paths | `src/features.cpp` |
| 3. field design | hard constraints: features, boundaries, strokes. soft guidance: principal curvature (scale-aware tensor fit), learned / external field; soft guidance is applied by an angle-limited pass so it cannot add singularities | `src/field.cpp`, `src/qflow_driver.cpp` |
| 4. sizing | target count or edge length; adaptive density map (per-vertex, sphere regions, curvature-driven) → scale field; **feature-adaptive size**: smaller elements along creases inside the solve, same face count | `src/field.cpp`, `src/remesh.cpp` |
| 5. integer grid + extraction | multiresolution orientation / position fields, min-cost-flow singularity optimisation, quad extraction (vendored, patched QuadriFlow); per connected component with coverage-checked retries for thin parts | `third_party/qflow_patched` |
| 5b. fallbacks | Instant Meshes extractor (second choice; best-of-two on dense scans), best approximate result for parts thinner than an element, QEM-decimated exact fallback | `src/im_driver.cpp`, `src/decimate.cpp` |
| 6. post-process | cleanup, triangle-pair merging, optional pure-quad split, feature/boundary/corner snapping, **feature edge-path recovery** (Dijkstra edge paths snapped onto every crease chain), optional feature refinement, guarded tangential relaxation + reprojection, untangling | `src/postprocess.cpp` |
| 6b. coverage stage | budgeted operations that put edges on uncovered creases: **near-miss snapping** (default), path-edge subdivision, loops, local refinement, curved splits, crease cuts (opt-in) | `src/remesh.cpp`, `src/crease_cut.cpp` |
| 7. symmetry | remesh one half (clipped at the plane, plane boundary preserved), mirror and weld | `src/preprocess.cpp`, `src/postprocess.cpp` |
| 8. attributes | per-corner UV transfer with UV-chart consistency; generic per-vertex attribute transfer | `src/transfer.cpp` |
| 9. report | quad ratio, scaled Jacobian, planarity, aspect, distance (mean / RMS / Hausdorff, two-sided), feature & boundary coverage, singularities / valence histogram, topology | `src/metrics.cpp` |

Deterministic: single-threaded solver, seeded PCG RNG, sort-based topology. The test suite checks
bit-identical output for repeated runs.

## Results (15-model benchmark, equal target face count)

<!-- SUMMARY:BEGIN -->
<!-- SUMMARY:END -->

Two settings are reported. The **default** keeps every other criterion on every model (the
room scan's singularities excepted, see the write-up). **Max coverage** -
`--coverage-mode --path-subdiv 3 --sing-budget-im --coverage-search`
(Python: `coverage_mode=True, path_subdivision=3, sing_budget_im=True, coverage_search=True`) -
trades runtime (4 solves) and some element quality for feature coverage, while keeping quads
≥ 98 %, SJ ≥ 0.8, distance < 0.5 % and the singularity count below Instant Meshes'.

Feature coverage per model:

<!-- PERMODEL:BEGIN -->
<!-- PERMODEL:END -->

The ≥ 90 % coverage target is not met on the mean. Every coverage experiment (22 iterations,
including the failed ones) is in the generated study [`docs/coverage_study.md`](docs/coverage_study.md)
(`python benchmark/coverage_study.py run NAME "FLAGS"` + `report`), and the geometric limits of the
metric - why the room scan cannot reach 90 % at 15k faces and why the organic scans need edges
shorter than the element size - are measured in [`docs/crease_bounds.md`](docs/crease_bounds.md)
(`python benchmark/crease_bounds.py`). `docs/technical_writeup.pdf` §6.1 has the analysis.

## Build

Requirements: CMake ≥ 3.16, a C++17 compiler, Boost headers (for QuadriFlow's lemon flow
solver), Python ≥ 3.9 with numpy for the bindings.

```
./scripts/fetch_deps.sh            # eigen, libigl, pybind11, cgltf, upstream QuadriFlow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cp build/_quadremesh*.so python/quadremesh/      # or: pip install -e .
ctest --test-dir build                            # C++ tests
PYTHONPATH=python python -m pytest tests          # Python tests
```

## CLI

```
quadremesh INPUT(.obj|.ply|.glb|.gltf|.off) -o OUT.obj [options]
  -f/--faces N  -e/--edge-length X  -a/--feature-angle DEG  --min-feature-length K
  --feature-density K [--feature-density-width W]  --snap-near D  --path-subdiv K  --sing-budget-im
  --coverage-search  --legacy  --crease-cuts
  --coverage-mode  --feature-refine N  --feature-loops  --partial-loops  --curved-splits
  --trace-all  --tri-budget F  --sing-budget S  --refine-max-len K  --no-face-match
  --no-boundary  --curvature-weight W  --strokes FILE [--strokes-as-edges]
  --guide FILE [--guide-strength W]  --density FILE  --density-region X Y Z R F
  --curvature-density K  --symmetry x|y|z [--symmetry-offset C]
  --pure-quad  --smooth N  --no-uv  --report FILE.json  --field FILE  --seed S  -v

quadeval INPUT_TRI OUT_QUAD.obj [--target N] [--feature-angle DEG] [-o report.json]
```

Strokes file: one `x y z` point per line, blank line between strokes.

## Python API

`quadremesh(input, target_faces=5000, feature_angle=35, *, output=None, report=True,
target_edge_length=None, strokes=None, strokes_as_edges=False, density=None,
density_regions=None, curvature_density=0, symmetry_axis=None, guide_field=None,
guide_weight=None, learned_field=None, pure_quad=False, seed=0, **core_options)`
→ `QuadRemeshResult(vertices, faces, uv, report, timing, log, field, feature_edges)`.

`input` may be a path, a `(V, F)` tuple or any object with `.vertices/.faces` (e.g. trimesh).
`evaluate(input, quad)` scores any quad mesh with the same metrics.

Optional PyTorch modules (CUDA used automatically when available):

* `quadremesh.learned` - frame-equivariant graph network predicting a cross field + confidence
  from noisy geometry (`learned_field=True`); `python -m quadremesh.learned train|eval`.
* `quadremesh.torch_field.optimize_field` - sparse CG cross-field solver (smoothness +
  curvature alignment + hard feature constraints) on CPU or GPU, output usable as `guide_field`.

## Benchmark & report

```
python data/prepare_benchmark.py                 # build data/models from the downloads
python scripts/prepare_lounge.py <lounge.ply> data/models/lounge_room.obj
python benchmark/run_benchmark.py                # all methods, all models -> benchmark/results
python benchmark/make_report.py                  # figures + docs/report/visual_report.html
```

See `docs/technical_writeup.md` for design choices, results, failure cases and future work.

## Layout

```
include/quadremesh/   public headers           src/           core implementation
apps/                 CLI tools                python/        bindings + Python package
third_party/qflow_patched/  vendored QuadriFlow + PATCHES.md
benchmark/            runner, Blender script, report builder, results
data/                 benchmark/training model preparation + manifest
docs/                 write-up and visual report       tests/  C++ and Python tests
```

## Licenses

quadremesh code: MIT. Vendored QuadriFlow: its own license (`third_party/qflow_patched/LICENSE.txt`). Instant Meshes (fetched by `scripts/fetch_deps.sh`): BSD.
Benchmark models keep their original licenses (Stanford, Khronos sample assets, Open3D datasets).
