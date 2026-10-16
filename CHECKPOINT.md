# Checkpoint: where the project stands and how to continue

Last updated: 16 Oct 2026, morning (IST). Submission deadline: EOD 16 Oct 2026, reply-all to the
Ctruh HR email. **No email has been sent yet - that is done by hand.**

## 1. State

| branch | what it holds |
|---|---|
| `main` | **the submission.** Code, benchmark results, coverage study it0-it24, write-up + PDF, README, code reference. |
| `search8` | experiment: coverage search with 8 candidates instead of 4 (study iteration it25). Merge into `main` only if every criterion still holds and it improves coverage. |
| `data` | `benchmark_models.tar.gz`: the 15 prepared benchmark models + `manifest.json` (46 MB). |
| `push90` | already merged into `main` (improved near-miss snapping). |

Final numbers on `main` (generated, see the tables at the top of `README.md`):

| | default | max coverage | Instant Meshes |
|---|---|---|---|
| quads | 100 % | 99.4 % | 100 % |
| mean scaled Jacobian (lowest model) | 0.903 (0.821) | 0.903 (0.821) | 0.935 |
| mean distance | 0.086 % | 0.082 % | 0.316 % |
| feature coverage, all 15 | 66.9 % | 73.1 % | 22.1 % |
| feature coverage, 13 non-scan | 75.9 % | 82.8 % | 25.3 % |
| fewer singularities than IM | 14 / 15 | 14 / 15 | - |

Feature coverage >= 90 % on 5 models (max coverage): fandisk 100 %, spot 99.4 %, sheen chair 92.9 %,
velvet sofa 92.4 %, leather sofa 91.5 %. Below: organic scans (zigzag creases finer than an
element), damask (dense carvings), room scans (scanner-noise creases). Limits measured in
`docs/crease_bounds.md`. Max coverage = `--coverage-mode --path-subdiv 3 --sing-budget-im --coverage-search`.

## 2. Set up on a laptop

The C++ build needs Linux tools; on Windows use **WSL2 (Ubuntu)**. A GTX 1650 works with CUDA inside
WSL2 once the normal NVIDIA Windows driver is installed (no driver inside WSL).

```bash
sudo apt install -y build-essential cmake git libboost-dev python3-pip python3-venv
git clone https://github.com/aasim-syed/researchstudy.git && cd researchstudy
./scripts/fetch_deps.sh                       # eigen, libigl, pybind11, cgltf, upstream QuadriFlow, Instant Meshes
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cp build/_quadremesh*.so python/quadremesh/

# benchmark models (from the data branch)
git fetch origin data
git show origin/data:benchmark_models.tar.gz > /tmp/bm.tar.gz && tar -xzf /tmp/bm.tar.gz -C data

python3 -m venv .venv && . .venv/bin/activate
pip install numpy trimesh matplotlib pillow markdown weasyprint
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build; works on a GTX 1650

build/test_core data/models                    # C++ tests -> "all tests passed"
PYTHONPATH=python python -m pytest -q tests     # Python tests
```

## 3. Everyday commands

```bash
# remesh one model (default / max coverage) and score it
build/quadremesh data/models/fandisk.obj -o out.obj -f 5000 -a 35 -v
build/quadremesh data/models/fandisk.obj -o out.obj -f 5000 -a 35 --coverage-mode --path-subdiv 3 --sing-budget-im --coverage-search
build/quadeval data/models/fandisk.obj out.obj --target 5000 --feature-angle 35 -o report.json

# coverage study: run an iteration, regenerate docs/coverage_study.md
python benchmark/coverage_study.py run NAME "FLAGS"      # results in benchmark/study/NAME/
python benchmark/coverage_study.py report

# main benchmark, visual report, generated tables, code reference, PDF
python benchmark/run_benchmark.py --methods ours           # or --import-study RUN:ours_coverage
python benchmark/make_report.py
python docs/gen_tables.py        # tables in README.md and docs/technical_writeup.md
python docs/gen_codemap.py       # docs/code_reference.md + write-up section 9
python docs/build_pdf.py         # docs/technical_writeup.pdf
python benchmark/crease_bounds.py   # docs/crease_bounds.md
```

The baselines (Instant Meshes via `pynanoinstantmeshes`, QuadriFlow CLI, Blender) are already
scored in `benchmark/results/reports/`; re-running them is only needed to change the protocol.

## 4. Open items

1. **it25 (8-candidate search) on `search8`.** It was running in the cloud session; partial
   results are committed on that branch. To finish or check:
   `git checkout search8 && python benchmark/coverage_study.py report` and compare it25 with it24
   in `docs/coverage_study.md`. A 4-model test gave +1 to +3 points on fandisk_noisy / rocker arm
   and nothing on bunny / beetle (about +0.3 points on the mean) at 2x runtime. Merge only if every
   criterion holds; then `run_benchmark.py --import-study it25_max_coverage_search8:ours_coverage`,
   `make_report.py`, `gen_tables.py`, `build_pdf.py`, and update the max-coverage flags in
   `README.md` / write-up / `run_benchmark.py` to include `--search-candidates 8`.
2. **CUDA check on the GTX 1650 (optional, about 15 min).** Follow `docs/cuda_checklist.md`;
   it ends with `PYTHONPATH=python python scripts/cuda_check.py --runs 5`, which prints one report
   (environment, CUDA vs CPU agreement, timings, end-to-end remesh, VERDICT). Paste it for review,
   then replace the "exercised only on CPU" sentence in the write-up section 3 with the speed-up.
3. **Send the submission** (by hand): reply-all to the Ctruh HR email with the repo link and
   `docs/technical_writeup.pdf`. Make sure the repo is public or shared with them.

## 5. Rules this project follows

* Every number in the docs comes from an evaluator report through a script (`gen_tables.py`,
  `coverage_study.py report`, `gen_codemap.py`); nothing is typed by hand.
* No model is dropped and the metric is not relaxed to improve a score; failed attempts stay in the study.
* Commits are authored as Syed Aasim <syedaasim133@gmail.com>.

## 6. Prompt to continue in a new Claude session

Paste this into Claude Code opened in the cloned repository:

```
You are continuing my Ctruh take-home assignment (triangle -> structured quad remeshing) in this
repository. Read CHECKPOINT.md first, then README.md and docs/technical_writeup.md section 6.1.
The submission is on main; do not break it. Rules: every number in the docs must come from the
evaluator through the scripts (gen_tables.py, coverage_study.py report); never drop models or
relax the metric; keep failed attempts in the study; commit as
"Syed Aasim <syedaasim133@gmail.com>". Do not send any email.
Tasks, in order: (1) set up the environment from CHECKPOINT.md section 2 and run the tests;
(2) open item 2 - verify the CUDA path on my GTX 1650 and record the GPU vs CPU timing in the
write-up section 3; (3) open item 1 - finish or evaluate it25 on the search8 branch and merge it
only if it improves coverage with every criterion holding; (4) regenerate tables, report and PDF,
commit and push. Show me the numbers before merging anything into main.
```
