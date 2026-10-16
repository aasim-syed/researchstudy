# CUDA checklist (GTX 1650, Windows + WSL2)

Goal: confirm the optional PyTorch path (`torch_field.optimize_field`, `learned.predict_guide_field`)
runs on the GPU, gives the same field as on the CPU, and measure the speed-up.
Run each step and paste the outputs of steps 3 and 6 back for review.

## 1. Windows side (once)
- [ ] Install the current NVIDIA **Game Ready / Studio driver** for the GTX 1650 on Windows.
      Do **not** install a Linux driver inside WSL.
- [ ] In PowerShell: `wsl --install -d Ubuntu` (skip if Ubuntu already exists), then `wsl --update`.

## 2. Inside Ubuntu (WSL2): build the project
Follow `CHECKPOINT.md` section 2 (apt packages, clone, `fetch_deps.sh`, cmake build, copy the
`.so`, extract the benchmark models from the `data` branch).

## 3. GPU visible to Linux - paste this output
```bash
nvidia-smi
```
- [ ] Shows `NVIDIA GeForce GTX 1650` and a driver version. If the command is missing or errors,
      the Windows driver / WSL2 step is wrong; stop and paste the error.

## 4. PyTorch with CUDA
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy trimesh
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- [ ] Prints `... True NVIDIA GeForce GTX 1650`. The GTX 1650 is compute capability 7.5 (sm_75),
      which the cu126 wheels support.

## 5. Project tests still pass
```bash
build/test_core data/models
PYTHONPATH=python python -m pytest -q tests
```
- [ ] `all tests passed` and `6 passed`.

## 6. CUDA check - paste the whole output
```bash
PYTHONPATH=python python scripts/cuda_check.py --runs 5
```
Expected:
- [ ] `[1]` shows the GPU, `cuda available True`, `sm_75`.
- [ ] `[2]` field angle CPU vs CUDA: p99 below 1 degree (float rounding only).
- [ ] `[3]` CPU and CUDA timings and a speed-up figure (on a 13K-triangle model the GPU may be only
      modestly faster; try `--model data/models/igea.obj` for a 269K-triangle input).
- [ ] `[4]` end-to-end remesh with the CUDA field: ~5000 faces, 100 % quads.
- [ ] Last line `VERDICT: PASS`.

## 7. After review
The measured speed-up replaces the "exercised only on CPU" sentence in
`docs/technical_writeup.md` section 3; then `python docs/build_pdf.py`, commit, push.

## Common problems
| symptom | fix |
|---|---|
| `nvidia-smi: command not found` in WSL | update the Windows NVIDIA driver; run `wsl --update`; restart WSL (`wsl --shutdown`) |
| `cuda available False` | the CPU wheel was installed: `pip uninstall torch` then reinstall with the cu126 index URL |
| `CUDA out of memory` (4 GB card) | use a smaller model (`--model data/models/fandisk.obj`) |
| `no kernel image is available` | torch build without sm_75: reinstall from the cu126 index |
