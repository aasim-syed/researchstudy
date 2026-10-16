# quadremesh web: interactive pipeline and results

A React + TypeScript site (Vite, Tailwind, shadcn-style components, three.js) that walks through the
pipeline stage by stage, shows each stage on the selected benchmark model, and lets you rotate,
compare and download every method's quad mesh.

* `src/data/stages.ts` - the 12 stages: file, function signature, input, output, features.
* `scripts/build_data.py` - builds `public/data/` from `benchmark/results/summary.json`, the
  scored meshes, the evaluator reports and the visual-report images. Numbers are never typed by hand;
  the scoreboard means are computed in the browser from that file.

```bash
python web/scripts/build_data.py     # from the repo root, after benchmark/make_report.py
cd web && npm install && npm run dev  # http://localhost:5173
npm run build                         # static site in web/dist
```

## Deploy on Vercel

1. vercel.com -> Add New -> Project -> import `aasim-syed/researchstudy`.
2. Root Directory: `web` (framework preset Vite is detected; `vercel.json` sets build `npm run build`, output `dist`).
3. Deploy. Every push to `main` redeploys.

Or with the CLI: `cd web && npx vercel --prod`.
