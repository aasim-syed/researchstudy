import { useEffect, useState } from "react";

export type MethodKey = "ours" | "ours_coverage" | "instant_meshes" | "quadriflow" | "blender_quadriflow";

export interface Metrics {
  status: string;
  faces: number | null;
  quad_ratio: number | null;
  sj_mean: number | null;
  sj_min: number | null;
  dist_mean_pct: number | null;
  hausdorff_pct: number | null;
  feature_coverage: number | null;
  singularities: number | null;
  sing_per_1k: number | null;
  runtime_s: number | null;
  mesh?: string;
  report?: string;
}

export interface Model {
  name: string;
  category: string;
  input_faces: number;
  target: number;
  methods: Partial<Record<MethodKey, Metrics>>;
  images: Partial<Record<"compare" | "edges" | "sj" | "field", string>>;
}

export interface Benchmark {
  source: string;
  models: Model[];
  demos: Record<string, string>;
}

export const METHOD_LABEL: Record<MethodKey, string> = {
  ours: "quadremesh",
  ours_coverage: "quadremesh · max coverage",
  instant_meshes: "Instant Meshes",
  quadriflow: "QuadriFlow",
  blender_quadriflow: "Blender QuadriFlow",
};
export const METHODS: MethodKey[] = ["ours", "ours_coverage", "instant_meshes", "quadriflow", "blender_quadriflow"];

export const pretty = (s: string) => s.replace(/_/g, " ");

export function useBenchmark() {
  const [data, setData] = useState<Benchmark | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("data/benchmark.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);
  return { data, error };
}

const ok = (m?: Metrics) => !!m && m.status === "ok";
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);

/** Benchmark-wide numbers, computed in the browser from the evaluator reports. */
export function summarize(models: Model[], method: MethodKey) {
  const rows = models.map((m) => m.methods[method]).filter(ok) as Metrics[];
  const nonScan = models.filter((m) => !/scene scan|room-scale scan/.test(m.category));
  let singWins = 0;
  let singOurs = 0;
  let singIM = 0;
  for (const m of models) {
    const a = m.methods[method];
    const b = m.methods.instant_meshes;
    if (ok(a) && ok(b)) {
      if ((a!.singularities ?? 0) < (b!.singularities ?? 0)) singWins++;
      singOurs += a!.singularities ?? 0;
      singIM += b!.singularities ?? 0;
    }
  }
  return {
    processed: rows.length,
    total: models.length,
    quad: mean(rows.map((r) => r.quad_ratio ?? 0)),
    sj: mean(rows.map((r) => r.sj_mean ?? 0)),
    sjMinModel: Math.min(...rows.map((r) => r.sj_mean ?? 1)),
    dist: mean(rows.map((r) => r.dist_mean_pct ?? 0)),
    cov: mean(rows.map((r) => r.feature_coverage ?? 0)),
    covNonScan: mean(nonScan.map((m) => m.methods[method]).filter(ok).map((r) => r!.feature_coverage ?? 0)),
    singWins,
    singOurs,
    singIM,
  };
}
