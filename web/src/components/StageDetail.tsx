import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, FileCode2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { GROUP_STYLE, type Stage, type Visual } from "@/data/stages";
import { pretty, type Benchmark, type Model } from "@/data/benchmark";
import { MeshViewer } from "./MeshViewer";
import { ICONS } from "./Pipeline";

const pct = (x?: number | null, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const num = (x?: number | null, d = 3) => (x == null ? "–" : x.toFixed(d));

function Img({ src, alt }: { src?: string; alt: string }) {
  if (!src) return <div className="grid h-64 place-items-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">No image for this model.</div>;
  return <img src={src} alt={alt} className="max-h-[420px] w-full rounded-lg bg-white object-contain" loading="lazy" />;
}

function InputStats({ model }: { model: Model }) {
  const out = model.methods.ours?.faces ?? model.target;
  const ratio = model.input_faces / Math.max(1, out);
  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-2 gap-3">
        {[
          ["model", pretty(model.name)],
          ["category", model.category],
          ["input triangles", model.input_faces.toLocaleString()],
          ["target quads", model.target.toLocaleString()],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-border bg-muted/40 p-3">
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{k}</div>
            <div className="mt-1 truncate text-lg font-semibold">{v}</div>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-border p-3 text-sm text-muted-foreground">
        <div className="mb-2 flex justify-between"><span>triangles in</span><span>quads out</span></div>
        <div className="relative h-3 overflow-hidden rounded bg-muted">
          <motion.div className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-400 to-sky-400" initial={{ width: "100%" }} animate={{ width: `${Math.max(3, 100 / ratio)}%` }} transition={{ duration: 1.2 }} />
        </div>
        <div className="mt-2 font-mono text-xs">{model.input_faces.toLocaleString()} → {out.toLocaleString()} faces ({ratio.toFixed(1)}× fewer)</div>
      </div>
    </div>
  );
}

function DefaultVsCoverage({ model }: { model: Model }) {
  const a = model.methods.ours, b = model.methods.ours_coverage;
  const rows: [string, string, string, boolean][] = [
    ["feature coverage", pct(a?.feature_coverage), pct(b?.feature_coverage), (b?.feature_coverage ?? 0) > (a?.feature_coverage ?? 0)],
    ["quads", pct(a?.quad_ratio), pct(b?.quad_ratio), false],
    ["mean scaled Jacobian", num(a?.sj_mean), num(b?.sj_mean), false],
    ["mean distance", `${num(a?.dist_mean_pct)}%`, `${num(b?.dist_mean_pct)}%`, false],
    ["singularities", String(a?.singularities ?? "–"), String(b?.singularities ?? "–"), false],
    ["faces", String(a?.faces ?? "–"), String(b?.faces ?? "–"), false],
  ];
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/60 text-xs text-muted-foreground">
          <tr><th className="p-2 text-left font-medium">{pretty(model.name)}</th><th className="p-2 text-right font-medium">default</th><th className="p-2 text-right font-medium">max coverage</th></tr>
        </thead>
        <tbody>
          {rows.map(([k, x, y, up]) => (
            <tr key={k} className="border-t border-border">
              <td className="p-2 text-muted-foreground">{k}</td>
              <td className="p-2 text-right font-mono">{x}</td>
              <td className={"p-2 text-right font-mono " + (up ? "text-emerald-400" : "")}>{y}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VisualFor({ v, model, data }: { v: Visual; model: Model; data: Benchmark }) {
  switch (v) {
    case "input": return <InputStats model={model} />;
    case "edges": case "field": case "sj": case "compare":
      return <Img src={model.images[v]} alt={`${model.name} ${v}`} />;
    case "density": return <Img src={data.demos.demo_density} alt="adaptive density demo" />;
    case "uv": return <Img src={data.demos.demo_uv_purequad} alt="UV transfer demo" />;
    case "viewer": return <MeshViewer url={model.methods.ours?.mesh} className="h-[380px]" />;
    case "metrics": return <DefaultVsCoverage model={model} />;
  }
}

export function StageDetail({ stage, model, data }: { stage: Stage; model: Model; data: Benchmark }) {
  const g = GROUP_STYLE[stage.group];
  const Icon = ICONS[stage.icon];
  return (
    <Card className="relative overflow-hidden">
      <div className="pointer-events-none absolute -top-24 left-1/3 h-48 w-2/3 rounded-full blur-3xl" style={{ background: g.color + "22" }} />
      <AnimatePresence mode="wait">
        <motion.div
          key={stage.id + model.name}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.28 }}
          className="relative grid gap-6 p-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]"
        >
          <div className="flex min-w-0 flex-col gap-4">
            <div className="flex items-center gap-3">
              <div className="grid size-10 place-items-center rounded-lg" style={{ background: g.color + "22", color: g.color }}><Icon className="size-5" /></div>
              <div>
                <div className="font-mono text-[11px] uppercase tracking-wider" style={{ color: g.color }}>step {stage.n} · {g.label}</div>
                <h3 className="text-xl font-semibold">{stage.title}</h3>
              </div>
            </div>
            <p className="text-sm leading-relaxed text-slate-300">{stage.what}</p>
            <div className="grid grid-cols-[1fr_auto_1fr] items-stretch gap-2 text-xs">
              <div className="rounded-lg border border-border bg-muted/40 p-2.5"><div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">input</div>{stage.input}</div>
              <ArrowRight className="self-center text-muted-foreground" />
              <div className="rounded-lg border p-2.5" style={{ borderColor: g.color + "66", background: g.color + "10" }}><div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">output</div>{stage.output}</div>
            </div>
            <div className="rounded-lg border border-border bg-slate-950 p-3 font-mono text-[11.5px] leading-relaxed">
              <div className="mb-1 flex items-center gap-1.5 text-muted-foreground"><FileCode2 className="size-3.5" />{stage.file}</div>
              <code className="break-words text-sky-300">{stage.signature}</code>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {stage.features.map((f, i) => (
                <motion.span key={f} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 + i * 0.08 }}>
                  <Badge variant="outline" className="border-border bg-muted/40 py-1 text-slate-200">{f}</Badge>
                </motion.span>
              ))}
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-3">
            <VisualFor v={stage.visual} model={model} data={data} />
            <p className="text-xs text-muted-foreground">{stage.visualCaption}</p>
            {stage.extra && (
              <>
                <VisualFor v={stage.extra.visual} model={model} data={data} />
                <p className="text-xs text-muted-foreground">{stage.extra.caption}</p>
              </>
            )}
          </div>
        </motion.div>
      </AnimatePresence>
    </Card>
  );
}
