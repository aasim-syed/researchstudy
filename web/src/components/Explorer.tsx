import { useState } from "react";
import { Download, FileJson, ImageDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { METHOD_LABEL, METHODS, pretty, type MethodKey, type Metrics, type Model } from "@/data/benchmark";
import { MeshViewer } from "./MeshViewer";

type Row = { key: keyof Metrics; label: string; fmt: (v: number) => string; better: "max" | "min" | "none" };
const ROWS: Row[] = [
  { key: "faces", label: "faces", fmt: (v) => v.toLocaleString(), better: "none" },
  { key: "quad_ratio", label: "quads", fmt: (v) => `${(v * 100).toFixed(1)}%`, better: "max" },
  { key: "sj_mean", label: "mean scaled Jacobian", fmt: (v) => v.toFixed(3), better: "max" },
  { key: "dist_mean_pct", label: "mean distance (% diag)", fmt: (v) => v.toFixed(3), better: "min" },
  { key: "feature_coverage", label: "feature coverage", fmt: (v) => `${(v * 100).toFixed(1)}%`, better: "max" },
  { key: "singularities", label: "singularities", fmt: (v) => String(v), better: "min" },
  { key: "runtime_s", label: "runtime (s)", fmt: (v) => v.toFixed(1), better: "min" },
];

export function Explorer({ model }: { model: Model }) {
  const [method, setMethod] = useState<MethodKey>("ours");
  const okMethods = METHODS.filter((m) => model.methods[m]?.status === "ok");
  const cur = model.methods[method];
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
      <Card className="flex flex-col gap-3 p-4">
        <Tabs value={method} onValueChange={(v) => setMethod(v as MethodKey)}>
          <TabsList>
            {METHODS.map((m) => (
              <TabsTrigger key={m} value={m} disabled={!okMethods.includes(m)}>{METHOD_LABEL[m]}</TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <MeshViewer url={cur?.status === "ok" ? cur.mesh : undefined} className="h-[440px]" />
        <div className="flex flex-wrap gap-2">
          {cur?.mesh && (
            <Button asChild size="sm"><a href={cur.mesh} download><Download /> {METHOD_LABEL[method]} · .obj</a></Button>
          )}
          {cur?.report && (
            <Button asChild size="sm" variant="outline"><a href={cur.report} download><FileJson /> evaluator report</a></Button>
          )}
          {model.images.compare && (
            <Button asChild size="sm" variant="outline"><a href={model.images.compare} download><ImageDown /> comparison image</a></Button>
          )}
        </div>
      </Card>
      <Card className="overflow-hidden">
        <div className="border-b border-border p-4">
          <div className="text-sm font-semibold">{pretty(model.name)} · all methods</div>
          <div className="text-xs text-muted-foreground">{model.category} · target {model.target.toLocaleString()} faces · best value per row highlighted</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="p-2 text-left font-medium" />
                {METHODS.map((m) => <th key={m} className={"p-2 text-right font-medium " + (m === method ? "text-primary" : "")}>{METHOD_LABEL[m].replace("quadremesh · ", "")}</th>)}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => {
                const vals = METHODS.map((m) => (model.methods[m]?.status === "ok" ? (model.methods[m]![r.key] as number | null) : null));
                const present = vals.filter((v): v is number => v != null);
                const best = r.better === "max" ? Math.max(...present) : r.better === "min" ? Math.min(...present) : NaN;
                return (
                  <tr key={r.key} className="border-t border-border">
                    <td className="p-2 text-muted-foreground">{r.label}</td>
                    {vals.map((v, i) => (
                      <td key={i} className={"p-2 text-right font-mono " + (v != null && v === best ? "font-semibold text-emerald-400" : "")}>
                        {v == null ? <span className="text-muted-foreground">failed</span> : r.fmt(v)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
      {model.images.compare && (
        <Card className="p-3 lg:col-span-2">
          <img src={model.images.compare} alt={`${model.name}: all methods side by side`} className="w-full rounded-md bg-white" loading="lazy" />
        </Card>
      )}
    </div>
  );
}
