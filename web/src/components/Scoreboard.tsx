import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import { motion } from "motion/react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { pretty, summarize, type Model } from "@/data/benchmark";

function Crit({ title, target, value, sub, pass, i }: { title: string; target: string; value: string; sub: string; pass: boolean; i: number }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.07 }}>
      <Card className="h-full p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="text-xs text-muted-foreground">{title}</div>
          {pass ? <CheckCircle2 className="size-4 text-emerald-400" /> : <XCircle className="size-4 text-rose-400" />}
        </div>
        <div className="mt-2 text-2xl font-semibold tabular-nums">{value}</div>
        <div className="mt-1 text-xs text-muted-foreground">target {target} · {sub}</div>
      </Card>
    </motion.div>
  );
}

export function Scoreboard({ models, onPick }: { models: Model[]; onPick: (name: string) => void }) {
  const d = summarize(models, "ours");
  const c = summarize(models, "ours_coverage");
  const im = summarize(models, "instant_meshes");
  const cov = models
    .map((m) => ({ name: m.name, v: m.methods.ours_coverage?.feature_coverage ?? 0 }))
    .sort((a, b) => b.v - a.v);
  const scans = models.filter((m) => /scene scan|room-scale scan/.test(m.category));
  const ceiling = (models.length - scans.length + scans.reduce((a, m) => a + (m.methods.ours_coverage?.feature_coverage ?? 0), 0)) / models.length;
  const tiers = [
    { label: "≥ 90 %", items: cov.filter((x) => x.v >= 0.9), variant: "success" as const },
    { label: "75 – 90 %", items: cov.filter((x) => x.v >= 0.75 && x.v < 0.9), variant: "warning" as const },
    { label: "< 75 %", items: cov.filter((x) => x.v < 0.75), variant: "danger" as const },
  ];
  return (
    <div className="grid gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Crit i={0} title="Quads" target="≥ 98 %" value={`${(d.quad * 100).toFixed(1)}%`} sub={`max coverage ${(c.quad * 100).toFixed(1)}%`} pass={d.quad >= 0.98 && c.quad >= 0.98} />
        <Crit i={1} title="Mean scaled Jacobian" target="≥ 0.8" value={d.sj.toFixed(3)} sub={`every model ≥ ${d.sjMinModel.toFixed(2)}`} pass={d.sj >= 0.8} />
        <Crit i={2} title="Mean distance error" target="< 0.5 %" value={`${d.dist.toFixed(3)}%`} sub={`Instant Meshes ${im.dist.toFixed(3)}%`} pass={d.dist < 0.5} />
        <Crit i={3} title="Fewer singularities than IM" target="equal faces" value={`${d.singWins} / ${d.total}`} sub={`${d.singOurs.toLocaleString()} vs ${d.singIM.toLocaleString()} total`} pass={d.singWins > d.total / 2} />
      </div>
      <Card className="p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="text-xs text-muted-foreground">Feature coverage per model · target ≥ 90 % · max-coverage setting</div>
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground"><CircleDashed className="size-3.5" /> click a model to load it</div>
        </div>
        <div className="mt-3 grid gap-3">
          {tiers.map((t) => (
            <div key={t.label} className="flex flex-wrap items-center gap-2">
              <Badge variant={t.variant} className="w-24 justify-center">{t.label} · {t.items.length}</Badge>
              {t.items.map((x) => (
                <button key={x.name} onClick={() => onPick(x.name)} className="rounded-md border border-border bg-muted/40 px-2 py-1 text-xs hover:border-primary hover:text-primary">
                  {pretty(x.name)} <span className="font-mono text-muted-foreground">{(x.v * 100).toFixed(1)}%</span>
                </button>
              ))}
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
          Mean over all {c.total}: <span className="font-mono text-foreground">{(c.cov * 100).toFixed(1)}%</span> ({(c.covNonScan * 100).toFixed(1)}% without the two room scans),
          vs Instant Meshes <span className="font-mono">{(im.cov * 100).toFixed(1)}%</span>. The room scans' 35° “creases” are scanner noise: the lounge would need more
          crease-aligned edges than its mesh has, so even 100 % on the other {models.length - scans.length} models gives a mean of {(ceiling * 100).toFixed(1)} % (docs/crease_bounds.md).
        </p>
      </Card>
    </div>
  );
}
