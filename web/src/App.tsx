import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { BookOpen, Github, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Pipeline } from "@/components/Pipeline";
import { StageDetail } from "@/components/StageDetail";
import { Scoreboard } from "@/components/Scoreboard";
import { Explorer } from "@/components/Explorer";
import { STAGES } from "@/data/stages";
import { pretty, useBenchmark } from "@/data/benchmark";
import { cn } from "@/lib/utils";

const REPO = "https://github.com/aasim-syed/researchstudy";
const STEP_MS = 4200;

function Section({ id, kicker, title, sub, children }: { id: string; kicker: string; title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="min-w-0 scroll-mt-20">
      <div className="mb-4">
        <div className="font-mono text-xs uppercase tracking-[0.16em] text-primary">{kicker}</div>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
        {sub && <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{sub}</p>}
      </div>
      {children}
    </section>
  );
}

export default function App() {
  const { data, error } = useBenchmark();
  const [modelName, setModelName] = useState("fandisk");
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => setActive((a) => (a + 1) % STAGES.length), STEP_MS);
    return () => clearTimeout(t);
  }, [playing, active]);

  const model = data?.models.find((m) => m.name === modelName) ?? data?.models[0];
  const pick = (n: string) => {
    setModelName(n);
    document.getElementById("pipeline")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-border/70 bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3">
          <a href="#" className="flex items-center gap-2 font-semibold">
            <span className="grid size-7 place-items-center rounded-md bg-primary/15 text-primary">
              <svg viewBox="0 0 16 16" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 2h12v12H2zM8 2v12M2 8h12" /></svg>
            </span>
            quadremesh
          </a>
          <nav className="hidden gap-5 text-sm text-muted-foreground md:flex">
            <a href="#results" className="hover:text-foreground">Results</a>
            <a href="#pipeline" className="hover:text-foreground">Pipeline</a>
            <a href="#output" className="hover:text-foreground">Output & compare</a>
          </nav>
          <div className="flex gap-2">
            <Button asChild variant="outline" size="sm"><a href={`${REPO}/blob/main/docs/technical_writeup.pdf`} target="_blank" rel="noreferrer"><BookOpen /> Write-up</a></Button>
            <Button asChild size="sm"><a href={REPO} target="_blank" rel="noreferrer"><Github /> Code</a></Button>
          </div>
        </div>
      </header>

      <div className="grid-bg">
        <div className="mx-auto max-w-7xl px-4 pb-10 pt-14">
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="font-mono text-xs uppercase tracking-[0.18em] text-primary">quality-driven remeshing · C++17 · Python · PyTorch</div>
            <h1 className="mt-3 max-w-4xl text-4xl font-bold tracking-tight sm:text-5xl">
              Triangles in, <span className="bg-gradient-to-r from-sky-400 to-emerald-300 bg-clip-text text-transparent">structured quads</span> out.
            </h1>
            <p className="mt-4 max-w-2xl text-muted-foreground">
              Follow one mesh through every stage of the pipeline, see what each stage changes, then rotate, compare and download the result against Instant Meshes,
              QuadriFlow and Blender. Every number on this page is read from the evaluator reports in the repository.
            </p>
          </motion.div>
        </div>
      </div>

      <main className="mx-auto grid max-w-7xl grid-cols-[minmax(0,1fr)] gap-14 px-4 pb-24">
        {error && <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-300">Could not load data/benchmark.json: {error}</div>}
        {!data && !error && <div className="grid h-60 place-items-center text-muted-foreground"><Loader2 className="size-6 animate-spin" /></div>}
        {data && model && (
          <>
            <Section id="results" kicker="success criteria" title="Results across the 15-model benchmark" sub="Defaults for the first four criteria; feature coverage with the max-coverage setting. Instant Meshes at the same face count is the reference.">
              <Scoreboard models={data.models} onPick={pick} />
            </Section>

            <Section id="pipeline" kicker="end-to-end" title="The pipeline, stage by stage" sub="The highlighted stage glows; press play to run the flow, or click any stage. Images and numbers follow the selected model.">
              <div className="mb-4 flex flex-wrap gap-1.5">
                {data.models.map((m) => (
                  <button
                    key={m.name}
                    onClick={() => setModelName(m.name)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs transition-colors",
                      m.name === model.name ? "border-primary bg-primary/15 text-primary" : "border-border text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {pretty(m.name)}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-[minmax(0,1fr)] gap-4">
                <Pipeline active={active} setActive={setActive} playing={playing} setPlaying={setPlaying} />
                <StageDetail stage={STAGES[active]} model={model} data={data} />
              </div>
            </Section>

            <Section id="output" kicker="output" title={`${pretty(model.name)}: rotate, compare, download`} sub="Pick a method to load its quad mesh. Downloads are the exact OBJ files that were scored.">
              <Explorer key={model.name} model={model} />
            </Section>
          </>
        )}
      </main>

      <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
        Data: benchmark/results/summary.json and docs/report, built by web/scripts/build_data.py · <a className="underline" href={REPO}>{REPO.replace("https://", "")}</a>
      </footer>
    </div>
  );
}
