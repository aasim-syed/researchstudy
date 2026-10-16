import { useEffect, useRef } from "react";
import { motion } from "motion/react";
import {
  Box, Compass, FileInput, Gauge, Grid3x3, Layers, Pause, Play, Route, Search, ShieldCheck, SkipBack, SkipForward, Spline, Target, Wrench,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { GROUP_STYLE, STAGES, type Stage } from "@/data/stages";
import { cn } from "@/lib/utils";

export const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  FileInput, Wrench, Spline, Compass, Grid3x3, Route, Target, ShieldCheck, Search, Layers, Box, Gauge,
};

interface Props {
  active: number;
  setActive: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
}

function Node({ s, state, onClick }: { s: Stage; state: "done" | "active" | "todo"; onClick: () => void }) {
  const Icon = ICONS[s.icon];
  const color = GROUP_STYLE[s.group].color;
  return (
    <button onClick={onClick} className="group flex w-[92px] shrink-0 flex-col items-center gap-2 focus:outline-none" aria-current={state === "active"}>
      <div
        style={{ ["--glow" as string]: color + "aa", borderColor: state === "todo" ? undefined : color, color: state === "todo" ? undefined : color }}
        className={cn(
          "relative grid size-14 place-items-center rounded-xl border-2 border-border bg-card text-muted-foreground transition-all duration-300 group-hover:scale-105",
          state === "active" && "glow-active scale-110 bg-slate-900",
          state === "done" && "bg-slate-900/60",
        )}
      >
        <Icon className="size-6" />
        <span
          className={cn(
            "absolute -right-2 -top-2 grid size-5 place-items-center rounded-full font-mono text-[10px] font-semibold",
            state === "todo" ? "bg-muted text-muted-foreground" : "text-slate-950",
          )}
          style={state === "todo" ? undefined : { background: color }}
        >
          {s.n}
        </span>
      </div>
      <span className={cn("text-center text-[11px] leading-tight transition-colors", state === "active" ? "font-semibold text-foreground" : "text-muted-foreground")}>
        {s.short}
      </span>
    </button>
  );
}

function Connector({ lit, flowing, color }: { lit: boolean; flowing: boolean; color: string }) {
  return (
    <div className="relative mt-7 h-[2px] w-8 shrink-0 overflow-hidden rounded bg-border sm:w-10">
      <motion.div
        className="absolute inset-y-0 left-0"
        style={{ background: color }}
        initial={false}
        animate={{ width: lit ? "100%" : "0%" }}
        transition={{ duration: 0.45 }}
      />
      {flowing && <div className="flow-light absolute inset-y-0 w-1/3" style={{ background: `linear-gradient(90deg, transparent, ${color}, white, transparent)` }} />}
    </div>
  );
}

export function Pipeline({ active, setActive, playing, setPlaying }: Props) {
  const rowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const row = rowRef.current;
    const el = row?.querySelectorAll("button")[active] as HTMLElement | undefined;
    if (row && el) row.scrollTo({ left: el.offsetLeft - row.clientWidth / 2 + el.clientWidth / 2, behavior: "smooth" });
  }, [active]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
          {Object.values(GROUP_STYLE).map((g) => (
            <span key={g.label} className="inline-flex items-center gap-1.5">
              <span className="size-2.5 rounded-sm" style={{ background: g.color }} /> {g.label}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="icon" aria-label="Previous stage" onClick={() => { setPlaying(false); setActive(Math.max(0, active - 1)); }}><SkipBack /></Button>
          <Button size="sm" onClick={() => setPlaying(!playing)} className="w-24">
            {playing ? <><Pause /> Pause</> : <><Play /> Play</>}
          </Button>
          <Button variant="outline" size="icon" aria-label="Next stage" onClick={() => { setPlaying(false); setActive(Math.min(STAGES.length - 1, active + 1)); }}><SkipForward /></Button>
        </div>
      </div>
      <div ref={rowRef} className="no-scrollbar relative overflow-x-auto pb-3 pt-3">
        <div className="flex w-max items-start px-2">
          {STAGES.map((s, i) => (
            <div key={s.id} className="flex items-start">
              {i > 0 && <Connector lit={i <= active} flowing={i === active} color={GROUP_STYLE[s.group].color} />}
              <Node s={s} state={i < active ? "done" : i === active ? "active" : "todo"} onClick={() => { setPlaying(false); setActive(i); }} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
