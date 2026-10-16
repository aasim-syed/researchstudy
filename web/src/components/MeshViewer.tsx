import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Loader2 } from "lucide-react";

interface Parsed {
  positions: Float32Array;
  tris: Uint32Array;
  edges: Float32Array;
  val3: Float32Array;
  val5: Float32Array;
  faces: number;
  quads: number;
}

const cache = new Map<string, Parsed>();

/** Parse an OBJ with polygon faces: triangulated surface, unique polygon edges and singular vertices. */
function parseObj(text: string): Parsed {
  const v: number[] = [];
  const faces: number[][] = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("v ")) {
      const p = line.trim().split(/\s+/);
      v.push(+p[1], +p[2], +p[3]);
    } else if (line.startsWith("f ")) {
      faces.push(line.trim().split(/\s+/).slice(1).map((t) => parseInt(t.split("/")[0], 10) - 1));
    }
  }
  const tris: number[] = [];
  const edgeFaces = new Map<string, number>();
  const edgeList: [number, number][] = [];
  let quads = 0;
  for (const f of faces) {
    if (f.length === 4) quads++;
    for (let i = 1; i + 1 < f.length; i++) tris.push(f[0], f[i], f[i + 1]);
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const k = a < b ? `${a}_${b}` : `${b}_${a}`;
      const c = edgeFaces.get(k);
      if (c === undefined) { edgeFaces.set(k, 1); edgeList.push([a, b]); } else edgeFaces.set(k, c + 1);
    }
  }
  const nv = v.length / 3;
  const deg = new Int32Array(nv);
  const onBoundary = new Uint8Array(nv);
  const edges = new Float32Array(edgeList.length * 6);
  edgeList.forEach(([a, b], i) => {
    deg[a]++; deg[b]++;
    const k = a < b ? `${a}_${b}` : `${b}_${a}`;
    if (edgeFaces.get(k) === 1) { onBoundary[a] = 1; onBoundary[b] = 1; }
    edges.set([v[3 * a], v[3 * a + 1], v[3 * a + 2], v[3 * b], v[3 * b + 1], v[3 * b + 2]], i * 6);
  });
  const s3: number[] = [], s5: number[] = [];
  for (let i = 0; i < nv; i++) {
    if (onBoundary[i] || deg[i] === 0 || deg[i] === 4) continue;
    (deg[i] < 4 ? s3 : s5).push(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
  }
  return {
    positions: new Float32Array(v), tris: new Uint32Array(tris), edges,
    val3: new Float32Array(s3), val5: new Float32Array(s5), faces: faces.length, quads,
  };
}

export function MeshViewer({ url, className }: { url?: string; className?: string }) {
  const host = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"loading" | "ready" | "error" | "none">("loading");
  const [info, setInfo] = useState<{ faces: number; quads: number; s3: number; s5: number } | null>(null);
  const [showSing, setShowSing] = useState(true);
  const singRef = useRef<THREE.Group | null>(null);

  useEffect(() => {
    if (singRef.current) singRef.current.visible = showSing;
  }, [showSing]);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    if (!url) { setState("none"); return; }
    let disposed = false;
    setState("loading");

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.8;
    controls.addEventListener("start", () => (controls.autoRotate = false));
    scene.add(new THREE.HemisphereLight(0xdbeafe, 0x0b1220, 1.6));
    const dl = new THREE.DirectionalLight(0xffffff, 1.4);
    dl.position.set(2, 3, 4);
    camera.add(dl);
    scene.add(camera);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let raf = 0;
    const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop); };
    loop();

    const build = (p: Parsed) => {
      if (disposed) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(p.positions, 3));
      g.setIndex(new THREE.BufferAttribute(p.tris, 1));
      g.computeVertexNormals();
      g.computeBoundingSphere();
      const bs = g.boundingSphere!;
      const surf = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        color: 0x94a3b8, roughness: 0.55, metalness: 0.05, flatShading: true,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, side: THREE.DoubleSide,
      }));
      const eg = new THREE.BufferGeometry();
      eg.setAttribute("position", new THREE.BufferAttribute(p.edges, 3));
      const wire = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x0b1220, transparent: true, opacity: 0.55 }));
      const root = new THREE.Group();
      root.add(surf, wire);
      const sing = new THREE.Group();
      const dot = (arr: Float32Array, color: number) => {
        const gg = new THREE.BufferGeometry();
        gg.setAttribute("position", new THREE.BufferAttribute(arr, 3));
        return new THREE.Points(gg, new THREE.PointsMaterial({ color, size: bs.radius * 0.04, sizeAttenuation: true }));
      };
      sing.add(dot(p.val3, 0x3b82f6), dot(p.val5, 0xef4444));
      sing.visible = showSing;
      singRef.current = sing;
      root.add(sing);
      root.position.sub(bs.center);
      scene.add(root);
      camera.position.set(bs.radius * 1.6, bs.radius * 1.1, bs.radius * 2.4);
      camera.near = bs.radius / 100;
      camera.far = bs.radius * 20;
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      setInfo({ faces: p.faces, quads: p.quads, s3: p.val3.length / 3, s5: p.val5.length / 3 });
      setState("ready");
    };

    const cached = cache.get(url);
    if (cached) build(cached);
    else
      fetch(url)
        .then((r) => (r.ok ? r.text() : Promise.reject(r.statusText)))
        .then((t) => { const p = parseObj(t); cache.set(url, p); build(p); })
        .catch(() => !disposed && setState("error"));

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | undefined;
        mat?.dispose?.();
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  return (
    <div className={"relative overflow-hidden rounded-lg border border-border bg-gradient-to-b from-slate-900/60 to-slate-950 " + (className ?? "")}>
      <div ref={host} className="absolute inset-0" />
      {state === "loading" && (
        <div className="absolute inset-0 grid place-items-center text-muted-foreground"><Loader2 className="size-6 animate-spin" /></div>
      )}
      {state === "none" && <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">No output for this method on this model.</div>}
      {state === "error" && <div className="absolute inset-0 grid place-items-center text-sm text-rose-400">Could not load the mesh.</div>}
      {info && state === "ready" && (
        <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2 font-mono text-[11px]">
          <span className="rounded bg-black/50 px-2 py-1">{info.faces.toLocaleString()} faces · {((info.quads / info.faces) * 100).toFixed(1)}% quads</span>
          <span className="rounded bg-black/50 px-2 py-1"><span className="text-blue-400">●</span> valence 3 <span className="ml-1 text-red-400">●</span> valence 5+</span>
        </div>
      )}
      {state === "ready" && (
        <button
          onClick={() => setShowSing((s) => !s)}
          className="absolute bottom-3 right-3 rounded-md bg-black/50 px-2 py-1 text-[11px] text-slate-200 hover:bg-black/70"
        >
          {showSing ? "hide" : "show"} singularities
        </button>
      )}
    </div>
  );
}
