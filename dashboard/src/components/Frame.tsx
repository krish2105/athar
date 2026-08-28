import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { animate } from "motion";
import { provenance } from "../data";

/** A figure with a title, an optional table view, and a caveat it cannot omit.
 *
 * The caveat is read from the backing artifact's provenance block. There is no
 * prop for it and no way to pass one in, which is the point: a chart drawn from
 * simulated data renders its badge or it does not render.
 */
export function Figure({
  title,
  artifactName,
  note,
  table,
  children,
}: {
  title: string;
  artifactName: string;
  note?: ReactNode;
  table?: ReactNode;
  children: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  const block = provenance(artifactName);

  return (
    <figure className="figure">
      <header>
        <h3>{title}</h3>
        <div style={{ display: "flex", gap: ".5rem", alignItems: "center" }}>
          {block && <ProvenanceBadge synthetic={block.synthetic} />}
          {table && (
            <button
              className="table-toggle"
              onClick={() => setShowTable((v) => !v)}
              aria-expanded={showTable}
            >
              {showTable ? "Chart" : "Table"}
            </button>
          )}
        </div>
      </header>
      {showTable && table ? <div className="scroll-x">{table}</div> : children}
      {note && <figcaption>{note}</figcaption>}
    </figure>
  );
}

export function ProvenanceBadge({ synthetic }: { synthetic: boolean }) {
  return (
    <span className={`badge ${synthetic ? "synthetic" : "real"}`}>
      <i className="dot" />
      {synthetic ? "Simulated" : "Real data"}
    </span>
  );
}

/** The synthetic caveat, rendered from the artifact rather than typed. */
export function Caveat({ artifactName }: { artifactName: string }) {
  const block = provenance(artifactName);
  if (!block?.synthetic || !block.caveat) return null;
  const [label, ...rest] = block.caveat.split(" — ");
  return (
    <p className="caveat">
      <strong>{label}</strong> — {rest.join(" — ")}
      {block.dgp_hash && (
        <>
          {" "}
          <span className="num" style={{ fontSize: ".82em", opacity: 0.75 }}>
            config {block.dgp_hash} · seed {block.seed}
          </span>
        </>
      )}
    </p>
  );
}

const reduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Reveal on scroll, implemented so that failure leaves the page readable.
 *
 * The hidden state lives in CSS behind an `html.motion-ready` class that
 * `enableMotion()` adds only after a frame has actually been painted. Nothing
 * here sets an inline opacity, so there is no override left behind if the
 * animation cannot run — which is what an earlier version did, and it stranded
 * the entire page at opacity zero in a context that delivers no frames.
 */
export function Reveal({ children, delay = 0 }: { children: ReactNode; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const show = () => node.setAttribute("data-shown", "");
    if (reduced() || !document.documentElement.classList.contains("motion-ready")) {
      show();
      return;
    }
    let done = false;
    const fire = () => {
      if (done) return;
      done = true;
      observer?.disconnect();
      clearTimeout(timer);
      show();
    };
    const observer =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(
            (entries) => {
              if (entries.some((entry) => entry.isIntersecting)) fire();
            },
            { rootMargin: "0px 0px -8% 0px", threshold: 0.01 },
          );
    observer?.observe(node);
    // IntersectionObserver is frame-driven and setTimeout is not, so this is the
    // guarantee rather than the optimisation.
    const timer = setTimeout(fire, 1600);
    return () => {
      done = true;
      observer?.disconnect();
      clearTimeout(timer);
    };
  }, []);

  return (
    <div ref={ref} data-reveal style={{ "--reveal-delay": `${delay}s` } as CSSProperties}>
      {children}
    </div>
  );
}

/** Turn the reveal behaviour on, but only once a frame has been painted.
 *
 * Called from the entry point. If the frame never arrives the class is never
 * added and every `Reveal` renders its children plainly.
 */
export function enableMotion(): void {
  if (typeof window === "undefined") return;
  requestAnimationFrame(() => {
    document.documentElement.classList.add("motion-ready");
  });
}

export function Counter({
  to,
  format,
  duration = 1.3,
}: {
  to: number;
  format: (value: number) => string;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  // The formatter is almost always written inline at the call site, so it is a
  // new function on every render. Depending on it would restart the count from
  // zero on every render and leave the figure frozen part-way — which it did.
  const formatRef = useRef(format);
  formatRef.current = format;

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const settle = () => {
      node.textContent = formatRef.current(to);
    };

    // The true value is written first and unconditionally. Everything after this
    // is decoration, and a decoration that stalls must not be able to leave a
    // wrong number on screen — which is exactly what happened when the count ran
    // first: a frame-starved context left the headline reading 2,454 instead of
    // 23,031, and a plausible wrong number is worse than no animation at all.
    settle();
    if (reduced() || !document.documentElement.classList.contains("motion-ready")) return;

    const state = { value: 0 };
    node.textContent = formatRef.current(0);
    const controls = animate(
      state,
      { value: to },
      {
        duration,
        ease: [0.16, 1, 0.3, 1],
        onUpdate: () => {
          node.textContent = formatRef.current(state.value);
        },
      },
    );
    controls.then(settle, settle);
    // Armed outside any frame callback, because setTimeout runs where
    // requestAnimationFrame does not.
    const timer = setTimeout(settle, duration * 1000 + 700);

    return () => {
      clearTimeout(timer);
      controls.stop();
      settle();
    };
  }, [to, duration]);

  return <span ref={ref} className="num" />;
}
