import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { enableMotion } from "./components/Frame";
import "./styles.css";

/** Smooth scroll, but only where scrolling can actually be driven.
 *
 * Lenis takes over the scroll position and advances it from a
 * requestAnimationFrame loop. In a context that delivers no frames — a headless
 * screenshotter, an embedded webview, a tab never brought forward — it takes
 * over and then never advances, leaving the page stuck part-way through a scroll
 * it will not finish. That was observed here, not hypothesised.
 *
 * So the first frame has to arrive before Lenis is allowed to exist, and anyone
 * who has asked for reduced motion never gets it at all. Native scrolling is the
 * fallback, which is no hardship.
 */
function enableSmoothScroll() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  requestAnimationFrame(async () => {
    const { default: Lenis } = await import("lenis");
    const lenis = new Lenis({ duration: 1.05, smoothWheel: true });
    const raf = (time: number) => {
      lenis.raf(time);
      requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
  });
}

enableMotion();
enableSmoothScroll();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
