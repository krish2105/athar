import { useCallback, useEffect, useState } from "react";

/** Three states, not two.
 *
 * "system" is the default and stamps nothing, so the page follows the operating
 * system. An explicit choice stamps `data-theme` on the root and wins over the
 * media query in both directions. Anything that treats a toggle as a boolean
 * loses the ability to go back to following the OS.
 */
export type Theme = "system" | "light" | "dark";

const KEY = "athar-theme";

function read(): Theme {
  try {
    const stored = localStorage.getItem(KEY);
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    return "system";
  }
}

export function useTheme(): [Theme, (next: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(read);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, theme);
    } catch {
      /* private browsing: the choice simply does not persist */
    }
  }, [theme]);

  return [theme, useCallback((next: Theme) => setTheme(next), [])];
}

/** The resolved light/dark value, for code that has to pick a colour in JS
 *  (the WebGL scene) rather than in CSS. */
export function useResolvedTheme(theme: Theme): "light" | "dark" {
  const [system, setSystem] = useState<"light" | "dark">(() =>
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light",
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = (event: MediaQueryListEvent) =>
      setSystem(event.matches ? "dark" : "light");
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, []);
  return theme === "system" ? system : theme;
}
