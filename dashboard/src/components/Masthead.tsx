import { useTheme, type Theme } from "../lib/useTheme";

const SECTIONS = [
  ["evidence", "The real number"],
  ["findable", "Findable?"],
  ["attribution", "Attribution"],
  ["recovery", "Recovery"],
  ["experiment", "Experiments"],
  ["clv", "Lifetime value"],
  ["decision", "The decision"],
  ["method", "Method"],
];

export function Masthead() {
  const [theme, setTheme] = useTheme();
  const options: { value: Theme; label: string; glyph: string }[] = [
    { value: "light", label: "Light", glyph: "☀" },
    { value: "system", label: "Follow the system", glyph: "◐" },
    { value: "dark", label: "Dark", glyph: "☾" },
  ];

  return (
    <header className="masthead">
      <a href="#top" className="wordmark" style={{ textDecoration: "none" }}>
        ATHAR <span>MAIB AI 208</span>
      </a>
      <nav>
        {SECTIONS.map(([id, label]) => (
          <a key={id} href={`#${id}`}>
            {label}
          </a>
        ))}
      </nav>
      <div className="theme-toggle" role="group" aria-label="Colour theme">
        {options.map((option) => (
          <button
            key={option.value}
            aria-pressed={theme === option.value}
            aria-label={option.label}
            title={option.label}
            onClick={() => setTheme(option.value)}
          >
            <span aria-hidden="true" style={{ fontSize: 12 }}>
              {option.glyph}
            </span>
          </button>
        ))}
      </div>
    </header>
  );
}
