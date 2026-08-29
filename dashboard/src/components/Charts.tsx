import { useState, type ReactNode } from "react";

/* Chart primitives, built to the data-viz mark specs: 2px lines, markers at
 * least 8px, a 2px surface gap between adjacent fills, recessive grid and axes,
 * selective direct labels rather than a number on every point, and a hover layer
 * on every form that plots something.
 *
 * Text always wears an ink token. A coloured mark beside a label carries the
 * identity; the label itself never takes the series colour. */

const INK = "var(--ink)";
const INK2 = "var(--ink-2)";
const INK3 = "var(--ink-3)";
const RULE = "var(--rule)";
const RULE2 = "var(--rule-2)";
const SURFACE = "var(--panel)";

export function useTooltip() {
  const [tip, setTip] = useState<{ x: number; y: number; content: ReactNode } | null>(null);
  const node = tip ? (
    <div
      className="tooltip"
      style={{ left: tip.x, top: tip.y, transform: "translate(-50%, calc(-100% - 10px))" }}
    >
      {tip.content}
    </div>
  ) : null;
  return { tip, setTip, node };
}

/** Dumbbell: two values per category, with the gap between them as the mark.
 *  The right form when the comparison IS the finding — a paired bar chart would
 *  make the reader do the subtraction the chart exists to show. */
export function Dumbbell({
  rows,
  leftLabel,
  rightLabel,
  leftColor,
  rightColor,
  format,
  domain,
}: {
  rows: { key: string; label: string; left: number; right: number }[];
  leftLabel: string;
  rightLabel: string;
  leftColor: string;
  rightColor: string;
  format: (value: number) => string;
  domain?: [number, number];
}) {
  const { setTip, node } = useTooltip();
  const width = 720;
  const rowHeight = 52;
  const padding = { top: 26, right: 84, bottom: 34, left: 168 };
  const height = padding.top + rows.length * rowHeight + padding.bottom;
  const values = rows.flatMap((r) => [r.left, r.right]);
  const [lo, hi] = domain ?? [0, Math.max(...values) * 1.12];
  const x = (value: number) =>
    padding.left + ((value - lo) / (hi - lo)) * (width - padding.left - padding.right);

  const ticks = 5;
  return (
    <div style={{ position: "relative" }}>
      <div className="legend">
        <span style={{ color: leftColor }}>
          <i /> <span style={{ color: INK2 }}>{leftLabel}</span>
        </span>
        <span style={{ color: rightColor }}>
          <i /> <span style={{ color: INK2 }}>{rightLabel}</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img"
           aria-label={`${leftLabel} against ${rightLabel} by channel`}>
        {Array.from({ length: ticks + 1 }, (_, i) => lo + ((hi - lo) * i) / ticks).map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={padding.top - 8} y2={height - padding.bottom}
                  stroke={RULE2} strokeWidth={1} />
            <text x={x(v)} y={height - padding.bottom + 18} fill={INK3} fontSize={11}
                  textAnchor="middle" className="num">{format(v)}</text>
          </g>
        ))}
        {rows.map((row, index) => {
          const y = padding.top + index * rowHeight + rowHeight / 2;
          const [a, b] = [x(row.left), x(row.right)];
          return (
            <g key={row.key}
               onMouseMove={(e) => {
                 const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                 setTip({
                   x: e.clientX - box.left,
                   y: e.clientY - box.top,
                   content: (
                     <>
                       <strong>{row.label}</strong>
                       <br />
                       {leftLabel} <span className="num">{format(row.left)}</span>
                       <br />
                       {rightLabel} <span className="num">{format(row.right)}</span>
                     </>
                   ),
                 });
               }}
               onMouseLeave={() => setTip(null)}>
              <rect x={0} y={y - rowHeight / 2} width={width} height={rowHeight} fill="transparent" />
              <text x={padding.left - 14} y={y + 4} fill={INK} fontSize={13} textAnchor="end">
                {row.label}
              </text>
              <line x1={a} x2={b} y1={y} y2={y} stroke={RULE} strokeWidth={2} strokeLinecap="round" />
              <circle cx={a} cy={y} r={5.5} fill={leftColor} stroke={SURFACE} strokeWidth={2} />
              <circle cx={b} cy={y} r={5.5} fill={rightColor} stroke={SURFACE} strokeWidth={2} />
              <text x={width - padding.right + 12} y={y + 4} fill={INK2} fontSize={12}
                    className="num">
                {format(row.right)}
              </text>
            </g>
          );
        })}
      </svg>
      {node}
    </div>
  );
}

/** Interval plot: a posterior range per category with the truth marked.
 *  Whether the interval covers the truth is the claim being made, so the truth
 *  is a rule through the chart rather than another dot to be compared by eye. */
export function Intervals({
  rows,
  color,
  format,
}: {
  rows: { key: string; label: string; low: number; high: number; mean: number; truth: number; covered: boolean }[];
  color: string;
  format: (value: number) => string;
}) {
  const { setTip, node } = useTooltip();
  const width = 720;
  const rowHeight = 54;
  const padding = { top: 26, right: 30, bottom: 34, left: 168 };
  const height = padding.top + rows.length * rowHeight + padding.bottom;
  const hi = Math.max(...rows.flatMap((r) => [r.high, r.truth])) * 1.08;
  const x = (v: number) => padding.left + (v / hi) * (width - padding.left - padding.right);

  return (
    <div style={{ position: "relative" }}>
      <div className="legend">
        <span style={{ color }}>
          <i /> <span style={{ color: INK2 }}>estimated ROI, 89% interval</span>
        </span>
        <span style={{ color: "var(--truth)" }}>
          <i style={{ width: 2, height: 12 }} /> <span style={{ color: INK2 }}>true ROI</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img"
           aria-label="Estimated ROI intervals against the true ROI, by channel">
        {rows.map((row, index) => {
          const y = padding.top + index * rowHeight + rowHeight / 2;
          return (
            <g key={row.key}
               onMouseMove={(e) => {
                 const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                 setTip({
                   x: e.clientX - box.left, y: e.clientY - box.top,
                   content: (
                     <>
                       <strong>{row.label}</strong><br />
                       true <span className="num">{format(row.truth)}</span><br />
                       estimate <span className="num">{format(row.mean)}</span><br />
                       interval <span className="num">{format(row.low)}–{format(row.high)}</span><br />
                       {row.covered ? "covers the truth" : "misses the truth"}
                     </>
                   ),
                 });
               }}
               onMouseLeave={() => setTip(null)}>
              <rect x={0} y={y - rowHeight / 2} width={width} height={rowHeight} fill="transparent" />
              <text x={padding.left - 14} y={y + 4} fill={INK} fontSize={13} textAnchor="end">
                {row.label}
              </text>
              <line x1={x(row.low)} x2={x(row.high)} y1={y} y2={y}
                    stroke={color} strokeWidth={7} strokeLinecap="round" opacity={0.34} />
              <circle cx={x(row.mean)} cy={y} r={5} fill={color} stroke={SURFACE} strokeWidth={2} />
              <line x1={x(row.truth)} x2={x(row.truth)} y1={y - 15} y2={y + 15}
                    stroke="var(--truth)" strokeWidth={2} strokeLinecap="round" />
              <text x={x(row.truth)} y={y - 20} fill={INK3} fontSize={10.5} textAnchor="middle"
                    className="num">{format(row.truth)}</text>
              {!row.covered && (
                <text x={width - padding.right} y={y + 4} fill="var(--signal)" fontSize={11}
                      textAnchor="end">missed</text>
              )}
            </g>
          );
        })}
      </svg>
      {node}
    </div>
  );
}

/** Stacked shares, one bar per allocation. A 2px surface gap separates segments,
 *  so adjacent fills never touch and the boundary is legible without relying on
 *  the hue difference alone. */
export function AllocationBars({
  rows,
  channels,
  colors,
  format,
}: {
  rows: { key: string; label: string; shares: number[]; note?: string }[];
  channels: string[];
  colors: string[];
  format: (value: number) => string;
}) {
  const { setTip, node } = useTooltip();
  const width = 720;
  const barHeight = 30;
  const gap = 30;
  const padding = { top: 8, right: 24, bottom: 30, left: 210 };
  const height = padding.top + rows.length * (barHeight + gap) + padding.bottom;
  const scale = width - padding.left - padding.right;

  return (
    <div style={{ position: "relative" }}>
      <div className="legend">
        {channels.map((channel, index) => (
          <span key={channel} style={{ color: colors[index] }}>
            <i /> <span style={{ color: INK2 }}>{channel}</span>
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img"
           aria-label="Budget shares by channel under each allocation">
        {rows.map((row, rowIndex) => {
          const y = padding.top + rowIndex * (barHeight + gap);
          let offset = 0;
          return (
            <g key={row.key}>
              <text x={padding.left - 14} y={y + barHeight / 2 + 4} fill={INK} fontSize={13}
                    textAnchor="end">{row.label}</text>
              {row.note && (
                <text x={padding.left - 14} y={y + barHeight / 2 + 19} fill={INK3} fontSize={10.5}
                      textAnchor="end">{row.note}</text>
              )}
              {row.shares.map((share, index) => {
                const w = share * scale;
                const x = padding.left + offset;
                offset += w;
                return (
                  <g key={channels[index]}
                     onMouseMove={(e) => {
                       const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                       setTip({
                         x: e.clientX - box.left, y: e.clientY - box.top,
                         content: (<><strong>{channels[index]}</strong><br />
                           {row.label}: <span className="num">{format(share)}</span></>),
                       });
                     }}
                     onMouseLeave={() => setTip(null)}>
                    <rect x={x + 1} y={y} width={Math.max(0, w - 2)} height={barHeight}
                          fill={colors[index]} rx={2} />
                    {share > 0.11 && (
                      <text x={x + w / 2} y={y + barHeight / 2 + 4} fill={SURFACE} fontSize={11}
                            textAnchor="middle" className="num">{format(share)}</text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      {node}
    </div>
  );
}

/** Multi-series lines with a crosshair. Used for response curves and power
 *  curves; direct end-labels rather than a legend box when four or fewer. */
export function Lines({
  series,
  xLabel,
  yLabel,
  formatX,
  formatY,
  marker,
}: {
  series: { key: string; label: string; color: string; points: [number, number][] }[];
  xLabel: string;
  yLabel: string;
  formatX: (value: number) => string;
  formatY: (value: number) => string;
  marker?: { x: number; label: string };
}) {
  const { setTip, node } = useTooltip();
  const [hover, setHover] = useState<number | null>(null);
  const width = 720;
  const height = 380;
  const padding = { top: 20, right: 128, bottom: 44, left: 66 };
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(0, ...ys), Math.max(...ys) * 1.06];
  const sx = (v: number) => padding.left + ((v - x0) / (x1 - x0)) * (width - padding.left - padding.right);
  const sy = (v: number) => height - padding.bottom - ((v - y0) / (y1 - y0)) * (height - padding.top - padding.bottom);

  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img"
           aria-label={`${yLabel} against ${xLabel}`}
           onMouseMove={(e) => {
             const box = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
             const px = ((e.clientX - box.left) / box.width) * width;
             const value = x0 + ((px - padding.left) / (width - padding.left - padding.right)) * (x1 - x0);
             if (value < x0 || value > x1) { setHover(null); setTip(null); return; }
             setHover(value);
             setTip({
               x: e.clientX - box.left, y: e.clientY - box.top,
               content: (
                 <>
                   <strong>{xLabel} {formatX(value)}</strong>
                   {series.map((s) => {
                     const nearest = s.points.reduce((best, p) =>
                       Math.abs(p[0] - value) < Math.abs(best[0] - value) ? p : best);
                     return (<div key={s.key}>{s.label} <span className="num">{formatY(nearest[1])}</span></div>);
                   })}
                 </>
               ),
             });
           }}
           onMouseLeave={() => { setHover(null); setTip(null); }}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = y0 + (y1 - y0) * t;
          return (
            <g key={t}>
              <line x1={padding.left} x2={width - padding.right} y1={sy(v)} y2={sy(v)}
                    stroke={RULE2} strokeWidth={1} />
              <text x={padding.left - 10} y={sy(v) + 4} fill={INK3} fontSize={11}
                    textAnchor="end" className="num">{formatY(v)}</text>
            </g>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = x0 + (x1 - x0) * t;
          return (
            <text key={t} x={sx(v)} y={height - padding.bottom + 18} fill={INK3} fontSize={11}
                  textAnchor="middle" className="num">{formatX(v)}</text>
          );
        })}
        <text x={width / 2} y={height - 6} fill={INK3} fontSize={11} textAnchor="middle">{xLabel}</text>

        {marker && (
          <g>
            <line x1={sx(marker.x)} x2={sx(marker.x)} y1={padding.top} y2={height - padding.bottom}
                  stroke={INK3} strokeWidth={1} strokeDasharray="3 3" />
            <text x={sx(marker.x) + 6} y={padding.top + 12} fill={INK3} fontSize={10.5}>
              {marker.label}
            </text>
          </g>
        )}
        {hover != null && (
          <line x1={sx(hover)} x2={sx(hover)} y1={padding.top} y2={height - padding.bottom}
                stroke={RULE} strokeWidth={1} />
        )}
        {series.map((s) => (
          <g key={s.key}>
            <path d={s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join(" ")}
                  fill="none" stroke={s.color} strokeWidth={2} strokeLinecap="round"
                  strokeLinejoin="round" />
            <text x={sx(s.points[s.points.length - 1][0]) + 8}
                  y={sy(s.points[s.points.length - 1][1]) + 4}
                  fill={INK2} fontSize={11.5}>{s.label}</text>
          </g>
        ))}
      </svg>
      {node}
    </div>
  );
}

/** A coverage grid. Sequential, one hue, light to dark — never a rainbow. */
export function Heatmap({
  rows,
  columns,
  values,
  format,
  caption,
}: {
  rows: string[];
  columns: string[];
  values: (number | null)[][];
  format: (value: number) => string;
  caption?: string;
}) {
  const { setTip, node } = useTooltip();
  const cell = 84;
  const padding = { top: 34, left: 176, right: 12, bottom: 12 };
  const width = padding.left + columns.length * cell + padding.right;
  const height = padding.top + rows.length * cell + padding.bottom;

  return (
    <div style={{ position: "relative", overflowX: "auto" }}>
      {/* A grid this narrow — a handful of columns — must not stretch to the
          container. Doing so preserves the aspect ratio and produced a heatmap
          twelve hundred pixels tall for eight cells. It scales down to fit a
          phone and stops growing at its natural size. */}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ minWidth: 340, maxWidth: width * 1.5, display: "block" }}
        role="img"
        aria-label={caption ?? "Coverage grid"}
      >
        {columns.map((column, columnIndex) => (
          <text key={column} x={padding.left + columnIndex * cell + cell / 2} y={padding.top - 12}
                fill={INK3} fontSize={11} textAnchor="middle">{column}</text>
        ))}
        {rows.map((row, rowIndex) => (
          <text key={row} x={padding.left - 12} y={padding.top + rowIndex * cell + cell / 2 + 4}
                fill={INK} fontSize={12.5} textAnchor="end">{row}</text>
        ))}
        {rows.map((row, rowIndex) =>
          columns.map((column, columnIndex) => {
            const value = values[rowIndex][columnIndex];
            const x = padding.left + columnIndex * cell;
            const y = padding.top + rowIndex * cell;
            return (
              <g key={`${row}-${column}`}
                 onMouseMove={(e) => {
                   const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                   setTip({
                     x: e.clientX - box.left, y: e.clientY - box.top,
                     content: (<><strong>{row}</strong><br />{column}<br />
                       {value == null ? "no converged fit" : format(value)}</>),
                   });
                 }}
                 onMouseLeave={() => setTip(null)}>
                <rect x={x + 1} y={y + 1} width={cell - 2} height={cell - 2} rx={3}
                      fill={value == null ? "var(--sunk)" : "var(--mmm)"}
                      opacity={value == null ? 1 : 0.14 + 0.76 * value} />
                <text x={x + cell / 2} y={y + cell / 2 + 5} fontSize={14} textAnchor="middle"
                      className="num"
                      fill={value != null && value > 0.55 ? "var(--panel)" : INK}>
                  {value == null ? "—" : format(value)}
                </text>
              </g>
            );
          }),
        )}
      </svg>
      {node}
    </div>
  );
}
