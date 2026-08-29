import { Suspense, lazy, useState } from "react";
import { Caveat, Figure, Reveal } from "./components/Frame";
import { AllocationBars, Dumbbell, Heatmap, Intervals, Lines } from "./components/Charts";
// Three.js is by far the heaviest dependency on the page and only one section
// needs it. Split out so the reading experience above it loads without waiting
// for a WebGL bundle.
const BudgetSurface = lazy(() =>
  import("./components/Surface").then((module) => ({ default: module.BudgetSurface })),
);
import { artifact, available, channelLabel } from "./data";
import { brl, count, pct, roi, signedPct } from "./lib/format";
import { useResolvedTheme, useTheme } from "./lib/useTheme";

const SERIES = ["var(--mmm)", "var(--attribution)", "var(--experiment)", "var(--neutral)", "var(--ink-3)"];

/* ------------------------------------------------------------------ */

export function AttributionSection() {
  const panel = artifact("panel");
  if (!panel) return null;
  const channels = panel.channels as any[];
  const rows = channels.map((c) => ({
    key: c.channel,
    label: channelLabel(c.channel),
    left: c.true_roi_average,
    right: c.lastclick_roas,
  }));
  const worst = channels.reduce((a, b) =>
    a.lastclick_bias_relative > b.lastclick_bias_relative ? a : b,
  );
  const best = channels.reduce((a, b) =>
    a.lastclick_bias_relative < b.lastclick_bias_relative ? a : b,
  );
  const nullCase = channels.find((c) => Math.abs(c.lastclick_bias_relative) < 1e-9);

  return (
    <section id="attribution">
      <div className="shell split reverse">
        <Reveal>
          <Figure
            title="What last-click reports, against what is true"
            artifactName="panel"
            note={
              <>
                Simulated panel. Last-click here is a parametric caricature with two stated
                knobs per channel — the share of true contribution it observes, and the share
                of baseline revenue it credits — not a simulated user journey. Simulating
                journeys would look more faithful without being more honest, since the journey
                parameters would be equally invented.
              </>
            }
            table={
              <table>
                <thead>
                  <tr>
                    <th>Channel</th>
                    <th>True ROI</th>
                    <th>Last-click ROAS</th>
                    <th>Error</th>
                    <th>Organic capture</th>
                  </tr>
                </thead>
                <tbody>
                  {channels.map((c) => (
                    <tr key={c.channel}>
                      <td>{channelLabel(c.channel)}</td>
                      <td className="num">{roi(c.true_roi_average)}</td>
                      <td className="num">{roi(c.lastclick_roas)}</td>
                      <td className="num">{signedPct(c.lastclick_bias_relative)}</td>
                      <td className="num">{pct(c.organic_capture, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <Dumbbell
              rows={rows}
              leftLabel="True ROI"
              rightLabel="Last-click ROAS"
              leftColor="var(--truth)"
              rightColor="var(--attribution)"
              format={(v) => roi(v)}
            />
          </Figure>
        </Reveal>
        <div className="sticky">
          <p className="eyebrow">Where attribution goes wrong</p>
          <h2>Not uniformly wrong. Wrong in a direction you can predict.</h2>
          <p style={{ marginTop: "1.3rem" }}>
            Last-click overstates <strong>{channelLabel(worst.channel)}</strong> by{" "}
            <span className="num">{signedPct(worst.lastclick_bias_relative)}</span>, because
            buyers who had already decided pass through it on the way to a purchase they were
            going to make anyway. It understates{" "}
            <strong>{channelLabel(best.channel)}</strong> by{" "}
            <span className="num">{signedPct(best.lastclick_bias_relative)}</span>, because most
            of an upper-funnel effect is never observed by a click at all.
          </p>
          {nullCase && (
            <p>
              And on <strong>{channelLabel(nullCase.channel)}</strong> it is exactly right. That
              case is in the design deliberately: a harness that only ever showed attribution
              failing would have had its answer chosen for it.
            </p>
          )}
          <Caveat artifactName="panel" />
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */

export function RecoverySection() {
  // Hooks first, unconditionally. The early return below is safe in practice —
  // the artifact set is fixed when the bundle is built — but a hook after a
  // conditional return is a rule-of-hooks violation regardless of whether this
  // particular condition can change, and it would become a real bug the moment
  // the data were loaded at runtime.
  const [arm, setArm] = useState<"misspecified" | "matched">("misspecified");
  if (!available("mmm")) return null;
  const mmm = artifact("mmm")!;
  const recovery = artifact("recovery");
  const fit = mmm.fits[arm];
  const channels = Object.entries(fit.average_roi.channels) as [string, any][];
  const summary = fit.average_roi.summary;

  const rows = channels.map(([key, entry]) => ({
    key,
    label: channelLabel(key),
    low: entry.hdi_low,
    high: entry.hdi_high,
    mean: entry.estimated_mean,
    truth: entry.true,
    covered: entry.covered,
  }));

  let heat = null;
  if (recovery) {
    const lengths = recovery.design.panel_lengths as number[];
    const levels = recovery.design.collinearity_levels as string[];
    const specs = Object.keys(recovery.design.specifications) as string[];
    const rowLabels = specs.flatMap((spec) => levels.map((l) => `${spec}, ${l} collinearity`));
    const columnLabels = lengths.map((w) => `${w} weeks`);
    const values = specs.flatMap((spec) =>
      levels.map((level) =>
        lengths.map((weeks) => {
          const slice = recovery.slices[`${weeks}w_${level}_${spec}`]?.average_roi;
          return slice && slice.converged > 0 ? slice.coverage_rate : null;
        }),
      ),
    );
    heat = { rowLabels, columnLabels, values, recovery };
  }

  return (
    <section id="recovery">
      <div className="shell">
        <Reveal>
          <p className="eyebrow">Can the model find what is there</p>
          <h2 style={{ maxWidth: "20ch" }}>
            The interval is the honest part.
          </h2>
        </Reveal>
        <div className="split" style={{ marginTop: "2.4rem" }}>
          <div>
            <p>
              A media-mix model does not return a number, it returns a distribution. The claim it
              makes is that the truth lies inside its interval — so the thing worth measuring is
              whether it does, not how close the point estimate happened to land on one draw.
            </p>
            <p>
              On the headline panel the {arm} fit covers{" "}
              <span className="num">
                {summary.channels_covered} of {summary.channels_total}
              </span>{" "}
              channels at the 89% level, with a median absolute error of{" "}
              <span className="num">{pct(summary.median_absolute_relative_error, 0)}</span>.
              The intervals are wide because the priors are the library's defaults and the design
              is weakly identified. Both of those are deliberate.
            </p>
            <div className="controls">
              <label>
                Specification
                <select value={arm} onChange={(e) => setArm(e.target.value as any)}>
                  <option value="misspecified">Misspecified (headline)</option>
                  <option value="matched">Matched (control)</option>
                </select>
              </label>
            </div>
            <p className="footnote">{mmm.design.specifications[arm]}</p>
            <Caveat artifactName="mmm" />
          </div>
          <Reveal>
            <Figure
              title={`Estimated against true ROI — ${arm}`}
              artifactName="mmm"
              note={`Sampler: ${fit.diagnostics.divergences} divergences in ${fit.diagnostics.post_warmup_draws} draws, max R-hat ${fit.diagnostics.max_r_hat}, min bulk ESS ${fit.diagnostics.min_ess_bulk}. ${fit.diagnostics.passed ? "Converged." : "Failed its diagnostics — reported, not hidden."}`}
              table={
                <table>
                  <thead>
                    <tr>
                      <th>Channel</th>
                      <th>True</th>
                      <th>Estimate</th>
                      <th>89% interval</th>
                      <th>Covered</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.key}>
                        <td>{r.label}</td>
                        <td className="num">{roi(r.truth)}</td>
                        <td className="num">{roi(r.mean)}</td>
                        <td className="num">
                          {roi(r.low)}–{roi(r.high)}
                        </td>
                        <td>{r.covered ? "yes" : "no"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              }
            >
              <Intervals rows={rows} color="var(--mmm)" format={(v) => roi(v)} />
            </Figure>
          </Reveal>
        </div>

        {heat && (
          <Reveal>
            <div style={{ marginTop: "3.5rem" }}>
              <Figure
                title="Coverage across the recovery grid"
                artifactName="recovery"
                note={
                  <>
                    {heat.recovery.convergence.converged} of {heat.recovery.convergence.fits}{" "}
                    full-posterior fits converged and are included; the rest are excluded and
                    counted rather than averaged in. Each cell is the share of channels whose
                    true ROI fell inside the 89% interval, across{" "}
                    {heat.recovery.design.seeds.length} seeds. Perfect calibration would read
                    0.89.
                  </>
                }
                table={
                  <table>
                    <thead>
                      <tr>
                        <th>Cell</th>
                        <th>Converged</th>
                        <th>Coverage</th>
                        <th>Median abs rel error</th>
                        <th>Mean interval width</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(heat.recovery.slices).map(([name, slice]: [string, any]) => {
                        const block = slice.average_roi;
                        return (
                          <tr key={name}>
                            <td>{name.replace(/_/g, " ")}</td>
                            <td className="num">
                              {block.converged ?? 0}/{block.fits ?? 0}
                            </td>
                            <td className="num">
                              {block.converged ? block.coverage_rate.toFixed(2) : "—"}
                            </td>
                            <td className="num">
                              {block.converged
                                ? block.median_absolute_relative_error.toFixed(2)
                                : "—"}
                            </td>
                            <td className="num">
                              {block.converged ? block.mean_interval_width.toFixed(2) : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                }
              >
                <Heatmap
                  rows={heat.rowLabels}
                  columns={heat.columnLabels}
                  values={heat.values}
                  format={(v) => v.toFixed(2)}
                  caption="Coverage of the 89% interval by panel length, collinearity and specification"
                />
                <p
                  className="footnote"
                  style={{ marginTop: "1rem", borderLeft: "2px solid var(--signal)", paddingLeft: "1rem" }}
                >
                  <strong>Coverage alone is not enough.</strong> An interval reaches perfect
                  coverage by being uselessly wide. Switch to the table to see the median
                  error and the interval width beside it — a cell at 1.00 coverage with a
                  median error above 1.0 has not recovered anything, it has declined to
                  answer.
                </p>
              </Figure>
            </div>
          </Reveal>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */

export function ExperimentSection() {
  const experiments = artifact("experiments");
  const channels = experiments ? Object.keys(experiments.power) : [];
  const [channel, setChannel] = useState(channels[0] ?? "");
  if (!experiments) return null;
  const curve = experiments.power[channel] as any[];

  return (
    <section id="experiment">
      <div className="shell split">
        <div className="sticky">
          <p className="eyebrow">The only unbiased estimator</p>
          <h2>Right on average. Rarely right enough.</h2>
          <p style={{ marginTop: "1.3rem" }}>
            A geo holdout is the one method here with no bias to argue about. The catch is the
            price: Olist sold into{" "}
            <span className="num">{experiments.geography.states}</span> Brazilian states, one of
            which carries{" "}
            <span className="num">{pct(experiments.geography.largest_state_share, 0)}</span> of
            revenue while{" "}
            <span className="num">{experiments.geography.states_below_one_percent}</span> carry
            under one percent each.
          </p>
          <p>
            Switching a channel off across nearly half the business for two months still leaves
            an estimate you would not want to reallocate on. And the sting is which channels are
            hardest: the ones with the smallest effects, which are exactly the ones worth
            catching.
          </p>
          <div className="controls">
            <label>
              Channel
              <select value={channel} onChange={(e) => setChannel(e.target.value)}>
                {channels.map((c) => (
                  <option key={c} value={c}>
                    {channelLabel(c)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <Caveat artifactName="experiments" />
        </div>
        <Reveal>
          <Figure
            title={`Holdout precision — ${channelLabel(channel)}`}
            artifactName="experiments"
            note={
              <>
                {experiments.design.replicates_per_point} random assignments of treated states
                per point, {experiments.design.test_weeks}-week holdout. "Detected" means the
                estimate had the right sign and landed within half the true effect — a blunt
                rule, fixed before the numbers were seen.
              </>
            }
            table={
              <table>
                <thead>
                  <tr>
                    <th>Treated states</th>
                    <th>Revenue switched off</th>
                    <th>Median error</th>
                    <th>Error SD</th>
                    <th>Detected</th>
                  </tr>
                </thead>
                <tbody>
                  {curve.map((row) => (
                    <tr key={row.treated_states}>
                      <td className="num">{row.treated_states}</td>
                      <td className="num">{pct(row.median_treated_revenue_share, 0)}</td>
                      <td className="num">{signedPct(row.median_relative_error, 1)}</td>
                      <td className="num">{row.relative_error_sd.toFixed(2)}</td>
                      <td className="num">{pct(row.detected, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <Lines
              series={[
                {
                  key: "detected",
                  label: "detected",
                  color: "var(--experiment)",
                  points: curve.map((r) => [r.median_treated_revenue_share, r.detected]),
                },
                {
                  key: "sd",
                  label: "error SD",
                  color: "var(--attribution)",
                  points: curve.map((r) => [
                    r.median_treated_revenue_share,
                    Math.min(1.5, r.relative_error_sd),
                  ]),
                },
              ]}
              xLabel="share of revenue switched off"
              yLabel=""
              formatX={(v) => pct(v, 0)}
              formatY={(v) => v.toFixed(2)}
            />
          </Figure>
        </Reveal>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */

export function ClvSection() {
  if (!available("clv")) return null;
  const clv = artifact("clv")!;
  const behaviour = clv.repeat_behaviour;
  const ml = clv.maximum_likelihood;
  const value = clv.lifetime_value;
  const validation = clv.validation;
  const bayesian = clv.models.bgnbd_bayesian_full_base;
  const sampler = bayesian.parameters?.sampler ?? {};

  return (
    <section id="clv">
      <div className="shell">
        <Reveal>
          <p className="eyebrow">Customer lifetime value</p>
          <h2 style={{ maxWidth: "24ch" }}>
            The standard model does not fit this business, and that is the answer.
          </h2>
        </Reveal>
        <div className="split" style={{ marginTop: "2.4rem" }}>
          <div>
            <p>
              <span className="num">{pct(behaviour.repeat_rate, 2)}</span> of Olist's{" "}
              <span className="num">{count(behaviour.customers)}</span> customers ever bought
              twice, and those who did averaged{" "}
              <span className="num">
                {behaviour.mean_repeats_among_repeaters.toFixed(2)}
              </span>{" "}
              repeat purchases.
            </p>
            <p>
              BG/NBD, the standard model, did not converge by maximum likelihood —{" "}
              <span className="num">
                {ml.converged_on_full_base} of {ml.attempts_full_base.length}
              </span>{" "}
              settings on the full base and{" "}
              <span className="num">
                {ml.converged_on_repeaters_only} of {ml.attempts_repeaters_only.length}
              </span>{" "}
              on the repeaters alone, across days, weeks and months and penalties from zero to
              ten. That is not a tuning problem. The model's dropout parameters describe how
              churn probability is distributed, and they are identified only by the pattern of
              repeat purchasing. There is no pattern here to identify them from.
            </p>
            <p>
              The Bayesian fit is the only one that runs at all, and running is not the same
              as working. Its sampler throws{" "}
              <span className="num">{count(sampler.divergences ?? 0)}</span> divergences with
              an R-hat of <span className="num">{sampler.max_r_hat}</span>, and{" "}
              <span className="num">alpha</span> collapses to about{" "}
              <span className="num">1e-306</span> — the smallest number a float can hold —
              against a prior mean near 9. An interval that tight is the sampler falling into
              a corner, not the data speaking.
            </p>
            <p>
              And on a time-based holdout it predicts{" "}
              <span className="num">{validation.predicted_total.toFixed(1)}</span> repeat
              purchases against <span className="num">{count(validation.actual_total)}</span>{" "}
              actual, losing to a baseline that predicts zero for everyone.
            </p>
          </div>
          <Reveal>
            <div className="tiles" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <div className="tile">
                <div className="k">Repeat rate</div>
                <div className="v">{pct(behaviour.repeat_rate, 2)}</div>
                <p className="n">{count(behaviour.repeaters)} customers of {count(behaviour.customers)}</p>
              </div>
              <div className="tile">
                <div className="k">Fits that can be believed</div>
                <div className="v" style={{ color: "var(--signal)" }}>
                  {ml.converged_on_full_base}
                </div>
                <p className="n">
                  {ml.converged_on_full_base} of {ml.attempts_full_base.length}{" "}
                  maximum-likelihood settings converged, and the MCMC fit failed its
                  diagnostics
                </p>
              </div>
              <div className="tile">
                <div className="k">Expected lifetime value</div>
                <div className="v" style={{ fontSize: "clamp(1.1rem, 2vw, 1.5rem)" }}>
                  {value.computable ? brl(value.mean_expected_clv_brl, 2) : "not computable"}
                </div>
                <p className="n">
                  {value.computable
                    ? "over a year, mean across the base"
                    : "there is no working transaction model to compute it from"}
                </p>
              </div>
              <div className="tile">
                <div className="k">First-order value</div>
                <div className="v">{brl(value.mean_first_order_value_brl, 0)}</div>
                <p className="n">
                  {value.computable ? (
                    <>
                      Lifetime value is{" "}
                      <span className="num">{pct(value.clv_over_first_order_value, 2)}</span> of
                      it
                    </>
                  ) : (
                    "the one lifetime-value figure this base does support"
                  )}
                </p>
              </div>
            </div>
          </Reveal>
        </div>
        <Reveal>
          <div style={{ marginTop: "2.2rem", maxWidth: "72ch" }}>
            {clv.verdict && (
              <p style={{ borderLeft: "2px solid var(--signal)", paddingLeft: "1rem" }}>
                <strong>Verdict.</strong> {clv.verdict}
              </p>
            )}
            {value.why_not && (
              <p style={{ color: "var(--ink-2)" }}>{value.why_not}</p>
            )}
            <p style={{ color: "var(--ink-2)" }}>{clv.finding}</p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */

export function DecisionSection() {
  const [theme] = useTheme();
  const dark = useResolvedTheme(theme) === "dark";
  if (!available("triangulation", "allocator")) return null;
  const triangulation = artifact("triangulation")!;
  const allocator = artifact("allocator")!;

  const channels = allocator.channels as string[];
  const headline = triangulation.headline;
  const allocations = allocator.allocations as Record<string, any>;

  const order = [
    "optimal_under_truth",
    "mmm_with_curvature",
    "mmm_average_roi",
    "experiment",
    "attribution_lastclick",
    "equal_split",
  ].filter((key) => allocations[key]);

  const LABELS: Record<string, string> = {
    optimal_under_truth: "Knowing the true curves",
    mmm_with_curvature: "Media-mix model, with curvature",
    mmm_average_roi: "Media-mix model, average ROI only",
    experiment: "Geo experiments",
    attribution_lastclick: "Last-click attribution",
    equal_split: "Equal split",
    true_average_roi: "True average ROI",
    true_marginal_roi: "True marginal ROI",
  };

  // The most uncomfortable number here, shown only when it actually holds.
  const estimatorKeys = [
    "attribution_lastclick",
    "mmm_average_roi",
    "mmm_with_curvature",
    "experiment",
  ].filter((key) => allocations[key]);
  const ranked = estimatorKeys
    .map((key) => [key, allocations[key]] as [string, any])
    .sort((a, b) => a[1].shortfall_share - b[1].shortfall_share);
  const bestEstimator = ranked[0] ?? ["", { shortfall_share: 0 }];
  const worstEstimator = ranked[ranked.length - 1] ?? ["", { shortfall_share: 0 }];
  const equalSplitWins =
    !!allocations.equal_split &&
    ranked.length > 0 &&
    ranked.every(([, entry]) => allocations.equal_split.shortfall_share < entry.shortfall_share);

  const surface = allocator.surface;
  // The truth marker takes ink; the estimators take the categorical slots in fixed
  // order. Counted separately from the loop index so the palette is never cycled —
  // reusing a hue would put two estimators in the same colour on one legend.
  const estimatorHues = dark
    ? ["#d95926", "#3987e5", "#199e70", "#6b6862"]
    : ["#eb6834", "#2a78d6", "#1baf7a", "#a9a49a"];
  let hue = 0;
  const markers = order
    .filter((key) => key !== "equal_split")
    .map((key) => ({
      key,
      label: LABELS[key] ?? key,
      x: allocations[key].spend[surface.channels[0]],
      y: allocations[key].spend[surface.channels[1]],
      z: allocations[key].revenue_under_truth,
      color:
        key === "optimal_under_truth"
          ? dark
            ? "#f0ede7"
            : "#16171b"
          : (estimatorHues[hue++] ?? "var(--ink-3)"),
    }));

  return (
    <section id="decision">
      <div className="shell">
        <Reveal>
          <p className="eyebrow">The decision</p>
          <h2 style={{ maxWidth: "22ch" }}>What believing the wrong number costs.</h2>
          <p className="lede" style={{ marginTop: "1.4rem", maxWidth: "62ch" }}>
            Each method's ROI table becomes a budget under identical governance constraints.
            Every budget is then scored against the <em>same</em> true response curves — so what
            is compared is the consequence of believing an estimator, not its opinion of itself.
          </p>
        </Reveal>

        <Reveal>
          <div
            className="tiles"
            style={{
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              marginTop: "2.4rem",
            }}
          >
            <div className="tile">
              <div className="k">Budget reallocated</div>
              <div className="v">{brl(headline.budget_brl)}</div>
              <p className="n">the panel's own media spend across five channels</p>
            </div>
            <div className="tile">
              <div className="k">Left on the table by last-click</div>
              <div className="v" style={{ color: "var(--signal)" }}>
                {brl(headline.cost_of_believing_attribution_brl)}
              </div>
              <p className="n">
                {pct(headline.cost_of_believing_attribution_share, 1)} of the incremental revenue
                the best-informed allocation earns
              </p>
            </div>
            <div className="tile">
              <div className="k">On an AED 1,000,000 budget</div>
              <div className="v">
                {count(
                  triangulation.scenario_in_aed.on_a_one_million_aed_budget
                    .cost_of_believing_attribution_aed,
                )}
              </div>
              <p className="n">
                SCENARIO — the shortfall is a share of budget, so it scales. Not an exchange-rate
                claim and not a measurement.
              </p>
            </div>
          </div>
        </Reveal>

        <div className="split" style={{ marginTop: "3rem" }}>
          <Reveal>
            <Figure
              title="The budget surface"
              artifactName="allocator"
              note={
                <>
                  {surface.note} Each marker is where one estimator's budget lands. The distance
                  down the slope from the ridge is what that estimator costs.
                </>
              }
            >
              <Suspense
                fallback={
                  <div
                    className="surface-wrap"
                    style={{ display: "grid", placeItems: "center", color: "var(--ink-3)" }}
                  >
                    <span className="eyebrow" style={{ margin: 0 }}>
                      loading the surface
                    </span>
                  </div>
                }
              >
                <BudgetSurface data={surface} markers={markers} dark={dark} />
              </Suspense>
              <div className="legend" style={{ marginTop: ".9rem" }}>
                {markers.map((m) => (
                  <span key={m.key} style={{ color: m.color }}>
                    <i /> <span style={{ color: "var(--ink-2)" }}>{m.label}</span>
                  </span>
                ))}
              </div>
            </Figure>
          </Reveal>
          <Reveal>
            <Figure
              title="Where each method sends the money"
              artifactName="allocator"
              note={
                <>
                  Floors of {pct(allocator.governance.floor_share, 0)} and caps of{" "}
                  {pct(allocator.governance.cap_share, 0)} per channel, applied identically to
                  every allocation. Without them a scalar-ROI table sends the whole budget to one
                  channel, and the comparison would be against a caricature.
                </>
              }
              table={
                <table>
                  <thead>
                    <tr>
                      <th>Allocation</th>
                      <th>Revenue under truth</th>
                      <th>Shortfall</th>
                    </tr>
                  </thead>
                  <tbody>
                    {order.map((key) => (
                      <tr key={key}>
                        <td>{LABELS[key] ?? key}</td>
                        <td className="num">{brl(allocations[key].revenue_under_truth)}</td>
                        <td className="num">
                          {pct(allocations[key].shortfall_share, 2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              }
            >
              <AllocationBars
                rows={order.map((key) => ({
                  key,
                  label: LABELS[key] ?? key,
                  shares: channels.map((c) => allocations[key].shares[c]),
                  note: `${pct(allocations[key].shortfall_share, 2)} short`,
                }))}
                channels={channels.map(channelLabel)}
                colors={SERIES}
                format={(v) => pct(v, 0)}
              />
            </Figure>
          </Reveal>
        </div>

        <Reveal>
          <div style={{ marginTop: "3rem" }} className="split">
            <div>
              <h3>Reconciliation is not averaging</h3>
              <p>{triangulation.method_notes.why_not_average}</p>
              <p>{triangulation.method_notes.linear_versus_curved}</p>
              <Caveat artifactName="triangulation" />
            </div>
            <Figure
              title="Response curves, true against fitted"
              artifactName="allocator"
              note="Total incremental revenue as each channel's plan is scaled. The dashed line marks the observed plan. Only a media-mix model produces a curve at all; an experiment and an attribution report each return a single number."
            >
              <Lines
                series={channels.slice(0, 3).map((channel, index) => ({
                  key: channel,
                  label: channelLabel(channel),
                  color: SERIES[index],
                  points: allocator.response_curves.spend[channel].map(
                    (spend: number, i: number) => [
                      spend,
                      allocator.response_curves.true_revenue[channel][i],
                    ],
                  ) as [number, number][],
                }))}
                xLabel="spend on the channel"
                yLabel="incremental revenue"
                formatX={(v) => `R$${count(v / 1000)}k`}
                formatY={(v) => `R$${count(v / 1000)}k`}
                marker={{
                  x: allocator.observed_spend[channels[0]],
                  label: "observed plan",
                }}
              />
            </Figure>
          </div>
        </Reveal>

        <Reveal>
          <div style={{ marginTop: "2.4rem", maxWidth: "72ch" }}>
            <p style={{ color: "var(--ink-2)" }}>{headline.statement}</p>
            {equalSplitWins && (
              <>
                <h3 style={{ marginTop: "2rem" }}>
                  And an equal split beats all three of them
                </h3>
                <p>
                  Dividing the budget evenly across the five channels, using no estimate at
                  all, forgoes{" "}
                  <span className="num">{pct(allocations.equal_split.shortfall_share, 1)}</span>{" "}
                  — against{" "}
                  <span className="num">{pct(bestEstimator[1].shortfall_share, 1)}</span> for
                  the best estimator here and{" "}
                  <span className="num">{pct(worstEstimator[1].shortfall_share, 1)}</span> for
                  the worst.
                </p>
                <p>
                  This is not an argument for allocating blindly, and it is not a quirk of
                  the arithmetic. It is what a linear allocation rule does to a biased
                  estimate. A scalar ROI table forces a corner solution — the best-ranked
                  channel to its ceiling, the worst to its floor — so the budget concentrates
                  exactly where the estimate is most wrong. Last-click drives brand search to
                  its cap, the channel it over-credits most, and video to its floor, the one
                  it understates most. Each estimator is confidently wrong in one large
                  place; the even split is never badly wrong anywhere.
                </p>
                <p style={{ borderLeft: "2px solid var(--signal)", paddingLeft: "1rem" }}>
                  A ranking is worth acting on only in proportion to how much better it is
                  than not knowing. A planner holding a biased table should shrink toward the
                  even split rather than optimise against it — a stronger conclusion than the
                  headline, and one that cuts against this project's own instrument.
                </p>
              </>
            )}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
