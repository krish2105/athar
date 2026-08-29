import { Masthead } from "./components/Masthead";
import { Counter, Reveal } from "./components/Frame";
import { artifact, available } from "./data";
import { count, pct, ratio } from "./lib/format";
import {
  AttributionSection,
  FindableSection,
  ClvSection,
  DecisionSection,
  ExperimentSection,
  RecoverySection,
} from "./Findings";

function Hero() {
  const criteo = artifact("criteo");
  if (!criteo) return null;
  const gap = criteo.platform_reported_versus_incremental;
  const naive = gap.naive_rate_framing;
  const itt = criteo.intent_to_treat;

  return (
    <section id="top" style={{ paddingBlockStart: "clamp(7rem, 18vh, 11rem)" }}>
      <div className="shell">
        <Reveal>
          <p className="eyebrow">
            MAIB AI 208 · AI in Marketing · SP Jain Dubai · Krishna Mathur
          </p>
        </Reveal>
        <Reveal delay={0.06}>
          <h1>
            Platform-reported ROAS
            <br />
            is not incremental.
          </h1>
        </Reveal>
        <Reveal delay={0.12}>
          <p className="lede" style={{ marginTop: "1.6rem", maxWidth: "58ch" }}>
            Here is the gap, measured on a real randomised trial of{" "}
            <span className="num">{count(criteo.population.rows)}</span> users. No model,
            no simulation, no assumption — a held-out control group and arithmetic.
          </p>
        </Reveal>

        <Reveal delay={0.2}>
          <div
            className="tiles"
            style={{
              gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
              marginTop: "2.6rem",
            }}
          >
            <div className="tile">
              <div className="k">The platform reports</div>
              <div className="v">
                <Counter to={gap.platform_reported_conversions} format={(v) => count(v)} />
              </div>
              <p className="n">
                Conversions by users who were shown an ad. This is not a causal quantity:
                who saw an ad depended on who was browsing, and browsing predicts buying.
              </p>
            </div>
            <div className="tile">
              <div className="k">Advertising actually caused</div>
              <div className="v" style={{ color: "var(--signal)" }}>
                <Counter to={gap.incremental_conversions} format={(v) => count(v)} />
              </div>
              <p className="n">
                The intent-to-treat difference between randomly assigned arms. Unbiased by
                construction, because assignment was random.
              </p>
            </div>
            <div className="tile">
              <div className="k">Overstatement</div>
              <div className="v">
                <Counter to={gap.overstatement_ratio} format={(v) => ratio(v)} />
              </div>
              <p className="n">
                Stated as a conversion <em>rate</em> instead, the same trial makes advertising
                look <span className="num">{ratio(naive.naive_rate_ratio)}</span> better than
                the control — against a true lift of{" "}
                <span className="num">{ratio(naive.true_relative_lift_ratio)}</span>.
              </p>
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.26}>
          <p style={{ marginTop: "2rem", color: "var(--ink-2)", maxWidth: "62ch" }}>
            Both framings are correct arithmetic on the same trial. Only{" "}
            <span className="num">{pct(naive.exposed_share_of_treated_arm, 1)}</span> of the
            treated arm was ever shown an ad, so a seventeen-fold exaggeration of the rate
            becomes a{" "}
            <span className="num">{ratio(gap.overstatement_ratio)}</span> exaggeration of the
            count. Reporting only the dramatic one would be rhetoric. The intent-to-treat lift
            is <span className="num">{itt.absolute_lift.toFixed(6)}</span>, with a 95% interval
            of{" "}
            <span className="num">
              {itt.ci_95[0].toFixed(6)}–{itt.ci_95[1].toFixed(6)}
            </span>
            .
          </p>
        </Reveal>
      </div>
    </section>
  );
}

function WhatIsReal() {
  const frame = artifact("frame");
  const panel = artifact("panel");
  const criteo = artifact("criteo");
  if (!frame || !panel || !criteo) return null;

  return (
    <section id="evidence">
      <div className="shell split">
        <div>
          <p className="eyebrow">What this rests on</p>
          <h2>Two findings, never blurred together.</h2>
          <p style={{ marginTop: "1.4rem" }}>
            No public dataset carries a media-mix model, an experiment and an attribution log
            for the same business. Pretending otherwise is the usual move, and it is the one
            thing this project refuses. So there are two separate claims here, and every chart
            on this page says which one it belongs to.
          </p>
          <p>
            The badge on each figure is read from the artifact behind it, not written into the
            page. A chart drawn from simulated data cannot be published without one.
          </p>
        </div>
        <div className="tiles" style={{ gridTemplateColumns: "1fr" }}>
          <div className="tile">
            <span className="badge real">
              <i className="dot" />
              Real data
            </span>
            <h3 style={{ marginTop: ".9rem" }}>Criteo, and Olist</h3>
            <p className="n" style={{ maxWidth: "52ch" }}>
              A genuine randomised trial over{" "}
              <span className="num">{count(criteo.population.rows)}</span> users, with{" "}
              <span className="num">{criteo.randomisation_check.exposure_in_control_arm}</span>{" "}
              exposures leaking into the control arm — so the randomisation is clean. And{" "}
              <span className="num">{count(frame.totals.orders)}</span> real Brazilian
              e-commerce orders across{" "}
              <span className="num">{frame.window.weeks}</span> gap-free weeks, worth{" "}
              <span className="num">R$ {count(frame.totals.revenue_brl)}</span>.
            </p>
          </div>
          <div className="tile">
            <span className="badge synthetic">
              <i className="dot" />
              Simulated
            </span>
            <h3 style={{ marginTop: ".9rem" }}>The media panel</h3>
            <p className="n" style={{ maxWidth: "52ch" }}>
              Five channels of spend, generated from a configuration committed before any
              model was fitted, layered on the real Olist revenue baseline. Its purpose is a
              known truth: no real advertiser knows its own true ROI, so "did the model recover
              it?" is a question that only exists inside a simulation. Media accounts for{" "}
              <span className="num">
                {pct(panel.design.media_as_share_of_revenue, 1)}
              </span>{" "}
              of revenue and{" "}
              <span className="num">
                {pct(panel.identification.media_share_of_detrended_variance, 1)}
              </span>{" "}
              of the variance a model has to work with.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Method() {
  const panel = artifact("panel");
  const gate = artifact("env_gate");
  const frame = artifact("frame");
  if (!panel || !frame) return null;

  return (
    <section id="method">
      <div className="shell split">
        <div className="sticky">
          <p className="eyebrow">Method and limitations</p>
          <h2>What would falsify this.</h2>
        </div>
        <div>
          <h3>The model is fitted wrong on purpose</h3>
          <p>
            The panel is generated with delayed-geometric adstock and a Hill saturation curve.
            The headline model is fitted with geometric adstock and a logistic curve — the first
            cannot represent a delayed peak at all, and the second cannot take the Hill shape. A
            model fitted with the form that generated the data recovers its own assumptions and
            measures nothing. A matched arm runs alongside as a control, so the cost of the wrong
            shape can be separated from the cost of a design that cannot identify the parameters.
          </p>

          <h3 style={{ marginTop: "2rem" }}>The answer was locked away during fitting</h3>
          <p>
            The ground truth lives outside the repository and is guarded by a function that
            refuses to return it until the fit being scored already exists on disk. Priors are
            the library's defaults, unchanged: a prior centred near the true ROI would produce
            excellent recovery and prove nothing.
          </p>

          <h3 style={{ marginTop: "2rem" }}>What this cannot tell you</h3>
          <p>
            Nothing here describes the effectiveness of any real marketing channel. The spend is
            invented, the effect sizes are chosen, and the ordering of attribution bias across
            channels is the only part with outside support — Blake, Nosko and Tadelis (2015)
            switched eBay's paid search off and found returns to branded keywords
            indistinguishable from zero. The magnitudes here are still chosen, not measured.
          </p>
          <p>
            The revenue baseline is treated as if it were a no-advertising counterfactual. Olist
            plainly did market itself over 2017–18, so the real series already contains real media
            effects. Layering a simulated effect on top does not recover a clean counterfactual;
            it produces a series whose <em>simulated</em> component has a known truth. That is the
            only claim made.
          </p>
          <p>
            Currency is Brazilian reais from 2017–18. Every dirham figure on this page is a stated
            scenario at a nominal rate, present so the magnitudes are legible at the scale the
            brief describes, and is not a measurement of anything.
          </p>

          <h3 style={{ marginTop: "2rem" }}>Reproducing it</h3>
          <p className="footnote">
            Window selected by rule: {frame.window.rule} Configuration{" "}
            <span className="num">{panel.provenance.dgp_hash}</span>, seed{" "}
            <span className="num">{panel.provenance.seed}</span>.
            {gate && (
              <>
                {" "}
                Fitted on {gate.platform.system} {gate.platform.machine}, Python{" "}
                {gate.platform.python}, pymc-marketing {gate.versions["pymc-marketing"]}.
              </>
            )}{" "}
            Every number on this page is read from a committed artifact; none is typed into the
            page.
          </p>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const ready = available("criteo", "frame", "panel");
  return (
    <>
      <Masthead />
      <main>
        {ready ? (
          <>
            <Hero />
            <WhatIsReal />
            <FindableSection />
            <AttributionSection />
            <RecoverySection />
            <ExperimentSection />
            <ClvSection />
            <DecisionSection />
            <Method />
          </>
        ) : (
          <section>
            <div className="shell">
              <h1>ATHAR</h1>
              <p className="lede">
                No metrics artifacts found. Run <code>make all</code> to generate them.
              </p>
            </div>
          </section>
        )}
      </main>
      <footer style={{ borderTop: "1px solid var(--rule-2)", padding: "3rem 0" }}>
        <div className="shell footnote">
          ATHAR · MAIB Term 4 · SP Jain School of Global Management, Dubai · Krishna Mathur.
          Nothing here is legal or financial advice, and no figure describes a real advertiser.
        </div>
      </footer>
    </>
  );
}
