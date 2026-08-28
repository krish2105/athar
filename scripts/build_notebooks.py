"""Generate the eight numbered notebooks.

The notebooks carry the argument; the scripts in this directory do the expensive
computation; ``metrics/*.json`` is the interface between them. A notebook
recomputes anything cheap enough to recompute — so a reader can see the working —
and reads the artifact for anything that took an hour of sampling.

Written by a generator rather than by hand because eight notebooks share a
preamble, a house style and a set of assertions, and eight hand-maintained copies
of those drift. Re-running this regenerates them from one place.

Run: `make notebooks-src`
"""

import json
import logging

from athar import paths

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_notebooks")

PREAMBLE = """import json
import warnings

import numpy as np
import pandas as pd

from athar import paths
from athar.provenance import read_metric

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

METRICS = paths.metrics_dir()
PROCESSED = paths.processed_dir()


def show(frame, caption=""):
    if caption:
        print(caption)
    print(frame.to_string(index=False))
    print()
"""


def _lines(source: str) -> list[str]:
    """Split into nbformat's source list, keeping the trailing newlines.

    nbformat stores a cell's source as a list of strings that concatenate back to
    the original text, so each entry has to carry its own newline. Splitting
    without them produces a file that loads and then runs every line jammed onto
    one — which is exactly what happened.
    """
    text = source.strip("\n")
    return [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def notebook(cells: list[dict]) -> dict:
    # Stable ids derived from position rather than randomly generated, so
    # regenerating an unchanged notebook produces an unchanged file.
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "ATHAR", "language": "python", "name": "athar"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> dict[str, dict]:
    books: dict[str, dict] = {}

    # ---------------------------------------------------------------- 01
    books["01_frame"] = notebook(
        [
            markdown("""
# 01 — The frame, and the window chosen by rule

Everything real in ATHAR rests on this notebook. Olist gives 99,441 orders across
two years of Brazilian e-commerce; what comes out is a weekly revenue series, a
state-by-week panel, and a per-customer summary.

Two defects bound the usable history, and neither is a business fact. Collection
does not begin cleanly — the 2016 weeks are interrupted by a hole where no orders
were recorded at all — and it stops mid-week at the end, so the final weeks tail
off toward a single order. Choosing a window by eye would be indefensible, so it
is chosen by a rule stated before the data was inspected.
"""),
            code(PREAMBLE),
            code("""
frame = read_metric("frame", METRICS)
print(frame["window"]["rule"])
print()
gaps = frame["verification"]["zero_order_weeks_in_window"]
print("window:", frame["window"]["start"], "to", frame["window"]["end"],
      f"({frame['window']['weeks']} weeks, {gaps} with no orders)")
"""),
            markdown("""
## What the rule keeps

98.5% of orders and 98.7% of revenue survive, and the window contains no gaps.
"""),
            code("""
totals = frame["totals"]
coverage = frame["coverage"]
show(pd.DataFrame([
    {"quantity": "orders", "in window": totals["orders"], "all dates": coverage["orders_all_dates"],
     "share kept": round(coverage["share_of_orders_in_window"], 4)},
    {"quantity": "revenue (BRL)", "in window": round(totals["revenue_brl"], 2),
     "all dates": round(coverage["revenue_all_dates_brl"], 2),
     "share kept": round(coverage["share_of_revenue_in_window"], 4)},
]))
print("mean order value:", totals["mean_order_value_brl"], "BRL")
print("states:", totals["states"], " people:", totals["people"])
"""),
            markdown("""
## Verification: an independent recomputation

`athar.frame` stages the join through CTEs; the check in `scripts/build_frame.py`
uses a flat join and filter, sharing no code path. The two must agree.

They agree on the order count exactly and on revenue to 1.0e-12 relative — not
bit-for-bit, and they cannot be. Float addition is not associative, the two plans
accumulate 96,731 terms in different orders, and DuckDB is already pinned to one
thread. So the test is agreement to the centavo, which is the resolution the
quantity actually has.
"""),
            code("""
verification = frame["verification"]
for key, value in verification.items():
    print(f"{key:52s} {value}")
"""),
            markdown("""
## The geography, and the power problem it creates

One state carries 38% of revenue; thirteen of the twenty-seven carry under 1%
each. That concentration is what makes a geo holdout expensive, and it is Olist's,
not this project's invention. Notebook 05 measures the consequence.
"""),
            code("""
geo = frame["geo_panel"]
print("states:", geo["states"], "| balanced panel:", geo["balanced"], "| rows:", geo["rows"])
print("largest state share of revenue:", round(geo["largest_state_revenue_share"], 4))
print("states below 1% of revenue:", geo["states_below_one_percent_of_revenue"])
"""),
            markdown("""
## The repeat rate, which decides notebook 06

A person is `customer_unique_id`. Olist issues a fresh `customer_id` per order, so
keying on that would report a repeat rate of exactly zero — a mistake that would
have made the CLV notebook meaningless rather than merely difficult.
"""),
            code("""
repeat = frame["repeat_behaviour"]
show(pd.DataFrame([
    {"basis": "all dates", **repeat["all_dates"]},
    {"basis": "in window", **{k: v for k, v in repeat["in_window"].items()}},
]))
print(repeat["note"])
"""),
            markdown("""
**Carried forward.** A gap-free 85-week weekly series with a real trend and real
seasonality, which notebook 02 uses as the non-media baseline; a balanced 27-state
panel for notebook 05; and a 3% repeat rate that notebook 06 has to take seriously.
"""),
        ]
    )

    # ---------------------------------------------------------------- 02
    books["02_panel"] = notebook(
        [
            markdown("""
# 02 — The instrument: a panel whose truth is known

No real advertiser knows the true ROI of its own media plan. That is the problem
the whole field is organised around, and it means "did the model recover the
truth?" cannot be asked of real data at all.

So it is asked here instead, of a panel constructed for the purpose, from a
configuration committed before any model was fitted. **This is an instrument, not
a substitute for data that was unavailable.** Nothing it produces describes the
effectiveness of a real marketing channel.
"""),
            code(PREAMBLE),
            code("""
from athar import dgp

config = dgp.load_config()
panel_metrics = read_metric("panel", METRICS)
print("configuration digest:", config.digest)
print("seed:", config.spec["seed"])
print()
print(panel_metrics["provenance"]["caveat"])
"""),
            markdown("""
## The generating process

Spend is generated in logs from two latent AR(1) factors — budget pressure, which
moves every channel together, and funnel tilt, which trades performance spend
against brand and upper-funnel spend. Two factors rather than one because a single
factor can only produce a rank-one correlation structure, in which every pair of
channels correlates in the same direction, which is not what a media plan looks
like.

Effect is delayed-geometric adstock followed by a Hill curve. The coefficient on
each channel is solved so its average ROI lands exactly on the pre-registered
target.
"""),
            code("""
show(pd.DataFrame(panel_metrics["channels"])[[
    "channel", "mean_weekly_spend", "true_roi_average", "true_roi_marginal",
    "adstock_alpha", "adstock_theta", "adstock_peak_lag_weeks", "hill_slope",
]], "The five channels as configured")
"""),
            markdown("""
## Average ROI is not marginal ROI, and the difference is not a constant

Average ROI divides total contribution by total spend, and is what a media-mix
model usually reports. Marginal ROI is the slope at the current plan, and is what
a reallocation decision turns on.

Which is larger is *not* fixed. A Hill curve with a slope above one is S-shaped:
below its inflection the response is convex, marginal exceeds average, and the
channel is under-invested. Above it, the curve is concave and marginal falls
below average. Both regimes are present here at the configured spend, which is
what makes notebook 08 an allocation problem rather than a ranking exercise.
"""),
            code("""
channels = pd.DataFrame(panel_metrics["channels"])
channels["regime"] = np.where(
    channels["true_roi_marginal"] > channels["true_roi_average"],
    "convex — under-invested", "concave — saturating")
show(channels[["channel", "true_roi_average", "true_roi_marginal",
               "marginal_over_average", "regime"]])
"""),
            markdown("""
## The deliberate misspecification

The panel is generated with a kernel that peaks at a positive lag. The headline
model in notebook 03 is fitted with plain geometric adstock, which is the same
kernel with its delay pinned to zero and therefore cannot express a delayed peak
at all, and with a logistic curve that cannot take the Hill shape.

Fitting the generating form to its own output recovers the assumptions and
measures nothing. A matched arm runs as a control so the two sources of error can
be told apart.

One practical note that shaped the design: the kernel was originally a Weibull
PDF. `WeibullPDFAdstock` raises `TypeError: x must be have an XTensorType` under
pymc-marketing 0.19.2 with pytensor 2.38.2, for every saturation. `WeibullCDFAdstock`
samples cleanly but decays monotonically and cannot represent a delayed peak
either. The delayed-geometric form both admits a delayed peak and can actually be
fitted, which is what makes the matched arm genuinely matched.
"""),
            code("""
for key, value in panel_metrics["generating_specification"].items():
    print(f"{key}:\\n  {value}\\n")
"""),
            markdown("""
## How hard the identification problem is

Two numbers bound what any model can do here, and both are reported per cell of
the recovery grid in notebook 04.

The condition number and VIF say how separable the channels are. The media share
of *detrended* revenue variance says how much signal there is to separate — the
raw share is dominated by Olist's fivefold growth, which any model absorbs into
its own trend terms rather than having to explain with media.
"""),
            code("""
ident = panel_metrics["identification"]
print("max pairwise correlation :", round(ident["max_pairwise_correlation"], 4))
print("condition number         :", round(ident["condition_number"], 3))
print("max VIF                  :", round(ident["max_vif"], 3))
print("media share of variance  :", round(ident["media_share_of_revenue_variance"], 4), "(raw)")
detrended = round(ident["media_share_of_detrended_variance"], 4)
print("                          ", detrended, "(detrended — the one to quote)")
print()
print(ident["note"])
print()
show(pd.DataFrame(ident["correlation"]).round(3).reset_index().rename(columns={"index": ""}),
     "Realised spend correlation")
"""),
            markdown("""
## Attribution, and its one pre-registered knob

Last-click here is a parametric caricature, not a simulated journey: a tracking
rate, the share of a channel's true contribution it observes at all, and an
organic capture rate, the share of *baseline* revenue it credits to that channel.
The second is the mechanism the whole project is about.

Simulating individual journeys would look more faithful without being more honest
— the journey parameters would be exactly as invented, only harder to state. The
cost of this choice is that no claim is made about attribution *mechanics*, only
about the consequences of a stated bias.

`search_nonbrand` carries a tracking rate of 1 and an organic capture of 0, so
last-click recovers its true ROI exactly. That case is in the design deliberately.
"""),
            code("""
show(channels[["channel", "true_roi_average", "lastclick_roas",
               "lastclick_bias_relative", "tracking_rate", "organic_capture"]],
     "What last-click would report")
print("most overstated :", panel_metrics["attribution_summary"]["most_overstated"])
print("most understated:", panel_metrics["attribution_summary"]["most_understated"])
print()
print(panel_metrics["attribution_summary"]["null_case"])
"""),
            markdown("""
## Verification: the stored truth is the panel's actual truth

Recomputed straight from the generated series, sharing no code path with the solve
that produced the coefficients. A drifted truth would make every recovery number in
notebooks 03 and 04 wrong in a way nothing downstream could detect.
"""),
            code("""
for key, value in panel_metrics["verification"].items():
    print(f"{key:48s} {value}")
"""),
        ]
    )

    # ---------------------------------------------------------------- 03
    books["03_mmm"] = notebook(
        [
            markdown("""
# 03 — The media-mix model, fitted wrong on purpose

Two fits on the panel with the real Olist baseline. The **misspecified** arm leads,
because it is the situation an analyst is actually in: nobody knows the true
functional form. The **matched** arm is a control, and reporting only it would be
circular.

Priors are pymc-marketing's defaults, unchanged. This matters more than it sounds:
prior choice is where circularity gets into a recovery study without anyone
noticing, and a prior centred near the true ROI would produce excellent recovery
and prove nothing. The consequence is wide intervals, which is a finding rather
than a defect.
"""),
            code(PREAMBLE),
            code("""
mmm = read_metric("mmm", METRICS)
print(mmm["truth_access"])
print()
for name, description in mmm["design"]["specifications"].items():
    print(f"{name:14s} {description}")
"""),
            markdown("""
## Did the sampler work

A fit that fails its diagnostics is reported as failed rather than folded into a
result. The threshold is a divergence *rate* rather than a count, because a count
tightens as you sample more and is therefore not a rule.
"""),
            code("""
show(pd.DataFrame([
    {"specification": name, **{k: v for k, v in fit["diagnostics"].items()
                                if k not in ("criteria", "criteria_strict")}}
    for name, fit in mmm["fits"].items()
]))
"""),
            markdown("""
## Recovery

Coverage leads. Whether the true value falls inside the interval is the property
the model actually claims, and it is checkable; a point estimate that lands close
on one draw is an anecdote.
"""),
            code("""
for name, fit in mmm["fits"].items():
    rows = [{"channel": c, **{k: round(v, 4) if isinstance(v, float) else v
                              for k, v in entry.items()}}
            for c, entry in fit["average_roi"]["channels"].items()]
    show(pd.DataFrame(rows)[["channel", "true", "estimated_mean", "hdi_low", "hdi_high",
                             "covered", "relative_error"]],
         f"{name} — average ROI")
    print("  summary:", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in fit["average_roi"]["summary"].items()})
    print()
"""),
            markdown("""
## Marginal ROI, which is what the budget decision needs

Scored separately, because a model can be respectable on average ROI and useless
on the slope — and the slope is the quantity notebook 08 allocates on.
"""),
            code("""
for name, fit in mmm["fits"].items():
    summary = fit["marginal_roi"]["summary"]
    print(f"{name:14s} coverage {summary['coverage_rate']:.2f}  "
          f"median |rel err| {summary['median_absolute_relative_error']:.2f}  "
          f"mean interval width {summary['mean_interval_width']:.2f}")
"""),
            markdown("""
## What the model recovered about the transforms

The misspecified arm cannot represent a delayed peak, so whatever the true delay
was has to be absorbed somewhere else — usually into the decay rate and the
coefficient. This is where that shows.
"""),
            code("""
print(json.dumps(mmm["fits"]["misspecified"]["recovered_parameters"], indent=2)[:1800])
"""),
        ]
    )

    # ---------------------------------------------------------------- 04
    books["04_meridian"] = notebook(
        [
            markdown("""
# 04 — Google Meridian, as an independent comparison

pymc-marketing and Meridian are the two open media-mix implementations in current
use. They differ in more than syntax: Meridian is written for a geo-hierarchical
design, uses its own priors, and samples with TensorFlow Probability rather than
PyMC.

Running both on the same panel is a cross-implementation check in the same spirit
as the two CLV fits in notebook 06 — agreement between independently written
implementations is evidence, in a way a model agreeing with itself is not.

The gate in `metrics/env_gate.json` records whether Meridian could be imported and
exercised at all on this machine. A comparison that could not run is recorded as
such rather than quietly dropped.
"""),
            code(PREAMBLE),
            code("""
gate = read_metric("env_gate", METRICS)
print("platform:", gate["platform"])
print()
for name, result in gate["gates"].items():
    print(f"{name:18s} {result['status']}")
    if result["status"] == "fail":
        print("   ", result["error"])
"""),
            code("""
meridian_gate = gate["gates"]["meridian"]
if meridian_gate["status"] != "pass":
    print("Meridian did not pass its gate on this machine; the comparison below did not run.")
    print(meridian_gate.get("error"))
else:
    print("Meridian imported and exercised successfully:")
    print(json.dumps(meridian_gate["detail"], indent=2))
"""),
            markdown("""
## Why the comparison is reported at the level of the gate

Meridian's data interface expects a geo-by-time array with its own coordinate
conventions and its own notion of media, reach and frequency inputs. Adapting the
panel to it faithfully is a piece of work in its own right, and adapting it
*unfaithfully* — flattening the geo dimension, guessing at the population scaling —
would produce a number that looks like a comparison and is not one.

What is claimed here is therefore narrow and true: Meridian installs and runs on
this machine, which was not obvious on Apple silicon with TensorFlow Probability,
and the version is recorded. What is **not** claimed is a fitted Meridian ROI for
this panel. An unfaithful adaptation would be worse than an absent one, and saying
so is more useful than a number nobody should trust.
"""),
            code("""
print("tensorflow:", gate["versions"].get("tensorflow"))
print("google-meridian:", gate["versions"].get("google-meridian"))
print("pymc-marketing:", gate["versions"].get("pymc-marketing"))
"""),
        ]
    )

    # ---------------------------------------------------------------- 05
    books["05_recovery"] = notebook(
        [
            markdown("""
# 05 — The recovery grid

Forty full-posterior fits over a factorial design: two panel lengths, two
collinearity levels, two specifications, five seeds. This notebook fits nothing;
it reads what `scripts/run_recovery.py` produced.

Both length arms use the *simulated* extended baseline, including the 85-week arm,
so that length is the only quantity changing along the length axis. Only the
headline fit in notebook 03 uses the real Olist baseline.
"""),
            code(PREAMBLE),
            code("""
recovery = read_metric("recovery", METRICS)
print(json.dumps(recovery["design"], indent=2))
print()
print(json.dumps(recovery["convergence"], indent=2))
"""),
            markdown("""
## Coverage, and why coverage alone is not enough

Coverage is the property a Bayesian model claims: the truth should fall inside the
89% interval about 89% of the time. But an interval can achieve perfect coverage by
being uselessly wide, so interval width is reported beside it. A cell with coverage
1.00 and a median error of twenty is not a success.
"""),
            code("""
rows = []
for name, slice_ in recovery["slices"].items():
    for quantity in ("average_roi", "marginal_roi"):
        block = slice_[quantity]
        if block.get("converged", 0) == 0:
            rows.append({"cell": name, "quantity": quantity, "converged": 0})
            continue
        rows.append({
            "cell": name, "quantity": quantity,
            "converged": block["converged"],
            "coverage": block["coverage_rate"],
            "median |rel err|": round(block["median_absolute_relative_error"], 3),
            "mean interval width": round(block["mean_interval_width"], 3),
        })
show(pd.DataFrame(rows).sort_values(["quantity", "cell"]), "Every cell of the grid")
"""),
            markdown("""
## Does the choice of convergence rule change the conclusion

The primary rule uses a divergence rate; the strict rule demands zero divergences
and rejected almost every fit when it was first applied. Both are computed for
every cell so the effect of the choice is visible rather than asserted.
"""),
            code("""
rows = []
for name, slice_ in recovery["slices"].items():
    primary = slice_["average_roi"]
    strict = slice_.get("average_roi_strict_convergence", {})
    rows.append({
        "cell": name,
        "converged (rate rule)": primary.get("converged", 0),
        "coverage (rate rule)": primary.get("coverage_rate"),
        "converged (strict)": strict.get("converged", 0),
        "coverage (strict)": strict.get("coverage_rate"),
    })
show(pd.DataFrame(rows))
print(recovery["convergence"]["rule_note"])
"""),
            markdown("""
## Error against the identification diagnostics

If recovery error tracks the condition number and the media signal share, then the
failures are a property of the *design* rather than of the method — which is the
more useful thing for a practitioner to know, because a design is something they
control.
"""),
            code("""
fits = pd.DataFrame([{
    "weeks": f["weeks"], "collinearity": f["collinearity"], "specification": f["specification"],
    "converged": f["diagnostics"]["passed"],
    "condition_number": f["identification"]["condition_number"],
    "media_signal": f["identification"]["media_share_of_detrended_variance"],
    "coverage": f["average_roi"]["summary"]["coverage_rate"],
    "median_abs_rel_error": f["average_roi"]["summary"]["median_absolute_relative_error"],
    "mean_interval_width": f["average_roi"]["summary"]["mean_interval_width"],
} for f in recovery["fits"]])
usable = fits[fits["converged"]]
show(usable.groupby(["specification", "collinearity", "weeks"]).agg(
    fits=("coverage", "size"),
    coverage=("coverage", "mean"),
    median_error=("median_abs_rel_error", "median"),
    interval_width=("mean_interval_width", "mean"),
    condition=("condition_number", "mean"),
    signal=("media_signal", "mean"),
).round(3).reset_index())
"""),
            markdown("""
## Per channel

Which channels are recoverable is not uniform, and the pattern is worth more than
the average: a channel that is never recovered is a channel a media-mix model
should not be used to budget.
"""),
            code("""
rows = []
for name, slice_ in recovery["slices"].items():
    block = slice_["average_roi"]
    for channel, entry in block.get("per_channel", {}).items():
        rows.append({"cell": name, "channel": channel, **entry})
if rows:
    per_channel = pd.DataFrame(rows)
    show(per_channel.groupby("channel").agg(
        true=("true", "first"),
        median_estimate=("median_estimate", "median"),
        coverage=("coverage_rate", "mean"),
        median_rel_error=("median_relative_error", "median"),
    ).round(3).reset_index(), "Across every converged cell")
"""),
        ]
    )

    # ---------------------------------------------------------------- 06
    books["06_uplift"] = notebook(
        [
            markdown("""
# 06 — Criteo: the real causal result, and whether the effect is findable

This is the only notebook in ATHAR whose causal claim rests on data rather than on
a construction. Criteo-UPLIFT v2.1 is a genuine randomised trial over 13,979,592
users, 85% of whom were eligible to be shown advertising and 15% held back.

Two questions. First, how far does what a platform reports sit from what actually
happened — computed on the full population, no sampling. Second, given twelve
anonymised features, can a model find *who* the advertising moved.
"""),
            code(PREAMBLE),
            code("""
criteo = read_metric("criteo", METRICS)
print("rows:", f"{criteo['population']['rows']:,}",
      "| treated share:", criteo["population"]["treated_share"])
print("exposure leaking into the control arm:",
      criteo["randomisation_check"]["exposure_in_control_arm"])
print()
print(criteo["randomisation_check"]["note"])
"""),
            markdown("""
## The intent-to-treat effect

Assignment was random, so the difference between arms is the causal effect of
being eligible for advertising. No adjustment, no model.
"""),
            code("""
itt = criteo["intent_to_treat"]
conversion = criteo["conversion"]
print(f"treated  {conversion['treated_rate']:.6f}  "
      f"[{conversion['treated_ci'][0]:.6f}, {conversion['treated_ci'][1]:.6f}]")
print(f"control  {conversion['control_rate']:.6f}  "
      f"[{conversion['control_ci'][0]:.6f}, {conversion['control_ci'][1]:.6f}]")
print()
print(f"absolute lift {itt['absolute_lift']:.8f}  95% CI "
      f"[{itt['ci_95'][0]:.8f}, {itt['ci_95'][1]:.8f}]")
print(f"relative lift {itt['relative_lift']:.4f}")
print(f"incremental conversions {itt['incremental_conversions']:,.0f}")
"""),
            markdown("""
## What a platform reports, against what happened

Both quantities below are correct arithmetic on the same trial. The first
conditions on exposure, which is decided *after* randomisation — who saw an ad
depended on who was browsing, and browsing predicts buying.

Stated as a count, advertising is overstated by a factor of about 1.7. Stated as a
conversion *rate*, by a factor of about 28. Both are reported, because only 3.6% of
the treated arm was ever exposed, and quoting only the dramatic framing would be
rhetoric rather than a finding.
"""),
            code("""
gap = criteo["platform_reported_versus_incremental"]
naive = gap["naive_rate_framing"]
print(f"platform-reported conversions {gap['platform_reported_conversions']:,}")
print(f"incremental conversions       {gap['incremental_conversions']:,.0f}")
print(f"overstatement                 {gap['overstatement_ratio']}x")
print()
print(f"conversion rate among exposed {naive['conversion_rate_among_exposed']:.6f}")
print(f"conversion rate in control    {naive['conversion_rate_in_control']:.6f}")
print(f"naive rate ratio              {naive['naive_rate_ratio']}x")
print(f"true intent-to-treat ratio    {naive['true_relative_lift_ratio']}x")
print(f"share of treated arm exposed  {naive['exposed_share_of_treated_arm']:.4f}")
"""),
            markdown("""
## The defensible version of "effect on the exposed"

Random assignment as an instrument for exposure. This recovers the effect on those
the advertising actually reached, without the selection that makes the naive
version meaningless.
"""),
            code("""
cace = criteo["complier_effect"]
print(f"compliance rate {cace['compliance_rate']:.6f}")
low, high = cace["ci_95"]
print(f"complier effect {cace['cace']:.6f}  95% CI [{low:.6f}, {high:.6f}]")
print()
print(cace["interpretation"])
"""),
            markdown("""
## Can a model find who was moved

Four learners and a random-targeting baseline, scored by Qini on a held-out half
with a bootstrap interval. Qini subtracts the random-targeting line, so it scores
only what the *ranking* contributed — a model with no uplift signal scores zero
however large the average effect happens to be.

If nothing beats random, that is the finding, and it is a statement about what
twelve anonymised and heavily repeated features support rather than about uplift
modelling.
"""),
            code("""
try:
    uplift = read_metric("uplift", METRICS)
except FileNotFoundError:
    uplift = None
    print("metrics/uplift.json not present; run `make uplift`.")

if uplift:
    print("test-half control converters:", f"{uplift['sample']['test_control_converters']:,}")
    print()
    rows = [{"model": name, "qini": round(r["qini"], 5),
             "ci_low": round(r["ci_low"], 5), "ci_high": round(r["ci_high"], 5),
             "beats random": r["beats_random"]}
            for name, r in uplift["models"].items()]
    show(pd.DataFrame(rows).sort_values("qini", ascending=False))
    print("verdict:", json.dumps(uplift["verdict"], indent=2))
"""),
            code("""
if uplift:
    best = uplift["verdict"]["best_model"]
    if best:
        show(pd.DataFrame(uplift["models"][best]["targeting_curve"]).round(4),
             f"Targeting curve — {best}")
        show(pd.DataFrame(uplift["models"][best]["deciles"]).round(6),
             f"Incremental response by decile — {best}")
"""),
            markdown("""
## Effective sample size

Criteo's twelve features are anonymised and heavily repeated. Where the distinct
share is well below one, the nominal row count overstates the precision available
to a model fitted on those features. It does not affect the treatment-effect
estimates above, which depend on the assignment rather than on the features.
"""),
            code("""
effective = criteo["effective_sample"]
print(f"distinct feature vectors {effective['distinct_feature_vectors']:,} "
      f"of {criteo['population']['rows']:,} ({effective['distinct_share']:.4f})")
print()
print(effective["note"])
"""),
        ]
    )

    # ---------------------------------------------------------------- 07
    books["07_clv"] = notebook(
        [
            markdown("""
# 07 — Customer lifetime value, and a model that does not fit

3.03% of Olist's customers ever bought twice, and those who did averaged 1.11
repeat purchases. This notebook establishes what follows from that, which is more
interesting than a fitted curve would have been.
"""),
            code(PREAMBLE),
            code("""
try:
    clv = read_metric("clv", METRICS)
except FileNotFoundError:
    clv = None
    print("metrics/clv.json not present; run `make clv`.")

if clv:
    print(json.dumps(clv["repeat_behaviour"], indent=2))
"""),
            markdown("""
## Maximum likelihood does not converge, and it is not a tuning problem

BG/NBD is the standard model and `lifetimes` is the reference implementation. It
does not converge here — at any time scale, at any penalty, on the full base or on
the repeaters alone. The likelihood returns NaN and the parameters run off in log
space.

The reason is structural. BG/NBD's dropout parameters describe the shape of a Beta
distribution over the probability of churning after each purchase, and they are
identified only by the *pattern* of repeat purchasing. With repeaters averaging
1.11 repeats there is no pattern for them to be estimated from, so the likelihood
is flat in those directions.
"""),
            code("""
if clv:
    ml = clv["maximum_likelihood"]
    show(pd.DataFrame(ml["attempts_full_base"])[["time_unit", "penalizer", "converged"]],
         "Full base")
    show(pd.DataFrame(ml["attempts_repeaters_only"])[["time_unit", "penalizer", "converged"]],
         "Repeaters only")
    print(ml["finding"])
"""),
            markdown("""
## The Bayesian fit converges because its priors do the work

This is not the Bayesian model being better. pymc-marketing parameterises dropout
with a Pareto prior whose shape parameter is one, which has no finite mean — an
extremely diffuse prior. If the posterior on those parameters barely narrows
against it, the data has told us nothing about the dropout process, which is
exactly what maximum likelihood was unable to estimate.

The ratio of posterior to prior standard deviation is reported per parameter so
that can be read rather than assumed.
"""),
            code("""
if clv:
    parameters = clv["models"]["bgnbd_bayesian_full_base"]["parameters"]
    print("declared priors:")
    for name, prior in parameters.get("declared_priors", {}).items():
        print(f"  {name:16s} {prior}")
    print()
    rows = [{"parameter": name, **{k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in entry.items()}}
            for name, entry in parameters.items() if name != "declared_priors"]
    if rows:
        show(pd.DataFrame(rows))
    print(clv["models"]["bgnbd_bayesian_full_base"]["note"])
"""),
            markdown("""
## Validation on a time-based holdout

Never a random split: the prediction is "how many purchases in the next N weeks",
and a random split would let the model see the future of the customers it is
forecasting.

The comparison that matters is against a baseline that predicts nothing. A
near-zero prediction matching a near-zero outcome is the model working; it is only
*skill* if it beats predicting zero.
"""),
            code("""
if clv:
    validation = {k: v for k, v in clv["validation"].items() if not isinstance(v, str)}
    for key, value in validation.items():
        print(f"{key:44s} {value}")
    print()
    print(clv["validation"]["metric_note"])
"""),
            markdown("""
## What lifetime value comes to, and why it decides notebook 08

If lifetime value is very nearly proportional to first-order value, then weighting
a media allocation by lifetime value and weighting it by immediate revenue rank the
channels identically and produce the same budget. The CLV-weighted reallocation the
brief invites is then not a different answer — it is the same answer with more
steps, and saying so is the finding.
"""),
            code("""
if clv:
    value = clv["lifetime_value"]
    for key, item in value.items():
        print(f"{key:44s} {item}")
    print()
    print(clv["finding"])
"""),
        ]
    )

    # ---------------------------------------------------------------- 08
    books["08_triangulate"] = notebook(
        [
            markdown("""
# 08 — Triangulation: where the three disagree, and what that costs

Three estimates of what each channel returns, and they do not agree.

**Attribution** is available daily, for every channel, at no cost, and is biased by
however much organic demand flows through that channel's last click.
**Experiments** are unbiased and are the only thing here that is, but cost weeks of
switched-off spend for one channel at a time. **A media-mix model** covers every
channel at once and is the only one that can say what the *next* dirham buys,
resting on assumptions about functional form that its own data cannot verify.

Reconciliation here does not mean averaging them. Averaging a biased estimator
with an unbiased one produces a biased estimator with a smaller variance, which is
worse to hand a decision-maker than either input because it looks more trustworthy
than it is.
"""),
            code(PREAMBLE),
            code("""
triangulation = read_metric("triangulation", METRICS)
allocator = read_metric("allocator", METRICS)
print(triangulation["method_notes"]["why_not_average"])
print()
print(triangulation["method_notes"]["linear_versus_curved"])
"""),
            markdown("""
## The three estimates, per channel

The divergence score is the coefficient of variation across the available
estimates. A high score is the signal a marketer should act on: it says the methods
are telling different stories about this channel, and that no amount of dashboard
polish will settle which is right without an experiment.
"""),
            code("""
rows = []
for channel, entry in triangulation["comparison"]["channels"].items():
    rows.append({
        "channel": channel,
        "true": round(entry["true_roi"], 3),
        "attribution": round(entry["attribution"]["estimate"], 3),
        "mmm": round(entry["mmm"]["estimate"], 3) if entry["mmm"]["estimate"] else None,
        "experiment": round(entry["experiment"]["estimate"], 3)
                       if entry["experiment"].get("estimate") else None,
        "divergence": round(entry["divergence"], 3),
        "closest": entry["closest_method"],
    })
show(pd.DataFrame(rows))
print("most divergent :", triangulation["comparison"]["summary"]["most_divergent_channel"])
print("least divergent:", triangulation["comparison"]["summary"]["least_divergent_channel"])
"""),
            markdown("""
## Turning each into a budget

Every allocation runs under identical governance constraints — a floor and a cap
per channel. Without them a scalar ROI table sends the whole budget to one channel,
and the comparison would be against a caricature no marketing organisation
resembles.

Only the media-mix model produces a *curve*, so only it can allocate on curvature.
An experiment and an attribution report each return a single number per channel and
say nothing about the next dirham, so a planner holding only those must treat
returns as constant.
"""),
            code("""
print(json.dumps(allocator["governance"], indent=2))
print()
rows = []
for name, entry in allocator["allocations"].items():
    row = {"allocation": name}
    row.update({channel: round(share, 3) for channel, share in entry["shares"].items()})
    row["revenue under truth"] = round(entry["revenue_under_truth"], 0)
    row["shortfall"] = round(entry["shortfall_share"], 4)
    rows.append(row)
show(pd.DataFrame(rows).sort_values("shortfall"))
"""),
            markdown("""
## The answer

Every allocation is scored against the *same* true response curves, so what is
compared is the consequence of believing an estimator rather than the estimator's
own opinion of itself. The benchmark is the allocation built from the true curves —
a ceiling nobody can reach, and the right benchmark precisely because it separates
"this estimator is wrong" from "this problem is hard".
"""),
            code("""
headline = triangulation["headline"]
for key, value in headline.items():
    if key != "statement":
        print(f"{key:44s} {value}")
print()
print(headline["statement"])
"""),
            markdown("""
## The same figure at the scale the brief describes

A stated scenario, not a measurement. Olist trades in 2017-18 Brazilian reais and
this project makes no claim about any exchange rate; the shortfall is a share of
budget, so it scales.
"""),
            code("""
scenario = triangulation["scenario_in_aed"]
print(scenario["caveat"])
print()
print(scenario["on_a_one_million_aed_budget"]["note"])
"""),
            markdown("""
## And the lifetime-value consequence

If lifetime value is proportional to first-order value, the CLV-weighted
reallocation is the same reallocation.
"""),
            code("""
if triangulation.get("clv_consequence"):
    for key, value in triangulation["clv_consequence"].items():
        print(f"{key}:\\n  {value}\\n")
else:
    print("metrics/clv.json was not present when this was generated.")
"""),
            markdown("""
---

**What would falsify this.** The ordering of attribution bias across channels is
pre-registered rather than measured, so a business whose brand search genuinely
drives incremental demand would invert the headline. The media-mix result is
conditional on a signal share of roughly 17% of detrended variance; a quieter media
plan is a harder recovery problem and the reported errors would grow. And the
allocation comparison assumes the governance floors and caps, without which every
scalar-ROI allocation collapses to a corner.
"""),
        ]
    )

    return books


def trim_preamble(content: dict) -> None:
    """Drop preamble imports the notebook never uses.

    Every notebook shares one preamble, but not every notebook touches json or
    numpy, and an unused import is a lint failure in a repository that lints its
    notebooks. Trimming here rather than afterwards keeps regeneration idempotent.
    """
    preamble = content["cells"][1]
    body = "".join(
        "".join(cell["source"]) for cell in content["cells"][2:] if cell["cell_type"] == "code"
    )
    source = "".join(preamble["source"])
    for statement, token in (("import json\n", "json."), ("import numpy as np\n", "np.")):
        if token not in body:
            source = source.replace(statement, "")
    preamble["source"] = _lines(source)


def main():
    directory = paths.project_root() / "notebooks"
    directory.mkdir(exist_ok=True)
    for name, content in build().items():
        trim_preamble(content)
        path = directory / f"{name}.ipynb"
        path.write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n")
        log.info("wrote %s (%d cells)", path.name, len(content["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
