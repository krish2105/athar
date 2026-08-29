# ATHAR

**Subject:** MAIB AI 208 — AI in Marketing
**Purpose:** Triangulated marketing incrementality: media-mix modelling, a randomised experiment and attribution, reconciled — and priced.

MAIB Term 4 · SP Jain School of Global Management, Dubai · Krishna Mathur

## The question

A CMO reallocating a budget has three sources of truth about what each channel
returns, and they disagree. **Attribution** is available daily for every channel at
no cost, and is biased by however much organic demand happens to flow through that
channel's last click. **An experiment** is unbiased and is the only thing here that
is, but costs weeks of switched-off spend for one channel at a time. **A media-mix
model** covers every channel at once and is the only one that can say what the
*next* dirham buys, resting on assumptions its own data cannot verify.

The deliverable is not a model. It is a **reconciliation**: what each method says,
how far each is from a truth none of them could see, and what believing the wrong
one costs in money on a fixed budget.

## Two findings, never blurred together

No public dataset carries a media-mix model, an experiment and an attribution log
for the same business. Pretending otherwise is the usual move, and it is the one
thing this project refuses. So there are two separate claims, and every chart,
table and paragraph says which one it belongs to.

### Real, no simulation

Criteo-UPLIFT v2.1 is a genuine randomised trial over **13,979,592** users.
Control-arm exposure is exactly **0**, so the randomisation is clean and `exposure`
is unambiguously post-treatment.

| | Value |
|---|---:|
| Platform-reported conversions | 23,031 |
| Incremental conversions (intent-to-treat) | 13,687 |
| **Overstatement** | **1.68×** |

Stated as a conversion *rate* instead, the same trial makes advertising look
**27.8×** better than control against a true intent-to-treat ratio of **1.59×**.
Both are correct arithmetic. Only 3.6% of the treated arm was ever exposed, so a
seventeen-fold exaggeration of the rate becomes a 1.68× exaggeration of the count.
Reporting only the dramatic framing would be rhetoric, so both are reported.

### Simulated, and labelled everywhere

A five-channel spend panel generated from `config/dgp.yaml`, committed before any
model was fitted, layered on the **real** Olist revenue baseline. Its purpose is a
known truth: no real advertiser knows its own true ROI, so "did the model recover
it?" is a question that exists only inside a simulation. Nothing derived from it
describes the effectiveness of a real marketing channel.

## Negative results, reported as results

- **Fitting the *right* functional form made recovery worse.** Across forty
  full-posterior fits, the matched arm — the one handed the shape that generated
  the data — reaches higher coverage (1.00 against 0.60) and is ten times further
  from the truth, with intervals six times wider and only 6 of 20 runs passing
  their sampler diagnostics against 15 of 20. It has more parameters to identify
  from the same weak signal, so it covers the truth by being unable to exclude
  anything. This is the project's clearest argument against reading coverage on
  its own.
- **More data made coverage worse.** For the misspecified model at high
  collinearity, going from 85 to 156 weeks moved coverage from 0.90 to 0.40 while
  the intervals narrowed. A misspecified model given more evidence becomes more
  confident about the wrong value.
- **BG/NBD cannot be fitted to Olist at all.** Maximum likelihood does not
  converge at any time scale or penalty, on the full base or on the repeaters
  alone. The MCMC fit runs but throws 3,507 divergences with `alpha` collapsing
  to 1.6e-306, and on a time-based holdout it predicts 43.6 repeat purchases
  against 557 actual — losing to a baseline that predicts zero. Its dropout
  parameters are identified only by the pattern of repeat purchasing, and with a
  3.03% repeat rate and repeaters averaging 1.11 repeats there is no pattern to
  identify them from.
- **Customer lifetime value is very nearly proportional to first-order value**, so
  a CLV-weighted reallocation and a revenue-weighted one produce the same budget.
  The story the brief invites cannot honestly be told on this data.
- **The channel most worth catching is the one experiments catch least.**
  `display_prog` has the lowest true ROI and is detected in 20–40% of holdout
  designs, against 78–90% for the channels with large effects.
- **`WeibullPDFAdstock` is broken** in pymc-marketing 0.19.2 with pytensor 2.38.2,
  which changed the design: see `reports/panel.md`.

## Setup

ATHAR depends on [SPINE](https://github.com/krish2105/spine), which supplies the
evaluation, splitting and decision primitives shared across this Term 4 portfolio.
It is consumed as an editable path dependency, so **the repositories must sit as
siblings**:

```
MAIB-Term4/
├── 01-spine/     # github.com/krish2105/spine
├── 03-athar/     # this repository
└── data/         # shared, outside every repo, never committed
```

```bash
uv sync --all-extras
cp .env.example .env      # DATA_ROOT points at the shared data store
```

Data lives outside the repository and is never committed, so two datasets have to
be fetched into `<DATA_ROOT>/raw/` before anything will run. Neither is
redistributable here.

| Directory | Dataset | Source |
|---|---|---|
| `raw/olist/` | Brazilian E-Commerce Public Dataset by Olist — 9 CSVs | Kaggle: `olistbr/brazilian-ecommerce` |
| `raw/criteo/` | Criteo Uplift Modeling Dataset v2.1 — `criteo-uplift-v2.1.csv`, 3.2 GB | [ailab.criteo.com/criteo-uplift-prediction-dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) |

`make frame` and `make criteo` fail with the expected path if either is missing,
rather than proceeding on partial data.

## Run

```bash
make all
```

Continuous integration runs the same lint and test gate, and separately builds the
dashboard and fails if the committed page no longer matches its source. It does
not run `make all`: the pipeline reads Olist and Criteo from a shared store that
is not redistributable, and forty full-posterior fits are not a CI job.

`lint` → `test` → `gate` → `frame` → `criteo` → `panel` → `mmm` → `uplift` → `clv`
→ `experiments` → `recovery` → `triangulate` → `notebooks` → `reports` → `card` →
`dashboard`.

`make gate` is worth knowing about: it imports and *exercises* the four
dependencies that can install cleanly and then fail on first use, recording the
outcome to `metrics/env_gate.json`. A comparison that could not run is recorded
rather than quietly dropped.

`make recovery` is the long pole — forty full-posterior fits, roughly an hour. It
caches each cell under a hash of its configuration, so an interruption costs one
fit rather than the batch.

## Layout

```
src/athar/      paths, provenance, frame, dgp, truth, attribution-in-dgp,
                experiments, mmm, uplift, clv, allocate, reconcile
scripts/        build_frame, build_criteo, build_panel, build_mmm, build_uplift,
                build_clv, build_experiments, run_recovery, build_triangulation,
                build_notebooks, render_reports, check_env, install_kernel
config/         dgp.yaml — the pre-registered data-generating process
notebooks/      01_frame … 08_triangulate
metrics/        every reported number, as JSON, each with a provenance block
reports/        committed markdown, generated from metrics/
dashboard/      Vite + React source for the results page
docs/           the built page — GitHub Pages target
tests/          mirrors src/athar
```

## Four things that are not obvious

**The synthetic caveat is enforced, not remembered.** Every artifact is written
through `athar.provenance`, which refuses one that descends from the simulated
panel without declaring itself synthetic, or one that cannot name the
configuration and seed behind it. `tests/test_provenance.py` walks every committed
artifact. The dashboard derives each badge from the artifact rather than taking it
as a prop, so a chart drawn from simulated data cannot render without one.

**The model is fitted wrong on purpose.** The panel is generated with
delayed-geometric adstock and a Hill curve; the headline model is fitted with
geometric adstock and a logistic curve, which cannot represent a delayed peak or
take the Hill shape. A model fitted with the form that generated the data recovers
its own assumptions and measures nothing. A matched arm runs as a control so the
cost of the wrong shape can be separated from the cost of a design that cannot
identify the parameters.

**The answer is locked away during fitting.** `athar.truth` refuses to return the
ground truth until the fit being scored already exists on disk. Priors are
pymc-marketing's defaults, unchanged — a prior centred near the true ROI would
produce excellent recovery and prove nothing. The consequence is wide intervals,
which is a finding rather than a defect.

**`spine.splitting` is deliberately unused on Criteo.** It carries no time column;
it is a randomised cross-section and is split as one. It *is* used everywhere this
project genuinely has a time axis — the media-mix holdout and the CLV calibration
window. The same treatment ADIL gives Home Credit.

## Limitations

Nothing here describes the effectiveness of any real marketing channel. The spend
is invented and the effect sizes are chosen. The one part with outside support is
the *ordering* of attribution bias across channels: Blake, Nosko and Tadelis
(2015), *Econometrica* 83(1), 155–174, switched eBay's paid search off across US
markets and found returns to branded keyword advertising indistinguishable from
zero. The magnitudes here remain chosen, not measured.

The revenue baseline is treated as if it were a no-advertising counterfactual.
Olist plainly did market itself over 2017–18, so the real series already contains
real media effects. Layering a simulated effect on top does not recover a clean
counterfactual; it produces a series whose *simulated* component has a known
truth. That is the only claim made.

Currency is Brazilian reais from 2017–18. Every dirham figure is a stated scenario
at a nominal rate, present so the magnitudes are legible at the scale the brief
describes, and is not a measurement of anything.

Nothing here is legal or financial advice.
