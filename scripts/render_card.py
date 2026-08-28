"""Render the ATHAR model card through SPINE's card renderer.

`spine.cards` refuses to render a metric without the split it was computed on. A
number without its split is not a result: "coverage 0.80" is unfalsifiable,
"coverage 0.80 on the 85-week high-collinearity misspecified cell, five seeds" can
be checked. Every metric below therefore names its split, and every value is read
from a `metrics/*.json` artifact rather than typed.

The card's data provenance is `simulated` for the media-mix half and the real
Criteo result is carried in its own metrics rather than blurred into it — the same
separation the rest of the project keeps.

Run: `make card`
"""

import json
import logging

import yaml
from spine.cards import CardValidationError, render_card, validate_card

from athar import paths

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("render_card")


def load(name):
    path = paths.metrics_dir() / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def metric(name, value, split, note=None):
    entry = {"name": name, "value": value, "split": split}
    if note:
        entry["note"] = note
    return entry


def build():
    criteo = load("criteo")
    panel = load("panel")
    mmm = load("mmm")
    recovery = load("recovery")
    clv = load("clv")
    triangulation = load("triangulation")
    frame = load("frame")

    metrics = []

    if criteo:
        gap = criteo["platform_reported_versus_incremental"]
        itt = criteo["intent_to_treat"]
        metrics += [
            metric(
                "Intent-to-treat lift (Criteo)",
                round(itt["absolute_lift"], 8),
                "full population, 13,979,592 rows, no sampling",
                f"95% CI {itt['ci_95'][0]:.8f} to {itt['ci_95'][1]:.8f}. Real randomised trial.",
            ),
            metric(
                "Platform-reported over incremental conversions",
                gap["overstatement_ratio"],
                "full population, 13,979,592 rows, no sampling",
                "Real data. The numerator conditions on exposure, which is post-treatment.",
            ),
        ]

    if mmm:
        for name, fit in mmm["fits"].items():
            summary = fit["average_roi"]["summary"]
            metrics.append(
                metric(
                    f"Average-ROI coverage, {name} fit",
                    summary["coverage_rate"],
                    "headline panel, 85 weeks, real Olist baseline",
                    f"{summary['channels_covered']}/{summary['channels_total']} channels "
                    f"inside the {summary['hdi_prob']:.0%} interval; median absolute "
                    f"relative error {summary['median_absolute_relative_error']:.2f}. "
                    f"Simulated media.",
                )
            )

    if recovery:
        convergence = recovery["convergence"]
        metrics.append(
            metric(
                "Recovery-grid fits converged",
                f"{convergence['converged']}/{convergence['fits']}",
                "40-cell factorial grid, 5 seeds per cell",
                convergence["excluded_from_coverage"],
            )
        )

    if clv:
        validation = clv["validation"]
        metrics += [
            metric(
                "BG/NBD maximum-likelihood fits that converged",
                f"{clv['maximum_likelihood']['converged_on_full_base']}/"
                f"{len(clv['maximum_likelihood']['attempts_full_base'])}",
                "full Olist base, 3 time scales x 5 penalties",
                "Real data. A negative result: the dropout parameters are unidentified.",
            ),
            metric(
                "Holdout mean absolute error, repeat purchases",
                round(validation["mean_absolute_error"], 6),
                f"{validation['holdout_weeks']}-week holdout to {validation['observation_end']}",
                f"Predicting zero scores {validation['mean_absolute_error_predicting_zero']:.6f}; "
                f"beats it: {validation['beats_predicting_zero']}.",
            ),
        ]

    if triangulation:
        headline = triangulation["headline"]
        metrics.append(
            metric(
                "Cost of allocating from last-click attribution",
                round(headline["cost_of_believing_attribution_share"], 4),
                "headline panel, full budget, evaluated under the true response curves",
                "Share of achievable incremental revenue forgone. Simulated media.",
            )
        )

    limitations = [
        "No figure describes the effectiveness of any real marketing channel. The "
        "channel-spend panel, the effect sizes and the attribution bias are all "
        "constructed from a pre-registered configuration.",
        "The revenue baseline is treated as if it were a no-advertising "
        "counterfactual. Olist did market itself over 2017-18, so the real series "
        "already contains real media effects; layering a simulated effect on top "
        "produces a series whose simulated component has a known truth, which is "
        "the only claim made.",
        "The ordering of attribution bias across channels follows Blake, Nosko and "
        "Tadelis (2015), Econometrica 83(1), 155-174, who found returns to branded "
        "keyword advertising indistinguishable from zero. The magnitudes here are "
        "chosen, not measured.",
        "Currency is Brazilian reais, 2017-18. Every dirham figure is a stated "
        "scenario at a nominal rate and is not a measurement.",
        "The media-mix results are conditional on a media signal share of roughly "
        "17% of detrended revenue variance on the headline panel. A quieter media "
        "plan is a harder recovery problem and the reported errors would grow.",
        "Criteo's twelve features are anonymised, so no uplift finding can be given "
        "a business interpretation, and they are heavily repeated, so the nominal "
        "row count overstates the precision available to a model fitted on them.",
        "Meridian was gated and runs on this machine, but no Meridian fit is "
        "reported: adapting the panel to its geo-hierarchical interface faithfully "
        "is separate work, and an unfaithful adaptation would be worse than none.",
        "Nothing here is legal or financial advice.",
    ]

    card = {
        "model": {
            "name": "ATHAR — triangulated marketing incrementality",
            "version": "0.1.0",
            "owner": "Krishna Mathur",
            "description": (
                "A media-mix model, a randomised-experiment analysis and an "
                "attribution model, reconciled against a known ground truth and "
                "priced as a budget decision."
            ),
        },
        # SPINE renders this section with str(), so it is written as markdown here
        # rather than as a mapping — a nested dict would come out as its repr.
        "intended_use": (
            "An academic demonstration, for MAIB AI 208, that platform-reported return "
            "is not incremental return, and a measurement of how far three standard "
            "methods sit from a truth none of them can see. Read by examiners and by "
            "readers of the public portfolio.\n\n"
            "**Out of scope:** budget decisions for any real advertiser; any claim "
            "about a named marketing channel's real effectiveness; any exchange-rate "
            "or currency claim."
        ),
        "data": {
            "source": (
                "Olist Brazilian E-Commerce (real, 96,731 orders across 85 weeks) and "
                "Criteo-UPLIFT v2.1 (real randomised trial, 13,979,592 rows), plus a "
                "simulated five-channel spend panel generated from config/dgp.yaml"
            ),
            "provenance": "simulated",
            "provenance_note": (
                "Mixed, and labelled as simulated because the strictest label wins. "
                "The Criteo result and the Olist frame are real and carry their own "
                "artifacts; everything downstream of the spend panel is simulated. "
                "athar.provenance enforces the distinction on every artifact."
            ),
            "window": (
                f"{frame['window']['start']} to {frame['window']['end']}"
                if frame
                else "see metrics/frame.json"
            ),
        },
        "metrics": metrics,
        "limitations": limitations,
    }
    if panel:
        card["data"]["configuration"] = (
            f"config/dgp.yaml, digest {panel['provenance']['dgp_hash']}, "
            f"seed {panel['provenance']['seed']}"
        )
    return card


def main():
    card = build()
    problems = validate_card(card)
    if problems:
        for problem in problems:
            log.error("  %s", problem)
        raise SystemExit("the card specification is not renderable")

    reports = paths.reports_dir()
    (reports / "athar_card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, width=88))
    try:
        (reports / "model_card.md").write_text(render_card(card).rstrip() + "\n")
    except CardValidationError as error:
        raise SystemExit(f"SPINE refused to render the card: {error}") from error

    log.info(
        "wrote reports/athar_card.yaml and reports/model_card.md (%d metrics)", len(card["metrics"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
