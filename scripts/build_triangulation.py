"""The decision: three methods, three budgets, and the cost of believing each.

Everything upstream produces estimates. This turns them into money.

Each estimator's ROI table becomes a budget under identical governance constraints,
and every budget is then scored against the *same* true response curves. What is
compared is therefore the consequence of believing an estimator, not the
estimator's own opinion of itself.

The benchmark is the allocation built from the true curves, including their
curvature. Nobody can reach it — that is the point. It separates "this estimator is
wrong" from "this problem is hard".

Run: `make triangulate`
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd

from athar import dgp, paths, reconcile, truth
from athar.provenance import Provenance, read_metric, write_metric

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_triangulation")

#: A scenario rate, for legibility only. Olist trades in 2017-18 Brazilian reais and
#: this project makes no claim about any exchange rate; the dirham figures exist so
#: a reader can hold the magnitudes, and are labelled SCENARIO wherever they appear.
BRL_PER_AED = 1.55


def main():
    metrics = paths.metrics_dir()
    processed = paths.processed_dir()
    for required in ("panel.json", "mmm.json", "experiments.json"):
        if not (metrics / required).exists():
            raise SystemExit(f"metrics/{required} is missing; run its build step first")

    config = dgp.load_config()
    weekly = pd.read_parquet(processed / "weekly.parquet")
    panel = dgp.generate_panel(
        config,
        weekly["revenue"].to_numpy(),
        week_index=pd.DatetimeIndex(weekly["week"]),
    )
    stored = truth.load_truth(after=metrics / "mmm.json")
    channels = config.channel_names

    truth_average = {c: stored["channels"][c]["roi_average"] for c in channels}
    truth_marginal = {c: stored["channels"][c]["roi_marginal"] for c in channels}
    attribution = {c: stored["channels"][c]["roas_attributed"] for c in channels}

    mmm_artifact = read_metric("mmm", metrics)
    headline = mmm_artifact["fits"]["misspecified"]
    mmm_average = {
        c: {
            "mean": entry["estimated_mean"],
            "hdi_low": entry["hdi_low"],
            "hdi_high": entry["hdi_high"],
        }
        for c, entry in headline["average_roi"]["channels"].items()
    }

    experiments_artifact = read_metric("experiments", metrics)
    experiment_estimates = {
        channel: {"estimate": result["implied_roi"]}
        for channel, result in experiments_artifact["headline_experiments"].items()
        if np.isfinite(result.get("implied_roi", np.nan))
    }

    comparison = reconcile.compare_estimates(
        truth_average, attribution, mmm_average, experiment_estimates
    )
    log.info(
        "most divergent channel: %s; least: %s",
        comparison["summary"]["most_divergent_channel"],
        comparison["summary"]["least_divergent_channel"],
    )

    budget = float(panel.spend.to_numpy().sum())
    log.info("allocating BRL %s across %d channels", f"{budget:,.0f}", len(channels))

    # The media-mix model's fitted response curve, interpolated from the posterior
    # medians it published. This is the one estimator that offers curvature at all.
    curve = headline["response_curve"]
    multipliers = np.array(curve["multipliers"])
    mmm_responses = {}
    for channel in channels:
        observed = float(panel.spend[channel].sum())
        revenue = np.array(curve["median_revenue"][channel])

        def response(spend, observed=observed, revenue=revenue):
            return float(np.interp(spend / observed, multipliers, revenue))

        mmm_responses[channel] = response

    estimator_rois = {
        "attribution_lastclick": attribution,
        "mmm_average_roi": {c: mmm_average[c]["mean"] for c in channels},
        "true_average_roi": truth_average,
        "true_marginal_roi": truth_marginal,
    }
    if len(experiment_estimates) == len(channels):
        estimator_rois["experiment"] = {c: experiment_estimates[c]["estimate"] for c in channels}

    cost = reconcile.cost_of_believing(panel, budget, estimator_rois, mmm_responses)
    for name, entry in sorted(
        cost["allocations"].items(), key=lambda kv: -kv[1]["revenue_under_truth"]
    ):
        log.info(
            "  %-24s revenue %12s  shortfall %10s (%.2f%%)",
            name,
            f"{entry['revenue_under_truth']:,.0f}",
            f"{entry['shortfall_against_best']:,.0f}",
            100 * entry["shortfall_share"],
        )

    clv_note = None
    if (metrics / "clv.json").exists():
        clv_artifact = read_metric("clv", metrics)
        correlation = clv_artifact["lifetime_value"]["correlation_clv_with_first_order_value"]
        clv_note = {
            "correlation_clv_with_first_order_value": correlation,
            "clv_over_first_order_value": clv_artifact["lifetime_value"][
                "clv_over_first_order_value"
            ],
            "consequence": (
                "Lifetime value on this base is very nearly proportional to first-order "
                "value, so weighting the allocation by lifetime value and weighting it by "
                "immediate revenue rank the channels identically and produce the same "
                "budget. The CLV-weighted reallocation the brief invites is not a "
                "different answer here; it is the same answer with more steps, and "
                "saying so is the finding."
            ),
        }

    best = cost["allocations"]["optimal_under_truth"]["revenue_under_truth"]
    believed = cost["allocations"]["attribution_lastclick"]
    payload = {
        "comparison": comparison,
        "allocation": cost,
        "headline": {
            "budget_brl": budget,
            "best_possible_revenue_brl": best,
            "attribution_revenue_brl": believed["revenue_under_truth"],
            "cost_of_believing_attribution_brl": believed["shortfall_against_best"],
            "cost_of_believing_attribution_share": believed["shortfall_share"],
            "statement": (
                f"On a budget of BRL {budget:,.0f}, allocating from last-click "
                f"attribution leaves BRL {believed['shortfall_against_best']:,.0f} "
                f"({believed['shortfall_share']:.1%}) of incremental revenue on the table "
                f"against an allocation built from the true response curves. Computed "
                f"inside the simulation; a statement about the methods, not about any "
                f"real market."
            ),
        },
        "scenario_in_aed": {
            "caveat": (
                "SCENARIO — Olist trades in 2017-18 Brazilian reais. This project makes "
                "no claim about any exchange rate and no dirham figure here is a "
                "measurement. The conversion exists so a reader can hold the magnitudes "
                "at the scale the brief describes."
            ),
            "assumed_brl_per_aed": BRL_PER_AED,
            "budget_aed": budget / BRL_PER_AED,
            "cost_of_believing_attribution_aed": believed["shortfall_against_best"] / BRL_PER_AED,
            "on_a_one_million_aed_budget": {
                "cost_of_believing_attribution_aed": 1_000_000 * believed["shortfall_share"],
                "note": (
                    "The shortfall is a share of budget, so it scales. On AED 1,000,000 "
                    f"the same misallocation costs about AED "
                    f"{1_000_000 * believed['shortfall_share']:,.0f} of incremental revenue."
                ),
            },
        },
        "clv_consequence": clv_note,
        "method_notes": {
            "why_not_average": (
                "Reconciliation here is not averaging. Averaging a biased estimator with "
                "an unbiased one gives a biased estimator with a smaller variance, which "
                "is worse to hand a decision-maker than either input because it looks "
                "more trustworthy than it is."
            ),
            "linear_versus_curved": (
                "An experiment and an attribution report each return one number per "
                "channel and say nothing about what the next dirham earns, so a planner "
                "holding only those must treat returns as constant. Only the media-mix "
                "model offers curvature, and the difference between its curved allocation "
                "and its own linear one isolates what that curvature is worth."
            ),
        },
    }

    path = write_metric(
        "triangulation",
        payload,
        Provenance(
            source="triangulation",
            synthetic=True,
            split="full panel",
            seed=stored["seed"],
            dgp_hash=stored["config_digest"],
        ),
        metrics,
    )
    log.info("wrote %s", path)

    surface = processed / "allocator_surface.json"
    surface.write_text(
        json.dumps(
            {
                "channels": channels,
                "budget": budget,
                "observed_spend": {c: float(panel.spend[c].sum()) for c in channels},
                "multipliers": multipliers.tolist(),
                "true_revenue": {
                    c: dgp.response_curve(panel, c, multipliers).tolist() for c in channels
                },
                "mmm_revenue": curve["median_revenue"],
                "allocations": {
                    name: entry["spend"] for name, entry in cost["allocations"].items()
                },
                "governance": cost["governance"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    log.info("wrote %s for the dashboard", surface.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
