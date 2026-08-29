"""Where the three methods disagree, and what believing the wrong one costs.

The project's question, finally posed in money. A chief marketing officer holds
three estimates of what each channel returns:

**Attribution** — available daily, for every channel, at no cost. Biased by however
much organic demand happens to flow through that channel's last click, which varies
by channel and is not knowable from the attribution data itself.

**Experiment** — unbiased, and the only one of the three that is. Available for one
channel at a time, after weeks of switched-off spend in a meaningful share of the
market, with a confidence interval wide enough to be uncomfortable.

**Media-mix model** — covers every channel at once, models saturation, and so is
the only one that can say what the *next* dirham buys rather than what the average
one did. Rests on assumptions about functional form that are never verifiable from
the data it is fitted to.

Reconciliation here does not mean averaging them. Averaging a biased estimator with
an unbiased one produces a biased estimator with a smaller variance, which is a
worse thing to hand a decision-maker than either input, because it looks more
trustworthy than it is. What this module does instead is:

1. Put the three side by side per channel with their intervals, and score how far
   each is from a truth none of them could see.
2. Turn each into a budget under the same governance constraints.
3. Evaluate every resulting budget against the *same* true response curves.

The last step is the answer. The gap between what the best-informed allocation earns
and what each estimator's allocation earns is the price of that estimator, in money,
on a stated budget — computed inside the simulation and labelled as such.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from athar import allocate, dgp

__all__ = ["compare_estimates", "cost_of_believing", "divergence_score"]

log = logging.getLogger(__name__)


def divergence_score(estimates: dict[str, float]) -> float:
    """How far apart the methods are for one channel, relative to their level.

    The coefficient of variation across the available estimates. Scale-free, so a
    channel returning 0.9 and one returning 2.8 are comparable, and undefined
    rather than infinite when every estimate is zero.

    A high score is the signal a marketer should act on: it says the methods are
    telling different stories about this channel, and that no amount of dashboard
    polish will resolve which is right without an experiment.

    Parameters
    ----------
    estimates : dict
        Method name to estimated ROI.

    Returns
    -------
    float
        Coefficient of variation, or NaN if the mean is zero.

    Examples
    --------
    >>> round(divergence_score({"mmm": 2.0, "experiment": 2.0, "attribution": 2.0}), 6)
    0.0
    >>> round(divergence_score({"mmm": 1.0, "experiment": 2.0, "attribution": 6.0}), 4)
    0.8819
    """
    values = np.array([v for v in estimates.values() if v is not None and np.isfinite(v)])
    if len(values) < 2 or values.mean() == 0:
        return float("nan")
    return float(values.std(ddof=1) / abs(values.mean()))


def compare_estimates(
    truth: dict[str, float],
    attribution: dict[str, float],
    mmm_estimates: dict[str, dict[str, float]],
    experiment: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Put the three estimates beside the truth, per channel.

    Parameters
    ----------
    truth : dict
        Channel to true ROI.
    attribution : dict
        Channel to last-click ROAS.
    mmm_estimates : dict
        Channel to a posterior summary with ``mean``, ``hdi_low``, ``hdi_high``.
    experiment : dict, optional
        Channel to ``estimate`` and optionally ``ci_low`` / ``ci_high``. Channels
        without an experiment are reported as such rather than filled in — in
        practice most channels never get one, and pretending otherwise would hide
        the method's real limitation.

    Returns
    -------
    dict
        Per-channel comparison and a summary of which method wins where.
    """
    experiment = experiment or {}
    channels: dict[str, Any] = {}

    for channel, actual in truth.items():
        mmm_summary = mmm_estimates.get(channel, {})
        mmm_mean = mmm_summary.get("mean")
        experiment_entry = experiment.get(channel, {})
        experiment_mean = experiment_entry.get("estimate")

        available = {"attribution": attribution.get(channel), "mmm": mmm_mean}
        if experiment_mean is not None:
            available["experiment"] = experiment_mean

        errors = {
            name: (value / actual - 1.0)
            for name, value in available.items()
            if value is not None and actual
        }
        channels[channel] = {
            "true_roi": actual,
            "attribution": {
                "estimate": attribution.get(channel),
                "relative_error": errors.get("attribution"),
                "interval": None,
                "note": "no interval — an attribution report states a number, not a range",
            },
            "mmm": {
                "estimate": mmm_mean,
                "relative_error": errors.get("mmm"),
                "interval": [mmm_summary.get("hdi_low"), mmm_summary.get("hdi_high")],
                "covers_truth": (
                    bool(mmm_summary["hdi_low"] <= actual <= mmm_summary["hdi_high"])
                    if mmm_summary
                    else None
                ),
            },
            "experiment": (
                {
                    "estimate": experiment_mean,
                    "relative_error": errors.get("experiment"),
                    "interval": [
                        experiment_entry.get("ci_low"),
                        experiment_entry.get("ci_high"),
                    ],
                    "covers_truth": (
                        bool(experiment_entry["ci_low"] <= actual <= experiment_entry["ci_high"])
                        if experiment_entry.get("ci_low") is not None
                        else None
                    ),
                }
                if experiment_mean is not None
                else {
                    "estimate": None,
                    "note": (
                        "not run for this channel — which is the normal case, since a "
                        "holdout costs weeks of switched-off spend and is affordable for "
                        "one channel at a time"
                    ),
                }
            ),
            "divergence": divergence_score(available),
            "closest_method": (min(errors, key=lambda name: abs(errors[name])) if errors else None),
        }

    ranked = sorted(
        (c for c in channels if not np.isnan(channels[c]["divergence"])),
        key=lambda c: channels[c]["divergence"],
        reverse=True,
    )
    return {
        "channels": channels,
        "summary": {
            "most_divergent_channel": ranked[0] if ranked else None,
            "least_divergent_channel": ranked[-1] if ranked else None,
            "closest_method_by_channel": {
                name: entry["closest_method"] for name, entry in channels.items()
            },
            "note": (
                "Reconciliation is not averaging. Averaging a biased estimate with an "
                "unbiased one yields a biased estimate with a smaller variance, which is "
                "more dangerous to hand a decision-maker than either input because it "
                "looks more trustworthy than it is."
            ),
        },
    }


def cost_of_believing(
    panel: dgp.Panel,
    budget: float,
    estimator_rois: dict[str, dict[str, float]],
    mmm_responses: dict[str, Any] | None = None,
    floor: float = allocate.DEFAULT_FLOOR,
    cap: float = allocate.DEFAULT_CAP,
) -> dict[str, Any]:
    """Allocate under each estimator, then score every allocation against the truth.

    The best-informed allocation uses the true response curves and their curvature.
    It is a ceiling nobody can reach in practice, and it is the right benchmark
    precisely because it separates "this estimator is wrong" from "this problem is
    hard".

    Parameters
    ----------
    panel : athar.dgp.Panel
        The generated panel supplying the true curves.
    budget : float
        Total to allocate, in the panel's currency.
    estimator_rois : dict
        Estimator name to a channel-to-ROI mapping. Each is allocated linearly,
        which is what a scalar ROI table permits.
    mmm_responses : dict, optional
        Channel to a callable spend-to-revenue curve fitted by the media-mix model.
        When present, an additional allocation uses its curvature — the thing only
        a media-mix model can offer.
    floor, cap : float, optional
        Governance constraints, applied identically to every allocation.

    Returns
    -------
    dict
        Every allocation, its revenue under the truth, and the shortfall against
        the best-informed allocation.
    """
    channels = list(panel.spend.columns)
    bounds = allocate.bounds_from_shares(channels, budget, floor=floor, cap=cap)
    truth_curves = allocate.truth_response_functions(panel)

    allocations: dict[str, dict[str, float]] = {
        "optimal_under_truth": allocate.allocate_concave(truth_curves, budget, bounds, channels)
    }
    for name, roi in estimator_rois.items():
        allocations[name] = allocate.allocate_linear(
            {channel: roi[channel] for channel in channels}, budget, bounds
        )
    failed: dict[str, str] = {}
    if mmm_responses:
        # The only allocation here that optimises over a *fitted* surface rather
        # than a known one. A weakly identified posterior can produce a response
        # curve flat or bumpy enough that no start converges, and that is a result
        # about the model rather than a reason to lose every other allocation.
        try:
            allocations["mmm_with_curvature"] = allocate.allocate_concave(
                mmm_responses, budget, bounds, channels
            )
        except RuntimeError as error:
            failed["mmm_with_curvature"] = str(error)
            log.warning("curved allocation from the fitted response did not converge: %s", error)

    # An even split: the allocation a planner reaches for when they distrust every
    # number on the table. Worth knowing, because an estimator that cannot beat it
    # has earned nothing.
    allocations["equal_split"] = dict.fromkeys(channels, budget / len(channels))

    scored = {
        name: allocate.evaluate_under_truth(allocation, panel)
        for name, allocation in allocations.items()
    }
    ceiling = scored["optimal_under_truth"]["total_revenue"]

    return {
        "budget": budget,
        "governance": {
            "floor_share": floor,
            "cap_share": cap,
            "note": (
                "Applied identically to every allocation. Without them a linear "
                "allocation puts the whole budget in one channel, and the comparison "
                "would be against a caricature no marketing organisation resembles."
            ),
        },
        "allocations_that_failed": failed,
        "allocations": {
            name: {
                "spend": allocation,
                "shares": {channel: value / budget for channel, value in allocation.items()},
                "revenue_under_truth": scored[name]["total_revenue"],
                "blended_roi_under_truth": scored[name]["blended_roi"],
                "shortfall_against_best": ceiling - scored[name]["total_revenue"],
                "shortfall_share": (ceiling - scored[name]["total_revenue"]) / ceiling,
            }
            for name, allocation in allocations.items()
        },
        "interpretation": (
            "Every allocation is evaluated against the same true response curves, so "
            "what is compared is the consequence of believing each estimator rather than "
            "each estimator's opinion of itself. The shortfall is computed inside the "
            "simulation and is a statement about the methods, not about any real market."
        ),
    }
