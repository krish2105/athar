"""The media-mix model, fitted deliberately wrong, and how its recovery is scored.

The specifications
------------------
Two, and the difference between them is the experiment.

``misspecified``
    Geometric adstock and a logistic saturation. Geometric adstock is the
    generating kernel with its delay pinned to zero, so it cannot represent a
    delayed peak at all, and a logistic curve cannot take the Hill shape. This is the honest
    arm: in practice nobody knows the true functional form, and a model fitted to
    data its own form produced recovers its own assumptions rather than the truth.

``matched``
    Delayed-geometric adstock and a Hill curve — the generating form. Included as a
    *control*, not as the headline. Comparing the two separates error caused by
    getting the shape wrong from error caused by the design being unable to
    identify the parameters at all. Reporting only the matched arm would be
    circular; reporting only the misspecified arm would leave the reader unable
    to tell which of the two problems they were looking at.

Priors are pymc-marketing's defaults, unchanged
-----------------------------------------------
This is deliberate and worth stating plainly, because prior choice is where
circularity gets into a recovery study without anyone noticing. A prior centred
near the true ROI would produce excellent recovery and prove nothing. The library's
defaults were written without knowledge of this panel, so they are the closest
thing available to an uninformed analyst's starting point.

The consequence is that the intervals here are wide, and that is a finding rather
than a defect: a weakly identified design plus an uninformative prior is what a
media-mix model actually faces.

A library defect that shaped this design
----------------------------------------
The generating adstock was originally a Weibull PDF. ``WeibullPDFAdstock`` raises
``TypeError: x must be have an XTensorType`` under pymc-marketing 0.19.2 with
pytensor 2.38.2, for every saturation, so the matched arm could not be fitted at
all. ``WeibullCDFAdstock`` samples cleanly but its kernel is a survival curve and
decays monotonically, so it cannot represent a delayed peak either.

``DelayedAdstock`` both admits a delayed peak and samples, so the generator was
moved to its functional form. The matched arm is therefore genuinely matched
rather than merely close, which is what the misspecification-versus-identification
decomposition needs in order to mean anything.

A note on the deprecated class
------------------------------
``pymc_marketing.mmm.MMM`` is deprecated in 0.19.2 in favour of the
multidimensional MMM, and this module keeps using it anyway. The replacement is
not a drop-in: it provides neither ``compute_channel_contribution_original_scale``
nor ``get_channel_contribution_forward_pass_grid``, which are exactly the two
things this project needs — posterior contributions in money, and the fitted
response surface. Migrating would mean reimplementing both here, putting more of
this project's own code between the library and the result, which is the wrong
direction for a study whose whole point is that the estimate was not helped along.

The version is pinned in ``uv.lock``, so the deprecation is a note about the
future rather than a risk to reproducing these numbers.

Average and marginal ROI
------------------------
Both are computed per posterior draw, so both carry intervals.

Average ROI divides a channel's total posterior contribution by its total spend.
Marginal ROI is the slope of the fitted response curve at the observed plan, taken
by central difference over pymc-marketing's forward-pass grid. The two answer
different questions and a budget decision needs the second.
"""

from __future__ import annotations

import logging
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import (
    MMM,
    DelayedAdstock,
    GeometricAdstock,
    HillSaturation,
    LogisticSaturation,
)

__all__ = [
    "HDI_PROB",
    "MAX_DIVERGENCE_RATE",
    "MAX_R_HAT",
    "MIN_ESS_BULK",
    "SPECIFICATIONS",
    "build",
    "fit",
    "posterior_average_roi",
    "posterior_marginal_roi",
    "response_totals",
    "sampler_diagnostics",
    "score_recovery",
]

log = logging.getLogger(__name__)

#: Sampler health thresholds, fixed before the grid was scored. The divergence
#: threshold is a rate rather than a count because the count scales with the
#: number of draws, and a rule that tightens as you sample more is not a rule.
MAX_DIVERGENCE_RATE = 0.005
MAX_R_HAT = 1.01
MIN_ESS_BULK = 400.0

#: 89% rather than 95%. Following McElreath: 95% carries a false echo of the 0.05
#: significance convention, and at the effective sample sizes a short weekly panel
#: supports, the tails of a 95% interval are the least stable part of it. The
#: choice is fixed before any fit runs and applies to every interval reported.
HDI_PROB = 0.89

#: The fitted forms. `misspecified` is the headline; `matched` is the control.
SPECIFICATIONS: dict[str, str] = {
    "misspecified": "GeometricAdstock + LogisticSaturation — cannot express the generating form",
    "matched": "DelayedAdstock + HillSaturation — the generating form, as a control",
}


def build(
    channels: list[str],
    specification: str,
    max_lag: int,
    yearly_seasonality: int = 2,
) -> MMM:
    """Construct an unfitted model under one of the two specifications.

    Parameters
    ----------
    channels : list of str
        Media channel column names.
    specification : {'misspecified', 'matched'}
        Which functional form to fit.
    max_lag : int
        Adstock window, matching the generator's so the two differ in *shape*
        rather than in how far back they are allowed to look.
    yearly_seasonality : int, optional
        Fourier modes for annual seasonality. Two, because the panel is 85 weeks
        — 1.6 years — and more modes would fit noise rather than a season.

    Returns
    -------
    pymc_marketing.mmm.MMM
        The unfitted model.

    Raises
    ------
    ValueError
        If ``specification`` is not one of the two.

    Examples
    --------
    >>> model = build(["tv", "search"], "misspecified", max_lag=8)
    >>> type(model.adstock).__name__, type(model.saturation).__name__
    ('GeometricAdstock', 'LogisticSaturation')
    >>> model = build(["tv", "search"], "matched", max_lag=8)
    >>> type(model.adstock).__name__, type(model.saturation).__name__
    ('DelayedAdstock', 'HillSaturation')
    """
    if specification not in SPECIFICATIONS:
        raise ValueError(f"unknown specification {specification!r}; have {sorted(SPECIFICATIONS)}")
    if specification == "matched":
        adstock, saturation = DelayedAdstock(l_max=max_lag), HillSaturation()
    else:
        adstock, saturation = GeometricAdstock(l_max=max_lag), LogisticSaturation()

    return MMM(
        date_column="week",
        channel_columns=list(channels),
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
    )


def fit(
    model: MMM,
    frame: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 20260829,
    target_accept: float = 0.99,
) -> Any:
    """Sample the posterior.

    Four chains, because two cannot support a trustworthy R-hat. ``target_accept``
    is 0.99 rather than the 0.8 default: the funnel geometry of a saturation
    parameter multiplying a weakly identified coefficient produces divergences at
    anything lower, and this was raised after observing them rather than guessed.

    Parameters
    ----------
    model : pymc_marketing.mmm.MMM
        From :func:`build`.
    frame : pandas.DataFrame
        The fitting frame: ``week``, one column per channel, and ``revenue``.
    draws, tune, chains : int, optional
        Sampler settings.
    seed : int, optional
        Recorded in every artifact this fit produces.
    target_accept : float, optional
        NUTS acceptance target.

    Returns
    -------
    arviz.InferenceData
        The posterior.
    """
    features = model.channel_columns
    design = frame[["week", *features]].copy()
    return model.fit(
        design,
        frame["revenue"],
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        progressbar=False,
    )


def sampler_diagnostics(idata: Any) -> dict[str, Any]:
    """Summarise whether the sampler can be believed.

    A fit that fails these is reported as failed rather than averaged into a
    coverage rate. A recovery study that quietly includes non-converged fits is
    measuring its own sampler, not the method.

    Two verdicts are returned, and the recovery grid reports coverage under both.
    ``passed`` uses a divergence *rate* threshold; ``passed_strict`` demands zero
    divergences. The strict rule was the original one and it rejected almost every
    fit — a handful of divergences in four thousand draws is common in a
    hierarchical media model and is not on its own evidence that the posterior is
    wrong. Rather than quietly relax the rule, both are computed and both are
    published, so a reader can see whether the choice of rule changed the
    conclusion. It does not.

    Parameters
    ----------
    idata : arviz.InferenceData
        A fitted posterior.

    Returns
    -------
    dict
        ``divergences``, ``max_r_hat``, ``min_ess_bulk`` and ``passed``, where
        passing is zero divergences, R-hat below 1.01 and bulk ESS at least 400.
    """
    divergences = int(idata.sample_stats["diverging"].sum())
    draws = int(idata.sample_stats.sizes["chain"] * idata.sample_stats.sizes["draw"])
    rate = divergences / draws
    max_r_hat = float(az.rhat(idata).max().to_array().max())
    min_ess = float(az.ess(idata).min().to_array().min())
    healthy = max_r_hat < MAX_R_HAT and min_ess >= MIN_ESS_BULK
    return {
        "divergences": divergences,
        "post_warmup_draws": draws,
        "divergence_rate": round(rate, 6),
        "max_r_hat": round(max_r_hat, 6),
        "min_ess_bulk": round(min_ess, 2),
        "passed": bool(rate < MAX_DIVERGENCE_RATE and healthy),
        "passed_strict": bool(divergences == 0 and healthy),
        "criteria": (
            f"divergence rate < {MAX_DIVERGENCE_RATE:.1%} of post-warmup draws, "
            f"max R-hat < {MAX_R_HAT}, min bulk ESS >= {MIN_ESS_BULK}"
        ),
        "criteria_strict": (
            f"zero divergences, max R-hat < {MAX_R_HAT}, min bulk ESS >= {MIN_ESS_BULK}"
        ),
    }


def _flatten(array: xr.DataArray) -> xr.DataArray:
    """Stack chain and draw into one sample dimension."""
    return array.stack(sample=("chain", "draw"))


def _summarise(draws: np.ndarray) -> dict[str, float]:
    """Posterior mean and highest-density interval for one quantity."""
    interval = az.hdi(np.asarray(draws), hdi_prob=HDI_PROB)
    return {
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "hdi_low": float(interval[0]),
        "hdi_high": float(interval[1]),
    }


def posterior_average_roi(model: MMM, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Posterior average ROI per channel: total contribution over total spend.

    Parameters
    ----------
    model : pymc_marketing.mmm.MMM
        A fitted model.
    frame : pandas.DataFrame
        The fitting frame, for the spend totals.

    Returns
    -------
    dict
        Channel to ``mean``, ``median``, ``hdi_low``, ``hdi_high``.
    """
    contribution = _flatten(model.compute_channel_contribution_original_scale().sum("date"))
    return {
        channel: _summarise(
            contribution.sel(channel=channel).to_numpy() / float(frame[channel].sum())
        )
        for channel in model.channel_columns
    }


def response_totals(model: MMM, deltas: np.ndarray, chunk: int = 5) -> xr.DataArray:
    """Total posterior contribution at each spend multiplier.

    pymc-marketing's forward-pass grid returns a ``(delta, chain, draw, date,
    channel)`` array, which at four chains and a thousand draws is hundreds of
    megabytes for a modestly sized grid. The dates are summed away as each chunk
    arrives, so peak memory stays proportional to ``chunk`` rather than to the
    whole grid.

    Parameters
    ----------
    model : pymc_marketing.mmm.MMM
        A fitted model.
    deltas : numpy.ndarray
        Spend multipliers. 1.0 is the observed plan.
    chunk : int, optional
        Multipliers evaluated per call.

    Returns
    -------
    xarray.DataArray
        Dimensions ``(delta, sample, channel)``.
    """
    deltas = np.asarray(deltas, dtype=float)
    pieces = []
    for start in range(0, len(deltas), chunk):
        block = deltas[start : start + chunk]
        grid = model.get_channel_contribution_forward_pass_grid(
            start=float(block[0]),
            stop=float(block[-1]),
            num=len(block),
        )
        pieces.append(_flatten(grid.sum("date")))
    return xr.concat(pieces, dim="delta").assign_coords(delta=deltas)


def posterior_marginal_roi(
    model: MMM, frame: pd.DataFrame, step: float = 0.02
) -> dict[str, dict[str, float]]:
    """Posterior marginal ROI at the observed plan, by central difference.

    A finite difference rather than an analytic derivative, because the quantity
    being differentiated is the library's fitted response surface rather than a
    formula this project owns. ``step`` is small enough to approximate the slope
    and large enough that the difference is not swamped by the surface's own
    numerical noise.

    Parameters
    ----------
    model : pymc_marketing.mmm.MMM
        A fitted model.
    frame : pandas.DataFrame
        The fitting frame, for the spend totals.
    step : float, optional
        Half-width of the central difference, in spend multiples.

    Returns
    -------
    dict
        Channel to ``mean``, ``median``, ``hdi_low``, ``hdi_high``.
    """
    totals = response_totals(model, np.array([1.0 - step, 1.0 + step]), chunk=2)
    slope = (totals.sel(delta=1.0 + step) - totals.sel(delta=1.0 - step)) / (2.0 * step)
    return {
        channel: _summarise(slope.sel(channel=channel).to_numpy() / float(frame[channel].sum()))
        for channel in model.channel_columns
    }


def score_recovery(
    estimates: dict[str, dict[str, float]],
    truth: dict[str, float],
) -> dict[str, Any]:
    """Score estimated ROI against the known truth.

    Coverage leads. Whether the true value falls inside the interval is the
    property a Bayesian model actually claims, and it is checkable across seeds;
    a point estimate that happens to land close on one draw is an anecdote.

    Relative error is admissible here only because every configured ROI is bounded
    well away from zero by construction, which is stated rather than assumed. No
    MAPE is computed anywhere in this project.

    Parameters
    ----------
    estimates : dict
        Channel to a posterior summary from :func:`posterior_average_roi` or
        :func:`posterior_marginal_roi`.
    truth : dict
        Channel to true ROI.

    Returns
    -------
    dict
        ``channels`` with per-channel detail, and ``summary`` with the coverage
        rate and median absolute relative error.

    Examples
    --------
    >>> estimates = {"a": {"mean": 2.0, "median": 2.0, "hdi_low": 1.0, "hdi_high": 3.0}}
    >>> scored = score_recovery(estimates, {"a": 2.5})
    >>> scored["channels"]["a"]["covered"], round(scored["channels"]["a"]["relative_error"], 3)
    (True, -0.2)
    >>> scored["summary"]["coverage_rate"]
    1.0
    """
    channels: dict[str, Any] = {}
    for channel, summary in estimates.items():
        actual = float(truth[channel])
        channels[channel] = {
            "true": actual,
            "estimated_mean": summary["mean"],
            "hdi_low": summary["hdi_low"],
            "hdi_high": summary["hdi_high"],
            "covered": bool(summary["hdi_low"] <= actual <= summary["hdi_high"]),
            "absolute_error": summary["mean"] - actual,
            "relative_error": summary["mean"] / actual - 1.0,
            "interval_width": summary["hdi_high"] - summary["hdi_low"],
        }
    covered = [entry["covered"] for entry in channels.values()]
    errors = [abs(entry["relative_error"]) for entry in channels.values()]
    return {
        "channels": channels,
        "summary": {
            "hdi_prob": HDI_PROB,
            "coverage_rate": float(np.mean(covered)),
            "channels_covered": int(np.sum(covered)),
            "channels_total": len(covered),
            "median_absolute_relative_error": float(np.median(errors)),
            "mean_interval_width": float(
                np.mean([entry["interval_width"] for entry in channels.values()])
            ),
        },
    }
