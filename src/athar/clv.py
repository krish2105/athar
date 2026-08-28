"""Customer lifetime value on a base that has almost none, and what follows from that.

Olist's repeat rate inside the modelling window is **3.03%** — 2,838 of 93,573
people bought more than once, and 230 bought more than twice. That is not a
modelling inconvenience to be worked around; it is the finding, and this module is
built to establish it rather than to route around it.

What is fitted
--------------
BG/NBD on the *whole* base, including the 96.97% with zero repeat purchases. Those
customers are informative — a long observation window with no second purchase is
evidence about the churn process, not a missing value — and dropping them is the
single most common way a CLV analysis flatters itself.

Gamma-Gamma on the 2,838 repeaters only, because monetary value conditional on
repeating cannot be estimated from people who never repeated. That restriction is
unavoidable and is labelled wherever the output appears.

Two implementations, on purpose
-------------------------------
``lifetimes`` gives the maximum-likelihood fit that the literature is written
against. ``pymc-marketing`` gives a Bayesian fit with intervals, which this
project's own conventions require of anything with a posterior. They are run on the
same data and their point estimates compared: agreement between two independent
implementations is evidence, in the way a model checking itself is not.

Validation
----------
A time-based calibration and holdout split via ``spine.splitting``, never a random
one. The prediction being validated is "how many purchases in the next N weeks",
and a random split would let the model see the future of the very customers it is
forecasting.

The expected result is that predicted repeat purchases are near zero and the
holdout agrees. That is the model *passing*, and the conclusion is that Olist is a
one-shot acquisition business whose customer lifetime value is approximately the
margin on the first order. Notebook 08 then carries the consequence: if lifetime
value is proportional to first-order value, weighting a budget allocation by
lifetime value and weighting it by immediate return give the same answer, and a
story the brief invites cannot be told honestly on this data.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "calibration_holdout",
    "compare_implementations",
    "fit_bgnbd",
    "fit_gamma_gamma",
    "holdout_accuracy",
    "summarise_repeat_behaviour",
]

log = logging.getLogger(__name__)


def summarise_repeat_behaviour(summary: pd.DataFrame) -> dict[str, Any]:
    """Describe how little repeat purchasing there is, before any model is fitted.

    Stated first and separately so the diagnosis stands on arithmetic rather than
    on a model's output.

    Parameters
    ----------
    summary : pandas.DataFrame
        From :func:`athar.frame.customer_summary`.

    Returns
    -------
    dict
        Counts and shares by number of repeat purchases.

    Examples
    --------
    >>> import pandas as pd
    >>> summary = pd.DataFrame({"frequency": [0, 0, 0, 1, 2], "monetary": [0, 0, 0, 10.0, 20.0]})
    >>> facts = summarise_repeat_behaviour(summary)
    >>> facts["customers"], facts["repeaters"], round(facts["repeat_rate"], 4)
    (5, 2, 0.4)
    """
    frequency = summary["frequency"].to_numpy()
    repeaters = int((frequency > 0).sum())
    return {
        "customers": int(len(summary)),
        "repeaters": repeaters,
        "repeat_rate": float(repeaters / len(summary)),
        "with_two_or_more_repeats": int((frequency > 1).sum()),
        "zero_repeat_share": float((frequency == 0).mean()),
        "max_repeats": int(frequency.max()),
        "mean_repeats_overall": float(frequency.mean()),
        "mean_repeats_among_repeaters": float(frequency[frequency > 0].mean())
        if repeaters
        else 0.0,
    }


def fit_bgnbd(summary: pd.DataFrame, penalizer: float = 0.01) -> Any:
    """Fit BG/NBD to the whole base with `lifetimes`.

    Parameters
    ----------
    summary : pandas.DataFrame
        Needs ``frequency``, ``recency`` and ``T``.
    penalizer : float, optional
        L2 penalty. Small but non-zero: at a 3% repeat rate the likelihood is
        nearly flat in some directions and an unpenalised fit wanders.

    Returns
    -------
    lifetimes.BetaGeoFitter
        The fitted model.
    """
    from lifetimes import BetaGeoFitter

    model = BetaGeoFitter(penalizer_coef=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(summary["frequency"], summary["recency"], summary["T"])
    return model


def fit_gamma_gamma(summary: pd.DataFrame, penalizer: float = 0.01) -> tuple[Any, int]:
    """Fit Gamma-Gamma to the repeaters, and report how few of them there are.

    The count is returned rather than logged because every figure derived from this
    model has to carry it: a monetary model fitted on 3% of the base does not
    describe the base.

    Parameters
    ----------
    summary : pandas.DataFrame
        Needs ``frequency`` and ``monetary``.
    penalizer : float, optional
        L2 penalty.

    Returns
    -------
    tuple
        The fitted model and the number of customers it was fitted on.

    Raises
    ------
    ValueError
        If no customer has both a repeat purchase and a positive monetary value.
    """
    from lifetimes import GammaGammaFitter

    repeaters = summary[(summary["frequency"] > 0) & (summary["monetary"] > 0)]
    if repeaters.empty:
        raise ValueError("no repeat customers with positive monetary value to fit on")

    model = GammaGammaFitter(penalizer_coef=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(repeaters["frequency"], repeaters["monetary"])
    return model, len(repeaters)


def calibration_holdout(
    orders: pd.DataFrame, cutoff: pd.Timestamp, observation_end: pd.Timestamp
) -> pd.DataFrame:
    """Build a time-split calibration and holdout summary.

    Everything before ``cutoff`` builds the customer's history; everything between
    the cutoff and ``observation_end`` is what the model is asked to predict and
    never sees. Customers acquired after the cutoff are excluded: there is no
    history to predict from, and including them would let a model that predicts
    zero look good for the wrong reason.

    Parameters
    ----------
    orders : pandas.DataFrame
        Order-level, from :func:`athar.frame.load_orders`, restricted to the window.
    cutoff : pandas.Timestamp
        End of the calibration period.
    observation_end : pandas.Timestamp
        End of the holdout period.

    Returns
    -------
    pandas.DataFrame
        ``customer_unique_id``, ``frequency``, ``recency``, ``T``, ``monetary``,
        ``holdout_frequency``, ``holdout_weeks``.

    Raises
    ------
    ValueError
        If the cutoff leaves no calibration history.
    """
    from athar.frame import customer_summary

    orders = orders.copy()
    orders["purchased_at"] = pd.to_datetime(orders["purchased_at"])
    calibration = orders[orders["purchased_at"] < cutoff]
    if calibration.empty:
        raise ValueError(f"no orders before {cutoff}; the cutoff leaves no history")

    summary = customer_summary(calibration, cutoff)
    holdout = orders[
        (orders["purchased_at"] >= cutoff) & (orders["purchased_at"] < observation_end)
    ]
    counts = holdout.groupby("customer_unique_id").size()
    summary["holdout_frequency"] = (
        summary["customer_unique_id"].map(counts).fillna(0.0).astype(float)
    )
    summary["holdout_weeks"] = (observation_end - cutoff).days / 7.0
    return summary


def holdout_accuracy(model: Any, summary: pd.DataFrame) -> dict[str, float]:
    """Compare predicted holdout purchases against what actually happened.

    Mean absolute error, not MAPE: the actual holdout count is zero for almost
    every customer, and a percentage error against zero is undefined. This project
    does not compute MAPE anywhere.

    A near-zero prediction that matches a near-zero outcome is the model working.
    The comparison against a predict-nothing baseline is what stops that being
    mistaken for skill.

    Parameters
    ----------
    model : lifetimes.BetaGeoFitter
        A fitted model.
    summary : pandas.DataFrame
        From :func:`calibration_holdout`.

    Returns
    -------
    dict
        Predicted and actual totals, mean absolute error, and the same error for a
        baseline that predicts zero for everyone.
    """
    horizon = float(summary["holdout_weeks"].iloc[0]) * 7.0
    predicted = model.conditional_expected_number_of_purchases_up_to_time(
        horizon, summary["frequency"], summary["recency"], summary["T"]
    ).to_numpy()
    actual = summary["holdout_frequency"].to_numpy()

    return {
        "horizon_days": horizon,
        "customers": int(len(summary)),
        "predicted_total": float(predicted.sum()),
        "actual_total": float(actual.sum()),
        "predicted_mean": float(predicted.mean()),
        "actual_mean": float(actual.mean()),
        "mean_absolute_error": float(np.mean(np.abs(predicted - actual))),
        "mean_absolute_error_predicting_zero": float(np.mean(np.abs(actual))),
        "beats_predicting_zero": bool(
            np.mean(np.abs(predicted - actual)) < np.mean(np.abs(actual))
        ),
        "share_of_customers_predicted_below_0_1": float((predicted < 0.1).mean()),
    }


def compare_implementations(
    summary: pd.DataFrame, draws: int = 1000, seed: int = 20260829
) -> dict[str, Any]:
    """Fit BG/NBD twice, by maximum likelihood and by MCMC, and compare.

    Agreement between two independently written implementations is evidence about
    the fit. A single implementation agreeing with itself is not.

    The Bayesian fit is the one the report quotes, because it carries intervals and
    this project's conventions require them wherever a posterior exists.

    Parameters
    ----------
    summary : pandas.DataFrame
        Needs ``customer_unique_id``, ``frequency``, ``recency``, ``T``.
    draws : int, optional
        Posterior draws.
    seed : int, optional
        Reproducibility.

    Returns
    -------
    dict
        Both parameter sets and the largest relative disagreement between them.
    """
    from pymc_marketing.clv import BetaGeoModel

    frequentist = fit_bgnbd(summary)
    mle = {name: float(value) for name, value in frequentist.params_.items()}

    data = pd.DataFrame(
        {
            "customer_id": summary["customer_unique_id"].to_numpy(),
            "frequency": summary["frequency"].to_numpy(),
            "recency": summary["recency"].to_numpy(),
            "T": summary["T"].to_numpy(),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bayesian = BetaGeoModel(data=data)
        bayesian.build_model()
        bayesian.fit(draws=draws, tune=draws, chains=4, random_seed=seed, progressbar=False)

    posterior = bayesian.fit_result
    bayes = {
        name: {
            "mean": float(posterior[name].mean()),
            "hdi_low": float(posterior[name].quantile(0.055)),
            "hdi_high": float(posterior[name].quantile(0.945)),
        }
        for name in ("a", "b", "alpha", "r")
        if name in posterior
    }

    disagreement = {
        name: abs(bayes[name]["mean"] / mle[name] - 1.0)
        for name in bayes
        if name in mle and mle[name] != 0
    }
    return {
        "maximum_likelihood": mle,
        "bayesian": bayes,
        "relative_disagreement": disagreement,
        "worst_relative_disagreement": max(disagreement.values()) if disagreement else None,
        "note": (
            "lifetimes fits by maximum likelihood; pymc-marketing fits the same model by "
            "MCMC. Two independent implementations agreeing is evidence about the fit in a "
            "way one implementation cannot be. The Bayesian parameters are what the report "
            "quotes, because they carry intervals."
        ),
    }
