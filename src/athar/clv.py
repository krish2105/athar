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

The maximum-likelihood fit does not exist, and that is the finding
------------------------------------------------------------------
``lifetimes`` is the reference implementation the literature is written against.
On this base it does not converge — at any time scale (days, weeks, months), at any
penalty (0 to 10), on the full base *or* on the repeaters alone. The likelihood
returns NaN and the parameters run off to the order of 1e-4 in log space.

That is not a library defect and it is not a tuning problem. BG/NBD's dropout
parameters ``a`` and ``b`` describe the shape of a Beta distribution over the
probability of churning after each purchase, and they are identified only by the
*pattern* of repeat purchases. Olist's repeaters average 1.11 repeat purchases
each. There is nothing in the data for those two parameters to be estimated from,
so the likelihood is flat in them and the optimiser walks off.

So the Bayesian fit is not a second opinion here; it is the only fit available. Its
priors supply the regularisation the data cannot, and it converges for exactly that
reason. That is not evidence the Bayesian model is better — the prior is doing the
work, and the posterior for ``a`` and ``b`` should be read as close to the prior
rather than as something learned. :func:`fit_bgnbd_bayesian` returns the prior and
posterior side by side so a reader can see how much moved.

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

import contextlib
import io
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "MaximumLikelihoodError",
    "calibration_holdout",
    "fit_bgnbd",
    "fit_bgnbd_bayesian",
    "fit_gamma_gamma",
    "holdout_accuracy",
    "maximum_likelihood_attempts",
    "summarise_repeat_behaviour",
]


class MaximumLikelihoodError(RuntimeError):
    """Raised when BG/NBD cannot be fitted by maximum likelihood on this data."""


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


def fit_bgnbd(summary: pd.DataFrame, penalizer: float = 0.01, time_divisor: float = 7.0) -> Any:
    """Fit BG/NBD by maximum likelihood with `lifetimes`, or say why it cannot be.

    Parameters
    ----------
    summary : pandas.DataFrame
        Needs ``frequency``, ``recency`` and ``T``.
    penalizer : float, optional
        L2 penalty on the log parameters.
    time_divisor : float, optional
        Divides ``recency`` and ``T``. 7 puts them in weeks, which conditions the
        optimisation better than days.

    Returns
    -------
    lifetimes.BetaGeoFitter
        The fitted model.

    Raises
    ------
    MaximumLikelihoodError
        If the optimiser does not converge. On Olist it never does, for the reason
        given in the module docstring, and the caller is expected to record that
        rather than retry with different settings.
    """
    from lifetimes import BetaGeoFitter

    model = BetaGeoFitter(penalizer_coef=penalizer)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                model.fit(
                    summary["frequency"],
                    summary["recency"] / time_divisor,
                    summary["T"] / time_divisor,
                )
    except Exception as error:
        raise MaximumLikelihoodError(
            f"BG/NBD did not converge at penalizer {penalizer} on a time divisor of "
            f"{time_divisor}: {type(error).__name__}"
        ) from error
    return model


def maximum_likelihood_attempts(
    summary: pd.DataFrame,
    penalizers: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0),
    time_units: tuple[tuple[str, float], ...] = (
        ("days", 1.0),
        ("weeks", 7.0),
        ("months", 30.44),
    ),
) -> list[dict[str, Any]]:
    """Try every reasonable maximum-likelihood setting and record what happened.

    Reported in full rather than summarised, because "the model did not converge"
    invites the reply "did you try a bigger penalty", and the answer needs to be a
    table rather than an assurance.

    Parameters
    ----------
    summary : pandas.DataFrame
        Needs ``frequency``, ``recency`` and ``T``.
    penalizers : tuple of float, optional
        L2 penalties to attempt.
    time_units : tuple, optional
        ``(name, divisor)`` pairs.

    Returns
    -------
    list of dict
        One entry per combination, with ``converged`` and the parameters if so.
    """
    attempts = []
    for unit, divisor in time_units:
        for penalizer in penalizers:
            entry: dict[str, Any] = {"time_unit": unit, "penalizer": penalizer}
            try:
                model = fit_bgnbd(summary, penalizer=penalizer, time_divisor=divisor)
                entry["converged"] = True
                entry["parameters"] = {k: float(v) for k, v in model.params_.items()}
            except MaximumLikelihoodError:
                entry["converged"] = False
            attempts.append(entry)
    return attempts


def fit_bgnbd_bayesian(
    summary: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 20260829,
    time_divisor: float = 7.0,
) -> tuple[Any, dict[str, Any]]:
    """Fit BG/NBD by MCMC, and report how far the posterior moved from the prior.

    The prior comparison is not decoration. Maximum likelihood fails here because
    the dropout parameters are unidentified; the Bayesian fit succeeds because its
    priors are proper. A parameter whose posterior sits on top of its prior has not
    been learned from the data, and saying which ones those are is the difference
    between using a Bayesian model and hiding behind one.

    Worth knowing before reading the output: pymc-marketing parameterises dropout
    as ``phi ~ Uniform(0, 1)`` and ``kappa ~ Pareto(alpha=1, m=1)``, with
    ``a = phi * kappa`` and ``b = (1 - phi) * kappa``. A Pareto with alpha of one
    has no finite mean, so the prior on ``a`` and ``b`` is extremely diffuse. If
    the posterior on those two barely narrows, the data has told us nothing about
    the dropout process — which is precisely what maximum likelihood was unable to
    estimate.

    Parameters
    ----------
    summary : pandas.DataFrame
        Needs ``customer_unique_id``, ``frequency``, ``recency``, ``T``.
    draws, tune, chains : int, optional
        Sampler settings.
    seed : int, optional
        Reproducibility.
    time_divisor : float, optional
        Divides ``recency`` and ``T``; 7 gives weeks.

    Returns
    -------
    tuple
        The fitted model, and a dict of per-parameter prior and posterior summaries.
    """
    from pymc_marketing.clv import BetaGeoModel

    data = pd.DataFrame(
        {
            "customer_id": summary["customer_unique_id"].to_numpy(),
            "frequency": summary["frequency"].to_numpy(),
            "recency": (summary["recency"] / time_divisor).to_numpy(),
            "T": (summary["T"] / time_divisor).to_numpy(),
        }
    )
    import pymc as pm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BetaGeoModel(data=data)
        model.build_model()
        # Drawn from the declared priors through PyMC directly. The model's own
        # prior-predictive helper is not available on this class, and what is
        # wanted here is the parameter prior rather than a predictive draw.
        with model.model:
            prior_draws = pm.sample_prior_predictive(
                draws=1000, random_seed=seed, var_names=["a", "b", "alpha", "r"]
            )
        model.fit(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)

    posterior = model.fit_result
    prior = prior_draws.prior

    declared = {
        name: str(value)
        for name, value in model.model_config.items()
        if name in ("alpha", "r", "phi_dropout", "kappa_dropout")
    }
    movement: dict[str, Any] = {"declared_priors": declared}
    for name in ("a", "b", "alpha", "r"):
        if name not in posterior:
            continue
        entry = {
            "posterior_mean": float(posterior[name].mean()),
            "posterior_sd": float(posterior[name].std()),
            "hdi_low": float(posterior[name].quantile(0.055)),
            "hdi_high": float(posterior[name].quantile(0.945)),
        }
        if prior is not None and name in prior:
            entry["prior_mean"] = float(prior[name].mean())
            entry["prior_sd"] = float(prior[name].std())
            entry["sd_ratio_posterior_over_prior"] = (
                entry["posterior_sd"] / entry["prior_sd"] if entry["prior_sd"] else None
            )
            entry["learned_from_data"] = bool(
                entry["prior_sd"] and entry["posterior_sd"] < 0.5 * entry["prior_sd"]
            )
        movement[name] = entry
    return model, movement


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
