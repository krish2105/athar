"""Geo holdout tests: unbiased, imprecise, and expensive — measured rather than asserted.

An experiment is the only one of the three methods in this project that is unbiased
by construction. It is also the one nobody runs often enough, and this module is
about why: on a real geographic footprint, the confidence interval around a single
channel's lift is wide enough that a well-powered test costs weeks of switched-off
spend in a meaningful share of the market.

The geography is real. Olist sold into 27 Brazilian states, and their revenue is
grotesquely unequal — the largest carries 38% of it and thirteen of the twenty-seven
carry under 1% each. That concentration is the power problem, and it is not one this
project invented.

The design
----------
A set of treated states goes dark on one channel for a block of weeks. Revenue is
compared before and after, treated against untreated — difference in differences.
The true effect is known exactly, because the contribution that was removed is the
contribution the generator put there, so the estimator can be scored rather than
merely reported.

Two quantities come out. **Bias**, which should be near zero and is checked rather
than assumed. And **spread** across random assignments of which states are treated,
which is what a single real experiment would have been one draw from.

The one assumption worth naming
-------------------------------
Media contribution is allocated across states in proportion to each state's share of
real revenue. That is an assumption, not a measurement: Olist carries no spend data,
so there is no way to know whether advertising worked harder in some states than
others. It makes the treated and untreated groups differ only in scale, which if
anything makes the experiment look *better* behaved than a real one, where regional
response heterogeneity is another source of variance. The direction of that
optimism is stated so it cannot be mistaken for realism.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "difference_in_differences",
    "power_curve",
    "run_holdout",
    "state_shares",
]

log = logging.getLogger(__name__)


def state_shares(panel: pd.DataFrame) -> pd.Series:
    """Each state's share of total revenue, from the real Olist geography.

    Parameters
    ----------
    panel : pandas.DataFrame
        The state-week frame from :func:`athar.frame.state_week_revenue`.

    Returns
    -------
    pandas.Series
        Shares indexed by state, summing to one, largest first.

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"state": ["SP", "SP", "AC"], "revenue": [60.0, 20.0, 20.0]})
    >>> shares = state_shares(panel)
    >>> round(float(shares["SP"]), 3), round(float(shares["AC"]), 3)
    (0.8, 0.2)
    """
    totals = panel.groupby("state")["revenue"].sum()
    return (totals / totals.sum()).sort_values(ascending=False)


def difference_in_differences(
    revenue: pd.DataFrame, treated: list[str], test_weeks: np.ndarray
) -> float:
    r"""Estimate the revenue lost to a holdout, against a scale-matched control.

    .. math::

        \hat{\Delta} = \left(\bar{Y}^{C}_{post}\,
                         \frac{\bar{Y}^{T}_{pre}}{\bar{Y}^{C}_{pre}}
                         - \bar{Y}^{T}_{post}\right) \times W

    The control group is rescaled to the treated group's pre-period level before the
    comparison. That rescaling is not a refinement, it is the whole estimator: three
    small states against the other twenty-four is a comparison between quantities
    differing by an order of magnitude, and a plain difference of differences on the
    raw sums is dominated by the control group's own growth rather than by the
    treatment. The first version of this function did exactly that and reported
    relative errors in the hundreds.

    Rescaling also absorbs anything that moved every state at once — a season, a
    promotion, the platform's own growth — which is the reason to run the test
    geographically rather than switching a channel off nationally and eyeballing the
    line.

    The identifying assumption is that treated and untreated revenue would have kept
    the same *ratio* absent the holdout. That is the parallel-trends assumption in
    multiplicative form, and it is the right form here because the series grows.

    Parameters
    ----------
    revenue : pandas.DataFrame
        Weeks by states.
    treated : list of str
        States that went dark.
    test_weeks : numpy.ndarray
        Boolean mask over rows marking the test window.

    Returns
    -------
    float
        Estimated revenue removed, positive when the holdout cost money.

    Raises
    ------
    ValueError
        If either group is empty, the window leaves no pre-period, or the control
        group's pre-period revenue is zero.

    Examples
    --------
    A treated state that loses a tenth of its revenue during the test, against a
    control that is flat:

    >>> import numpy as np, pandas as pd
    >>> revenue = pd.DataFrame({"T": [100.0] * 6 + [90.0] * 4, "C": [500.0] * 10})
    >>> mask = np.array([False] * 6 + [True] * 4)
    >>> round(difference_in_differences(revenue, ["T"], mask), 6)
    40.0
    """
    untreated = [state for state in revenue.columns if state not in treated]
    if not treated or not untreated:
        raise ValueError("difference in differences needs a non-empty group on both sides")
    if test_weeks.all() or not test_weeks.any():
        raise ValueError("the test window must leave a pre-period and a test period")

    treated_series = revenue[treated].sum(axis=1).to_numpy()
    control_series = revenue[untreated].sum(axis=1).to_numpy()

    control_pre = control_series[~test_weeks].mean()
    if control_pre == 0:
        raise ValueError("the control group has no pre-period revenue to scale against")

    scale = treated_series[~test_weeks].mean() / control_pre
    counterfactual = control_series[test_weeks].mean() * scale
    return float((counterfactual - treated_series[test_weeks].mean()) * test_weeks.sum())


def run_holdout(
    panel: Any,
    state_panel: pd.DataFrame,
    channel: str,
    treated: list[str],
    first_test_week: int,
    test_weeks: int,
    rng: np.random.Generator,
    noise_share: float = 0.05,
) -> dict[str, float]:
    """Simulate one geo holdout and score the estimator against the known truth.

    Parameters
    ----------
    panel : athar.dgp.Panel
        The generated panel supplying the true contributions.
    state_panel : pandas.DataFrame
        Real Olist state-week revenue, for the geographic shares.
    channel : str
        The channel switched off.
    treated : list of str
        States that go dark.
    first_test_week, test_weeks : int
        Test window, as an index into the panel's weeks.
    rng : numpy.random.Generator
        Seeded generator for the region-level observation noise.
    noise_share : float, optional
        Region-level noise as a share of each state's mean weekly revenue. Present
        because a state-week series is far noisier than the national aggregate, and
        an experiment run on smooth data would overstate how easy this is.

    Returns
    -------
    dict
        The true removed revenue, the estimate, and the error.

    Raises
    ------
    ValueError
        If the test window falls outside the panel.
    """
    weeks = len(panel.weeks)
    if first_test_week + test_weeks > weeks:
        raise ValueError(f"test window runs past the end of a {weeks}-week panel")

    shares = state_shares(state_panel)
    states = list(shares.index)
    share_vector = shares.to_numpy()

    national_baseline = panel.baseline
    contribution = panel.contribution[channel].to_numpy()
    other = panel.contribution.drop(columns=[channel]).sum(axis=1).to_numpy()

    # Allocate national revenue across states by revenue share.
    revenue = np.outer(national_baseline + other, share_vector) + np.outer(
        contribution, share_vector
    )

    mask = np.zeros(weeks, dtype=bool)
    mask[first_test_week : first_test_week + test_weeks] = True

    # The holdout: treated states lose this channel's contribution during the test.
    treated_index = [states.index(state) for state in treated]
    removed = np.outer(contribution, share_vector)[np.ix_(mask, treated_index)]
    revenue[np.ix_(mask, treated_index)] -= removed

    revenue = revenue + rng.normal(0.0, noise_share * revenue.mean(axis=0), revenue.shape)

    frame = pd.DataFrame(revenue, columns=states)
    estimate = difference_in_differences(frame, treated, mask)
    actual = float(removed.sum())
    return {
        "true_removed_revenue": actual,
        "estimated_removed_revenue": estimate,
        "absolute_error": estimate - actual,
        "relative_error": estimate / actual - 1.0 if actual else float("nan"),
        "treated_states": len(treated),
        "treated_revenue_share": float(shares[treated].sum()),
        "test_weeks": test_weeks,
    }


def power_curve(
    panel: Any,
    state_panel: pd.DataFrame,
    channel: str,
    treated_counts: tuple[int, ...] = (3, 5, 8, 13),
    test_weeks: int = 8,
    replicates: int = 60,
    seed: int = 20260829,
) -> list[dict[str, float]]:
    """How the estimate behaves as more of the market is switched off.

    Each replicate draws a different set of treated states, so the spread reported
    here is the spread a real experimenter faces when they pick regions — the single
    experiment they actually run is one draw from this.

    ``detected`` is the share of replicates whose estimate has the right sign and
    lands within half the true effect. A blunt criterion, chosen before the numbers
    were seen and reported as it is, rather than a p-value that would imply a
    testing framework this design does not have.

    Parameters
    ----------
    panel : athar.dgp.Panel
        The generated panel.
    state_panel : pandas.DataFrame
        Real Olist state-week revenue.
    channel : str
        The channel switched off.
    treated_counts : tuple of int, optional
        Numbers of treated states to try.
    test_weeks : int, optional
        Length of the holdout.
    replicates : int, optional
        Random assignments per point.
    seed : int, optional
        Reproducibility.

    Returns
    -------
    list of dict
        One entry per treated-state count.
    """
    shares = state_shares(state_panel)
    states = list(shares.index)
    rng = np.random.default_rng(seed)
    first_test_week = len(panel.weeks) - test_weeks - 1

    rows = []
    for count in treated_counts:
        if count >= len(states):
            continue
        errors, estimates, covered, treated_share = [], [], [], []
        for _ in range(replicates):
            treated = list(rng.choice(states, size=count, replace=False))
            result = run_holdout(
                panel, state_panel, channel, treated, first_test_week, test_weeks, rng
            )
            errors.append(result["relative_error"])
            estimates.append(result["estimated_removed_revenue"])
            treated_share.append(result["treated_revenue_share"])
            covered.append(abs(result["relative_error"]) < 0.5)
        rows.append(
            {
                "treated_states": count,
                "median_treated_revenue_share": float(np.median(treated_share)),
                "median_relative_error": float(np.median(errors)),
                "mean_relative_error": float(np.mean(errors)),
                "relative_error_sd": float(np.std(errors, ddof=1)),
                "estimate_sd": float(np.std(estimates, ddof=1)),
                "detected": float(np.mean(covered)),
                "replicates": replicates,
            }
        )
    return rows
