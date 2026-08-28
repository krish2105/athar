"""Turning an ROI estimate into a budget, and finding out what that estimate cost.

This is where the project stops being a measurement exercise and becomes a
decision. Three methods each hand a chief marketing officer a different set of
numbers for the same five channels. Each set implies a different split of the same
budget. Only one of those splits is best, and because the panel's truth is known,
the gap between them is computable in money rather than argued about.

Two allocators, because planners think in two different ways
------------------------------------------------------------
:func:`allocate_concave` maximises the total of fitted response curves under a
budget constraint. It knows that spending more on a channel buys less per unit, so
it stops before a channel saturates. This is what a media-mix model lets you do,
and it is the only one of the two that can use curvature.

:func:`allocate_linear` is what everything else forces. An experiment returns one
number for a channel; so does an attribution report. Neither says anything about
what the *next* dirham would earn, so a planner holding only those numbers has no
choice but to treat return as constant and pour budget into whatever ranks
highest. That is not a straw man — it is the arithmetic that a ROAS table permits.

Its solution is a corner: fill the best-ranked channel to its ceiling, then the
next. Which is why the ceilings matter.

The bounds are the honest part
------------------------------
Without per-channel floors and caps, every linear allocation puts the entire budget
into one channel, which no marketing organisation would ever do, and comparing that
to a curved optimum would be beating up a caricature. So both allocators run under
the same pre-registered governance constraints — no channel below a floor, none
above a cap — and the comparison is between two planners operating under identical
real-world restraint, differing only in what they believe about returns.

Evaluation
----------
:func:`evaluate_under_truth` scores any allocation against the generator's true
response curves. Every allocation is scored the same way, so what is being compared
is the *consequence* of believing an estimator, not the estimator's own opinion of
itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from athar import dgp

__all__ = [
    "DEFAULT_CAP",
    "DEFAULT_FLOOR",
    "allocate_concave",
    "allocate_linear",
    "bounds_from_shares",
    "evaluate_under_truth",
    "truth_response_functions",
]

log = logging.getLogger(__name__)

#: Pre-registered governance constraints. No channel may fall below 5% of the
#: budget or rise above 40%. Chosen to be unremarkable rather than convenient: a
#: real plan carries contractual minimums and internal concentration limits, and
#: without them every linear allocation collapses onto a single channel and the
#: comparison in this module would be against a caricature.
DEFAULT_FLOOR = 0.05
DEFAULT_CAP = 0.40


def bounds_from_shares(
    channels: list[str],
    budget: float,
    floor: float = DEFAULT_FLOOR,
    cap: float = DEFAULT_CAP,
) -> list[tuple[float, float]]:
    """Per-channel spend bounds in money, from budget shares.

    Parameters
    ----------
    channels : list of str
        Channel names, fixing the order.
    budget : float
        Total to allocate.
    floor, cap : float, optional
        Minimum and maximum share of the budget per channel.

    Returns
    -------
    list of tuple
        ``(low, high)`` per channel, in the order given.

    Raises
    ------
    ValueError
        If the bounds cannot contain the budget — floors summing above it, or caps
        summing below it — which would make the problem infeasible.

    Examples
    --------
    >>> bounds_from_shares(["a", "b"], 1000.0, floor=0.1, cap=0.6)
    [(100.0, 600.0), (100.0, 600.0)]
    """
    count = len(channels)
    if floor * count > 1.0 + 1e-12:
        raise ValueError(
            f"a floor of {floor:.0%} across {count} channels needs {floor * count:.0%} "
            f"of the budget"
        )
    if cap * count < 1.0 - 1e-12:
        raise ValueError(
            f"a cap of {cap:.0%} across {count} channels reaches only {cap * count:.0%} "
            f"of the budget"
        )
    return [(floor * budget, cap * budget) for _ in channels]


def truth_response_functions(panel: dgp.Panel) -> dict[str, Callable[[float], float]]:
    """The generator's true revenue curves, as functions of spend in money.

    Parameters
    ----------
    panel : athar.dgp.Panel
        A generated panel.

    Returns
    -------
    dict
        Channel to a callable mapping total spend to total incremental revenue.
    """
    functions: dict[str, Callable[[float], float]] = {}
    for channel in panel.spend.columns:
        observed = float(panel.spend[channel].sum())

        def response(spend: float, channel: str = channel, observed: float = observed) -> float:
            return float(dgp.response_curve(panel, channel, np.array([spend / observed]))[0])

        functions[channel] = response
    return functions


def allocate_concave(
    responses: dict[str, Callable[[float], float]],
    budget: float,
    bounds: list[tuple[float, float]],
    channels: list[str] | None = None,
) -> dict[str, float]:
    """Maximise total response under a budget, using the curvature of the responses.

    Sequential least squares, started from an equal split and repeated from several
    random starts. A saturating response is concave above its inflection, but a Hill
    curve with a slope above one is convex below it, so the objective is not concave
    everywhere and a single start can settle on a local optimum. The restarts are
    cheap and the alternative is silently reporting the worse of two answers.

    Parameters
    ----------
    responses : dict
        Channel to a callable mapping spend to revenue.
    budget : float
        Total to allocate.
    bounds : list of tuple
        Per-channel ``(low, high)``, in channel order.
    channels : list of str, optional
        Order. Defaults to the order of ``responses``.

    Returns
    -------
    dict
        Channel to allocated spend, summing to ``budget``.

    Raises
    ------
    RuntimeError
        If no start converges to a feasible allocation.
    """
    channels = list(responses) if channels is None else list(channels)
    lows = np.array([low for low, _ in bounds])
    highs = np.array([high for _, high in bounds])

    def total(allocation: np.ndarray) -> float:
        return -sum(
            responses[name](float(value)) for name, value in zip(channels, allocation, strict=True)
        )

    constraint = {"type": "eq", "fun": lambda x: float(x.sum() - budget)}
    rng = np.random.default_rng(20260829)
    starts = [np.full(len(channels), budget / len(channels))]
    for _ in range(6):
        weights = rng.dirichlet(np.ones(len(channels)))
        starts.append(np.clip(weights * budget, lows, highs))

    best, best_value = None, np.inf
    for start in starts:
        start = np.clip(start, lows, highs)
        start = start * budget / start.sum()
        start = np.clip(start, lows, highs)
        result = minimize(
            total,
            start,
            method="SLSQP",
            bounds=list(zip(lows, highs, strict=True)),
            constraints=[constraint],
            options={"maxiter": 400, "ftol": 1e-9},
        )
        if (
            result.success
            and abs(result.x.sum() - budget) < 1e-4 * budget
            and result.fun < best_value
        ):
            best, best_value = result.x, result.fun

    if best is None:
        raise RuntimeError("no start converged to a feasible allocation")
    return {name: float(value) for name, value in zip(channels, best, strict=True)}


def allocate_linear(
    roi: dict[str, float], budget: float, bounds: list[tuple[float, float]]
) -> dict[str, float]:
    """Allocate as a planner holding only a ROAS table must: constant returns.

    An experiment and an attribution report each give one number per channel and
    say nothing about what the next dirham earns. Under constant returns the
    optimum is a corner — every channel at its floor, then the budget poured into
    the highest-ranked channel until it hits its cap, then the next.

    Solved exactly rather than numerically. The problem is a linear programme with
    a single budget constraint and box bounds, whose greedy solution is provably
    optimal, and reaching for an optimiser here would add a convergence failure mode
    to a problem that has a closed form.

    Parameters
    ----------
    roi : dict
        Channel to estimated return per unit of spend.
    budget : float
        Total to allocate.
    bounds : list of tuple
        Per-channel ``(low, high)``, in the order of ``roi``.

    Returns
    -------
    dict
        Channel to allocated spend.

    Raises
    ------
    ValueError
        If the caps cannot absorb the budget.

    Examples
    --------
    The better channel fills to its cap; the worse one takes the remainder:

    >>> allocate_linear({"good": 3.0, "bad": 1.0}, 100.0, [(10.0, 60.0), (10.0, 60.0)])
    {'good': 60.0, 'bad': 40.0}
    """
    channels = list(roi)
    allocation = {name: low for name, (low, _) in zip(channels, bounds, strict=True)}
    caps = dict(zip(channels, [high for _, high in bounds], strict=True))

    remaining = budget - sum(allocation.values())
    if remaining < -1e-9:
        raise ValueError("the floors already exceed the budget")
    if sum(caps.values()) < budget - 1e-9:
        raise ValueError("the caps cannot absorb the budget")

    for name in sorted(channels, key=lambda n: roi[n], reverse=True):
        if remaining <= 1e-12:
            break
        room = caps[name] - allocation[name]
        take = min(room, remaining)
        allocation[name] += take
        remaining -= take
    return {name: float(round(value, 10)) for name, value in allocation.items()}


def evaluate_under_truth(
    allocation: dict[str, float], panel: dgp.Panel
) -> dict[str, float | dict[str, float]]:
    """Score an allocation against the generator's true response curves.

    Every allocation in this project is scored here, whichever estimator produced
    it, so the comparison is between the consequences of believing each estimator
    rather than between each estimator's opinion of itself.

    Parameters
    ----------
    allocation : dict
        Channel to spend.
    panel : athar.dgp.Panel
        The generated panel supplying the truth.

    Returns
    -------
    dict
        ``total_revenue``, ``total_spend``, ``blended_roi``, and the per-channel
        revenue behind them.
    """
    responses = truth_response_functions(panel)
    by_channel = {name: responses[name](spend) for name, spend in allocation.items()}
    total_revenue = float(sum(by_channel.values()))
    total_spend = float(sum(allocation.values()))
    return {
        "total_revenue": total_revenue,
        "total_spend": total_spend,
        "blended_roi": total_revenue / total_spend if total_spend else float("nan"),
        "by_channel": {name: float(value) for name, value in by_channel.items()},
    }
