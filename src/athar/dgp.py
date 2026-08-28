r"""The pre-registered data-generating process, and the truth it saves.

Why this exists
---------------
"Did the media-mix model recover the true ROI?" cannot be asked of real data,
because no real advertiser knows the true ROI of its own media plan. That is the
whole difficulty the field is organised around. So the question is asked here
instead, in a panel whose truth is known because it was constructed — and the
construction is committed to ``config/dgp.yaml`` before any model is fitted.

This module is an instrument, not a substitute for data that was unavailable.
Nothing it produces describes the effectiveness of a real marketing channel.

The model
---------
Spend is generated in logs from two latent AR(1) factors plus channel noise:

.. math::

    \log s_{c,t} = \mu_c + \kappa\,(\lambda_c z_t + \gamma_c w_t)
                   + \epsilon_{c,t} + f_{c,t}

``z`` is budget pressure, which moves every channel together; ``w`` is funnel
tilt, which trades performance spend against brand and upper-funnel spend. Two
factors rather than one because a single factor can only produce a rank-one
correlation structure, in which every pair of channels correlates in the same
direction — which is not what a media plan looks like. ``κ`` is the single knob
the recovery grid turns to make the design matrix well or badly conditioned.

Effect is Weibull-PDF adstock followed by a Hill curve:

.. math::

    h_{c,t} = \mathrm{Hill}\!\left(\sum_{l=0}^{L-1} w_l\, s_{c,t-l};\ k_c, \nu_c\right),
    \qquad \mathrm{Hill}(x) = \frac{x^{\nu}}{k^{\nu} + x^{\nu}}

and contribution is :math:`\beta_c h_{c,t}`, with :math:`\beta_c` solved so that
each channel's average ROI lands exactly on its pre-registered target.

**The fitted model is deliberately a different model.** The MMM in
:mod:`athar.mmm` uses geometric adstock and a logistic saturation, neither of
which can express what generated this. Fitting the generating form to its own
output recovers the assumptions and measures nothing; the matched-specification
arm of the recovery grid exists to separate misspecification error from
identification error, not to flatter the result.

Average and marginal ROI are both computed and they are not the same number.
Average ROI is what the MMM literature usually reports; marginal ROI at the
observed spend is what a budget optimiser actually needs, and under a saturating
response it is always the smaller of the two. Conflating them is a real error
this module makes it awkward to commit.

Attribution
-----------
:func:`attributed_revenue` is a parametric caricature of last-click, not a
simulation of user journeys. Two knobs per channel: ``tracking_rate``, the share
of a channel's true contribution last-click observes at all, and
``organic_capture``, the share of *baseline* revenue it hands to that channel.
The second knob is the mechanism the whole project is about.

Simulating individual journeys would look more faithful and would not be more
honest: the journey parameters would be exactly as invented as the bias is here,
only harder to state. Making the bias an explicit pre-registered input means the
question "how much does last-click over-credit brand search" has an answer that
can be read off the configuration rather than reverse-engineered from a
simulation. The cost is that no claim is made about attribution *mechanics* —
only about the consequences of a stated bias.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from athar import paths

__all__ = [
    "DgpConfig",
    "Panel",
    "apply_adstock",
    "attributed_revenue",
    "collinearity_diagnostics",
    "extended_baseline",
    "generate_panel",
    "hill",
    "load_config",
    "marginal_roi",
    "response_curve",
    "weibull_adstock_weights",
]


@dataclass(frozen=True)
class DgpConfig:
    """A parsed ``dgp.yaml`` and the hash of the exact bytes it was parsed from.

    The hash travels with every artifact the configuration produces, so a result
    can always name the configuration that produced it, and an edited config
    cannot be mistaken for the one a stored fit was run under.

    Parameters
    ----------
    spec : dict
        The parsed YAML.
    digest : str
        First 16 hex characters of the SHA-256 of the raw file bytes.
    """

    spec: dict[str, Any]
    digest: str

    @property
    def channels(self) -> list[dict[str, Any]]:
        """The channel blocks, in file order.

        Returns
        -------
        list of dict
            One block per channel.
        """
        return list(self.spec["channels"])

    @property
    def channel_names(self) -> list[str]:
        """Channel names, in file order.

        Returns
        -------
        list of str
            The names.
        """
        return [channel["name"] for channel in self.channels]


def load_config(path: str | Path | None = None) -> DgpConfig:
    """Read and hash the pre-registered configuration.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Defaults to ``config/dgp.yaml`` in the repository.

    Returns
    -------
    DgpConfig
        The parsed spec and its digest.

    Raises
    ------
    FileNotFoundError
        If the configuration is absent.
    """
    path = Path(path) if path is not None else paths.config_dir() / "dgp.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no DGP configuration at {path}")
    raw = path.read_bytes()
    return DgpConfig(spec=yaml.safe_load(raw), digest=hashlib.sha256(raw).hexdigest()[:16])


def weibull_adstock_weights(shape: float, scale: float, max_lag: int) -> NDArray[np.float64]:
    r"""Normalised Weibull-PDF carryover weights.

    .. math::

        w_l \propto \frac{k}{\lambda}\left(\frac{l+1}{\lambda}\right)^{k-1}
                    e^{-((l+1)/\lambda)^k},
        \qquad l = 0 \dots L-1

    normalised to sum to one, so adstock redistributes spend across weeks rather
    than inflating it. A shape above one puts the peak at a positive lag, which
    is the behaviour geometric adstock cannot express and which the misspecified
    fit therefore has to absorb somewhere else.

    Parameters
    ----------
    shape : float
        Weibull shape. Above 1 gives a delayed peak; below 1 is front-loaded.
    scale : float
        Weibull scale, in weeks.
    max_lag : int
        Number of lags retained.

    Returns
    -------
    numpy.ndarray of shape (max_lag,)
        Weights summing to 1.

    Raises
    ------
    ValueError
        If ``shape`` or ``scale`` is not positive, or ``max_lag`` is below 1.

    Examples
    --------
    Weights sum to one, and a shape above one peaks away from lag zero:

    >>> weights = weibull_adstock_weights(2.5, 4.0, 8)
    >>> round(float(weights.sum()), 12)
    1.0
    >>> int(weights.argmax())
    2

    A shape below one decays immediately:

    >>> int(weibull_adstock_weights(0.8, 1.0, 8).argmax())
    0
    """
    if shape <= 0 or scale <= 0:
        raise ValueError(f"shape and scale must be positive, got {shape} and {scale}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be at least 1, got {max_lag}")
    lags = np.arange(1, max_lag + 1, dtype=np.float64)
    density = (shape / scale) * (lags / scale) ** (shape - 1.0) * np.exp(-((lags / scale) ** shape))
    return density / density.sum()


def apply_adstock(spend: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convolve a spend series with carryover weights.

    Linear in ``spend``, which is what lets :func:`response_curve` scale a whole
    plan by a multiplier without re-running the convolution's structure, and what
    makes the analytic marginal ROI in :func:`marginal_roi` exact rather than a
    finite difference.

    The first ``len(weights) - 1`` positions carry incomplete history. Callers
    generate a burn-in prefix and discard it rather than accepting the ramp.

    Parameters
    ----------
    spend : numpy.ndarray of shape (n,)
        Spend per period, oldest first.
    weights : numpy.ndarray of shape (L,)
        Carryover weights, lag 0 first.

    Returns
    -------
    numpy.ndarray of shape (n,)
        Adstocked spend.

    Examples
    --------
    A single unit of spend spread over two weeks:

    >>> import numpy as np
    >>> apply_adstock(np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.4]))
    array([0.6, 0.4, 0. ])

    Scaling the input scales the output, exactly:

    >>> spend = np.array([1.0, 2.0, 3.0])
    >>> weights = np.array([0.6, 0.4])
    >>> bool(np.allclose(apply_adstock(2 * spend, weights), 2 * apply_adstock(spend, weights)))
    True
    """
    padded = np.concatenate([np.zeros(len(weights) - 1), np.asarray(spend, dtype=np.float64)])
    return np.convolve(padded, weights, mode="valid")


def hill(x: NDArray[np.float64], half_saturation: float, slope: float) -> NDArray[np.float64]:
    r"""Hill saturation curve.

    .. math::

        \mathrm{Hill}(x) = \frac{x^{\nu}}{k^{\nu} + x^{\nu}}

    Returns 0.5 at ``x = k`` for every slope, which is what makes ``k`` readable
    as a half-saturation point rather than an abstract scale parameter.

    Parameters
    ----------
    x : numpy.ndarray
        Adstocked spend, non-negative.
    half_saturation : float
        Spend at which the response reaches half its ceiling.
    slope : float
        Hill coefficient. Above 1 gives an S-shape with a convex toe.

    Returns
    -------
    numpy.ndarray
        Response in [0, 1).

    Raises
    ------
    ValueError
        If ``half_saturation`` or ``slope`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> float(hill(np.array([10.0]), half_saturation=10.0, slope=1.5)[0])
    0.5
    >>> [round(float(v), 4) for v in hill(np.array([0.0, 5.0, 20.0]), 10.0, 1.0)]
    [0.0, 0.3333, 0.6667]
    """
    if half_saturation <= 0 or slope <= 0:
        raise ValueError(
            f"half_saturation and slope must be positive, got {half_saturation} and {slope}"
        )
    powered = np.power(np.asarray(x, dtype=np.float64), slope)
    return powered / (half_saturation**slope + powered)


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> NDArray[np.float64]:
    """Draw a stationary AR(1) path, started from its stationary distribution."""
    innovation_sd = sd * np.sqrt(1.0 - phi**2)
    path = np.empty(n, dtype=np.float64)
    path[0] = rng.normal(0.0, sd)
    for index in range(1, n):
        path[index] = phi * path[index - 1] + rng.normal(0.0, innovation_sd)
    return path


def _detrended_sd(series: NDArray[np.float64]) -> float:
    """Residual standard deviation after a log-linear trend and one annual harmonic.

    Olist grew roughly fivefold across the window, so the raw standard deviation
    of the baseline is mostly trend. Two quantities here need the part a model
    would still have to explain — the observation-noise scale and the media share
    of variance — and the raw standard deviation distorts both.
    """
    series = np.asarray(series, dtype=np.float64)
    positions = np.arange(len(series), dtype=np.float64)
    design = np.column_stack(
        [
            np.ones_like(positions),
            positions,
            np.cos(2 * np.pi * positions / 52.18),
            np.sin(2 * np.pi * positions / 52.18),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, series, rcond=None)
    return float((series - design @ coefficients).std(ddof=1))


def collinearity_diagnostics(spend: pd.DataFrame) -> dict[str, Any]:
    """Measure how badly the spend design is conditioned.

    Reported alongside every recovery result so that a poor recovery is
    interpretable rather than mysterious. A media-mix model that cannot separate
    two channels which moved together has not failed; it has been asked a
    question the data does not answer, and these three numbers say so.

    Parameters
    ----------
    spend : pandas.DataFrame
        Weeks by channels, in levels.

    Returns
    -------
    dict
        ``correlation`` (channel to channel to float), ``vif`` (per channel),
        ``condition_number`` of the standardised design, and
        ``max_pairwise_correlation``.

    Examples
    --------
    Two identical channels are perfectly collinear and say so:

    >>> import pandas as pd
    >>> frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    >>> round(collinearity_diagnostics(frame)["max_pairwise_correlation"], 6)
    1.0
    """
    correlation = spend.corr()
    centred = spend - spend.mean()
    scaled = centred / centred.std(ddof=1).replace(0.0, np.nan)
    singular = np.linalg.svd(scaled.dropna(axis=1, how="all").to_numpy(), compute_uv=False)
    condition = float(singular.max() / singular.min()) if singular.min() > 0 else float("inf")

    vif: dict[str, float] = {}
    matrix = correlation.to_numpy()
    if np.linalg.matrix_rank(matrix) == len(matrix):
        diagonal = np.diag(np.linalg.inv(matrix))
        vif = {name: float(value) for name, value in zip(spend.columns, diagonal, strict=True)}
    else:
        vif = dict.fromkeys(spend.columns, float("inf"))

    offdiag = correlation.to_numpy()[~np.eye(len(correlation), dtype=bool)]
    return {
        "correlation": {
            row: {column: float(correlation.loc[row, column]) for column in correlation.columns}
            for row in correlation.index
        },
        "vif": vif,
        "condition_number": condition,
        "max_pairwise_correlation": float(np.max(np.abs(offdiag))) if len(offdiag) else 0.0,
    }


@dataclass(frozen=True)
class Panel:
    """A generated panel and everything needed to interrogate it.

    ``adstocked`` is kept because the response curves and the marginal ROI are
    both functions of it, and recomputing the convolution for every point of an
    optimiser's search would be the slowest part of the project for no reason.

    Parameters
    ----------
    weeks : pandas.DatetimeIndex
        The retained weeks, burn-in already removed.
    spend : pandas.DataFrame
        Weeks by channels, BRL.
    adstocked : pandas.DataFrame
        Spend after carryover, weeks by channels.
    contribution : pandas.DataFrame
        Incremental revenue by channel and week, BRL.
    baseline : numpy.ndarray
        The non-media component. Real Olist revenue, or a simulated extension.
    revenue : numpy.ndarray
        ``baseline + contribution.sum(axis=1) + noise``.
    truth : dict
        Everything a model is later scored against.
    """

    weeks: pd.DatetimeIndex
    spend: pd.DataFrame
    adstocked: pd.DataFrame
    contribution: pd.DataFrame
    baseline: NDArray[np.float64]
    revenue: NDArray[np.float64]
    truth: dict[str, Any]

    def frame(self) -> pd.DataFrame:
        """Render the panel as the single table a model is fitted to.

        The model sees week, spend per channel, and revenue. It does not see the
        contributions, the baseline or the noise, because those are the answer.

        Returns
        -------
        pandas.DataFrame
            Columns ``week``, one per channel, and ``revenue``.
        """
        table = pd.DataFrame({"week": self.weeks})
        for channel in self.spend.columns:
            table[channel] = self.spend[channel].to_numpy()
        table["revenue"] = self.revenue
        return table


def response_curve(
    panel: Panel, channel: str, multipliers: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Total incremental revenue from a channel when its plan is scaled.

    .. math::

        R_c(m) = \beta_c \sum_t \mathrm{Hill}(m\, a_{c,t};\ k_c, \nu_c)

    Scaling the whole plan rather than a single week is what a budget decision
    actually is, and adstock's linearity makes :math:`a_{c,t}` scale with ``m``
    exactly, so the curve needs no re-convolution. The half-saturation ``k`` is
    held at the value calibrated from observed spend: it is a property of the
    channel, not of the budget being considered.

    Parameters
    ----------
    panel : Panel
        A generated panel.
    channel : str
        Channel name.
    multipliers : numpy.ndarray
        Spend multipliers, 1.0 being the observed plan.

    Returns
    -------
    numpy.ndarray
        Total incremental revenue at each multiplier, BRL.

    Raises
    ------
    KeyError
        If the channel is not in the panel.
    """
    if channel not in panel.spend.columns:
        raise KeyError(f"unknown channel {channel!r}; panel has {list(panel.spend.columns)}")
    parameters = panel.truth["channels"][channel]
    adstocked = panel.adstocked[channel].to_numpy()
    return np.array(
        [
            parameters["beta"]
            * float(
                hill(
                    multiplier * adstocked,
                    parameters["half_saturation"],
                    parameters["hill_slope"],
                ).sum()
            )
            for multiplier in np.asarray(multipliers, dtype=np.float64)
        ]
    )


def marginal_roi(panel: Panel, channel: str, multiplier: float = 1.0) -> float:
    r"""Incremental revenue from the next BRL of spend on a channel.

    .. math::

        \frac{\partial R_c}{\partial S_c}
        = \frac{1}{S_c}\,\beta_c \sum_t a_{c,t}\,
          \frac{\nu k^{\nu} (m a_{c,t})^{\nu-1}}{\left(k^{\nu} + (m a_{c,t})^{\nu}\right)^2}

    Computed analytically rather than by finite difference, and checked against a
    finite difference in the test suite.

    This is not the average ROI, and the difference is the point. Average ROI
    divides total contribution by total spend and is what a media-mix model
    usually reports; marginal ROI is the slope at the current plan and is what a
    reallocation decision turns on.

    Which of the two is larger is not fixed, and assuming it is fixed is the
    error worth avoiding. A Hill curve with slope above one is S-shaped: below
    its inflection the response is *convex*, marginal exceeds average, and the
    channel is under-invested. Above the inflection the curve is concave,
    marginal falls below average, and a planner reallocating on average ROI
    overestimates what the next BRL buys. Both regimes occur in this panel at the
    configured spend levels, which is what makes the allocation problem worth
    solving rather than a ranking exercise.

    Parameters
    ----------
    panel : Panel
        A generated panel.
    channel : str
        Channel name.
    multiplier : float, optional
        Point on the response curve, 1.0 being the observed plan.

    Returns
    -------
    float
        Marginal revenue per BRL of spend.

    Raises
    ------
    KeyError
        If the channel is not in the panel.
    """
    if channel not in panel.spend.columns:
        raise KeyError(f"unknown channel {channel!r}; panel has {list(panel.spend.columns)}")
    parameters = panel.truth["channels"][channel]
    adstocked = panel.adstocked[channel].to_numpy()
    half_saturation = parameters["half_saturation"]
    slope = parameters["hill_slope"]

    scaled = multiplier * adstocked
    numerator = slope * half_saturation**slope * np.power(scaled, slope - 1.0)
    denominator = (half_saturation**slope + np.power(scaled, slope)) ** 2
    derivative = parameters["beta"] * float((adstocked * numerator / denominator).sum())
    return derivative / float(panel.spend[channel].sum())


def extended_baseline(
    observed: NDArray[np.float64], weeks: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Simulate a baseline of arbitrary length in the shape of the observed one.

    The recovery grid compares an 85-week panel against a 156-week panel, and
    Olist supplies only 85. Tiling the real series would fabricate a history;
    switching baselines between the two arms would confound length with baseline.
    So *both* arms of the grid use a simulated baseline from this function, and
    only the headline fit uses the real series. Length is then the only thing
    that changes across the length axis.

    The model is a log-linear trend, one annual Fourier pair, and an AR(1)
    residual. One harmonic, not three: 85 weeks is 1.6 years and will not support
    more without fitting noise. What this reproduces is the trend, the broad
    annual shape and the residual persistence — not Black Friday, which Olist has
    and this does not.

    Parameters
    ----------
    observed : numpy.ndarray
        The real weekly revenue series.
    weeks : int
        Length to simulate.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
        The simulated baseline and the fit diagnostics, including the R-squared
        of the deterministic part and the residual AR(1) coefficient.

    Raises
    ------
    ValueError
        If ``observed`` contains a non-positive value, which cannot be logged.
    """
    observed = np.asarray(observed, dtype=np.float64)
    if np.any(observed <= 0):
        raise ValueError("baseline extension needs a strictly positive revenue series")

    period = 52.18  # weeks in a year, so the harmonic does not drift over 3 years
    index = np.arange(len(observed), dtype=np.float64)

    def design(positions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            [
                np.ones_like(positions),
                positions,
                np.cos(2 * np.pi * positions / period),
                np.sin(2 * np.pi * positions / period),
            ]
        )

    logged = np.log(observed)
    matrix = design(index)
    coefficients, *_ = np.linalg.lstsq(matrix, logged, rcond=None)
    fitted = matrix @ coefficients
    residual = logged - fitted

    phi = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    residual_sd = float(residual.std(ddof=1))
    r_squared = float(1.0 - residual.var(ddof=1) / logged.var(ddof=1))

    future = np.arange(weeks, dtype=np.float64)
    simulated = design(future) @ coefficients + _ar1(rng, weeks, phi, residual_sd)
    return np.exp(simulated), {
        "r_squared": round(r_squared, 6),
        "residual_ar1": round(phi, 6),
        "residual_sd_log": round(residual_sd, 6),
        "trend_per_week_log": round(float(coefficients[1]), 8),
    }


def attributed_revenue(
    contribution: pd.DataFrame, baseline: NDArray[np.float64], config: DgpConfig
) -> pd.DataFrame:
    r"""What last-click would report, given the pre-registered bias.

    .. math::

        A_{c,t} = \tau_c\, \mathrm{contribution}_{c,t} + o_c\, \mathrm{baseline}_t

    The first term is under-counting: a channel whose effect surfaces later, or
    through someone else's click, is only partly observed. The second is
    over-counting, and it is the mechanism the project is about — revenue that
    would have happened anyway, credited to whichever channel the buyer touched
    last on the way to a purchase they had already decided on.

    ``search_nonbrand`` is configured with :math:`\tau = 1` and :math:`o = 0`, so
    last-click is exactly unbiased for it. That case is in the design on purpose.

    Parameters
    ----------
    contribution : pandas.DataFrame
        True incremental revenue, weeks by channels.
    baseline : numpy.ndarray
        The non-media component, per week.
    config : DgpConfig
        Supplies ``tracking_rate`` and ``organic_capture`` per channel.

    Returns
    -------
    pandas.DataFrame
        Attributed revenue, weeks by channels, on the same index.

    Raises
    ------
    ValueError
        If the configured ``organic_capture`` values sum above one, which would
        have last-click crediting more baseline revenue than exists.
    """
    settings = {channel["name"]: channel["attribution"] for channel in config.channels}
    total_capture = sum(value["organic_capture"] for value in settings.values())
    if total_capture > 1.0:
        raise ValueError(
            f"organic_capture sums to {total_capture:.3f}; last-click cannot credit more "
            f"baseline revenue than exists"
        )
    baseline = np.asarray(baseline, dtype=np.float64)
    return pd.DataFrame(
        {
            name: settings[name]["tracking_rate"] * contribution[name].to_numpy()
            + settings[name]["organic_capture"] * baseline
            for name in contribution.columns
        },
        index=contribution.index,
    )


def generate_panel(
    config: DgpConfig,
    baseline: NDArray[np.float64],
    *,
    collinearity: str | float | None = None,
    seed: int | None = None,
    week_index: pd.DatetimeIndex | None = None,
) -> Panel:
    """Generate one panel and the truth it will later be scored against.

    Spend is generated with a burn-in prefix of ``max_lag - 1`` weeks which is
    then discarded, so every retained week carries a complete carryover history
    rather than a ramp from an assumed zero.

    Each channel's spend is rescaled after generation so its realised mean over
    the retained weeks equals the configured ``mean_weekly_spend`` exactly. The
    alternative — setting the mean of the underlying log-normal and accepting the
    drift — makes the panel's total budget a function of the seed, which would
    make the recovery grid's cells differ in budget as well as in the thing being
    varied.

    ``beta`` is then solved so that each channel's average ROI lands exactly on
    its pre-registered target. The targets are the primary input because ROI is
    what this project is about; the resulting share of revenue variance
    attributable to media is a *diagnostic*, reported rather than imposed. Fixing
    the variance share instead would make every effect size depend on the noise
    level, which is the wrong dependency.

    Parameters
    ----------
    config : DgpConfig
        The pre-registered configuration.
    baseline : numpy.ndarray
        The non-media revenue component, one value per retained week.
    collinearity : str or float, optional
        A named level from the configuration, or a raw scaling. Defaults to the
        configured default.
    seed : int, optional
        Defaults to the configured seed.
    week_index : pandas.DatetimeIndex, optional
        Weeks to label the panel with. Defaults to Mondays from 2017-01-02, the
        start of the Olist window.

    Returns
    -------
    Panel
        The panel, and its truth.

    Raises
    ------
    ValueError
        If ``collinearity`` names an unknown level, or the generated revenue goes
        non-positive, which would mean the noise level is implausible for the
        baseline it was applied to.
    """
    spec = config.spec
    baseline = np.asarray(baseline, dtype=np.float64)
    n_weeks = len(baseline)

    if collinearity is None:
        collinearity = spec["collinearity"]["default"]
    if isinstance(collinearity, str):
        levels = spec["collinearity"]["levels"]
        if collinearity not in levels:
            raise ValueError(f"unknown collinearity level {collinearity!r}; have {sorted(levels)}")
        level_name, kappa = collinearity, float(levels[collinearity])
    else:
        level_name, kappa = "custom", float(collinearity)

    rng = np.random.default_rng(spec["seed"] if seed is None else seed)
    max_lag = int(spec["adstock"]["max_lag"])
    burn = max_lag - 1
    total_weeks = n_weeks + burn

    factors = spec["collinearity"]["factors"]
    budget_pressure = _ar1(
        rng, total_weeks, factors["budget_pressure"]["ar_phi"], factors["budget_pressure"]["sd"]
    )
    funnel_tilt = _ar1(
        rng, total_weeks, factors["funnel_tilt"]["ar_phi"], factors["funnel_tilt"]["sd"]
    )
    idiosyncratic = spec["collinearity"]["idiosyncratic"]

    # Total log-spend volatility is held fixed and only its *composition* changes
    # with the level, so the recovery grid's collinearity axis varies collinearity
    # and nothing else. Raising correlation by raising the loadings would also
    # make spend more volatile, and the grid could not then say which of the two
    # moved the result.
    log_spend_sd = float(spec["collinearity"]["log_spend_sd"])
    factor_sd = kappa * log_spend_sd
    idiosyncratic_sd = float(np.sqrt(max(0.0, 1.0 - kappa**2))) * log_spend_sd

    spend_columns: dict[str, NDArray[np.float64]] = {}
    adstock_columns: dict[str, NDArray[np.float64]] = {}
    full_spend: dict[str, NDArray[np.float64]] = {}

    for channel in config.channels:
        name = channel["name"]
        loading = channel["loading"]
        noise = _ar1(rng, total_weeks, idiosyncratic["ar_phi"], idiosyncratic_sd)

        # Loadings are normalised to unit length so every channel receives the
        # same total factor variance regardless of how its exposure splits
        # between the two factors. Without this, a channel loading on both would
        # simply be more volatile than one loading on a single factor — a
        # volatility difference masquerading as a structural one.
        norm = float(np.hypot(loading["budget_pressure"], loading["funnel_tilt"]))

        flighting = np.zeros(total_weeks)
        period = int(channel["flighting"]["period_weeks"])
        if period > 0:
            positions = np.arange(total_weeks, dtype=np.float64)
            flighting = channel["flighting"]["amplitude"] * np.sin(2 * np.pi * positions / period)

        logged = (
            factor_sd
            / norm
            * (loading["budget_pressure"] * budget_pressure + loading["funnel_tilt"] * funnel_tilt)
            + noise
            + flighting
        )
        series = np.exp(logged)
        # Rescale on the retained weeks so the panel's budget is exactly the
        # configured one; apply the same factor to the burn-in so carryover into
        # week zero is consistent with the weeks that follow.
        series = series * (channel["mean_weekly_spend"] / series[burn:].mean())

        weights = weibull_adstock_weights(
            channel["weibull"]["shape"], channel["weibull"]["scale"], max_lag
        )
        full_spend[name] = series
        spend_columns[name] = series[burn:]
        adstock_columns[name] = apply_adstock(series, weights)[burn:]

    weeks = (
        week_index
        if week_index is not None
        else pd.date_range("2017-01-02", periods=n_weeks, freq="W-MON")
    )
    spend = pd.DataFrame(spend_columns, index=weeks)
    adstocked = pd.DataFrame(adstock_columns, index=weeks)

    channel_truth: dict[str, Any] = {}
    contribution_columns: dict[str, NDArray[np.float64]] = {}

    for channel in config.channels:
        name = channel["name"]
        slope = float(channel["hill"]["slope"])
        half_saturation = float(
            channel["hill"]["half_saturation_multiple"] * adstocked[name].mean()
        )
        response = hill(adstocked[name].to_numpy(), half_saturation, slope)
        total_spend = float(spend[name].sum())
        beta = float(channel["true_roi"]) * total_spend / float(response.sum())
        contribution_columns[name] = beta * response

        channel_truth[name] = {
            "beta": beta,
            "half_saturation": half_saturation,
            "hill_slope": slope,
            "adstock_weights": weibull_adstock_weights(
                channel["weibull"]["shape"], channel["weibull"]["scale"], max_lag
            ).tolist(),
            "weibull_shape": float(channel["weibull"]["shape"]),
            "weibull_scale": float(channel["weibull"]["scale"]),
            "total_spend": total_spend,
            "mean_weekly_spend": float(spend[name].mean()),
            "tracking_rate": float(channel["attribution"]["tracking_rate"]),
            "organic_capture": float(channel["attribution"]["organic_capture"]),
        }

    contribution = pd.DataFrame(contribution_columns, index=weeks)

    noise_sd = float(spec["panel"]["noise_share_of_detrended_baseline_sd"]) * _detrended_sd(
        baseline
    )
    observation_noise = rng.normal(0.0, noise_sd, n_weeks)
    revenue = baseline + contribution.sum(axis=1).to_numpy() + observation_noise
    if np.any(revenue <= 0):
        raise ValueError("generated revenue went non-positive; the noise level is implausible")

    panel = Panel(
        weeks=pd.DatetimeIndex(weeks),
        spend=spend,
        adstocked=adstocked,
        contribution=contribution,
        baseline=baseline,
        revenue=revenue,
        truth={"channels": channel_truth},
    )

    attributed = attributed_revenue(contribution, baseline, config)

    for name, block in channel_truth.items():
        total_contribution = float(contribution[name].sum())
        total_attributed = float(attributed[name].sum())
        roi_average = total_contribution / block["total_spend"]
        roi_attributed = total_attributed / block["total_spend"]
        block.update(
            {
                "total_contribution": total_contribution,
                # Contribution is zero at zero spend, so the counterfactual
                # "incremental revenue if this channel were switched off" is the
                # channel's whole contribution. Stated rather than implied,
                # because it is what a holdout experiment is trying to estimate.
                "incremental_revenue_vs_zero_spend": total_contribution,
                "roi_average": roi_average,
                "roi_marginal": marginal_roi(panel, name),
                "attributed_revenue": total_attributed,
                "roas_attributed": roi_attributed,
                "attribution_bias_absolute": roi_attributed - roi_average,
                "attribution_bias_relative": roi_attributed / roi_average - 1.0,
            }
        )

    media = contribution.sum(axis=1).to_numpy()
    panel.truth.update(
        {
            "config_digest": config.digest,
            "seed": int(spec["seed"] if seed is None else seed),
            "collinearity_level": level_name,
            "collinearity_kappa": kappa,
            "weeks": n_weeks,
            "max_lag": max_lag,
            "totals": {
                "spend": float(spend.to_numpy().sum()),
                "media_contribution": float(media.sum()),
                "baseline": float(baseline.sum()),
                "revenue": float(revenue.sum()),
                "blended_roi": float(media.sum() / spend.to_numpy().sum()),
            },
            "collinearity": collinearity_diagnostics(spend),
            "variance": {
                "baseline": float(np.var(baseline, ddof=1)),
                "media": float(np.var(media, ddof=1)),
                "noise": float(np.var(observation_noise, ddof=1)),
                "revenue": float(np.var(revenue, ddof=1)),
                "media_share_of_revenue_variance": float(
                    np.var(media, ddof=1) / np.var(revenue, ddof=1)
                ),
                # The share above is dominated by the baseline's trend, which any
                # media-mix model absorbs into its own trend and seasonality terms
                # rather than having to explain with media. The share that
                # actually bounds identification is measured after that
                # deterministic part is removed, so both are reported and the
                # second is the one to quote.
                "detrended_baseline_sd": _detrended_sd(baseline),
                "media_share_of_detrended_variance": float(
                    np.var(media, ddof=1)
                    / (np.var(media, ddof=1) + _detrended_sd(baseline) ** 2 + noise_sd**2)
                ),
                "noise_sd": noise_sd,
            },
        }
    )
    return panel
