"""Criteo: who advertising actually moves, and whether a model can find them.

The intent-to-treat effect in ``metrics/criteo.json`` says advertising worked, on
average, by 11.5 basis points. That is an average over 13,979,592 people, almost
all of whom were never going to convert and a few of whom would have converted
anyway. The question a marketer actually faces is narrower: *is the effect
concentrated somewhere I can find with the features I have?*

If it is, targeting the top slice of predicted uplift buys most of the incremental
conversions for a fraction of the impressions. If it is not, the honest answer is
that the twelve anonymised features do not locate the effect, and that is reported
as the finding rather than dressed up.

Evaluation
----------
Ranking quality is measured with :func:`spine.metrics.qini_auc`, which subtracts
the random-targeting line and therefore scores only what the *ranking* contributed
— a model with no uplift signal scores zero however large the average effect is.
Every Qini figure here is reported with a bootstrap interval and with the number of
control-arm converters behind it, because that count, not the nominal row count, is
what bounds the precision.

Why the split is random, and why that is not an oversight
---------------------------------------------------------
Criteo carries no time column. It is a randomised cross-section, and it is split as
one. ``spine.splitting`` is deliberately unused here — the same treatment ADIL gives
Home Credit — and used everywhere in this project that a time axis genuinely exists:
the media-mix panel and the CLV calibration window.

The one thing never conditioned on
----------------------------------
``exposure``. It is decided after randomisation, so conditioning on it compares a
self-selected group against everyone else. Every model here is trained and scored
against ``treatment``, the randomised arm. ``exposure`` appears in this project
exactly once, in ``scripts/build_criteo.py``, to quantify what a platform reports.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from spine.metrics import qini_auc

from athar import paths

__all__ = [
    "FEATURES",
    "bootstrap_qini",
    "incremental_by_decile",
    "load_sample",
    "split_random",
    "targeting_curve",
]

log = logging.getLogger(__name__)

#: The twelve anonymised features. Criteo publishes no descriptions, so no feature
#: here can be given a business interpretation and none is attempted.
FEATURES: list[str] = [f"f{index}" for index in range(12)]


def load_sample(rows: int, seed: int = 20260829) -> pd.DataFrame:
    """Draw a uniform random sample from the Parquet conversion.

    Uniform rather than stratified on the outcome. Case-control sampling would put
    more converters in the frame, but it also breaks the arm balance that makes a
    Qini curve interpretable, and correcting for that needs weights the shared Qini
    implementation does not take. A larger uniform sample is the simpler honest
    answer, and the headline causal estimates are computed on the full population
    anyway, in ``scripts/build_criteo.py``.

    Parameters
    ----------
    rows : int
        Approximate number of rows to draw.
    seed : int, optional
        Recorded with every result computed from the sample.

    Returns
    -------
    pandas.DataFrame
        Features plus ``treatment``, ``conversion``, ``visit``.

    Raises
    ------
    FileNotFoundError
        If the Parquet conversion has not been built.
    """
    parquet = paths.criteo_parquet()
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} is missing; run `make criteo` first")

    connection = duckdb.connect()
    try:
        connection.execute("SET threads TO 1")
        total = connection.execute(f"SELECT count(*) FROM read_parquet('{parquet}')").fetchone()[0]
        share = min(1.0, rows / total)
        columns = ", ".join([*FEATURES, "treatment", "conversion", "visit"])
        return connection.execute(f"""
            SELECT {columns}
            FROM read_parquet('{parquet}')
            USING SAMPLE {share * 100:.6f} PERCENT (bernoulli, {seed})
        """).df()
    finally:
        connection.close()


def split_random(
    frame: pd.DataFrame, test_share: float = 0.5, seed: int = 20260829
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the randomised cross-section at random.

    Half and half rather than the usual 80/20: the binding constraint is the number
    of control-arm converters in the *test* set, since that is what a Qini interval
    is built from, and at a 0.19% control conversion rate a 20% test set is too thin
    to separate models from each other.

    Parameters
    ----------
    frame : pandas.DataFrame
        From :func:`load_sample`.
    test_share : float, optional
        Share held out.
    seed : int, optional
        Reproducibility.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test)``.
    """
    rng = np.random.default_rng(seed)
    held_out = rng.random(len(frame)) < test_share
    return frame.loc[~held_out].reset_index(drop=True), frame.loc[held_out].reset_index(drop=True)


def bootstrap_qini(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    replicates: int = 100,
    seed: int = 20260829,
) -> dict[str, float]:
    """Qini coefficient with a bootstrap interval.

    A Qini figure without an interval is not interpretable at a 0.19% control
    conversion rate: the whole curve is built from a few thousand events, and two
    models differing in the third decimal are indistinguishable.

    Parameters
    ----------
    outcome, treatment, score : numpy.ndarray
        Binary outcome, randomised arm, and predicted uplift.
    replicates : int, optional
        Bootstrap resamples.
    seed : int, optional
        Reproducibility.

    Returns
    -------
    dict
        ``qini``, ``ci_low``, ``ci_high``, ``beats_random`` — the last being whether
        the interval excludes zero, which is the only claim worth making.
    """
    outcome = np.asarray(outcome)
    treatment = np.asarray(treatment)
    score = np.asarray(score)
    point = float(qini_auc(outcome, treatment, score))

    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        pick = rng.integers(0, len(outcome), len(outcome))
        draws[index] = qini_auc(outcome[pick], treatment[pick], score[pick])

    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "qini": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "beats_random": bool(low > 0),
        "replicates": replicates,
    }


def targeting_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    depths: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> list[dict[str, float]]:
    """Incremental conversions captured by targeting the top slice of a ranking.

    The decision a marketer actually makes. At each depth the treated and control
    responses within the targeted slice are compared, with the control rescaled to
    the treated arm's size, which is the same construction the Qini curve uses.

    ``lift_over_random`` is the honest summary: targeting the top 10% by predicted
    uplift should capture appreciably more than 10% of the incremental conversions,
    or the ranking is not worth acting on.

    Parameters
    ----------
    outcome, treatment, score : numpy.ndarray
        Binary outcome, randomised arm, predicted uplift.
    depths : tuple of float, optional
        Targeting depths to report.

    Returns
    -------
    list of dict
        One entry per depth.
    """
    outcome = np.asarray(outcome)
    treatment = np.asarray(treatment)
    order = np.argsort(-np.asarray(score), kind="stable")
    outcome, treatment = outcome[order], treatment[order]

    total_treated = outcome[treatment == 1].sum()
    total_control = outcome[treatment == 0].sum()
    treated_n, control_n = int((treatment == 1).sum()), int((treatment == 0).sum())
    total_incremental = total_treated - total_control * treated_n / control_n

    rows = []
    for depth in depths:
        cut = max(1, int(round(depth * len(outcome))))
        arm = treatment[:cut]
        response = outcome[:cut]
        treated_here, control_here = int((arm == 1).sum()), int((arm == 0).sum())
        if control_here == 0:
            continue
        incremental = response[arm == 1].sum() - response[arm == 0].sum() * (
            treated_here / control_here
        )
        rows.append(
            {
                "depth": depth,
                "targeted": cut,
                "control_converters_in_slice": int(response[arm == 0].sum()),
                "incremental_conversions": float(incremental),
                "share_of_total_incremental": float(incremental / total_incremental)
                if total_incremental
                else float("nan"),
                "lift_over_random": float(incremental / total_incremental / depth)
                if total_incremental
                else float("nan"),
            }
        )
    return rows


def incremental_by_decile(
    outcome: np.ndarray, treatment: np.ndarray, score: np.ndarray
) -> list[dict[str, float]]:
    """Incremental response within each decile of the ranking.

    The cumulative targeting curve can look convincing while the underlying deciles
    are noise, because cumulation smooths. This is the unsmoothed view, and where a
    ranking that does not work shows it: the deciles should decline, and if they do
    not, the model is ordering people by something other than uplift.

    Parameters
    ----------
    outcome, treatment, score : numpy.ndarray
        Binary outcome, randomised arm, predicted uplift.

    Returns
    -------
    list of dict
        Ten entries, best-predicted decile first.
    """
    outcome = np.asarray(outcome)
    treatment = np.asarray(treatment)
    order = np.argsort(-np.asarray(score), kind="stable")
    outcome, treatment = outcome[order], treatment[order]

    rows = []
    for decile, block in enumerate(np.array_split(np.arange(len(outcome)), 10), start=1):
        arm, response = treatment[block], outcome[block]
        treated_n, control_n = int((arm == 1).sum()), int((arm == 0).sum())
        if treated_n == 0 or control_n == 0:
            continue
        treated_rate = float(response[arm == 1].mean())
        control_rate = float(response[arm == 0].mean())
        rows.append(
            {
                "decile": decile,
                "n": int(len(block)),
                "treated_rate": treated_rate,
                "control_rate": control_rate,
                "uplift": treated_rate - control_rate,
                "control_converters": int(response[arm == 0].sum()),
            }
        )
    return rows
