"""Lifetime value: the descriptive facts, the split, and the fit that does not work."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from athar import clv


def make_summary(repeat_share=0.03, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    repeats = rng.random(n) < repeat_share
    frequency = np.where(repeats, rng.integers(1, 3, n), 0)
    observed = rng.uniform(60, 500, n)
    recency = np.where(frequency > 0, rng.uniform(0, 1, n) * observed, 0.0)
    return pd.DataFrame(
        {
            "customer_unique_id": [f"c{i}" for i in range(n)],
            "frequency": frequency,
            "recency": recency,
            "T": observed,
            "monetary": np.where(frequency > 0, rng.gamma(3, 40, n), 0.0),
            "first_order_value": rng.gamma(3, 45, n),
            "state": "SP",
        }
    )


def test_repeat_behaviour_counts_what_it_says_it_counts():
    summary = make_summary(repeat_share=0.25, n=400, seed=1)
    facts = clv.summarise_repeat_behaviour(summary)
    assert facts["customers"] == 400
    assert facts["repeaters"] == int((summary["frequency"] > 0).sum())
    assert facts["repeat_rate"] == pytest.approx(facts["repeaters"] / 400)
    assert facts["zero_repeat_share"] == pytest.approx(1 - facts["repeat_rate"])


def test_a_base_with_no_repeaters_does_not_divide_by_zero():
    summary = make_summary(repeat_share=0.0, n=100)
    facts = clv.summarise_repeat_behaviour(summary)
    assert facts["repeaters"] == 0
    assert facts["mean_repeats_among_repeaters"] == 0.0


def test_gamma_gamma_refuses_a_base_with_nothing_to_fit_on():
    summary = make_summary(repeat_share=0.0, n=100)
    with pytest.raises(ValueError, match="no repeat customers"):
        clv.fit_gamma_gamma(summary)


def test_maximum_likelihood_attempts_report_every_setting_tried():
    """Every setting tried is reported, not summarised.

    The reply to "the model did not converge" is always "did you try a bigger
    penalty", and that deserves a table rather than an assurance.
    """
    summary = make_summary(repeat_share=0.02, n=800, seed=2)
    attempts = clv.maximum_likelihood_attempts(
        summary, penalizers=(0.0, 1.0), time_units=(("weeks", 7.0), ("days", 1.0))
    )
    assert len(attempts) == 4
    assert {a["time_unit"] for a in attempts} == {"weeks", "days"}
    assert all("converged" in a for a in attempts)


def test_the_calibration_split_is_by_time_and_excludes_later_arrivals():
    """The split is by time, and later arrivals are excluded.

    A random split would let the model see the future of the very customers it is
    forecasting. Customers acquired after the cutoff are dropped: there is no
    history to predict from, and keeping them would let a model that predicts zero
    look good for the wrong reason.
    """
    orders = pd.DataFrame(
        {
            "customer_unique_id": ["early", "early", "late"],
            "state": ["SP"] * 3,
            "purchased_at": pd.to_datetime(["2020-01-05", "2020-03-05", "2020-03-10"]),
            "revenue": [50.0, 60.0, 70.0],
        }
    )
    cutoff = pd.Timestamp("2020-02-01")
    split = clv.calibration_holdout(orders, cutoff, pd.Timestamp("2020-04-01"))
    assert list(split["customer_unique_id"]) == ["early"]
    assert split.loc[0, "frequency"] == 0  # the second order is in the holdout
    assert split.loc[0, "holdout_frequency"] == 1.0
    assert split.loc[0, "holdout_weeks"] == pytest.approx(60 / 7)


def test_a_cutoff_with_no_history_is_rejected():
    orders = pd.DataFrame(
        {
            "customer_unique_id": ["a"],
            "state": ["SP"],
            "purchased_at": pd.to_datetime(["2020-05-01"]),
            "revenue": [10.0],
        }
    )
    with pytest.raises(ValueError, match="no orders before"):
        clv.calibration_holdout(orders, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))


def test_the_fitter_raises_rather_than_returning_a_broken_model():
    """A failed fit raises rather than returning nonsense.

    On Olist this never converges, and the caller has to be able to record that
    rather than receive a model whose parameters are meaningless.
    """
    summary = make_summary(repeat_share=0.001, n=300, seed=5)
    try:
        model = clv.fit_bgnbd(summary)
    except clv.MaximumLikelihoodError as error:
        assert "did not converge" in str(error)
    else:
        assert bool(np.isfinite(np.asarray(model.params_, dtype=float)).all())
