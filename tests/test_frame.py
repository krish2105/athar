"""The window rule and the customer summary, which everything real rests on.

The window rule is load-bearing: it decides how much history the MMM sees, and
85 weeks against 156 is one axis of the recovery grid. A rule that quietly moved
would move every downstream number, so it is pinned here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from athar.frame import (
    customer_summary,
    select_window,
    state_week_revenue,
    weekly_revenue,
)


def make_weekly(orders):
    return pd.DataFrame(
        {"week": pd.date_range("2020-01-06", periods=len(orders), freq="W-MON"), "orders": orders}
    )


def test_window_starts_after_the_last_collection_gap():
    """Olist's 2016 history is interrupted by nine weeks of no orders at all.

    The rule must start after the *last* gap, not the first non-empty week, or
    the panel carries a hole and every adstock in the DGP decays across nothing.
    """
    weekly = make_weekly([50, 0, 60, 0, 100, 100, 100, 100])
    start, end = select_window(weekly, trailing=3, floor=0.7)
    assert str(start.date()) == "2020-02-03"  # the week after the second zero
    assert str(end.date()) == "2020-02-24"


def test_window_drops_a_truncated_tail_even_when_it_beats_the_global_median():
    """A truncated final week can still beat the global median.

    Olist grew throughout, so a half-collected final week sits above the median
    of the whole series while being half of its own neighbourhood. Judging
    truncation against recent weeks rather than the whole series is what catches
    it.
    """
    weekly = make_weekly([10, 10, 10, 10, 10, 200, 200, 200, 200, 100])
    assert weekly["orders"].iloc[-1] > weekly["orders"].median()  # the trap
    _, end = select_window(weekly, trailing=4, floor=0.7)
    assert str(end.date()) == "2020-03-02"  # the last full week, not the 100


def test_window_keeps_a_final_week_that_is_merely_quiet():
    weekly = make_weekly([100, 100, 100, 100, 90])
    _, end = select_window(weekly, trailing=4, floor=0.7)
    assert str(end.date()) == "2020-02-03"


def test_window_rejects_a_grid_with_no_survivors():
    with pytest.raises(ValueError, match="no week survives"):
        select_window(make_weekly([0, 0, 0]), trailing=2, floor=0.7)


def test_weekly_grid_exposes_gaps_as_zeros_rather_than_closing_them():
    """A missing row silently shortens the series; a zero is visible to the rule."""
    orders = pd.DataFrame(
        {
            "purchased_at": pd.to_datetime(["2020-01-07", "2020-02-04"]),
            "revenue": [10.0, 20.0],
        }
    )
    weekly = weekly_revenue(orders)
    assert len(weekly) == 5
    assert (weekly["orders"] == 0).sum() == 3


def test_state_panel_is_balanced_so_small_states_are_not_dropped():
    """Small states must survive the panel.

    13 of Olist's 27 states carry under 1% of revenue. An unbalanced panel drops
    them from a difference-in-differences fit, which is exactly where a geo
    holdout's power problem lives.
    """
    orders = pd.DataFrame(
        {
            "purchased_at": pd.to_datetime(["2020-01-07", "2020-01-21", "2020-01-21"]),
            "state": ["SP", "SP", "AC"],
            "revenue": [10.0, 20.0, 1.0],
        }
    )
    panel = state_week_revenue(orders)
    assert len(panel) == 3 * 2  # three weeks, two states, none missing
    assert set(panel["state"]) == {"SP", "AC"}
    assert panel.groupby("state").size().nunique() == 1


def test_customer_summary_gives_a_one_time_buyer_zero_recency():
    """A one-time buyer has zero recency, as the fitter demands.

    `lifetimes` requires recency == 0 wherever frequency == 0, and 96.9% of this
    base is one-and-done. A summary that violated it would fail at fit time on
    almost every row.
    """
    orders = pd.DataFrame(
        {
            "customer_unique_id": ["solo", "repeat", "repeat"],
            "state": ["SP", "RJ", "RJ"],
            "purchased_at": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
            "revenue": [50.0, 10.0, 30.0],
        }
    )
    summary = customer_summary(orders, pd.Timestamp("2020-03-01")).set_index("customer_unique_id")
    assert summary.loc["solo", "frequency"] == 0
    assert summary.loc["solo", "recency"] == 0.0
    zero_frequency = summary["frequency"] == 0
    assert (summary.loc[zero_frequency, "recency"] == 0).all()


def test_monetary_excludes_the_first_order():
    """Monetary value is the mean of repeat orders only.

    Gamma-Gamma is fitted on repeat-order value. Including the first order would
    let one-time buyers contribute a monetary observation they do not have, and
    would bias the repeaters' mean toward their acquisition basket.
    """
    orders = pd.DataFrame(
        {
            "customer_unique_id": ["a", "a", "a"],
            "state": ["SP"] * 3,
            "purchased_at": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "revenue": [100.0, 10.0, 20.0],
        }
    )
    summary = customer_summary(orders, pd.Timestamp("2020-04-01"))
    assert summary.loc[0, "frequency"] == 2
    assert summary.loc[0, "monetary"] == pytest.approx(15.0)  # mean(10, 20), not mean(100, 10, 20)
    assert summary.loc[0, "first_order_value"] == pytest.approx(100.0)


def test_observation_window_is_shared_by_everyone():
    """Everyone is observed to the same horizon.

    T is measured to the end of the window, not to each customer's last order, or
    every customer would look like they had just churned.
    """
    orders = pd.DataFrame(
        {
            "customer_unique_id": ["a", "b"],
            "state": ["SP", "RJ"],
            "purchased_at": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "revenue": [10.0, 20.0],
        }
    )
    summary = customer_summary(orders, pd.Timestamp("2020-03-01")).set_index("customer_unique_id")
    assert summary.loc["a", "T"] == 60.0
    assert summary.loc["b", "T"] == 29.0
    assert np.all(summary["T"] > 0)
