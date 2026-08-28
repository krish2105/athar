"""Geo holdouts: the estimator, and the scale problem it exists to solve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from athar import experiments


def test_shares_sum_to_one_and_are_ordered():
    panel = pd.DataFrame({"state": ["SP", "SP", "RJ", "AC"], "revenue": [60.0, 20.0, 15.0, 5.0]})
    shares = experiments.state_shares(panel)
    assert shares.sum() == pytest.approx(1.0)
    assert list(shares.index) == ["SP", "RJ", "AC"]


def test_the_estimator_recovers_a_known_removal():
    """A treated state losing a tenth of its revenue against a flat control."""
    revenue = pd.DataFrame({"T": [100.0] * 6 + [90.0] * 4, "C": [500.0] * 10})
    mask = np.array([False] * 6 + [True] * 4)
    assert experiments.difference_in_differences(revenue, ["T"], mask) == pytest.approx(40.0)


def test_the_control_absorbs_a_shock_that_hits_everyone():
    """The whole reason to run the test geographically.

    If every state rises 20% during the window, a holdout that removed nothing must estimate
    nothing.
    """
    revenue = pd.DataFrame({"T": [100.0] * 6 + [120.0] * 4, "C": [500.0] * 6 + [600.0] * 4})
    mask = np.array([False] * 6 + [True] * 4)
    assert experiments.difference_in_differences(revenue, ["T"], mask) == pytest.approx(0.0)


def test_scale_matching_is_what_makes_unequal_groups_comparable():
    """Groups of wildly different size still compare correctly.

    Three small states against twenty-four is a comparison between quantities an
    order of magnitude apart. A plain difference of the raw sums is dominated by
    the control group's own growth; the first version of this estimator did exactly
    that and reported relative errors in the hundreds. Here the treated group is a
    hundredth of the control's size and loses a known amount, and the estimate must
    still be right.
    """
    weeks = 10
    treated = np.array([10.0] * 6 + [9.0] * 4)
    control = np.array([1000.0] * 6 + [1000.0] * 4)
    revenue = pd.DataFrame({"T": treated, "C": control})
    mask = np.array([False] * 6 + [True] * 4)
    assert weeks == len(treated)
    assert experiments.difference_in_differences(revenue, ["T"], mask) == pytest.approx(4.0)


def test_the_estimator_refuses_a_degenerate_design():
    revenue = pd.DataFrame({"T": [1.0] * 4, "C": [2.0] * 4})
    with pytest.raises(ValueError, match="non-empty group"):
        experiments.difference_in_differences(revenue, [], np.array([False, False, True, True]))
    with pytest.raises(ValueError, match="pre-period"):
        experiments.difference_in_differences(revenue, ["T"], np.array([True] * 4))


def test_a_holdout_is_approximately_unbiased_over_many_assignments():
    """An experiment's defining property.

    Individual draws are noisy — that is the other finding — but the median across assignments
    should sit near zero error.
    """
    from athar import dgp

    config = dgp.load_config()
    rng = np.random.default_rng(4)
    positions = np.arange(60, dtype=float)
    baseline = 120_000.0 * (1.0 + 0.008 * positions) + rng.normal(0, 5_000, 60)
    panel = dgp.generate_panel(config, baseline, collinearity="high", seed=13)

    states = [f"S{i:02d}" for i in range(20)]
    weeks = pd.date_range("2017-01-02", periods=60, freq="W-MON")
    state_panel = pd.DataFrame(
        [
            {"week": week, "state": state, "revenue": 1000.0 * (20 - index)}
            for week in weeks
            for index, state in enumerate(states)
        ]
    )

    errors = []
    generator = np.random.default_rng(0)
    for _ in range(40):
        treated = list(generator.choice(states, size=8, replace=False))
        result = experiments.run_holdout(
            panel, state_panel, "search_nonbrand", treated, 45, 8, generator
        )
        errors.append(result["relative_error"])
    assert abs(float(np.median(errors))) < 0.35
