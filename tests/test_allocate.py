"""The allocator, and the corner solution that makes the comparison fair."""

from __future__ import annotations

import numpy as np
import pytest

from athar import allocate, dgp


@pytest.fixture(scope="module")
def panel():
    config = dgp.load_config()
    rng = np.random.default_rng(3)
    positions = np.arange(60, dtype=float)
    baseline = 100_000.0 * (1.0 + 0.01 * positions) + rng.normal(0, 6_000, 60)
    return dgp.generate_panel(config, baseline, collinearity="high", seed=9)


def test_bounds_reject_an_infeasible_governance_rule():
    with pytest.raises(ValueError, match="needs"):
        allocate.bounds_from_shares(["a", "b", "c"], 100.0, floor=0.4, cap=0.9)
    with pytest.raises(ValueError, match="reaches only"):
        allocate.bounds_from_shares(["a", "b", "c"], 100.0, floor=0.05, cap=0.2)


def test_linear_allocation_is_a_corner_solution():
    """Constant returns plus a budget means fill the best channel to its cap.

    This is not a straw man: it is the only arithmetic a scalar ROAS table
    permits, and it is what makes the caps load-bearing.
    """
    roi = {"best": 5.0, "middle": 3.0, "worst": 1.0}
    bounds = allocate.bounds_from_shares(list(roi), 300.0, floor=0.1, cap=0.5)
    result = allocate.allocate_linear(roi, 300.0, bounds)
    assert result["best"] == pytest.approx(150.0)  # its cap
    assert result["middle"] == pytest.approx(120.0)  # the remainder
    assert result["worst"] == pytest.approx(30.0)  # its floor
    assert sum(result.values()) == pytest.approx(300.0)


def test_linear_allocation_respects_every_bound():
    roi = {"a": 4.0, "b": 2.0, "c": 1.0, "d": 0.5}
    bounds = allocate.bounds_from_shares(list(roi), 1000.0)
    result = allocate.allocate_linear(roi, 1000.0, bounds)
    for name, (low, high) in zip(roi, bounds, strict=True):
        assert low - 1e-9 <= result[name] <= high + 1e-9
    assert sum(result.values()) == pytest.approx(1000.0)


def test_concave_allocation_beats_the_linear_one_under_the_truth(panel):
    """The whole point of a media-mix model, stated as a testable claim.

    Knowing the curvature must be worth something: an allocator that can see
    saturation should not do worse than one that assumes constant returns and
    ranks on the same true average ROI.
    """
    budget = float(panel.spend.to_numpy().sum())
    channels = list(panel.spend.columns)
    bounds = allocate.bounds_from_shares(channels, budget)
    curves = allocate.truth_response_functions(panel)

    curved = allocate.allocate_concave(curves, budget, bounds, channels)
    flat = allocate.allocate_linear(
        {c: panel.truth["channels"][c]["roi_average"] for c in channels}, budget, bounds
    )
    curved_revenue = allocate.evaluate_under_truth(curved, panel)["total_revenue"]
    flat_revenue = allocate.evaluate_under_truth(flat, panel)["total_revenue"]
    assert curved_revenue >= flat_revenue


def test_allocations_spend_exactly_the_budget(panel):
    budget = 2_000_000.0
    channels = list(panel.spend.columns)
    bounds = allocate.bounds_from_shares(channels, budget)
    curves = allocate.truth_response_functions(panel)
    curved = allocate.allocate_concave(curves, budget, bounds, channels)
    assert sum(curved.values()) == pytest.approx(budget, rel=1e-4)


def test_response_functions_reproduce_the_panel_at_observed_spend(panel):
    """The response curve agrees with the panel at the observed plan.

    At the observed plan the curve must return the contribution the generator
    actually produced, or every allocation is scored against the wrong surface.
    """
    curves = allocate.truth_response_functions(panel)
    for channel in panel.spend.columns:
        observed = float(panel.spend[channel].sum())
        assert curves[channel](observed) == pytest.approx(
            float(panel.contribution[channel].sum()), rel=1e-10
        )


def test_zero_spend_earns_nothing(panel):
    curves = allocate.truth_response_functions(panel)
    for channel in panel.spend.columns:
        assert curves[channel](0.0) == pytest.approx(0.0)
