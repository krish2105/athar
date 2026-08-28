"""Triangulation: the divergence signal, and the cost of believing an estimator."""

from __future__ import annotations

import numpy as np
import pytest

from athar import dgp, reconcile


@pytest.fixture(scope="module")
def panel():
    config = dgp.load_config()
    rng = np.random.default_rng(3)
    positions = np.arange(60, dtype=float)
    baseline = 100_000.0 * (1.0 + 0.01 * positions) + rng.normal(0, 6_000, 60)
    return dgp.generate_panel(config, baseline, collinearity="high", seed=9)


def test_agreement_scores_zero_and_disagreement_scores_high():
    assert reconcile.divergence_score({"a": 2.0, "b": 2.0, "c": 2.0}) == pytest.approx(0.0)
    assert reconcile.divergence_score({"a": 1.0, "b": 8.0}) > 1.0


def test_divergence_is_undefined_rather_than_infinite_on_a_single_estimate():
    assert np.isnan(reconcile.divergence_score({"only": 3.0}))
    assert np.isnan(reconcile.divergence_score({"a": 0.0, "b": 0.0}))


def test_a_channel_without_an_experiment_says_so_rather_than_guessing(panel):
    """Most channels never get a holdout.

    Filling one in would hide the method's real limitation, which is that it is affordable one
    channel at a time.
    """
    truth = {c: b["roi_average"] for c, b in panel.truth["channels"].items()}
    attribution = {c: b["roas_attributed"] for c, b in panel.truth["channels"].items()}
    mmm = {c: {"mean": v * 1.1, "hdi_low": v * 0.5, "hdi_high": v * 1.8} for c, v in truth.items()}
    tested = list(truth)[0]
    experiment = {tested: {"estimate": truth[tested], "ci_low": 0.0, "ci_high": 9.0}}

    result = reconcile.compare_estimates(truth, attribution, mmm, experiment)
    assert result["channels"][tested]["experiment"]["estimate"] is not None
    for channel in list(truth)[1:]:
        entry = result["channels"][channel]["experiment"]
        assert entry["estimate"] is None
        assert "not run" in entry["note"]


def test_the_null_channel_shows_the_least_divergence(panel):
    """search_nonbrand is configured so last-click is exactly right.

    If the three methods disagree least anywhere, it should be there.
    """
    truth = {c: b["roi_average"] for c, b in panel.truth["channels"].items()}
    attribution = {c: b["roas_attributed"] for c, b in panel.truth["channels"].items()}
    mmm = {c: {"mean": v, "hdi_low": v * 0.5, "hdi_high": v * 1.5} for c, v in truth.items()}
    result = reconcile.compare_estimates(truth, attribution, mmm)
    assert result["summary"]["least_divergent_channel"] == "search_nonbrand"
    assert result["summary"]["most_divergent_channel"] == "search_brand"


def test_the_best_informed_allocation_is_the_ceiling(panel):
    """Nothing can beat the allocation that knew the true curves.

    Every estimator's allocation is scored on the same true response curves, so
    none can beat the one built from those curves directly.
    """
    budget = float(panel.spend.to_numpy().sum())
    truth = {c: b["roi_average"] for c, b in panel.truth["channels"].items()}
    attribution = {c: b["roas_attributed"] for c, b in panel.truth["channels"].items()}
    result = reconcile.cost_of_believing(
        panel, budget, {"attribution": attribution, "true_average_roi": truth}
    )
    ceiling = result["allocations"]["optimal_under_truth"]["revenue_under_truth"]
    for name, entry in result["allocations"].items():
        assert entry["revenue_under_truth"] <= ceiling + 1e-6, name
        assert entry["shortfall_against_best"] >= -1e-6
    assert result["allocations"]["optimal_under_truth"]["shortfall_share"] == pytest.approx(0.0)


def test_believing_attribution_costs_money(panel):
    """The project's thesis as a test.

    Attribution over-credits brand search and under-credits video, so a budget built on it
    should earn less under the truth than one built on the true average returns.
    """
    budget = float(panel.spend.to_numpy().sum())
    truth = {c: b["roi_average"] for c, b in panel.truth["channels"].items()}
    attribution = {c: b["roas_attributed"] for c, b in panel.truth["channels"].items()}
    result = reconcile.cost_of_believing(
        panel, budget, {"attribution": attribution, "true_average_roi": truth}
    )
    assert (
        result["allocations"]["attribution"]["revenue_under_truth"]
        < result["allocations"]["true_average_roi"]["revenue_under_truth"]
    )


def test_every_allocation_spends_the_budget_under_the_same_governance(panel):
    budget = 1_000_000.0
    truth = {c: b["roi_average"] for c, b in panel.truth["channels"].items()}
    result = reconcile.cost_of_believing(panel, budget, {"truth": truth})
    for name, entry in result["allocations"].items():
        assert sum(entry["spend"].values()) == pytest.approx(budget, rel=1e-4), name
        assert sum(entry["shares"].values()) == pytest.approx(1.0, rel=1e-4)
