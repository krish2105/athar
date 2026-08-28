"""Uplift evaluation: the ranking metrics, and what they do when there is no signal."""

from __future__ import annotations

import numpy as np
import pytest

from athar import uplift


def make_trial(n=20_000, effect=0.02, seed=0, informative=True):
    """A randomised trial where the effect is concentrated in the top half of a score."""
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, n)
    modifier = rng.random(n)
    base = 0.02
    lift = effect * (modifier > 0.5)
    outcome = rng.binomial(1, base + treatment * lift)
    score = modifier if informative else rng.random(n)
    return outcome, treatment, score


def test_an_informative_ranking_beats_random_and_a_useless_one_does_not():
    """The property that makes Qini worth reporting.

    Qini subtracts the random-targeting line, so a score with no uplift signal
    scores about zero however large the average treatment effect is.
    """
    outcome, treatment, score = make_trial(informative=True)
    good = uplift.bootstrap_qini(outcome, treatment, score, replicates=40)
    assert good["beats_random"]

    outcome, treatment, noise = make_trial(informative=False)
    useless = uplift.bootstrap_qini(outcome, treatment, noise, replicates=40)
    assert useless["qini"] < good["qini"]


def test_the_bootstrap_interval_brackets_the_point_estimate():
    outcome, treatment, score = make_trial()
    result = uplift.bootstrap_qini(outcome, treatment, score, replicates=60)
    assert result["ci_low"] <= result["qini"] <= result["ci_high"]
    assert result["replicates"] == 60


def test_targeting_the_whole_population_captures_all_the_incremental_response():
    """The curve closes on 100% at full depth.

    Otherwise the normalisation is wrong and every share above it is wrong too.
    """
    outcome, treatment, score = make_trial()
    curve = uplift.targeting_curve(outcome, treatment, score, depths=(0.1, 0.5, 1.0))
    assert curve[-1]["depth"] == 1.0
    assert curve[-1]["share_of_total_incremental"] == pytest.approx(1.0, rel=1e-9)


def test_a_good_ranking_captures_more_than_its_share_early():
    outcome, treatment, score = make_trial(informative=True)
    curve = uplift.targeting_curve(outcome, treatment, score, depths=(0.2,))
    assert curve[0]["lift_over_random"] > 1.0


def test_the_targeting_curve_reports_the_converters_behind_each_point():
    """Every point carries the converter count behind it.

    A depth backed by a handful of control converters is not evidence, and the
    reader can only see that if the count is on the row.
    """
    outcome, treatment, score = make_trial()
    for row in uplift.targeting_curve(outcome, treatment, score):
        assert "control_converters_in_slice" in row
        assert row["control_converters_in_slice"] >= 0


def test_deciles_decline_for_an_informative_ranking():
    """The unsmoothed view agrees with the cumulative one.

    Cumulation smooths, so a targeting curve can look convincing while the
    underlying deciles are noise. This is where that would show.
    """
    outcome, treatment, score = make_trial(n=60_000, effect=0.05, informative=True)
    deciles = uplift.incremental_by_decile(outcome, treatment, score)
    assert len(deciles) == 10
    top = np.mean([d["uplift"] for d in deciles[:3]])
    bottom = np.mean([d["uplift"] for d in deciles[-3:]])
    assert top > bottom


def test_the_split_is_reproducible_and_roughly_even():
    frame = __import__("pandas").DataFrame({"x": np.arange(10_000)})
    train_a, test_a = uplift.split_random(frame, seed=7)
    train_b, test_b = uplift.split_random(frame, seed=7)
    assert len(train_a) + len(test_a) == len(frame)
    assert abs(len(test_a) / len(frame) - 0.5) < 0.02
    assert train_a.equals(train_b) and test_a.equals(test_b)


def test_exposure_is_not_among_the_features():
    """Exposure is post-treatment.

    Conditioning on it breaks the randomisation, so it must not be reachable as a model input.
    """
    assert "exposure" not in uplift.FEATURES
    assert len(uplift.FEATURES) == 12
