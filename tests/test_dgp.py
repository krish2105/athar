"""The generator, and the four properties that make its truth trustworthy.

A recovery study is worthless if the truth it scores against is wrong, if the
panel is not reproducible, or if the grid's axes move more than one thing at a
time. Each of those is checked here rather than assumed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from athar import dgp

BASELINE_WEEKS = 60


@pytest.fixture(scope="module")
def config():
    return dgp.load_config()


@pytest.fixture(scope="module")
def baseline():
    """A stand-in baseline with a trend, so tests do not need the Olist extract."""
    rng = np.random.default_rng(7)
    positions = np.arange(BASELINE_WEEKS, dtype=float)
    return 100_000.0 * (1.0 + 0.01 * positions) + rng.normal(0, 8_000, BASELINE_WEEKS)


@pytest.fixture(scope="module")
def panel(config, baseline):
    return dgp.generate_panel(config, baseline, collinearity="high", seed=11)


# --- the transforms ---------------------------------------------------------


@pytest.mark.parametrize("slope", [0.8, 1.0, 1.5, 2.5])
def test_hill_is_one_half_at_half_saturation_for_every_slope(slope):
    """What makes `half_saturation` readable as a spend level rather than a scale."""
    assert dgp.hill(np.array([250.0]), 250.0, slope)[0] == pytest.approx(0.5)


def test_hill_is_monotone_and_bounded():
    values = dgp.hill(np.linspace(0, 5_000, 200), 800.0, 1.7)
    assert np.all(np.diff(values) >= 0)
    assert values[0] == 0.0
    assert values.max() < 1.0


def test_adstock_is_linear_which_the_response_curve_depends_on():
    """Adstock is linear in spend.

    response_curve() scales a whole plan by a multiplier without re-convolving,
    and marginal_roi() is analytic rather than a finite difference. Both rest on
    this.
    """
    rng = np.random.default_rng(3)
    spend = rng.gamma(4.0, 900.0, 40)
    weights = dgp.delayed_adstock_weights(0.7, 2, 8)
    assert np.allclose(
        dgp.apply_adstock(3.7 * spend, weights), 3.7 * dgp.apply_adstock(spend, weights)
    )


def test_delayed_adstock_can_peak_away_from_lag_zero():
    """Geometric adstock cannot, which is the deliberate misspecification."""
    assert dgp.delayed_adstock_weights(0.8, 3, 8).argmax() == 3
    assert dgp.delayed_adstock_weights(0.3, 0, 8).argmax() == 0


@pytest.mark.parametrize(("alpha", "theta"), [(0.0, 1.0), (1.0, 1.0), (-0.1, 1.0), (0.5, -1.0)])
def test_delayed_adstock_rejects_impossible_parameters(alpha, theta):
    with pytest.raises(ValueError):
        dgp.delayed_adstock_weights(alpha, theta, 8)


def test_generator_kernel_matches_pymc_marketing_exactly():
    """The matched arm is matched, not merely close.

    The generator implements the delayed-geometric kernel in NumPy so it is
    readable and self-contained. If that implementation drifted from
    pymc-marketing's `DelayedAdstock`, the "matched" arm of the recovery grid would
    silently become another misspecified one, and the decomposition of error into
    misspecification and identification would mean nothing.
    """
    import numpy as np
    import pytensor
    import xarray as xr
    from pymc_marketing.mmm.transformers import delayed_adstock

    max_lag = 8
    for alpha, theta in [(0.3, 0), (0.6, 1), (0.8, 3)]:
        spend = np.zeros(20)
        spend[5] = 1.0  # an impulse: the response IS the kernel
        impulse = xr.DataArray(spend, dims=["date"])
        library = pytensor.function(
            [],
            delayed_adstock(
                impulse, alpha=alpha, theta=theta, l_max=max_lag, normalize=True, dim="date"
            ).values,
        )()
        ours = dgp.apply_adstock(spend, dgp.delayed_adstock_weights(alpha, theta, max_lag))
        assert np.allclose(ours[5 : 5 + max_lag], library[5 : 5 + max_lag], atol=1e-12)


# --- the truth --------------------------------------------------------------


def test_saved_roi_equals_the_roi_implied_by_the_generated_contributions(panel):
    """The core check: the answer stored is the answer the panel actually has.

    Recomputed straight from the generated series, sharing no code path with the
    solve that produced beta.
    """
    for name, block in panel.truth["channels"].items():
        recomputed = float(panel.contribution[name].sum()) / float(panel.spend[name].sum())
        assert recomputed == pytest.approx(block["roi_average"], rel=1e-12)


def test_saved_roi_equals_the_pre_registered_target(panel, config):
    """The stored ROI is the one that was pre-registered.

    Beta is solved so each channel lands exactly on its configured ROI. If this
    drifts, the recovery study is scoring against something nobody registered.
    """
    for channel in config.channels:
        stored = panel.truth["channels"][channel["name"]]["roi_average"]
        assert stored == pytest.approx(channel["true_roi"], rel=1e-12)


def test_analytic_marginal_roi_matches_a_numerical_derivative(panel):
    """The hand-differentiated Hill slope is correct.

    marginal_roi() differentiates the Hill curve by hand. A sign slip or a
    missing chain-rule term would be invisible in the output and fatal to the
    allocator, so it is checked against a central difference.
    """
    step = 1e-6
    for name in panel.spend.columns:
        curve = dgp.response_curve(panel, name, np.array([1.0 - step, 1.0 + step]))
        numerical = (curve[1] - curve[0]) / (2 * step) / float(panel.spend[name].sum())
        assert dgp.marginal_roi(panel, name) == pytest.approx(numerical, rel=1e-6)


def test_incremental_revenue_versus_zero_spend_is_the_whole_contribution(panel):
    """Switching a channel off removes exactly its contribution.

    What a holdout experiment is trying to estimate. Zero spend gives zero
    adstock and so zero response, so the counterfactual is the full contribution.
    """
    for name, block in panel.truth["channels"].items():
        assert dgp.response_curve(panel, name, np.array([0.0]))[0] == pytest.approx(0.0)
        assert block["incremental_revenue_vs_zero_spend"] == pytest.approx(
            block["total_contribution"], rel=1e-12
        )


# --- reproducibility --------------------------------------------------------


def test_same_seed_reproduces_the_panel_exactly(config, baseline):
    first = dgp.generate_panel(config, baseline, collinearity="high", seed=42)
    second = dgp.generate_panel(config, baseline, collinearity="high", seed=42)
    assert np.array_equal(first.spend.to_numpy(), second.spend.to_numpy())
    assert np.array_equal(first.revenue, second.revenue)
    assert first.truth["channels"] == second.truth["channels"]


def test_a_different_seed_gives_a_different_panel(config, baseline):
    first = dgp.generate_panel(config, baseline, collinearity="high", seed=42)
    second = dgp.generate_panel(config, baseline, collinearity="high", seed=43)
    assert not np.array_equal(first.spend.to_numpy(), second.spend.to_numpy())


def test_spend_hits_the_configured_budget_regardless_of_seed(config, baseline):
    """Every cell of the grid spends the same money.

    Otherwise the recovery grid's cells differ in budget as well as in the
    thing being varied, and the comparison across them means nothing.
    """
    for seed in (1, 2, 3):
        panel = dgp.generate_panel(config, baseline, collinearity="high", seed=seed)
        for channel in config.channels:
            realised = float(panel.spend[channel["name"]].mean())
            assert realised == pytest.approx(channel["mean_weekly_spend"], rel=1e-12)


# --- the grid axis is clean -------------------------------------------------


def test_collinearity_level_changes_correlation_but_not_volatility(config, baseline):
    """The axis must vary collinearity and nothing else.

    A knob that raised correlation by raising the factor loadings would also make
    spend more volatile, and the grid could not then say which of the two moved
    the recovery.
    """
    low = dgp.generate_panel(config, baseline, collinearity="low", seed=5)
    high = dgp.generate_panel(config, baseline, collinearity="high", seed=5)

    assert (
        high.truth["collinearity"]["max_pairwise_correlation"]
        > low.truth["collinearity"]["max_pairwise_correlation"] + 0.4
    )
    assert (
        high.truth["collinearity"]["condition_number"]
        > low.truth["collinearity"]["condition_number"]
    )

    # Unflighted channels carry only factor plus idiosyncratic variance, so their
    # log-spend volatility is what the construction holds fixed.
    unflighted = [c["name"] for c in config.channels if c["flighting"]["period_weeks"] == 0]
    for name in unflighted:
        low_sd = float(np.std(np.log(low.spend[name].to_numpy()), ddof=1))
        high_sd = float(np.std(np.log(high.spend[name].to_numpy()), ddof=1))
        assert high_sd == pytest.approx(low_sd, rel=0.20)


# --- attribution ------------------------------------------------------------


def test_the_null_channel_is_attributed_without_bias(panel, config):
    """A harness that only ever shows attribution failing has chosen its answer.

    search_nonbrand is configured with full tracking and no organic capture, so
    last-click recovers its true ROI exactly.
    """
    null = next(
        c["name"]
        for c in config.channels
        if c["attribution"]["organic_capture"] == 0 and c["attribution"]["tracking_rate"] == 1
    )
    assert panel.truth["channels"][null]["attribution_bias_relative"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_attribution_over_and_under_credits_in_the_configured_directions(panel):
    """The two failure directions are both present.

    Brand search over-credited by capturing organic demand; upper-funnel video
    under-credited because most of its effect is never observed by a click.
    """
    assert panel.truth["channels"]["search_brand"]["attribution_bias_relative"] > 1.0
    assert panel.truth["channels"]["video_ctv"]["attribution_bias_relative"] < -0.5


def test_organic_capture_cannot_exceed_the_baseline_it_credits(config, baseline, panel):
    greedy = copy.deepcopy(config.spec)
    for channel in greedy["channels"]:
        channel["attribution"]["organic_capture"] = 0.4
    with pytest.raises(ValueError, match="cannot credit more"):
        dgp.attributed_revenue(
            panel.contribution, baseline, dgp.DgpConfig(spec=greedy, digest="x" * 16)
        )


# --- the model must not see the answer --------------------------------------


def test_the_fitting_frame_carries_no_trace_of_the_truth(panel, config):
    """The model cannot see the answer from the data it is given.

    The frame a model is fitted to holds week, spend and revenue. Contributions,
    baseline and noise are the answer and must not be reachable from it.
    """
    table = panel.frame()
    assert set(table.columns) == {"week", "revenue", *config.channel_names}
    for forbidden in ("baseline", "contribution", "noise", "beta", "roi"):
        assert not any(forbidden in column for column in table.columns)


def test_config_digest_changes_when_the_configuration_changes(tmp_path, config):
    """An edited configuration is a different configuration.

    The digest travels with every artifact, so an edited config cannot be
    mistaken for the one a stored fit was run under.
    """
    import yaml

    path = tmp_path / "dgp.yaml"
    path.write_text(yaml.safe_dump(config.spec))
    first = dgp.load_config(path)

    spec = copy.deepcopy(config.spec)
    spec["channels"][0]["true_roi"] = 99.0
    path.write_text(yaml.safe_dump(spec))
    assert dgp.load_config(path).digest != first.digest


def test_extended_baseline_reproduces_the_shape_it_was_fitted_to(baseline):
    """A simulated baseline of any length keeps the shape it was fitted to.

    The recovery grid needs 156 weeks and Olist supplies 85. Both arms use this
    so that length is the only thing changing along the length axis.
    """
    rng = np.random.default_rng(0)
    simulated, diagnostics = dgp.extended_baseline(baseline, 156, rng)
    assert len(simulated) == 156
    assert np.all(simulated > 0)
    assert 0.0 <= diagnostics["r_squared"] <= 1.0
    # The trend is the dominant feature and must survive extrapolation.
    assert simulated[-20:].mean() > simulated[:20].mean()


def test_extended_baseline_rejects_a_series_it_cannot_log(baseline):
    broken = baseline.copy()
    broken[3] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        dgp.extended_baseline(broken, 100, np.random.default_rng(0))
