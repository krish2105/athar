"""Geo holdout tests on Olist's real 27-state footprint: unbiased but expensive.

An experiment is the only unbiased estimator in this project. This measures what it
costs to get one: how much of the market has to go dark, for how long, before the
estimate is tight enough to act on.

The geography is real and brutally concentrated — one state carries 38% of revenue
and thirteen carry under 1% each — so the power problem here is Olist's, not one
this project invented.

Run: `make experiments`
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd

from athar import dgp, experiments, paths, truth
from athar.provenance import Provenance, write_metric

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_experiments")

TEST_WEEKS = 8
TREATED_COUNTS = (3, 5, 8, 13)
SEED = 20260829


def main():
    processed = paths.processed_dir()
    for required in ("weekly.parquet", "state_week.parquet"):
        if not (processed / required).exists():
            raise SystemExit(f"{processed / required} is missing; run `make frame` first")

    config = dgp.load_config()
    weekly = pd.read_parquet(processed / "weekly.parquet")
    state_panel = pd.read_parquet(processed / "state_week.parquet")
    panel = dgp.generate_panel(
        config,
        weekly["revenue"].to_numpy(),
        week_index=pd.DatetimeIndex(weekly["week"]),
    )

    shares = experiments.state_shares(state_panel)
    log.info(
        "geography: %d states, largest %.1f%% of revenue, %d below 1%%",
        len(shares),
        100 * shares.iloc[0],
        int((shares < 0.01).sum()),
    )

    curves = {}
    for channel in config.channel_names:
        log.info("power curve for %s", channel)
        curves[channel] = experiments.power_curve(
            panel,
            state_panel,
            channel,
            treated_counts=TREATED_COUNTS,
            test_weeks=TEST_WEEKS,
            seed=SEED,
        )
        for row in curves[channel]:
            log.info(
                "  %2d states (%.0f%% of revenue): median rel err %+.3f, sd %.3f, detected %.0f%%",
                row["treated_states"],
                100 * row["median_treated_revenue_share"],
                row["median_relative_error"],
                row["relative_error_sd"],
                100 * row["detected"],
            )

    # One representative experiment per channel, at the largest tested design, to
    # report as the estimate a marketer would actually take away.
    rng = np.random.default_rng(SEED)
    states = list(shares.index)
    headline = {}
    for channel in config.channel_names:
        treated = list(rng.choice(states, size=max(TREATED_COUNTS), replace=False))
        result = experiments.run_holdout(
            panel,
            state_panel,
            channel,
            treated,
            first_test_week=len(panel.weeks) - TEST_WEEKS - 1,
            test_weeks=TEST_WEEKS,
            rng=rng,
        )
        spend_in_window = float(
            panel.spend[channel].to_numpy()[-TEST_WEEKS - 1 : -1].sum()
            * result["treated_revenue_share"]
        )
        result["implied_roi"] = (
            result["estimated_removed_revenue"] / spend_in_window
            if spend_in_window
            else float("nan")
        )
        result["spend_withheld"] = spend_in_window
        headline[channel] = result

    # The estimates are committed to disk before the truth becomes readable. A geo
    # holdout fits no model, so there is no posterior to gate on — but the gate is
    # only meaningful if it is satisfied by this step's own output rather than by
    # an artifact that happened to already exist.
    estimates_path = paths.processed_dir() / "experiment_estimates.json"
    estimates_path.write_text(
        json.dumps({"power": curves, "headline": headline}, indent=2, sort_keys=True) + "\n"
    )
    log.info("wrote %s; the ground truth becomes readable only now", estimates_path.name)

    stored = truth.load_truth(after=estimates_path)

    payload = {
        "geography": {
            "states": int(len(shares)),
            "largest_state_share": float(shares.iloc[0]),
            "states_below_one_percent": int((shares < 0.01).sum()),
            "top_five_share": float(shares.head(5).sum()),
            "source": "real Olist customer_state distribution",
        },
        "design": {
            "test_weeks": TEST_WEEKS,
            "treated_state_counts": list(TREATED_COUNTS),
            "replicates_per_point": 60,
            "estimator": "difference in differences, treated states against the rest",
            "detected_criterion": (
                "the estimate has the right sign and lands within half the true effect. "
                "A blunt rule, chosen before the numbers were seen and reported as it is, "
                "rather than a p-value that would imply a testing framework this design "
                "does not have."
            ),
            "assumption": (
                "Media contribution is allocated across states in proportion to each "
                "state's share of real revenue. Olist carries no spend data, so regional "
                "response heterogeneity cannot be known. That makes this experiment look "
                "better behaved than a real one, and the direction of the optimism is "
                "stated so it is not mistaken for realism."
            ),
        },
        "power": curves,
        "headline_experiments": {
            channel: {
                **result,
                "true_roi": stored["channels"][channel]["roi_average"],
            }
            for channel, result in headline.items()
        },
        "cost": {
            "note": (
                "The revenue share switched off is the price of the estimate. A design "
                "large enough to be precise is a design that turned advertising off "
                "across a meaningful share of the business for two months, for one "
                "channel. That is why experiments are the estimator nobody runs often "
                "enough, and why the other two exist at all."
            )
        },
    }
    path = write_metric(
        "experiments",
        payload,
        Provenance(
            source="geo-experiment",
            synthetic=True,
            split="holdout window",
            seed=SEED,
            dgp_hash=stored["config_digest"],
        ),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
