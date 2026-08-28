"""Generate the headline synthetic panel and store the truth it will be scored against.

Writes three things:

`<DATA_ROOT>/processed/athar/panel.parquet`
    What a model is allowed to see: week, spend per channel, revenue. Nothing else.

`<DATA_ROOT>/processed/athar/truth.json`
    What the model will be scored against, behind the lock in `athar.truth`.
    Outside the repository, and not readable until a fit exists on disk.

`metrics/panel.json`
    The committed description of the design — how collinear the spend is, how much
    of the revenue variance media accounts for, and what last-click would report
    against what is true. Carries `synthetic: true`, which every artifact
    downstream of it will inherit.

The attribution comparison is published here rather than after the MMM runs
because it involves no fitting at all: it is arithmetic on the generator's own
output, and holding it back until later would imply it depended on a model.

Run: `make panel`
"""

import logging

import numpy as np
import pandas as pd

from athar import dgp, paths, truth
from athar.provenance import Provenance, write_metric

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_panel")


def main():
    processed = paths.processed_dir()
    weekly_path = processed / "weekly.parquet"
    if not weekly_path.exists():
        raise SystemExit(f"{weekly_path} is missing; run `make frame` first")

    config = dgp.load_config()
    weekly = pd.read_parquet(weekly_path)
    baseline = weekly["revenue"].to_numpy()
    week_index = pd.DatetimeIndex(weekly["week"])

    mode = config.spec["panel"]["baseline_mode"]
    if mode != "olist":
        raise SystemExit(
            f"the headline panel must use the real Olist baseline; config says {mode!r}. "
            f"The extended baseline exists for the recovery grid, where both length "
            f"arms must share a construction."
        )

    log.info("config digest %s, seed %d", config.digest, config.spec["seed"])
    log.info(
        "baseline: %d real Olist weeks, %s to %s",
        len(baseline),
        week_index[0].date(),
        week_index[-1].date(),
    )

    panel = dgp.generate_panel(config, baseline, week_index=week_index)
    stored = panel.truth

    frame = panel.frame()
    frame.to_parquet(processed / "panel.parquet", index=False)
    log.info("wrote panel.parquet (%d weeks, %d channels)", len(frame), len(panel.spend.columns))

    truth_file = truth.write_truth(stored)
    log.info("wrote %s (quarantined)", truth_file.name)

    # --- verification ------------------------------------------------------
    # Recomputed from the generated series, sharing no code path with the solve
    # that produced beta. A drifted truth would make every recovery number wrong
    # in a way nothing downstream could detect.
    worst = 0.0
    for name, block in stored["channels"].items():
        recomputed = float(panel.contribution[name].sum()) / float(panel.spend[name].sum())
        worst = max(worst, abs(recomputed / block["roi_average"] - 1.0))
    log.info("truth reconciliation: worst relative difference = %.3e", worst)
    if worst > 1e-12:
        raise SystemExit(f"stored ROI disagrees with the generated panel by {worst:.3e}")

    # The fitting frame must not leak the answer.
    leaked = set(frame.columns) - {"week", "revenue", *config.channel_names}
    if leaked:
        raise SystemExit(f"the fitting frame exposes {sorted(leaked)}, which is the answer")

    channels = []
    for name, block in stored["channels"].items():
        channels.append(
            {
                "channel": name,
                "total_spend": round(block["total_spend"], 2),
                "mean_weekly_spend": round(block["mean_weekly_spend"], 2),
                "true_roi_average": round(block["roi_average"], 6),
                "true_roi_marginal": round(block["roi_marginal"], 6),
                "marginal_over_average": round(block["roi_marginal"] / block["roi_average"], 6),
                "response_regime": (
                    "convex — under-invested, the next BRL buys more than the average"
                    if block["roi_marginal"] > block["roi_average"]
                    else "concave — saturating, the next BRL buys less than the average"
                ),
                "true_incremental_revenue": round(block["incremental_revenue_vs_zero_spend"], 2),
                "lastclick_roas": round(block["roas_attributed"], 6),
                "lastclick_bias_absolute": round(block["attribution_bias_absolute"], 6),
                "lastclick_bias_relative": round(block["attribution_bias_relative"], 6),
                "tracking_rate": block["tracking_rate"],
                "organic_capture": block["organic_capture"],
                "weibull_shape": block["weibull_shape"],
                "weibull_scale": block["weibull_scale"],
                "adstock_peak_lag_weeks": int(np.argmax(block["adstock_weights"])),
                "hill_slope": block["hill_slope"],
                "half_saturation": round(block["half_saturation"], 2),
            }
        )

    collinearity = stored["collinearity"]
    variance = stored["variance"]
    payload = {
        "design": {
            "weeks": stored["weeks"],
            "baseline_mode": mode,
            "baseline_is_real": True,
            "window_start": str(week_index[0].date()),
            "window_end": str(week_index[-1].date()),
            "channels": config.channel_names,
            "collinearity_level": stored["collinearity_level"],
            "collinearity_kappa": stored["collinearity_kappa"],
            "adstock_max_lag": stored["max_lag"],
            "total_spend": round(stored["totals"]["spend"], 2),
            "total_media_contribution": round(stored["totals"]["media_contribution"], 2),
            "total_baseline": round(stored["totals"]["baseline"], 2),
            "total_revenue": round(stored["totals"]["revenue"], 2),
            "blended_true_roi": round(stored["totals"]["blended_roi"], 6),
            "spend_as_share_of_revenue": round(
                stored["totals"]["spend"] / stored["totals"]["revenue"], 6
            ),
            "media_as_share_of_revenue": round(
                stored["totals"]["media_contribution"] / stored["totals"]["revenue"], 6
            ),
        },
        "generating_specification": {
            "adstock": "Weibull PDF, which admits a delayed peak",
            "saturation": "Hill",
            "fitted_with": (
                "Geometric adstock and logistic saturation, neither of which can express "
                "the generating form. The mismatch is deliberate: a model fitted to data "
                "its own functional form produced recovers its own assumptions and "
                "measures nothing. The matched-specification arm of the recovery grid "
                "exists to separate misspecification error from identification error."
            ),
        },
        "identification": {
            "max_pairwise_correlation": round(collinearity["max_pairwise_correlation"], 6),
            "condition_number": round(collinearity["condition_number"], 6),
            "max_vif": round(max(collinearity["vif"].values()), 6),
            "vif": {k: round(v, 6) for k, v in collinearity["vif"].items()},
            "correlation": {
                row: {col: round(value, 6) for col, value in columns.items()}
                for row, columns in collinearity["correlation"].items()
            },
            "media_share_of_revenue_variance": round(
                variance["media_share_of_revenue_variance"], 6
            ),
            "media_share_of_detrended_variance": round(
                variance["media_share_of_detrended_variance"], 6
            ),
            "note": (
                "The detrended share is the one that bounds identification. The raw share "
                "is dominated by Olist's fivefold growth across the window, which any "
                "media-mix model absorbs into its own trend and seasonality terms rather "
                "than having to explain with media."
            ),
        },
        "channels": channels,
        "attribution_summary": {
            "mechanism": (
                "Last-click here is a parametric caricature, not a simulated journey: "
                "tracking_rate is the share of a channel's true contribution it observes, "
                "and organic_capture is the share of baseline revenue it credits to that "
                "channel. No claim is made about attribution mechanics, only about the "
                "consequences of a stated bias."
            ),
            "null_case": (
                "search_nonbrand carries tracking_rate 1.0 and organic_capture 0.0, so "
                "last-click recovers its true ROI exactly. A harness that only ever showed "
                "attribution failing would have had its answer chosen for it."
            ),
            "most_overstated": max(channels, key=lambda c: c["lastclick_bias_relative"])["channel"],
            "most_understated": min(channels, key=lambda c: c["lastclick_bias_relative"])[
                "channel"
            ],
        },
        "verification": {
            "truth_reconciliation_worst_relative_difference": worst,
            "fitting_frame_columns": sorted(frame.columns),
            "fitting_frame_leaks_truth": False,
        },
    }

    path = write_metric(
        "panel",
        payload,
        Provenance(
            source="panel",
            synthetic=True,
            split="full",
            seed=stored["seed"],
            dgp_hash=stored["config_digest"],
        ),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
