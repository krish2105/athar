"""Customer lifetime value on Olist, and the negative result that follows.

Runs the standard stack — BG/NBD on the whole base, Gamma-Gamma on the repeaters —
validates it on a time-based calibration and holdout split, and cross-checks the
maximum-likelihood fit against an independent Bayesian one.

The expected outcome is that predicted repeat purchases are near zero and the
holdout agrees. That is the model working, not failing, and the conclusion is that
Olist is a one-shot acquisition business whose lifetime value is approximately the
margin on a first order. `scripts/build_triangulation.py` then carries the
consequence into the budget decision.

Run: `make clv`
"""

import logging
import warnings

import numpy as np
import pandas as pd

from athar import clv, paths
from athar.provenance import Provenance, write_metric

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_clv")
for noisy in ("pymc", "pymc.sampling", "pytensor"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

#: The holdout is the last 16 weeks of the window. Long enough that a genuine
#: repeat purchaser would plausibly return within it, short enough to leave 69
#: weeks of history to predict from. Fixed before any fit was run.
HOLDOUT_WEEKS = 16


def main():
    processed = paths.processed_dir()
    orders_path = processed / "orders.parquet"
    customers_path = processed / "customers.parquet"
    if not orders_path.exists():
        raise SystemExit(f"{orders_path} is missing; run `make frame` first")

    orders = pd.read_parquet(orders_path)
    orders["purchased_at"] = pd.to_datetime(orders["purchased_at"])
    summary = pd.read_parquet(customers_path)

    behaviour = clv.summarise_repeat_behaviour(summary)
    log.info(
        "repeat rate %.4f (%s of %s customers)",
        behaviour["repeat_rate"],
        f"{behaviour['repeaters']:,}",
        f"{behaviour['customers']:,}",
    )

    log.info("fitting BG/NBD on the full base")
    bgnbd = clv.fit_bgnbd(summary)
    gamma, repeaters_fitted = clv.fit_gamma_gamma(summary)
    log.info("Gamma-Gamma fitted on %s repeaters", f"{repeaters_fitted:,}")

    observation_end = orders["purchased_at"].max().normalize() + pd.Timedelta(days=1)
    cutoff = observation_end - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    log.info("calibration ends %s, holdout runs to %s", cutoff.date(), observation_end.date())

    split = clv.calibration_holdout(orders, cutoff, observation_end)
    calibration_model = clv.fit_bgnbd(split)
    accuracy = clv.holdout_accuracy(calibration_model, split)
    log.info(
        "holdout: predicted %.1f purchases, actual %.0f, MAE %.5f (predict-zero MAE %.5f)",
        accuracy["predicted_total"],
        accuracy["actual_total"],
        accuracy["mean_absolute_error"],
        accuracy["mean_absolute_error_predicting_zero"],
    )

    log.info("cross-checking against an independent Bayesian fit")
    comparison = clv.compare_implementations(summary, draws=1000)
    log.info(
        "worst relative disagreement between implementations: %.4f",
        comparison["worst_relative_disagreement"],
    )

    # Expected lifetime value over a one-year horizon, for the customers where the
    # monetary model can be evaluated at all.
    horizon_days = 365.0
    predicted_purchases = bgnbd.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, summary["frequency"], summary["recency"], summary["T"]
    ).to_numpy()
    repeaters = summary["frequency"] > 0
    expected_value = np.zeros(len(summary))
    expected_value[repeaters.to_numpy()] = gamma.conditional_expected_average_profit(
        summary.loc[repeaters, "frequency"], summary.loc[repeaters, "monetary"]
    ).to_numpy()

    lifetime_value = predicted_purchases * expected_value
    first_order = summary["first_order_value"].to_numpy()

    payload = {
        "repeat_behaviour": behaviour,
        "models": {
            "bgnbd_full_base": {
                "fitted_on": int(len(summary)),
                "parameters": {k: float(v) for k, v in bgnbd.params_.items()},
                "note": (
                    "Fitted on everyone, including the 96.97% with no repeat purchase. "
                    "A long observation window with no second order is evidence about "
                    "the churn process, not a missing value, and dropping those "
                    "customers is the most common way a CLV analysis flatters itself."
                ),
            },
            "gamma_gamma_repeaters_only": {
                "fitted_on": repeaters_fitted,
                "share_of_base": repeaters_fitted / len(summary),
                "parameters": {k: float(v) for k, v in gamma.params_.items()},
                "note": (
                    "Monetary value conditional on repeating cannot be estimated from "
                    "customers who never repeated. Every figure derived from this model "
                    "describes 3% of the base and does not generalise to the rest."
                ),
            },
        },
        "validation": {
            "holdout_weeks": HOLDOUT_WEEKS,
            "calibration_end": str(cutoff.date()),
            "observation_end": str(observation_end.date()),
            "customers_with_history": int(len(split)),
            **accuracy,
            "metric_note": (
                "Mean absolute error, not MAPE: the actual holdout count is zero for "
                "almost every customer and a percentage error against zero is undefined. "
                "No MAPE is computed anywhere in this project."
            ),
        },
        "cross_implementation_check": comparison,
        "lifetime_value": {
            "horizon_days": horizon_days,
            "mean_predicted_purchases": float(predicted_purchases.mean()),
            "median_predicted_purchases": float(np.median(predicted_purchases)),
            "share_predicted_below_0_1_purchases": float((predicted_purchases < 0.1).mean()),
            "mean_expected_clv_brl": float(lifetime_value.mean()),
            "mean_first_order_value_brl": float(first_order.mean()),
            "clv_over_first_order_value": float(lifetime_value.mean() / first_order.mean()),
            "correlation_clv_with_first_order_value": float(
                np.corrcoef(lifetime_value, first_order)[0, 1]
            ),
        },
        "finding": (
            "Olist is a one-shot acquisition business. With a 3% repeat rate, expected "
            "lifetime value is a small multiple of first-order value and is very nearly "
            "proportional to it. The consequence is carried into the budget decision in "
            "metrics/triangulation.json: when lifetime value is proportional to immediate "
            "value, weighting a media allocation by one or the other gives the same "
            "answer, and a story the brief invites cannot honestly be told on this data."
        ),
    }
    path = write_metric(
        "clv",
        payload,
        Provenance(source="olist", synthetic=False, split="calibration/holdout"),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
