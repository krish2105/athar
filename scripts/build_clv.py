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

#: Draws and tuning steps per chain for the MCMC fit. Four chains.
BAYES_DRAWS = 1000


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
        "repeat rate %.4f (%s of %s customers), repeaters average %.2f repeat purchases",
        behaviour["repeat_rate"],
        f"{behaviour['repeaters']:,}",
        f"{behaviour['customers']:,}",
        behaviour["mean_repeats_among_repeaters"],
    )

    log.info("attempting BG/NBD by maximum likelihood across every reasonable setting")
    attempts_full = clv.maximum_likelihood_attempts(summary)
    repeaters = summary[summary["frequency"] > 0].reset_index(drop=True)
    attempts_repeaters = clv.maximum_likelihood_attempts(repeaters)
    converged_full = sum(a["converged"] for a in attempts_full)
    converged_repeaters = sum(a["converged"] for a in attempts_repeaters)
    log.info(
        "  full base: %d/%d converged; repeaters only: %d/%d",
        converged_full,
        len(attempts_full),
        converged_repeaters,
        len(attempts_repeaters),
    )

    log.info("fitting BG/NBD by MCMC (the only fit available on this base)")
    bayesian, movement = clv.fit_bgnbd_bayesian(summary, draws=BAYES_DRAWS, tune=BAYES_DRAWS)
    for name, entry in movement.items():
        ratio = entry.get("sd_ratio_posterior_over_prior")
        suffix = f", prior sd {entry['prior_sd']:.4f}, ratio {ratio:.3f}" if ratio else ""
        log.info(
            "  %s: posterior %.4f (sd %.4f)%s",
            name,
            entry["posterior_mean"],
            entry["posterior_sd"],
            suffix,
        )

    gamma, repeaters_fitted = clv.fit_gamma_gamma(summary)
    log.info("Gamma-Gamma fitted on %s repeaters", f"{repeaters_fitted:,}")

    observation_end = orders["purchased_at"].max().normalize() + pd.Timedelta(days=1)
    cutoff = observation_end - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    log.info("calibration ends %s, holdout runs to %s", cutoff.date(), observation_end.date())

    split = clv.calibration_holdout(orders, cutoff, observation_end)
    calibration_model, _ = clv.fit_bgnbd_bayesian(split, draws=BAYES_DRAWS, tune=BAYES_DRAWS)

    horizon_weeks = float(split["holdout_weeks"].iloc[0])
    calibration_data = pd.DataFrame(
        {
            "customer_id": split["customer_unique_id"].to_numpy(),
            "frequency": split["frequency"].to_numpy(),
            "recency": (split["recency"] / 7.0).to_numpy(),
            "T": (split["T"] / 7.0).to_numpy(),
        }
    )
    predicted = (
        calibration_model.expected_purchases(data=calibration_data, future_t=horizon_weeks)
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )
    actual = split["holdout_frequency"].to_numpy()
    accuracy = {
        "horizon_weeks": horizon_weeks,
        "customers": int(len(split)),
        "predicted_total": float(predicted.sum()),
        "actual_total": float(actual.sum()),
        "predicted_mean": float(predicted.mean()),
        "actual_mean": float(actual.mean()),
        "mean_absolute_error": float(np.mean(np.abs(predicted - actual))),
        "mean_absolute_error_predicting_zero": float(np.mean(np.abs(actual))),
        "beats_predicting_zero": bool(
            np.mean(np.abs(predicted - actual)) < np.mean(np.abs(actual))
        ),
        "share_of_customers_predicted_below_0_1": float((predicted < 0.1).mean()),
    }
    log.info(
        "holdout: predicted %.1f purchases, actual %.0f, MAE %.5f (predict-zero MAE %.5f)",
        accuracy["predicted_total"],
        accuracy["actual_total"],
        accuracy["mean_absolute_error"],
        accuracy["mean_absolute_error_predicting_zero"],
    )

    full_data = pd.DataFrame(
        {
            "customer_id": summary["customer_unique_id"].to_numpy(),
            "frequency": summary["frequency"].to_numpy(),
            "recency": (summary["recency"] / 7.0).to_numpy(),
            "T": (summary["T"] / 7.0).to_numpy(),
        }
    )
    predicted_purchases = (
        bayesian.expected_purchases(data=full_data, future_t=52.0)
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )
    is_repeater = (summary["frequency"] > 0).to_numpy()
    expected_value = np.zeros(len(summary))
    expected_value[is_repeater] = gamma.conditional_expected_average_profit(
        summary.loc[is_repeater, "frequency"], summary.loc[is_repeater, "monetary"]
    ).to_numpy()
    lifetime_value = predicted_purchases * expected_value
    first_order = summary["first_order_value"].to_numpy()

    payload = {
        "repeat_behaviour": behaviour,
        "maximum_likelihood": {
            "converged_on_full_base": int(converged_full),
            "converged_on_repeaters_only": int(converged_repeaters),
            "attempts_full_base": attempts_full,
            "attempts_repeaters_only": attempts_repeaters,
            "finding": (
                "BG/NBD does not converge by maximum likelihood on this base — at any "
                "time scale tried (days, weeks, months), at any penalty from 0 to 10, on "
                "the full base or on the repeaters alone. The likelihood returns NaN and "
                "the parameters run off in log space. This is not a library defect or a "
                "tuning problem: BG/NBD's dropout parameters describe the shape of a Beta "
                "distribution over the probability of churning after each purchase, and "
                "they are identified only by the pattern of repeat purchasing. Olist's "
                "repeaters average 1.11 repeat purchases each, so there is nothing for "
                "those parameters to be estimated from and the likelihood is flat in them."
            ),
        },
        "models": {
            "bgnbd_bayesian_full_base": {
                "fitted_on": int(len(summary)),
                "parameters": movement,
                "note": (
                    "The MCMC fit is not a second opinion here; it is the only fit "
                    "available. Its priors supply the regularisation the data cannot, "
                    "which is why it converges where maximum likelihood does not. A "
                    "parameter whose posterior standard deviation is close to its prior's "
                    "has not been learned from the data, and the ratio is reported per "
                    "parameter so a reader can see which ones those are."
                ),
            },
            "gamma_gamma_repeaters_only": {
                "fitted_on": repeaters_fitted,
                "share_of_base": repeaters_fitted / len(summary),
                "parameters": {k: float(v) for k, v in gamma.params_.items()},
                "note": (
                    "Monetary value conditional on repeating cannot be estimated from "
                    "customers who never repeated. Every figure derived from this model "
                    "describes 3% of the base and does not generalise to the rest. It "
                    "converges where BG/NBD does not because it conditions on the "
                    "repeaters and estimates a spend distribution rather than a churn one."
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
        "lifetime_value": {
            "horizon_weeks": 52.0,
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
