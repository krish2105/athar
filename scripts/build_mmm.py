"""The headline media-mix fit, on the panel with the real Olist baseline.

One fit under each specification. The misspecified arm leads, because it is the
situation an analyst is actually in: nobody knows the true functional form, and a
model fitted with the form that generated the data recovers its own assumptions.
The matched arm is the control that says how much of the error was the wrong shape
and how much was a design that cannot identify the parameters at all.

The ground truth is not read until both fits are on disk. `athar.truth.load_truth`
enforces that by refusing until the artifact it is asked to score exists, so a
recovery number here cannot have been informed by the answer.

Run: `make mmm`
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd

from athar import dgp, mmm, paths, truth
from athar.provenance import Provenance, write_metric

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_mmm")
for noisy in ("pymc", "pymc.sampling", "pytensor"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

RESPONSE_MULTIPLIERS = np.round(np.arange(0.0, 2.01, 0.1), 2)


def main():
    processed = paths.processed_dir()
    panel_path = processed / "panel.parquet"
    if not panel_path.exists():
        raise SystemExit(f"{panel_path} is missing; run `make panel` first")

    config = dgp.load_config()
    frame = pd.read_parquet(panel_path)
    channels = config.channel_names
    max_lag = int(config.spec["adstock"]["max_lag"])

    fits = {}
    for specification in ("misspecified", "matched"):
        log.info("fitting %s", specification)
        model = mmm.build(channels, specification, max_lag=max_lag)
        mmm.fit(model, frame, seed=config.spec["seed"])
        diagnostics = mmm.sampler_diagnostics(model.idata)
        log.info(
            "  divergences %d (%.3f%%), max r_hat %.4f, min ess %.0f, %s",
            diagnostics["divergences"],
            100 * diagnostics["divergence_rate"],
            diagnostics["max_r_hat"],
            diagnostics["min_ess_bulk"],
            "converged" if diagnostics["passed"] else "DIAGNOSTICS FAILED",
        )
        average = mmm.posterior_average_roi(model, frame)
        marginal = mmm.posterior_marginal_roi(model, frame)
        curve = mmm.response_totals(model, RESPONSE_MULTIPLIERS)
        fits[specification] = {
            "diagnostics": diagnostics,
            "average_roi": average,
            "marginal_roi": marginal,
            "response_curve": {
                "multipliers": RESPONSE_MULTIPLIERS.tolist(),
                "median_revenue": {
                    channel: [
                        float(np.median(curve.sel(delta=m, channel=channel).to_numpy()))
                        for m in RESPONSE_MULTIPLIERS
                    ]
                    for channel in channels
                },
            },
            "recovered_parameters": model.format_recovered_transformation_parameters(quantile=0.5),
        }

    # The fits exist on disk before the truth is reachable. This is the quarantine.
    fit_artifact = processed / "mmm_fits.json"
    fit_artifact.write_text(json.dumps(fits, indent=2, sort_keys=True, default=float) + "\n")
    log.info("wrote %s; the ground truth becomes readable only now", fit_artifact.name)

    stored = truth.load_truth(after=fit_artifact)
    truth_average = {n: b["roi_average"] for n, b in stored["channels"].items()}
    truth_marginal = {n: b["roi_marginal"] for n, b in stored["channels"].items()}

    scored = {}
    for specification, fit in fits.items():
        scored[specification] = {
            "diagnostics": fit["diagnostics"],
            "average_roi": mmm.score_recovery(fit["average_roi"], truth_average),
            "marginal_roi": mmm.score_recovery(fit["marginal_roi"], truth_marginal),
            "response_curve": fit["response_curve"],
            "recovered_parameters": fit["recovered_parameters"],
        }
        summary = scored[specification]["average_roi"]["summary"]
        log.info(
            "%s: coverage %.2f (%d/%d), median |rel err| %.2f",
            specification,
            summary["coverage_rate"],
            summary["channels_covered"],
            summary["channels_total"],
            summary["median_absolute_relative_error"],
        )

    payload = {
        "design": {
            "weeks": int(len(frame)),
            "channels": channels,
            "baseline": "real Olist weekly revenue",
            "specifications": mmm.SPECIFICATIONS,
            "headline": "misspecified",
            "headline_reason": (
                "It is the situation an analyst is actually in. The matched arm is a "
                "control that separates the cost of the wrong functional form from the "
                "cost of a design that cannot identify the parameters."
            ),
            "priors": "pymc-marketing defaults, unchanged",
            "hdi_prob": mmm.HDI_PROB,
        },
        "fits": scored,
        "truth_access": (
            "Both fits were written to disk before the ground truth was read. "
            "athar.truth.load_truth refuses until the artifact it is asked to score "
            "exists, so no estimate here can have been informed by the answer."
        ),
    }
    path = write_metric(
        "mmm",
        payload,
        Provenance(
            source="mmm",
            synthetic=True,
            split="full panel",
            seed=stored["seed"],
            dgp_hash=stored["config_digest"],
        ),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
