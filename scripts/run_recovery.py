"""The recovery grid: can a media-mix model find the ROI that is actually there.

Forty full-posterior fits over a factorial design:

    panel length   {85 weeks (the Olist window), 156 weeks}
    collinearity   {low, high}
    specification  {misspecified, matched}
    seeds          5

Both length arms use the *simulated* baseline from `athar.dgp.extended_baseline`,
including the 85-week arm, so that length is the only quantity changing along the
length axis. Only the headline fit in notebook 03 uses the real Olist baseline.

Scoring is pre-registered and coverage leads: does the true ROI fall inside the 89%
highest-density interval, and how often across seeds. A point estimate that lands
close on one draw is an anecdote; coverage is the property the model actually
claims. Median absolute relative error is reported alongside, admissible only
because every configured ROI is bounded well away from zero by construction.

Fits that fail their sampler diagnostics are recorded as failed and excluded from
the coverage rates, with the exclusion counted in the output. A recovery study that
quietly averages in non-converged fits is measuring its own sampler.

Resumable. Each cell is cached under a hash of its configuration, so an interrupted
run costs one fit rather than the batch. Re-running adds nothing.

Run: `make recovery`
"""

import hashlib
import json
import logging
import time
import warnings

import numpy as np
import pandas as pd

from athar import dgp, mmm, paths
from athar.provenance import Provenance, write_metric

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_recovery")
for noisy in ("pymc", "pymc.sampling", "pytensor"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

PANEL_LENGTHS = (85, 156)
COLLINEARITY = ("low", "high")
SPECIFICATIONS = ("misspecified", "matched")
SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)

DRAWS, TUNE, CHAINS = 1000, 1000, 4


def cell_key(weeks, collinearity, specification, seed, digest):
    """A stable identity for one cell, so a resumed run reuses exactly what it ran."""
    payload = json.dumps(
        {
            "weeks": weeks,
            "collinearity": collinearity,
            "specification": specification,
            "seed": seed,
            "digest": digest,
            "draws": DRAWS,
            "tune": TUNE,
            "chains": CHAINS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_cell(config, baseline_source, weeks, collinearity, specification, seed):
    """Generate one panel, fit one model, score it against the truth it never saw."""
    rng = np.random.default_rng(seed)
    baseline, baseline_fit = dgp.extended_baseline(baseline_source, weeks, rng)
    week_index = pd.date_range("2017-01-02", periods=weeks, freq="W-MON")
    panel = dgp.generate_panel(
        config, baseline, collinearity=collinearity, seed=seed, week_index=week_index
    )
    frame = panel.frame()

    model = mmm.build(config.channel_names, specification, max_lag=panel.truth["max_lag"])
    started = time.time()
    idata = mmm.fit(model, frame, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=seed)
    elapsed = time.time() - started

    diagnostics = mmm.sampler_diagnostics(idata)
    average = mmm.posterior_average_roi(model, frame)
    marginal = mmm.posterior_marginal_roi(model, frame)

    truth_average = {n: b["roi_average"] for n, b in panel.truth["channels"].items()}
    truth_marginal = {n: b["roi_marginal"] for n, b in panel.truth["channels"].items()}

    return {
        "weeks": weeks,
        "collinearity": collinearity,
        "specification": specification,
        "seed": seed,
        "seconds": round(elapsed, 1),
        "diagnostics": diagnostics,
        "identification": {
            "condition_number": round(panel.truth["collinearity"]["condition_number"], 4),
            "max_pairwise_correlation": round(
                panel.truth["collinearity"]["max_pairwise_correlation"], 4
            ),
            "max_vif": round(max(panel.truth["collinearity"]["vif"].values()), 4),
            "media_share_of_detrended_variance": round(
                panel.truth["variance"]["media_share_of_detrended_variance"], 4
            ),
        },
        "baseline_fit": baseline_fit,
        "average_roi": mmm.score_recovery(average, truth_average),
        "marginal_roi": mmm.score_recovery(marginal, truth_marginal),
    }


def aggregate(results, key, verdict="passed"):
    """Coverage and error across the converged fits in one slice of the grid."""
    usable = [r for r in results if r["diagnostics"][verdict]]
    if not usable:
        return {
            "fits": len(results),
            "converged": 0,
            "note": "every fit in this cell failed its sampler diagnostics",
        }
    coverage = [r[key]["summary"]["coverage_rate"] for r in usable]
    errors = [r[key]["summary"]["median_absolute_relative_error"] for r in usable]
    widths = [r[key]["summary"]["mean_interval_width"] for r in usable]
    per_channel = {}
    for channel in usable[0][key]["channels"]:
        entries = [r[key]["channels"][channel] for r in usable]
        per_channel[channel] = {
            "coverage_rate": round(float(np.mean([e["covered"] for e in entries])), 4),
            "median_relative_error": round(
                float(np.median([e["relative_error"] for e in entries])), 4
            ),
            "true": entries[0]["true"],
            "median_estimate": round(float(np.median([e["estimated_mean"] for e in entries])), 4),
        }
    return {
        "fits": len(results),
        "converged": len(usable),
        "excluded_for_diagnostics": len(results) - len(usable),
        "coverage_rate": round(float(np.mean(coverage)), 4),
        "median_absolute_relative_error": round(float(np.median(errors)), 4),
        "mean_interval_width": round(float(np.mean(widths)), 4),
        "per_channel": per_channel,
    }


def main():
    config = dgp.load_config()
    weekly_path = paths.processed_dir() / "weekly.parquet"
    if not weekly_path.exists():
        raise SystemExit(f"{weekly_path} is missing; run `make frame` first")
    baseline_source = pd.read_parquet(weekly_path)["revenue"].to_numpy()

    cache = paths.recovery_dir()
    cells = [
        (weeks, collinearity, specification, seed)
        for weeks in PANEL_LENGTHS
        for collinearity in COLLINEARITY
        for specification in SPECIFICATIONS
        for seed in SEEDS
    ]
    log.info("recovery grid: %d cells, %d draws x %d chains each", len(cells), DRAWS, CHAINS)

    results = []
    for index, (weeks, collinearity, specification, seed) in enumerate(cells, 1):
        key = cell_key(weeks, collinearity, specification, seed, config.digest)
        cached = cache / f"{key}.json"
        label = f"{weeks}w/{collinearity}/{specification}/seed {seed}"
        if cached.exists():
            log.info("[%2d/%d] %-46s cached", index, len(cells), label)
            results.append(json.loads(cached.read_text()))
            continue
        log.info("[%2d/%d] %-46s fitting", index, len(cells), label)
        result = run_cell(config, baseline_source, weeks, collinearity, specification, seed)
        cached.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        scored = result["average_roi"]["summary"]
        log.info(
            "        %.0fs | %s | coverage %.2f | median |rel err| %.2f",
            result["seconds"],
            "converged" if result["diagnostics"]["passed"] else "DIAGNOSTICS FAILED",
            scored["coverage_rate"],
            scored["median_absolute_relative_error"],
        )

    slices = {}
    for weeks in PANEL_LENGTHS:
        for collinearity in COLLINEARITY:
            for specification in SPECIFICATIONS:
                subset = [
                    r
                    for r in results
                    if r["weeks"] == weeks
                    and r["collinearity"] == collinearity
                    and r["specification"] == specification
                ]
                slices[f"{weeks}w_{collinearity}_{specification}"] = {
                    "average_roi": aggregate(subset, "average_roi"),
                    "marginal_roi": aggregate(subset, "marginal_roi"),
                    # The same slice scored under the stricter zero-divergence
                    # rule, so a reader can check that the choice of rule did not
                    # manufacture the conclusion.
                    "average_roi_strict_convergence": aggregate(
                        subset, "average_roi", verdict="passed_strict"
                    ),
                }

    converged = [r for r in results if r["diagnostics"]["passed"]]
    payload = {
        "design": {
            "panel_lengths": list(PANEL_LENGTHS),
            "collinearity_levels": list(COLLINEARITY),
            "specifications": {name: mmm.SPECIFICATIONS[name] for name in SPECIFICATIONS},
            "seeds": list(SEEDS),
            "cells": len(cells),
            "draws": DRAWS,
            "tune": TUNE,
            "chains": CHAINS,
            "hdi_prob": mmm.HDI_PROB,
            "baseline": (
                "Both length arms use the simulated extended baseline, including the "
                "85-week arm, so that length is the only quantity changing along the "
                "length axis. Only the headline fit uses the real Olist baseline."
            ),
            "priors": (
                "pymc-marketing defaults, unchanged. A prior centred near the true ROI "
                "would produce excellent recovery and prove nothing."
            ),
        },
        "convergence": {
            "fits": len(results),
            "converged": len(converged),
            "failed": len(results) - len(converged),
            "converged_under_strict_rule": sum(
                1 for r in results if r["diagnostics"]["passed_strict"]
            ),
            "criteria": results[0]["diagnostics"]["criteria"] if results else None,
            "criteria_strict": results[0]["diagnostics"]["criteria_strict"] if results else None,
            "rule_note": (
                "The primary rule uses a divergence rate rather than a count, because a "
                "count tightens as you sample more and is therefore not a rule. Every "
                "slice is also scored under the stricter zero-divergence rule so the "
                "effect of that choice is visible rather than asserted."
            ),
            "excluded_from_coverage": (
                "Fits failing the diagnostics are excluded from every coverage rate and "
                "counted here, rather than averaged in."
            ),
            "total_sampling_seconds": round(sum(r["seconds"] for r in results), 1),
        },
        "slices": slices,
        "fits": results,
    }

    path = write_metric(
        "recovery",
        payload,
        Provenance(
            source="recovery",
            synthetic=True,
            split="grid",
            seed=SEEDS[0],
            dgp_hash=config.digest,
        ),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    log.info("converged %d/%d fits", len(converged), len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
