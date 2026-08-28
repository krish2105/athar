"""Convert the Criteo uplift file once, and compute the project's one real causal result.

Criteo-UPLIFT v2.1 is 3.2 GB of CSV holding a genuine randomised trial: 13,979,592
users, 85% assigned to a treated arm that could be shown advertising and 15% held
back. Nothing here is simulated, and this is the only place in ATHAR where a causal
claim rests on data rather than on a construction.

Two numbers, and the distance between them is the thesis:

**What a platform reports.** Conversions among users who were actually shown an ad.
This is not a causal quantity. `exposure` is decided *after* randomisation — who
sees an ad depends on who browsed, and browsing predicts buying — so conditioning
on it compares a self-selected group to everyone else. That is precisely the
comparison ad platforms publish.

**What actually happened.** The intent-to-treat difference between the assigned arms.
Assignment is random, so this is unbiased by construction, and it is what an
advertiser's revenue actually experienced.

Also computed is the complier effect (CACE), recovering the effect on those the
advertising actually reached by using random assignment as an instrument for
exposure. That is the defensible version of "effect on the exposed", and it is
reported alongside the naive version so the size of the gap between them is visible.

Everything runs in DuckDB over the full population — no sampling. The Parquet
written here is what the uplift *models* later read, because 14 million rows of
CSV parsing per notebook run is not a thing anyone should do twice.

Run: `make criteo`
"""

import logging

import duckdb
import numpy as np

from athar import paths
from athar.provenance import Provenance, write_metric

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_criteo")

FEATURES = [f"f{index}" for index in range(12)]


def convert(connection, source, destination):
    """Stream the CSV into Parquet, narrowing the numeric types on the way."""
    columns = ", ".join(f"CAST({name} AS FLOAT) AS {name}" for name in FEATURES)
    connection.execute(f"""
        COPY (
            SELECT {columns},
                   CAST(treatment  AS TINYINT) AS treatment,
                   CAST(conversion AS TINYINT) AS conversion,
                   CAST(visit      AS TINYINT) AS visit,
                   CAST(exposure   AS TINYINT) AS exposure
            FROM read_csv_auto('{source}')
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def proportion_interval(successes, trials, z=1.959963984540054):
    """Wald interval for a proportion. At n in the millions the approximation is exact enough."""
    rate = successes / trials
    half = z * np.sqrt(rate * (1 - rate) / trials)
    return rate, rate - half, rate + half


def main():
    source = paths.criteo_dir() / paths.CRITEO_FILE
    destination = paths.criteo_parquet()
    if not source.exists():
        raise SystemExit(f"{source} is missing; see data/README.md")

    connection = duckdb.connect()
    connection.execute("SET threads TO 1")  # float aggregation order must be stable

    if destination.exists():
        log.info("%s already exists; reusing it", destination.name)
    else:
        log.info("converting %.1f GB of CSV to Parquet (once)", source.stat().st_size / 1e9)
        convert(connection, source, destination)
    log.info("parquet is %.0f MB", destination.stat().st_size / 1e6)

    table = f"read_parquet('{destination}')"

    log.info("computing full-population statistics (no sampling)")
    (
        rows,
        treated,
        control,
        treated_conversions,
        control_conversions,
        treated_visits,
        control_visits,
        treated_exposed,
        control_exposed,
        exposed_conversions,
    ) = connection.execute(f"""
        SELECT count(*),
               sum(treatment), sum(1 - treatment),
               sum(conversion * treatment), sum(conversion * (1 - treatment)),
               sum(visit * treatment), sum(visit * (1 - treatment)),
               sum(exposure * treatment), sum(exposure * (1 - treatment)),
               sum(conversion * exposure)
        FROM {table}
    """).fetchone()

    treated, control = int(treated), int(control)
    treated_conversions, control_conversions = int(treated_conversions), int(control_conversions)
    treated_exposed, control_exposed = int(treated_exposed), int(control_exposed)
    exposed_conversions = int(exposed_conversions)

    treated_rate, treated_lo, treated_hi = proportion_interval(treated_conversions, treated)
    control_rate, control_lo, control_hi = proportion_interval(control_conversions, control)

    # Intent to treat. Assignment is random, so this needs no adjustment.
    itt = treated_rate - control_rate
    itt_se = float(
        np.sqrt(
            treated_rate * (1 - treated_rate) / treated
            + control_rate * (1 - control_rate) / control
        )
    )
    itt_lo, itt_hi = itt - 1.959963984540054 * itt_se, itt + 1.959963984540054 * itt_se

    # Compliance: the share of the treated arm that advertising actually reached,
    # net of any exposure leaking into control.
    compliance = treated_exposed / treated - control_exposed / control
    cace = itt / compliance
    cace_se = itt_se / compliance

    # The naive quantity a platform reports, and the incremental one.
    exposed = treated_exposed + control_exposed
    platform_reported_conversions = exposed_conversions
    incremental_conversions = itt * treated

    log.info("  rows %s, treated %.4f", f"{rows:,}", treated / rows)
    log.info("  ITT %.6f  (%.6f, %.6f)", itt, itt_lo, itt_hi)
    log.info(
        "  platform-reported %s vs incremental %.0f conversions",
        f"{platform_reported_conversions:,}",
        incremental_conversions,
    )

    log.info("measuring near-duplication (nominal N overstates precision if it is heavy)")
    distinct_features = int(
        connection.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT {', '.join(FEATURES)} FROM {table})"
        ).fetchone()[0]
    )
    log.info(
        "  %s distinct feature vectors from %s rows (%.3f)",
        f"{distinct_features:,}",
        f"{rows:,}",
        distinct_features / rows,
    )

    connection.close()

    payload = {
        "population": {
            "rows": int(rows),
            "treated": treated,
            "control": control,
            "treated_share": round(treated / rows, 6),
            "sampling": "none — every statistic below is computed on all rows",
        },
        "randomisation_check": {
            "exposure_in_control_arm": control_exposed,
            "note": (
                "Exposure is a post-treatment variable: who was shown an ad depends on who "
                "browsed, and browsing predicts buying. Conditioning on it breaks the "
                "randomisation, which is why the platform-reported figure below is not a "
                "causal quantity and the intent-to-treat figure is."
            ),
        },
        "conversion": {
            "treated_rate": round(treated_rate, 8),
            "treated_ci": [round(treated_lo, 8), round(treated_hi, 8)],
            "control_rate": round(control_rate, 8),
            "control_ci": [round(control_lo, 8), round(control_hi, 8)],
            "treated_conversions": treated_conversions,
            "control_conversions": control_conversions,
        },
        "visit": {
            "treated_rate": round(treated_visits / treated, 8),
            "control_rate": round(control_visits / control, 8),
        },
        "intent_to_treat": {
            "absolute_lift": round(itt, 10),
            "standard_error": round(itt_se, 10),
            "ci_95": [round(itt_lo, 10), round(itt_hi, 10)],
            "relative_lift": round(treated_rate / control_rate - 1.0, 6),
            "incremental_conversions": round(incremental_conversions, 1),
            "interpretation": (
                "Unbiased by construction: assignment is random, so the difference between "
                "arms is the causal effect of being eligible for advertising."
            ),
        },
        "complier_effect": {
            "compliance_rate": round(compliance, 8),
            "cace": round(cace, 10),
            "standard_error": round(cace_se, 10),
            "ci_95": [
                round(cace - 1.959963984540054 * cace_se, 10),
                round(cace + 1.959963984540054 * cace_se, 10),
            ],
            "interpretation": (
                "Random assignment used as an instrument for exposure. This is the "
                "defensible version of 'the effect on users who saw the ad'; the naive "
                "version below is not."
            ),
        },
        "platform_reported_versus_incremental": {
            "exposed_users": exposed,
            "platform_reported_conversions": platform_reported_conversions,
            "incremental_conversions": round(incremental_conversions, 1),
            "overstatement_ratio": round(
                platform_reported_conversions / incremental_conversions, 4
            ),
            "interpretation": (
                "The numerator counts every conversion by a user who was shown an ad, which "
                "is what an ad platform reports. The denominator counts the conversions that "
                "would not have happened otherwise. Both are computed from the same real "
                "randomised trial."
            ),
            # The same bias stated as a rate rather than a count, which is how a
            # dashboard usually shows it. The two framings differ by an order of
            # magnitude and both are correct arithmetic on the same trial: only
            # 3.6% of the treated arm was ever exposed, so an enormous inflation
            # of the conversion RATE turns into a much smaller inflation of the
            # conversion COUNT. Reporting only the dramatic one would be a
            # rhetorical choice rather than a finding.
            "naive_rate_framing": {
                "conversion_rate_among_exposed": round(platform_reported_conversions / exposed, 8),
                "conversion_rate_in_control": round(control_rate, 8),
                "naive_rate_ratio": round(
                    (platform_reported_conversions / exposed) / control_rate, 4
                ),
                "true_relative_lift_ratio": round(treated_rate / control_rate, 4),
                "interpretation": (
                    "Comparing the exposed against the held-out control makes advertising "
                    "look roughly 28 times more effective than it was. The honest relative "
                    "figure is the intent-to-treat ratio. The gap is selection: users who "
                    "were shown an ad are users who were browsing, and browsing predicts "
                    "buying."
                ),
                "exposed_share_of_treated_arm": round(treated_exposed / treated, 6),
            },
        },
        "effective_sample": {
            "distinct_feature_vectors": distinct_features,
            "distinct_share": round(distinct_features / rows, 6),
            "note": (
                "Criteo's twelve features are anonymised and heavily repeated. Where the "
                "share below is well under one, the nominal row count overstates the "
                "precision available to any model fitted on those features, and precision "
                "claims should follow the effective count rather than the nominal one. "
                "It does not affect the treatment-effect estimates above, which depend on "
                "the assignment rather than on the features."
            ),
        },
    }

    path = write_metric(
        "criteo",
        payload,
        Provenance(source="criteo", synthetic=False, split="full population"),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
