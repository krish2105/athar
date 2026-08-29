"""Criteo uplift models: is the treatment effect findable, or only real on average.

The intent-to-treat effect is established in `metrics/criteo.json` on the full
population. This asks the narrower question a marketer faces: given the twelve
anonymised features, can a model rank people by how much advertising moves them, so
that targeting a slice captures more than its share of the incremental conversions?

Four learners plus a random-targeting baseline, evaluated by Qini on a held-out half
with a bootstrap interval. If none beats random, that is reported as the finding.
The features are anonymised and heavily repeated, so a null result here would be a
statement about what those twelve columns support, not about uplift modelling.

Run: `make uplift`
"""

import logging
import warnings

import numpy as np

from athar import paths, uplift
from athar.provenance import Provenance, write_metric

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_uplift")

#: Rows drawn from the 13,979,592 available. The binding constraint on a Qini
#: interval is not the row count but the number of control-arm converters behind
#: it: the control arm is 15% of the sample and converts at 0.19%, so ten million
#: rows leave roughly 1,450 of them in the held-out half. That count is reported
#: with every result so precision claims follow it rather than the row count.
SAMPLE_ROWS = 10_000_000

#: Bootstrap resamples per model. Each one re-sorts five million rows, so this is
#: the cost driver; sixty is enough to place a 95% interval at the precision the
#: converter count supports, and more would be false comfort.
BOOTSTRAP_REPLICATES = 60
SEED = 20260829


def main():
    log.info("sampling %s rows from the Parquet conversion", f"{SAMPLE_ROWS:,}")
    frame = uplift.load_sample(SAMPLE_ROWS, seed=SEED)
    train, test = uplift.split_random(frame, seed=SEED)
    log.info("train %s, test %s", f"{len(train):,}", f"{len(test):,}")

    control_converters = int(((test["treatment"] == 0) & (test["conversion"] == 1)).sum())
    log.info("control-arm converters in the test half: %s", f"{control_converters:,}")

    features_train = train[uplift.FEATURES].to_numpy(dtype=np.float32)
    features_test = test[uplift.FEATURES].to_numpy(dtype=np.float32)
    treatment_train = train["treatment"].to_numpy()
    outcome_train = train["conversion"].to_numpy()
    treatment_test = test["treatment"].to_numpy()
    outcome_test = test["conversion"].to_numpy()

    from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
    from lightgbm import LGBMClassifier, LGBMRegressor

    def learner():
        return LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED
        )

    def effect_learner():
        """The X-learner's second stage regresses an imputed treatment effect.

        It needs a regressor, not a classifier: the target is a continuous
        pseudo-outcome rather than a class. Passing None — which reads as "use the
        default" — raises instead.
        """
        return LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED
        )

    # The propensity is known by design. Criteo randomised at a fixed rate, so the
    # probability of treatment is a constant of the experiment rather than
    # something to be recovered from twelve anonymised features. causalml will
    # estimate it if it is not supplied, and on five million rows that took
    # thirty-four minutes of CPU to reconstruct a number the design already fixes
    # — and reconstructing it adds estimation noise to an otherwise exact quantity.
    propensity = float(treatment_train.mean())
    log.info("supplying the known propensity %.4f rather than estimating it", propensity)
    propensity_train = np.full(len(treatment_train), propensity)
    propensity_test = np.full(len(features_test), propensity)

    scores: dict[str, np.ndarray] = {}

    for name, builder in (
        ("s_learner", lambda: BaseSClassifier(learner=learner())),
        ("t_learner", lambda: BaseTClassifier(learner=learner())),
        (
            "x_learner",
            lambda: BaseXClassifier(outcome_learner=learner(), effect_learner=effect_learner()),
        ),
    ):
        log.info("fitting %s", name)
        try:
            model = builder()
            model.fit(
                X=features_train, treatment=treatment_train, y=outcome_train, p=propensity_train
            )
            # The X-learner re-derives the propensity at predict time too, and
            # raises looking for a model it never fitted because one was supplied.
            # Handing it the same design constant is both correct and what stops it.
            predicted = (
                model.predict(X=features_test, p=propensity_test)
                if name == "x_learner"
                else model.predict(X=features_test)
            )
            scores[name] = np.asarray(predicted).ravel()[: len(features_test)]
        except Exception as error:  # noqa: BLE001 - a failed learner is recorded, not hidden
            log.warning("  %s failed: %s: %s", name, type(error).__name__, error)

    log.info("fitting causal_forest")
    try:
        from econml.dml import CausalForestDML

        # Fitted on a subsample: a causal forest on three million rows exhausts
        # memory on this machine, and the subsample size is reported so the
        # comparison against the meta-learners is not read as like-for-like.
        forest_rows = min(400_000, len(features_train))
        # discrete_treatment is not optional here. The arm is binary, and the
        # default treats it as continuous — which on features this repetitive
        # produced a singular matrix in the first stage rather than a fit.
        forest = CausalForestDML(
            n_estimators=200,
            min_samples_leaf=50,
            discrete_treatment=True,
            random_state=SEED,
        )
        forest.fit(
            outcome_train[:forest_rows],
            treatment_train[:forest_rows],
            X=features_train[:forest_rows],
        )
        scores["causal_forest"] = np.asarray(forest.effect(features_test)).ravel()
    except Exception as error:  # noqa: BLE001
        log.warning("  causal_forest failed: %s: %s", type(error).__name__, error)
        forest_rows = 0

    rng = np.random.default_rng(SEED)
    scores["random_baseline"] = rng.random(len(features_test))

    results = {}
    for name, score in scores.items():
        log.info("scoring %s", name)
        qini = uplift.bootstrap_qini(
            outcome_test, treatment_test, score, replicates=BOOTSTRAP_REPLICATES, seed=SEED
        )
        results[name] = {
            **qini,
            "targeting_curve": uplift.targeting_curve(outcome_test, treatment_test, score),
            "deciles": uplift.incremental_by_decile(outcome_test, treatment_test, score),
        }
        log.info(
            "  qini %.4f [%.4f, %.4f] %s",
            qini["qini"],
            qini["ci_low"],
            qini["ci_high"],
            "beats random" if qini["beats_random"] else "does NOT beat random",
        )

    baseline = results["random_baseline"]["qini"]
    ranked = sorted(
        (n for n in results if n != "random_baseline"),
        key=lambda n: results[n]["qini"],
        reverse=True,
    )
    best = ranked[0] if ranked else None

    payload = {
        "sample": {
            "requested_rows": SAMPLE_ROWS,
            "rows": int(len(frame)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_control_converters": control_converters,
            "seed": SEED,
            "sampling": (
                "Uniform. Case-control sampling would put more converters in the frame "
                "but breaks the arm balance a Qini curve depends on, and correcting for "
                "it needs weights the shared implementation does not take. The headline "
                "causal estimates are computed on the full population in "
                "metrics/criteo.json and use no sample at all."
            ),
            "causal_forest_train_rows": forest_rows,
        },
        "evaluation": {
            "propensity": (
                f"Supplied as the known design constant {propensity:.4f}, not estimated. "
                f"Criteo randomised at a fixed rate, so the probability of treatment is a "
                f"property of the experiment; recovering it from the features would add "
                f"estimation noise to an exact quantity."
            ),
            "metric": "Qini coefficient (spine.metrics.qini_auc)",
            "why": (
                "Qini subtracts the random-targeting line, so it scores only what the "
                "ranking contributed. A model with no uplift signal scores zero however "
                "large the average treatment effect happens to be."
            ),
            "interval": f"95% bootstrap over the test half, {BOOTSTRAP_REPLICATES} replicates",
            "split": (
                "Random. Criteo carries no time column; it is a randomised cross-section "
                "and is split as one. spine.splitting is deliberately unused here and "
                "used wherever this project does have a time axis."
            ),
            "exposure_note": (
                "Every model is trained and scored against `treatment`, the randomised "
                "arm. `exposure` is post-treatment and is never conditioned on."
            ),
        },
        "models": results,
        "verdict": {
            "best_model": best,
            "best_qini": results[best]["qini"] if best else None,
            "random_baseline_qini": baseline,
            "any_model_beats_random": bool(
                any(results[n]["beats_random"] for n in results if n != "random_baseline")
            ),
        },
    }
    path = write_metric(
        "uplift",
        payload,
        Provenance(source="criteo", synthetic=False, split="random test half"),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
