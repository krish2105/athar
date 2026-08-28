"""Gate the four dependencies that can plausibly fail on this machine.

pymc-marketing needs PyTensor to compile C, Meridian pulls TensorFlow Probability,
causalml ships Cython extensions, and lifetimes is unmaintained and predates
NumPy 2. Any of the four can install cleanly and then fail on first use, which is
the worst moment to discover it.

So each is imported *and made to do its job on toy data*, and the outcome is
written to `metrics/env_gate.json`. A component that fails is recorded with its
error rather than quietly disappearing from the pipeline: a negative result about
tooling is still a result, and the report needs to be able to say which comparison
did not run and why.

Run: `make gate`
"""

import logging
import platform
import sys
import traceback

import numpy as np

from athar import paths
from athar.provenance import Provenance, write_metric

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("check_env")


def _version(module_name):
    try:
        import importlib.metadata as md

        return md.version(module_name)
    except Exception:  # noqa: BLE001 - a missing version is not a gate failure
        return "unknown"


def gate_pymc_marketing():
    """Compile a PyTensor graph and build the MMM transforms the DGP mirrors."""
    import pytensor
    import pytensor.tensor as pt
    from pymc_marketing.mmm import (
        GeometricAdstock,
        HillSaturation,
        LogisticSaturation,
        WeibullPDFAdstock,
    )

    x = pt.dvector("x")
    compiled = pytensor.function([x], (x**2).sum())
    value = float(compiled(np.array([1.0, 2.0, 3.0])))
    if value != 14.0:
        raise AssertionError(f"PyTensor compiled but computed {value}, expected 14.0")
    built = [
        type(t).__name__
        for t in (
            WeibullPDFAdstock(l_max=8),
            GeometricAdstock(l_max=8),
            HillSaturation(),
            LogisticSaturation(),
        )
    ]
    return {"pytensor_compiles": True, "transforms_built": built}


def gate_meridian():
    """Import the Meridian model and data layers and run one TensorFlow op."""
    import tensorflow as tf
    from meridian.data import input_data  # noqa: F401
    from meridian.model import model, spec  # noqa: F401

    product = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2))).numpy()
    if not np.allclose(product, 2.0):
        raise AssertionError("TensorFlow matmul returned an unexpected result")
    return {
        "tensorflow": _version("tensorflow"),
        "tensorflow_probability": _version("tensorflow-probability"),
        "devices": [d.device_type for d in tf.config.list_physical_devices()],
    }


def gate_causal():
    """Fit a causalml meta-learner and an econml causal forest on toy data."""
    from causalml.inference.meta import BaseTClassifier
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 4_000
    features = rng.normal(size=(n, 5))
    treatment = rng.binomial(1, 0.5, n)
    modifier = features[:, 0] + 0.8 * treatment * features[:, 1]
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-modifier)))

    learner = BaseTClassifier(learner=LogisticRegression(max_iter=200))
    t_effect = learner.fit_predict(X=features, treatment=treatment, y=outcome)

    # n_estimators must be divisible by econml's subforest_size, which defaults
    # to 4. A value that is not divisible raises rather than rounding.
    forest = CausalForestDML(n_estimators=48, random_state=0)
    forest.fit(outcome, treatment, X=features)
    effect = forest.effect(features)

    return {
        "causalml_t_learner_mean_effect": round(float(np.mean(t_effect)), 6),
        "econml_forest_mean_effect": round(float(np.mean(effect)), 6),
        # The true effect is modulated by feature 1. A forest that recovers the
        # ranking correlates with it; one that silently returns noise does not.
        "econml_correlation_with_true_modifier": round(
            float(np.corrcoef(effect, features[:, 1])[0, 1]), 4
        ),
    }


def gate_lifetimes():
    """Fit BG/NBD and Gamma-Gamma on an Olist-shaped toy sample.

    Olist-shaped means almost every customer has frequency zero, which is both the
    hard case for these models and the case this project actually faces.
    """
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    rng = np.random.default_rng(0)
    n = 5_000
    frequency = rng.poisson(0.05, n)
    observed = rng.uniform(30, 600, n)
    # lifetimes requires recency to be exactly zero wherever frequency is zero.
    recency = np.where(frequency > 0, rng.uniform(0, 1, n) * observed, 0.0)

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(frequency, recency, observed)
    repeaters = frequency > 0
    monetary = rng.gamma(3, 40, n)
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(frequency[repeaters], monetary[repeaters])

    return {
        "zero_frequency_share": round(float((frequency == 0).mean()), 4),
        "bgnbd_params": {k: round(float(v), 4) for k, v in bgf.params_.items()},
        "gamma_gamma_fitted_on": int(repeaters.sum()),
    }


GATES = {
    "pymc_marketing": gate_pymc_marketing,
    "meridian": gate_meridian,
    "causal": gate_causal,
    "lifetimes": gate_lifetimes,
}


def main():
    results = {}
    for name, gate in GATES.items():
        log.info("gate: %s", name)
        try:
            results[name] = {"status": "pass", "detail": gate()}
            log.info("  pass")
        except Exception as error:  # noqa: BLE001 - recording the failure is the point
            results[name] = {
                "status": "fail",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc().splitlines()[-6:],
            }
            log.warning("  FAIL — %s: %s", type(error).__name__, error)

    payload = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "versions": {
            name: _version(name)
            for name in (
                "pymc-marketing",
                "pymc",
                "pytensor",
                "google-meridian",
                "tensorflow",
                "causalml",
                "econml",
                "lifetimes",
                "duckdb",
                "numpy",
                "pandas",
                "scipy",
            )
        },
        "gates": results,
    }
    path = write_metric(
        "env_gate",
        payload,
        Provenance(source="environment", synthetic=False, split="n/a"),
        paths.metrics_dir(),
    )
    failed = [name for name, r in results.items() if r["status"] == "fail"]
    log.info("wrote %s", path)
    if failed:
        log.warning("failed gates: %s — recorded, not fatal", ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
