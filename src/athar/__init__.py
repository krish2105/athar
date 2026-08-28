"""ATHAR — triangulated marketing incrementality.

Platform-reported ROAS is not incremental. This package holds the machinery for
saying so with numbers: a media-mix model, a randomised-experiment analysis, a
simulated attribution log, and the reconciliation layer that puts the three
estimates of "incremental revenue per unit of spend" side by side and reports
what trusting the wrong one costs on a fixed budget.

Two findings live here and are never blurred together.

**Real, no simulation.** :mod:`athar.uplift` works on Criteo-UPLIFT v2.1, a
genuine randomised trial over 13,979,592 rows. The gap between conversions among
the exposed — what a platform reports — and the intent-to-treat lift — what
actually happened — is measured on real data.

**Simulated, and labelled everywhere.** :mod:`athar.dgp` generates a channel-spend
panel with a *saved ground-truth ROI*, because no real dataset knows its own true
ROI and "did the model recover it?" is otherwise unanswerable. The panel is a
measuring instrument, not a stand-in for data that was unavailable.

The modules
-----------

:mod:`athar.paths`
    Where the raw tables are, and where this project's artifacts land.

:mod:`athar.provenance`
    The synthetic caveat, enforced rather than remembered. Every artifact
    declares whether it came from simulated data, and writing one that does not
    raises.

:mod:`athar.frame`
    Olist to a weekly and state-week revenue spine.

:mod:`athar.dgp`
    The pre-registered data-generating process: spend paths with deliberate
    collinearity, Weibull adstock, Hill saturation, and the ground truth.

:mod:`athar.truth`
    The ground-truth registry, quarantined so a model cannot see the answer
    while it is being fitted.

:mod:`athar.attribution`
    A simulated touchpoint log, and the attribution estimators that misread it.

:mod:`athar.experiments`
    Geo holdout design over Olist's 27 real states, and the lift estimator.

:mod:`athar.mmm`
    The media-mix model, fitted deliberately misspecified.

:mod:`athar.uplift`
    Criteo: intent-to-treat, the complier effect, and uplift model evaluation.

:mod:`athar.clv`
    Customer lifetime value, and the diagnosis that Olist has almost none.

:mod:`athar.allocate`
    Response curves and the constrained budget optimiser.

:mod:`athar.reconcile`
    The triangulation layer, and the cost of believing the wrong estimator.
"""

__version__ = "0.1.0"
