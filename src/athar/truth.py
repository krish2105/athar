"""The ground truth, and the lock that keeps a model from reading it too early.

The whole value of a recovery study rests on the model not having seen the answer.
That is easy to say and easy to violate: a notebook cell that loads the truth "just
to check the scale" while tuning priors has quietly turned a recovery study into a
curve-fitting exercise, and nothing in the output would show it.

So the truth is not importable on demand. :func:`load_truth` requires the caller to
name the fit artifact it is about to score, and refuses if that file does not yet
exist on disk. Peeking before committing a fit is then not a matter of discipline;
it raises.

The truth itself lives outside the repository, in
``<DATA_ROOT>/processed/athar/truth.json``, alongside the panel it describes. It is
written once by ``scripts/build_panel.py`` and never by a notebook.

What this does not protect against
----------------------------------
Someone can open the JSON in an editor. The lock is not a security boundary and is
not meant to be one — it makes the honest path the path of least resistance, and it
makes the dishonest path something you have to mean to do. That is the same standard
SPINE applies when ``mase()`` takes the training window as a required argument: the
leaking version is still writable, just awkward enough that you cannot reach it by
accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from athar import paths

__all__ = ["TruthAccessError", "load_truth", "truth_path", "write_truth"]


class TruthAccessError(RuntimeError):
    """Raised when the ground truth is read before there is a fit to score."""


def truth_path() -> Path:
    """Locate the stored ground truth.

    Deliberately outside the repository and alongside the panel, not in
    ``metrics/``. Committed artifacts are the ones a report quotes; the truth is
    an input to scoring, and keeping it out of the committed surface means a
    notebook cannot reach it through the ordinary metrics reader.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/processed/athar/truth.json``.
    """
    return paths.processed_dir() / "truth.json"


def write_truth(truth: dict[str, Any]) -> Path:
    """Store the ground truth for a generated panel.

    Called once, by ``scripts/build_panel.py``. Written with sorted keys and no
    timestamp, so regenerating an unchanged panel produces an unchanged file.

    Parameters
    ----------
    truth : dict
        The ``truth`` block of a :class:`athar.dgp.Panel`.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    path = truth_path()
    path.write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")
    return path


def load_truth(after: str | Path) -> dict[str, Any]:
    """Read the ground truth, but only once there is a fit to score against it.

    Parameters
    ----------
    after : str or pathlib.Path
        The fit artifact whose results are about to be scored — a saved posterior,
        a metrics file, whatever the step actually produced. It must already exist.
        Naming it is the point: the caller has to have finished fitting before the
        answer becomes reachable.

    Returns
    -------
    dict
        The stored truth.

    Raises
    ------
    TruthAccessError
        If ``after`` does not exist, or the truth has not been generated yet.

    Examples
    --------
    Asking for the truth before a fit exists is refused. The message is matched
    without an ellipsis so the example asserts under a bare ``doctest`` run as
    well as under pytest, which enables ``ELLIPSIS`` and would otherwise hide a
    drift in the wording:

    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     try:
    ...         load_truth(Path(tmp) / "posterior.nc")
    ...     except TruthAccessError as error:
    ...         print(str(error).split(":")[0])
    refusing to load the ground truth
    """
    after = Path(after)
    if not after.exists():
        raise TruthAccessError(
            f"refusing to load the ground truth: no fit at {after}. "
            f"The truth is readable only once the model has committed its estimate "
            f"to disk, so that a recovery result cannot have been informed by the "
            f"answer it is scored against."
        )
    path = truth_path()
    if not path.exists():
        raise TruthAccessError(f"no ground truth at {path}; run `make panel` first")
    return json.loads(path.read_text())
