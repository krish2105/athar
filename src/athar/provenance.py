"""The synthetic caveat, enforced rather than remembered.

ATHAR mixes a real randomised experiment, a real transaction history, and a
simulated media panel. The project is only honest if a reader can always tell
which of the three produced a given number, and a rule that depends on the author
remembering to write "synthetic" under every chart will fail somewhere.

So the caveat is a property of the artifact, not of the prose. Every number this
project reports is written through :func:`write_metric`, which refuses to write an
artifact whose declared source and ``synthetic`` flag disagree, and refuses to
write a synthetic artifact that cannot name the configuration and seed that
produced it. The dashboard and the markdown reports derive their caveat text from
that block, so a chart cannot be shipped without one.

This is the same move SPINE makes when ``cards.render_card()`` refuses to emit a
metric without the split it was computed on: the thing that is easy to forget is
made impossible to omit.

Sources
-------
Three sources are real and carry no caveat:

``olist``
    Olist Brazilian E-Commerce. Real transactions, 2016-2018.
``criteo``
    Criteo-UPLIFT v2.1. A real randomised trial.
``environment``
    Facts about the machine and the installed toolchain.

The rest are simulated, and every artifact derived from them is caveated:

``panel``
    The channel-spend panel and the revenue series built on it. Note that this is
    caveated *even though its baseline is real Olist revenue*: once a simulated
    media contribution is added, the series as a whole is no longer a measurement
    of anything that happened.
``attribution``, ``geo-experiment``, ``mmm``, ``recovery``, ``allocation``,
``triangulation``
    Everything downstream of the panel.

There is deliberately no ``mixed`` source. An artifact that would need one is an
artifact that has put a real finding and a simulated finding in the same table,
which is the specific failure this module exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "REAL_SOURCES",
    "SYNTHETIC_CAVEAT",
    "SYNTHETIC_SOURCES",
    "Provenance",
    "ProvenanceError",
    "caveat_for",
    "markdown_caveat",
    "read_metric",
    "write_metric",
]

#: Sources that are measurements of something that happened.
REAL_SOURCES: frozenset[str] = frozenset({"olist", "criteo", "environment"})

#: Sources that are, or descend from, the simulated media panel.
SYNTHETIC_SOURCES: frozenset[str] = frozenset(
    {"panel", "attribution", "geo-experiment", "mmm", "recovery", "allocation", "triangulation"}
)

#: The sentence that accompanies every synthetic artifact. Written once here so
#: that it cannot drift between a chart, a report and a notebook.
SYNTHETIC_CAVEAT = (
    "SYNTHETIC — the channel-spend panel and every media effect in it are simulated "
    "from a pre-registered data-generating process, not measured. The revenue "
    "baseline is real Olist history; the media contribution layered on it is not. "
    "No figure here describes the effectiveness of any real marketing channel, and "
    "none may be quoted as one. What is measured is whether a method recovers a "
    "truth that is known only because it was constructed."
)


class ProvenanceError(ValueError):
    """Raised when an artifact's declared provenance is inconsistent or incomplete."""


@dataclass(frozen=True)
class Provenance:
    """Where a reported number came from.

    Parameters
    ----------
    source : str
        One of :data:`REAL_SOURCES` or :data:`SYNTHETIC_SOURCES`.
    synthetic : bool
        Whether the number descends from the simulated panel. Must agree with
        ``source``; the redundancy is the point, because it makes a mislabelled
        artifact a failure rather than a typo nobody notices.
    split : str, optional
        The split the number was computed on. Every metric should name one.
    seed : int, optional
        Required for synthetic artifacts.
    dgp_hash : str, optional
        Hash of ``config/dgp.yaml``. Required for synthetic artifacts, so a result
        can always name the configuration that produced it.

    Raises
    ------
    ProvenanceError
        If the source is unknown, if ``synthetic`` contradicts the source, or if a
        synthetic artifact omits its seed or configuration hash.

    Examples
    --------
    >>> Provenance(source="criteo", synthetic=False, split="full").source
    'criteo'

    A real source cannot be flagged synthetic, or the reverse:

    >>> Provenance(source="criteo", synthetic=True)
    Traceback (most recent call last):
        ...
    athar.provenance.ProvenanceError: source 'criteo' is real but synthetic=True was declared

    A synthetic artifact must name the configuration that produced it:

    >>> Provenance(source="mmm", synthetic=True, seed=1)
    Traceback (most recent call last):
        ...
    athar.provenance.ProvenanceError: synthetic artifact from 'mmm' must declare dgp_hash
    """

    source: str
    synthetic: bool
    split: str | None = None
    seed: int | None = None
    dgp_hash: str | None = None

    def __post_init__(self) -> None:
        """Reject a provenance block that cannot be true.

        Raises
        ------
        ProvenanceError
            If the source is unknown, contradicts ``synthetic``, or a synthetic
            artifact is missing its seed or configuration hash.
        """
        if self.source in REAL_SOURCES:
            if self.synthetic:
                raise ProvenanceError(
                    f"source {self.source!r} is real but synthetic=True was declared"
                )
        elif self.source in SYNTHETIC_SOURCES:
            if not self.synthetic:
                raise ProvenanceError(
                    f"source {self.source!r} descends from the simulated panel but "
                    f"synthetic=False was declared"
                )
            if self.dgp_hash is None:
                raise ProvenanceError(
                    f"synthetic artifact from {self.source!r} must declare dgp_hash"
                )
            if self.seed is None:
                raise ProvenanceError(f"synthetic artifact from {self.source!r} must declare seed")
        else:
            known = sorted(REAL_SOURCES | SYNTHETIC_SOURCES)
            raise ProvenanceError(f"unknown source {self.source!r}; known sources are {known}")

    def to_dict(self) -> dict[str, Any]:
        """Render the block as it is stored in an artifact.

        Returns
        -------
        dict
            The provenance fields, with the caveat text attached when synthetic.
            Keys whose value is ``None`` are omitted, so an artifact does not
            carry empty fields that look like unanswered questions.

        Examples
        --------
        >>> block = Provenance(source="olist", synthetic=False, split="full").to_dict()
        >>> sorted(block)
        ['source', 'split', 'synthetic']
        >>> "caveat" in Provenance(
        ...     source="panel", synthetic=True, seed=1, dgp_hash="abc"
        ... ).to_dict()
        True
        """
        block: dict[str, Any] = {"source": self.source, "synthetic": self.synthetic}
        for name in ("split", "seed", "dgp_hash"):
            value = getattr(self, name)
            if value is not None:
                block[name] = value
        if self.synthetic:
            block["caveat"] = SYNTHETIC_CAVEAT
        return block


def caveat_for(block: Any) -> str | None:
    """Read the caveat out of an artifact's provenance block.

    Parameters
    ----------
    block : dict
        A loaded artifact, or its ``provenance`` sub-block.

    Returns
    -------
    str or None
        The caveat text when the artifact is synthetic, otherwise ``None``.

    Raises
    ------
    ProvenanceError
        If the artifact carries no provenance block at all.

    Examples
    --------
    >>> caveat_for({"provenance": {"source": "criteo", "synthetic": False}}) is None
    True
    >>> caveat_for({"source": "panel", "synthetic": True})[:9]
    'SYNTHETIC'
    """
    inner = block.get("provenance", block) if isinstance(block, dict) else None
    if not isinstance(inner, dict) or "synthetic" not in inner:
        raise ProvenanceError("artifact carries no provenance block")
    return SYNTHETIC_CAVEAT if inner["synthetic"] else None


def markdown_caveat(block: Any) -> str:
    """Render the caveat as a markdown blockquote, or an empty string.

    Reports emit their caveat through this rather than typing it, so the sentence
    in a report and the sentence on the dashboard cannot diverge.

    Parameters
    ----------
    block : dict
        A loaded artifact, or its ``provenance`` sub-block.

    Returns
    -------
    str
        A blockquote ending in a blank line, or ``""`` for a real artifact.

    Examples
    --------
    >>> markdown_caveat({"source": "olist", "synthetic": False})
    ''
    >>> markdown_caveat({"source": "mmm", "synthetic": True}).startswith("> **SYNTHETIC")
    True
    """
    caveat = caveat_for(block)
    if caveat is None:
        return ""
    label, rest = caveat.split(" — ", 1)
    return f"> **{label}** — {rest}\n\n"


def write_metric(name: str, payload: Any, provenance: Provenance, directory: Path) -> Path:
    """Write a metrics artifact, refusing one whose provenance does not hold up.

    The artifact carries no timestamp, so rerunning an unchanged computation
    produces a byte-identical file. That is what makes "the dashboard matches the
    code" checkable by ``git diff`` rather than asserted.

    Parameters
    ----------
    name : str
        Artifact name without the extension, e.g. ``"frame"``.
    payload : dict
        The numbers. Must not itself contain a ``provenance`` key.
    provenance : Provenance
        Where the numbers came from. Validated on construction.
    directory : pathlib.Path
        Destination, normally :func:`athar.paths.metrics_dir`.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ProvenanceError
        If ``payload`` already carries a ``provenance`` key, which would let a
        caller smuggle in a block that bypassed validation; or if it contains a
        non-finite value, which is both invalid JSON and a computation that
        failed without saying so.

    Examples
    --------
    >>> import tempfile, json
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = write_metric(
    ...         "demo",
    ...         {"qini": 0.25},
    ...         Provenance(source="criteo", synthetic=False, split="test"),
    ...         Path(tmp),
    ...     )
    ...     loaded = json.loads(path.read_text())
    >>> loaded["qini"], loaded["provenance"]["synthetic"]
    (0.25, False)
    """
    if isinstance(payload, dict) and "provenance" in payload:
        raise ProvenanceError(
            "payload must not carry its own 'provenance' key; pass a Provenance instead, "
            "so the block is validated rather than asserted"
        )
    directory.mkdir(parents=True, exist_ok=True)
    document = {**payload, "provenance": provenance.to_dict()}
    path = directory / f"{name}.json"
    try:
        rendered = json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_plain,
            allow_nan=False,
        )
    except ValueError as error:
        raise ProvenanceError(
            f"{name}.json contains a non-finite value ({error}). Python writes NaN and "
            f"Infinity as bare tokens, which are not valid JSON and which every other "
            f"reader rejects — the dashboard build is where this surfaced. More to the "
            f"point, a NaN in a metrics artifact is a number that failed to compute, and "
            f"it belongs in the report as an explained absence rather than as a token "
            f"that looks like a value. Replace it with null and say why."
        ) from error
    path.write_text(rendered + "\n")
    return path


def _plain(value: Any) -> Any:
    """Convert NumPy scalars and arrays to something json can write.

    Library outputs arrive as ``np.float64`` and friends more often than not, and
    ``json.dumps`` refuses them. Failing at the final write, after an hour of
    sampling, is the worst possible moment to discover that.

    Raises
    ------
    TypeError
        For anything genuinely unserialisable, rather than coercing it to a string
        and quietly writing nonsense into an artifact.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__} into a metrics artifact")


def read_metric(name: str, directory: Path) -> dict[str, Any]:
    """Read a metrics artifact and confirm it declares its provenance.

    Parameters
    ----------
    name : str
        Artifact name without the extension.
    directory : pathlib.Path
        Where to look, normally :func:`athar.paths.metrics_dir`.

    Returns
    -------
    dict
        The loaded artifact.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist.
    ProvenanceError
        If it carries no provenance block.
    """
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no metrics artifact at {path}; run the notebook that writes it")
    document: dict[str, Any] = json.loads(path.read_text())
    caveat_for(document)  # raises if the block is absent
    return document
