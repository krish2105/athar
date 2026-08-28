"""Where the raw tables are, and where this project's artifacts land.

Holds the one answer to "where is the data" — resolved through
:func:`spine.io.data_root`, so ATHAR and its sibling repos cannot drift — plus the
project-scoped directories underneath it.

Processed artifacts live under ``<DATA_ROOT>/processed/athar/`` rather than inside
the repository. The shared store sits outside every git repo and is never
committed, and namespacing by project stops four repos writing over each other's
frames. Reports, metrics and the pre-registered DGP configuration *are* committed,
so they stay in the repository.
"""

from __future__ import annotations

from pathlib import Path

from spine.io import data_root

__all__ = [
    "CRITEO_FILE",
    "OLIST_TABLES",
    "config_dir",
    "criteo_dir",
    "criteo_parquet",
    "dashboard_data_dir",
    "metrics_dir",
    "olist_dir",
    "processed_dir",
    "project_root",
    "recovery_dir",
    "reports_dir",
]

#: The Olist tables ATHAR reads, and the column each is keyed on. The geolocation
#: table is deliberately absent: ATHAR needs state, which ``customers`` already
#: carries, and the 61 MB of zip-code centroids buy nothing at weekly resolution.
OLIST_TABLES: dict[str, str] = {
    "olist_orders_dataset": "order_id",
    "olist_order_items_dataset": "order_id",
    "olist_order_payments_dataset": "order_id",
    "olist_customers_dataset": "customer_id",
    "olist_products_dataset": "product_id",
    "product_category_name_translation": "product_category_name",
}

#: The single Criteo file. 3.2 GB of CSV, converted once to Parquet.
CRITEO_FILE = "criteo-uplift-v2.1.csv"


def project_root() -> Path:
    """Locate the repository root.

    Resolved from this file's position rather than the working directory, so a
    notebook running in ``notebooks/`` and a script running at the root agree.

    Returns
    -------
    pathlib.Path
        The ``03-athar`` directory.

    Examples
    --------
    >>> project_root().name
    '03-athar'
    """
    return Path(__file__).resolve().parents[2]


def olist_dir() -> Path:
    """Locate the raw Olist tables.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/raw/olist``.

    Raises
    ------
    RuntimeError
        If ``DATA_ROOT`` is not configured, or the directory is absent.
    """
    directory = data_root(project_root()) / "raw" / "olist"
    if not directory.is_dir():
        raise RuntimeError(
            f"expected the Olist tables at {directory}, which does not exist; "
            f"see data/README.md for the download checklist"
        )
    return directory


def criteo_dir() -> Path:
    """Locate the raw Criteo uplift file's directory.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/raw/criteo``.

    Raises
    ------
    RuntimeError
        If ``DATA_ROOT`` is not configured, or the directory is absent.
    """
    directory = data_root(project_root()) / "raw" / "criteo"
    if not directory.is_dir():
        raise RuntimeError(
            f"expected the Criteo file at {directory}, which does not exist; "
            f"see data/README.md for the download checklist"
        )
    return directory


def processed_dir() -> Path:
    """Locate the directory for this project's processed artifacts, creating it.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/processed/athar``.
    """
    directory = data_root(project_root()) / "processed" / "athar"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def criteo_parquet() -> Path:
    """Locate the Parquet conversion of the Criteo file.

    The CSV is read exactly once, by ``scripts/build_criteo.py``. Everything
    downstream reads this file, which is roughly a fifth of the size and loads
    columns rather than rows.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/processed/athar/criteo.parquet``.
    """
    return processed_dir() / "criteo.parquet"


def recovery_dir() -> Path:
    """Locate the cache of fitted recovery-grid posteriors, creating it.

    Each fit is cached under a hash of its cell configuration so an interrupted
    overnight run costs one fit rather than the batch.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/processed/athar/recovery``.
    """
    directory = processed_dir() / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def metrics_dir() -> Path:
    """Locate the committed directory holding every reported number, creating it.

    Returns
    -------
    pathlib.Path
        ``<repo>/metrics``.
    """
    directory = project_root() / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def reports_dir() -> Path:
    """Locate the committed markdown reports directory, creating it.

    Returns
    -------
    pathlib.Path
        ``<repo>/reports``.
    """
    directory = project_root() / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def config_dir() -> Path:
    """Locate the committed configuration directory.

    Holds ``dgp.yaml``, the pre-registered data-generating process. It is
    committed and hashed into every artifact derived from it, so a result can
    always name the configuration that produced it.

    Returns
    -------
    pathlib.Path
        ``<repo>/config``.
    """
    return project_root() / "config"


def dashboard_data_dir() -> Path:
    """Locate the directory the dashboard build reads its numbers from, creating it.

    The dashboard imports the committed ``metrics/*.json`` artifacts. This is that
    directory under its build-facing name, kept separate so the coupling is
    visible rather than implied by a relative path buried in a bundler config.

    Returns
    -------
    pathlib.Path
        ``<repo>/metrics``.
    """
    return metrics_dir()
