"""Olist to a weekly revenue spine, a state-week panel, and a customer summary.

Three frames come out of here and everything real in ATHAR rests on them.

``weekly_revenue``
    The national series the media-mix model explains. This is the *real* part of
    the MMM's target: the simulated media contribution in :mod:`athar.dgp` is
    layered on top of it, so the baseline the model has to see through is genuine
    Brazilian e-commerce history rather than one this project invented.

``state_week_revenue``
    The same series split across Olist's 27 real states. Geo holdout experiments
    in :mod:`athar.experiments` are simulated against this panel, so their
    heterogeneity and their power problem are real even though the treatment is
    not.

``customer_summary``
    Recency, frequency, monetary value per person, for :mod:`athar.clv`.

Three decisions worth defending
-------------------------------

**Revenue is item price, excluding freight.** Freight is pass-through logistics
billed at cost; a marketing campaign does not make it grow, and including it
would inflate every ROI in the project by roughly 17% for no reason a media
planner would accept.

**Revenue is recognised at the purchase timestamp, not at delivery.** Marketing
drives the order, not the courier. Recognising at delivery would also shift
revenue by a variable lag and truncate the end of the window twice over.

**Cancelled and unavailable orders are excluded; in-flight ones are kept.** An
order that was cancelled produced no revenue. One that is merely ``invoiced`` or
``processing`` at the end of the extract is a real purchase that a real campaign
could have caused, and dropping it would bias the most recent weeks downward
precisely where the series is already thinnest.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from athar import paths

__all__ = [
    "EXCLUDED_STATUSES",
    "TRAILING_WEEKS",
    "TRUNCATION_FLOOR",
    "customer_summary",
    "load_orders",
    "select_window",
    "state_week_revenue",
    "weekly_revenue",
]

#: Statuses that produced no revenue. Everything else is a real purchase.
EXCLUDED_STATUSES: tuple[str, ...] = ("canceled", "unavailable")

#: Weeks of context used to judge whether a trailing week is truncated.
TRAILING_WEEKS = 8

#: A trailing week carrying less than this share of the preceding weeks' median
#: order count is collection truncation rather than a business fact. Olist's
#: extract stops mid-week, so the final weeks tail off to single orders.
TRUNCATION_FLOOR = 0.70


def load_orders() -> pd.DataFrame:
    """Read Olist into one order-level frame, keyed on the purchase timestamp.

    DuckDB does the join out-of-core and is pinned to a single thread, because
    parallel float aggregation is not associative and a revenue total that
    changes between runs makes every downstream number unreproducible. The result
    is sorted explicitly: parallel joins do not promise a row order.

    Returns
    -------
    pandas.DataFrame
        One row per order with items, sorted by ``purchased_at``. Columns:
        ``order_id``, ``customer_unique_id``, ``state``, ``purchased_at``,
        ``revenue``, ``freight``, ``items``.

    Raises
    ------
    RuntimeError
        If the Olist directory is absent.
    """
    directory = paths.olist_dir()
    statuses = ", ".join(f"'{status}'" for status in EXCLUDED_STATUSES)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads TO 1")
        return connection.execute(f"""
            WITH orders AS (
                SELECT order_id, customer_id, order_purchase_timestamp AS purchased_at
                FROM read_csv_auto('{directory}/olist_orders_dataset.csv')
                WHERE order_status NOT IN ({statuses})
            ),
            items AS (
                SELECT order_id,
                       sum(price)         AS revenue,
                       sum(freight_value) AS freight,
                       count(*)           AS items
                FROM read_csv_auto('{directory}/olist_order_items_dataset.csv')
                GROUP BY order_id
            ),
            people AS (
                SELECT customer_id, customer_unique_id, customer_state AS state
                FROM read_csv_auto('{directory}/olist_customers_dataset.csv')
            )
            SELECT orders.order_id, people.customer_unique_id, people.state,
                   orders.purchased_at, items.revenue, items.freight, items.items
            FROM orders
            JOIN items  ON items.order_id = orders.order_id
            JOIN people ON people.customer_id = orders.customer_id
            ORDER BY orders.purchased_at, orders.order_id
        """).df()
    finally:
        connection.close()


def weekly_revenue(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate orders to a Monday-anchored weekly series.

    Parameters
    ----------
    orders : pandas.DataFrame
        Output of :func:`load_orders`.

    Returns
    -------
    pandas.DataFrame
        Columns ``week``, ``orders``, ``revenue``, reindexed onto a complete
        weekly grid so a collection gap shows up as a zero rather than as a
        missing row that silently closes up.

    Examples
    --------
    >>> import pandas as pd
    >>> orders = pd.DataFrame({
    ...     "purchased_at": pd.to_datetime(["2020-01-07", "2020-01-08", "2020-01-21"]),
    ...     "revenue": [10.0, 20.0, 5.0],
    ... })
    >>> weekly_revenue(orders)[["week", "orders", "revenue"]].to_dict("records")
    [{'week': Timestamp('2020-01-06 00:00:00'), 'orders': 2, 'revenue': 30.0}, \
{'week': Timestamp('2020-01-13 00:00:00'), 'orders': 0, 'revenue': 0.0}, \
{'week': Timestamp('2020-01-20 00:00:00'), 'orders': 1, 'revenue': 5.0}]
    """
    stamps = pd.to_datetime(orders["purchased_at"])
    week = stamps.dt.to_period("W-SUN").dt.start_time
    grouped = (
        pd.DataFrame({"week": week, "revenue": orders["revenue"].to_numpy()})
        .groupby("week", as_index=False)
        .agg(orders=("revenue", "size"), revenue=("revenue", "sum"))
    )
    grid = pd.date_range(grouped["week"].min(), grouped["week"].max(), freq="W-MON")
    return (
        grouped.set_index("week")
        .reindex(grid, fill_value=0)
        .rename_axis("week")
        .reset_index()
        .astype({"orders": int, "revenue": float})
    )


def state_week_revenue(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate orders to a state-by-week panel.

    Every state appears in every week, filled with zero where it sold nothing, so
    the panel is balanced. An unbalanced geo panel silently drops the small
    states from a difference-in-differences estimate, which is exactly where a
    holdout test's power problem lives.

    Parameters
    ----------
    orders : pandas.DataFrame
        Output of :func:`load_orders`.

    Returns
    -------
    pandas.DataFrame
        Columns ``week``, ``state``, ``orders``, ``revenue``, sorted by week then
        state.

    Examples
    --------
    >>> import pandas as pd
    >>> orders = pd.DataFrame({
    ...     "purchased_at": pd.to_datetime(["2020-01-07", "2020-01-08"]),
    ...     "state": ["SP", "RJ"],
    ...     "revenue": [10.0, 20.0],
    ... })
    >>> panel = state_week_revenue(orders)
    >>> len(panel), sorted(panel["state"].unique())
    (2, ['RJ', 'SP'])
    """
    stamps = pd.to_datetime(orders["purchased_at"])
    frame = pd.DataFrame(
        {
            "week": stamps.dt.to_period("W-SUN").dt.start_time,
            "state": orders["state"].to_numpy(),
            "revenue": orders["revenue"].to_numpy(),
        }
    )
    grouped = frame.groupby(["week", "state"], as_index=False).agg(
        orders=("revenue", "size"), revenue=("revenue", "sum")
    )
    grid = pd.MultiIndex.from_product(
        [
            pd.date_range(grouped["week"].min(), grouped["week"].max(), freq="W-MON"),
            sorted(grouped["state"].unique()),
        ],
        names=["week", "state"],
    )
    return (
        grouped.set_index(["week", "state"])
        .reindex(grid, fill_value=0)
        .reset_index()
        .astype({"orders": int, "revenue": float})
        .sort_values(["week", "state"], ignore_index=True)
    )


def select_window(
    weekly: pd.DataFrame,
    trailing: int = TRAILING_WEEKS,
    floor: float = TRUNCATION_FLOOR,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Choose the modelling window by rule rather than by eye.

    Two defects bound Olist's usable history and neither is a business fact.
    Collection does not begin cleanly — the 2016 weeks are interrupted by a
    nine-week hole where no orders were recorded at all — and it stops mid-week
    at the end, so the final weeks tail off toward a single order.

    So the window starts the week *after* the last week with no orders, and ends
    at the last week still carrying ``floor`` of the median of the ``trailing``
    weeks before it. Comparing against recent weeks rather than the whole series
    matters: Olist grew throughout, so a truncated final week can still sit above
    the global median while being half of its own neighbourhood.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Output of :func:`weekly_revenue`, on a complete weekly grid.
    trailing : int, optional
        Weeks of context used to judge truncation.
    floor : float, optional
        Share of the trailing median a final week must carry to be kept.

    Returns
    -------
    tuple of pandas.Timestamp
        Inclusive ``(start, end)`` of the modelling window.

    Raises
    ------
    ValueError
        If no week survives the rule.

    Examples
    --------
    A leading collection gap and a truncated final week, both removed:

    >>> import pandas as pd
    >>> weekly = pd.DataFrame({
    ...     "week": pd.date_range("2020-01-06", periods=6, freq="W-MON"),
    ...     "orders": [0, 100, 100, 100, 100, 20],
    ... })
    >>> start, end = select_window(weekly, trailing=3, floor=0.7)
    >>> str(start.date()), str(end.date())
    ('2020-01-13', '2020-02-03')
    """
    counts = weekly["orders"].to_numpy()
    weeks = pd.to_datetime(weekly["week"]).to_numpy()

    empty = [index for index, count in enumerate(counts) if count == 0]
    first = empty[-1] + 1 if empty else 0

    last = len(counts) - 1
    while last > first:
        context = counts[max(first, last - trailing) : last]
        if len(context) == 0 or counts[last] >= floor * float(pd.Series(context).median()):
            break
        last -= 1

    if first > last:
        raise ValueError("no week survives the window rule; check the weekly grid for gaps")
    return pd.Timestamp(weeks[first]), pd.Timestamp(weeks[last])


def customer_summary(orders: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Summarise each person into the quantities BG/NBD and Gamma-Gamma need.

    A person is ``customer_unique_id``. Olist issues a fresh ``customer_id`` per
    order, so keying on it would make every customer a first-time buyer and put
    the repeat rate at exactly zero.

    Units are days. ``frequency`` counts *repeat* purchases, so a one-time buyer
    scores zero and ``recency`` is then zero as well — a constraint ``lifetimes``
    enforces, and one that 96.9% of this base satisfies.

    Parameters
    ----------
    orders : pandas.DataFrame
        Output of :func:`load_orders`, already restricted to the window.
    observation_end : pandas.Timestamp
        End of the observation period, used for ``T``.

    Returns
    -------
    pandas.DataFrame
        One row per person: ``customer_unique_id``, ``frequency``, ``recency``,
        ``T``, ``monetary``, ``first_order_value``, ``state``.

    Examples
    --------
    A one-time buyer and a repeat buyer:

    >>> import pandas as pd
    >>> orders = pd.DataFrame({
    ...     "customer_unique_id": ["a", "b", "b"],
    ...     "state": ["SP", "RJ", "RJ"],
    ...     "purchased_at": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-31"]),
    ...     "revenue": [50.0, 10.0, 30.0],
    ... })
    >>> summary = customer_summary(orders, pd.Timestamp("2020-02-10"))
    >>> summary[["customer_unique_id", "frequency", "recency", "T", "monetary"]].to_dict("records")
    [{'customer_unique_id': 'a', 'frequency': 0, 'recency': 0.0, 'T': 40.0, 'monetary': 0.0}, \
{'customer_unique_id': 'b', 'frequency': 1, 'recency': 30.0, 'T': 40.0, 'monetary': 30.0}]
    """
    frame = orders.copy()
    frame["purchased_at"] = pd.to_datetime(frame["purchased_at"])
    frame = frame.sort_values(["customer_unique_id", "purchased_at"], kind="stable")

    grouped = frame.groupby("customer_unique_id", sort=True)
    first_at = grouped["purchased_at"].min()
    last_at = grouped["purchased_at"].max()

    # Gamma-Gamma is fitted on the value of *repeat* orders, so the first order of
    # each person is excluded from the monetary mean. A person with no repeats
    # contributes no monetary observation at all, which is why the model can only
    # be fitted on the repeaters.
    is_first = frame["purchased_at"] == frame["customer_unique_id"].map(first_at)
    repeats = frame[~is_first]
    monetary = repeats.groupby("customer_unique_id")["revenue"].mean()

    summary = pd.DataFrame(
        {
            "customer_unique_id": first_at.index,
            "frequency": (grouped.size() - 1).to_numpy(),
            "recency": (last_at - first_at).dt.days.to_numpy().astype(float),
            "T": (pd.Timestamp(observation_end) - first_at).dt.days.to_numpy().astype(float),
            "monetary": monetary.reindex(first_at.index).fillna(0.0).to_numpy(),
            "first_order_value": grouped["revenue"].first().to_numpy(),
            "state": grouped["state"].first().to_numpy(),
        }
    ).reset_index(drop=True)
    return summary
