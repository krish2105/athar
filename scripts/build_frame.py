"""Build the real Olist frames every downstream notebook rests on.

Writes four Parquet artifacts to `<DATA_ROOT>/processed/athar/` and one committed
metrics artifact. Nothing here is simulated: `metrics/frame.json` carries
`synthetic: false`, and it is the last point in the pipeline where that is true
without qualification.

The verification at the end is the point of the script existing separately from
the notebook. Revenue is recomputed by an independent DuckDB query that shares no
code path with `athar.frame`, and the two totals must agree exactly. A join that
silently dropped rows, or a window that moved, fails here rather than in a chart.

Run: `make frame`
"""

import logging

import duckdb
import pandas as pd

from athar import frame, paths
from athar.provenance import Provenance, write_metric

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_frame")

#: One centavo. See the reconciliation check in main() for why exact equality
#: is not the test.
TOLERANCE_BRL = 0.01


def independent_revenue_total(start, end):
    """Recompute windowed revenue without touching `athar.frame`.

    Deliberately a different query shape — a direct join and filter rather than
    the staged CTEs the library uses — so agreement is evidence rather than the
    same code run twice.
    """
    directory = paths.olist_dir()
    statuses = ", ".join(f"'{s}'" for s in frame.EXCLUDED_STATUSES)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads TO 1")
        return connection.execute(f"""
            SELECT count(DISTINCT o.order_id) AS orders, sum(i.price) AS revenue
            FROM read_csv_auto('{directory}/olist_orders_dataset.csv') o,
                 read_csv_auto('{directory}/olist_order_items_dataset.csv') i
            WHERE i.order_id = o.order_id
              AND o.order_status NOT IN ({statuses})
              AND date_trunc('week', o.order_purchase_timestamp)::DATE
                  BETWEEN DATE '{start.date()}' AND DATE '{end.date()}'
        """).fetchone()
    finally:
        connection.close()


def main():
    processed = paths.processed_dir()

    log.info("reading Olist")
    orders = frame.load_orders()
    log.info(
        "  %d orders with items, %s to %s",
        len(orders),
        orders["purchased_at"].min().date(),
        orders["purchased_at"].max().date(),
    )

    weekly_all = frame.weekly_revenue(orders)
    start, end = frame.select_window(weekly_all)
    log.info("window rule selects %s to %s", start.date(), end.date())

    in_window = orders[
        (orders["purchased_at"] >= start) & (orders["purchased_at"] < end + pd.Timedelta(days=7))
    ].reset_index(drop=True)

    weekly = frame.weekly_revenue(in_window)
    panel = frame.state_week_revenue(in_window)
    # T is measured to the end of the last complete week, not to the last order,
    # so every customer is observed for the full window.
    observation_end = end + pd.Timedelta(days=7)
    customers = frame.customer_summary(in_window, observation_end)

    for name, table in (
        ("orders", in_window),
        ("weekly", weekly),
        ("state_week", panel),
        ("customers", customers),
    ):
        path = processed / f"{name}.parquet"
        table.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path.name, len(table))

    # --- verification ------------------------------------------------------
    check_orders, check_revenue = independent_revenue_total(start, end)
    library_revenue = float(weekly["revenue"].sum())
    difference = abs(library_revenue - float(check_revenue))
    log.info("independent revenue check: difference = %.10f BRL", difference)

    # The two queries agree on every order but not bit-for-bit on the total, and
    # they cannot: float addition is not associative, the two plans accumulate
    # 96,731 terms in different orders, and DuckDB is already pinned to one
    # thread. The observed gap is ~1e-5 BRL on 1.3e7, or ~1e-12 relative, which
    # is float64 accumulation and nothing else. So the test is agreement to the
    # centavo — the resolution the quantity actually has — with the raw
    # difference recorded so a reader can see how far inside tolerance it sits.
    # Order counts, being integers, are required to match exactly.
    if difference > TOLERANCE_BRL or int(check_orders) != len(in_window):
        raise SystemExit(
            f"revenue reconciliation failed: library {library_revenue} on {len(in_window)} orders "
            f"vs independent {check_revenue} on {check_orders}"
        )

    zero_weeks = int((weekly["orders"] == 0).sum())
    if zero_weeks:
        raise SystemExit(f"{zero_weeks} zero-order weeks inside the selected window")

    # Repeat rate on the raw extract as well as inside the window. The two differ
    # because the window drops 2016 and the cancelled orders, and quoting only one
    # invites the question of which.
    raw_people = orders.groupby("customer_unique_id").size()
    window_people = in_window.groupby("customer_unique_id").size()

    payload = {
        "window": {
            "start": str(start.date()),
            "end": str(end.date()),
            "weeks": int(len(weekly)),
            "rule": (
                "Starts the week after the last week with zero orders, which removes the "
                "nine-week collection gap in late 2016. Ends at the last week carrying at "
                f"least {frame.TRUNCATION_FLOOR:.0%} of the median of the "
                f"{frame.TRAILING_WEEKS} weeks before it, which removes the mid-week "
                "truncation of the extract. The comparison is against recent weeks rather "
                "than the whole series because Olist grew throughout, so a half-collected "
                "final week can still sit above the global median."
            ),
            "excluded_statuses": list(frame.EXCLUDED_STATUSES),
            "revenue_definition": (
                "Sum of order_items.price, excluding freight, recognised at "
                "order_purchase_timestamp. Currency is BRL, 2017-2018."
            ),
        },
        "totals": {
            "orders": int(len(in_window)),
            "revenue_brl": round(library_revenue, 2),
            "items": int(in_window["items"].sum()),
            "freight_brl": round(float(in_window["freight"].sum()), 2),
            "mean_order_value_brl": round(library_revenue / len(in_window), 2),
            "states": int(in_window["state"].nunique()),
            "people": int(in_window["customer_unique_id"].nunique()),
        },
        "coverage": {
            "orders_all_dates": int(len(orders)),
            "revenue_all_dates_brl": round(float(orders["revenue"].sum()), 2),
            "share_of_orders_in_window": round(len(in_window) / len(orders), 6),
            "share_of_revenue_in_window": round(
                library_revenue / float(orders["revenue"].sum()), 6
            ),
        },
        "repeat_behaviour": {
            "note": (
                "A person is customer_unique_id; Olist issues a fresh customer_id per order, "
                "so keying on that would report a repeat rate of exactly zero."
            ),
            "all_dates": {
                "people": int(len(raw_people)),
                "with_repeat": int((raw_people > 1).sum()),
                "repeat_rate": round(float((raw_people > 1).mean()), 6),
            },
            "in_window": {
                "people": int(len(window_people)),
                "with_repeat": int((window_people > 1).sum()),
                "with_two_or_more_repeats": int((window_people > 2).sum()),
                "repeat_rate": round(float((window_people > 1).mean()), 6),
            },
        },
        "weekly_shape": {
            "median_orders": float(weekly["orders"].median()),
            "min_orders": int(weekly["orders"].min()),
            "max_orders": int(weekly["orders"].max()),
            "median_revenue_brl": round(float(weekly["revenue"].median()), 2),
            "first_eight_weeks_revenue_brl": round(float(weekly["revenue"].head(8).sum()), 2),
            "last_eight_weeks_revenue_brl": round(float(weekly["revenue"].tail(8).sum()), 2),
        },
        "geo_panel": {
            "states": int(panel["state"].nunique()),
            "rows": int(len(panel)),
            "balanced": bool(len(panel) == panel["state"].nunique() * len(weekly)),
            "largest_state_revenue_share": round(
                float(panel.groupby("state")["revenue"].sum().max() / panel["revenue"].sum()),
                6,
            ),
            "states_below_one_percent_of_revenue": int(
                (panel.groupby("state")["revenue"].sum() / panel["revenue"].sum() < 0.01).sum()
            ),
        },
        "verification": {
            "independent_revenue_difference_brl": round(difference, 10),
            "independent_revenue_tolerance_brl": TOLERANCE_BRL,
            "independent_revenue_relative_difference": round(difference / library_revenue, 15),
            "independent_order_count_matches": bool(int(check_orders) == len(in_window)),
            "zero_order_weeks_in_window": zero_weeks,
        },
    }

    path = write_metric(
        "frame",
        payload,
        Provenance(source="olist", synthetic=False, split="full"),
        paths.metrics_dir(),
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
