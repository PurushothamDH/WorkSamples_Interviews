"""Identify the single lowest-sales product/location row for last month."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(__file__).resolve().parents[1] / "Orders.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "LowerSales.xlsx"
GROUP_COLUMNS = ["Region", "State/Province", "City", "Category", "Sub-Category", "Product ID"]
OUTPUT_COLUMNS = [
    "region",
    "state",
    "city",
    "category",
    "subCategory",
    "productid",
    "why_1",
    "why_2",
    "why_3",
    "why_4",
    "why_5",
]


def create_lower_sales_report(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create an Excel report containing one bottom-half sales observation."""
    if not input_path.exists():
        raise FileNotFoundError(f"Orders file was not found: {input_path}")

    orders = pd.read_csv(input_path)
    required_columns = set(GROUP_COLUMNS + ["Order Date", "Sales", "Quantity", "Discount", "Profit"])
    missing_columns = sorted(required_columns.difference(orders.columns))
    if missing_columns:
        raise ValueError(f"Orders.csv is missing required columns: {', '.join(missing_columns)}")

    orders["Order Date"] = pd.to_datetime(orders["Order Date"], format="%d-%m-%Y", errors="coerce")
    if orders["Order Date"].isna().any():
        raise ValueError("Orders.csv contains an invalid Order Date.")
    for column in ["Sales", "Quantity", "Discount", "Profit"]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    if orders[["Sales", "Quantity", "Discount", "Profit"]].isna().any().any():
        raise ValueError("Orders.csv contains invalid numeric values.")

    reference_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    last_month = reference_date.to_period("M") - 1
    last_month_orders = orders[orders["Order Date"].dt.to_period("M") == last_month].copy()
    if last_month_orders.empty:
        raise ValueError(f"No orders were found for {last_month}.")

    summary = (
        last_month_orders.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            sales=("Sales", "sum"),
            quantity=("Quantity", "sum"),
            discount=("Discount", "mean"),
            profit=("Profit", "sum"),
        )
        .sort_values(GROUP_COLUMNS, kind="stable")
    )
    median_sales = summary["sales"].median()
    bottom_half = summary[summary["sales"] <= median_sales]
    lowest = bottom_half.sort_values(["sales"] + GROUP_COLUMNS, kind="stable").iloc[0]

    discount_text = f"{lowest['discount']:.0%}"
    profit_reason = (
        f"Profit is negative ({lowest['profit']:.2f}), indicating an unprofitable sales mix."
        if lowest["profit"] < 0
        else f"Profit is only {lowest['profit']:.2f} for the recorded sales, limiting contribution."
    )
    report = pd.DataFrame(
        [
            {
                "region": lowest["Region"],
                "state": lowest["State/Province"],
                "city": lowest["City"],
                "category": lowest["Category"],
                "subCategory": lowest["Sub-Category"],
                "productid": lowest["Product ID"],
                "why_1": f"Sales of {lowest['sales']:.2f} are the lowest observation in the bottom 50% for {last_month}.",
                "why_2": f"Only {int(lowest['quantity'])} unit(s) were sold in the period, indicating weak demand.",
                "why_3": f"The average discount was {discount_text}, which may be reducing realized sales value.",
                "why_4": profit_reason,
                "why_5": (
                    f"The weakness is concentrated in {lowest['City']}, {lowest['State/Province']} "
                    f"for {lowest['Sub-Category']} in the {lowest['Region']} region."
                ),
            }
        ],
        columns=OUTPUT_COLUMNS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_excel(output_path, index=False, sheet_name="Lower Sales")
    return report


if __name__ == "__main__":
    result = create_lower_sales_report()
    print(result.to_string(index=False))
    print(f"\nSaved report to: {OUTPUT_PATH}")
