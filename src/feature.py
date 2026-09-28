import pandas as pd


def create_features(sales_data):
    sales_data = sales_data.copy()

    sales_data = sales_data.sort_values(
        ["Store ID", "Product ID", "Date"]
    ).reset_index(drop=True)

    group = sales_data.groupby(["Store ID", "Product ID"])

    # Previous day's sales
    sales_data["Previous_Day_Sales"] = (
        group["Units Sold"].shift(1)
    )

    # 7-day average sales using previous days only
    sales_data["Avg_7_Day_Sales"] = (
        group["Units Sold"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    # Previous day's demand
    sales_data["Previous_Day_Demand"] = (
        group["Demand"].shift(1)
    )

    # Remove rows where historical features are unavailable
    sales_data = sales_data.dropna(
        subset=[
            "Previous_Day_Sales",
            "Avg_7_Day_Sales",
            "Previous_Day_Demand"
        ]
    ).copy()

    return sales_data