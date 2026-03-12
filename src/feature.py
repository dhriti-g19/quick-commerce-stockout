import pandas as pd

def create_features(sales_data):
    # avg demand per city and category
    sales_data["avg_units_city"] = (
        sales_data.groupby(["City", "Category"])["Units"]
        .transform("mean")
    )

    # inventory coverage
    sales_data["inventory_days_left"] = (
        sales_data["Stock_On_Hand"] / (sales_data["avg_units_city"] + 1)
    )

    # demand velocity
    sales_data["demand_velocity"] = (
        sales_data["Units"] / (sales_data["inventory_days_left"] + 1)
    )
    return sales_data

def encode_features(sales_data):
    X = sales_data.drop("StockoutRisk_flag", axis=1)
    y = sales_data["StockoutRisk_flag"]

    # one hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # remove leakage columns
    X_encoded = X_encoded.drop(
        ["Stock_On_Hand", "Reorder_Level"],
        axis=1,
        errors="ignore"
    )
    return X_encoded, y