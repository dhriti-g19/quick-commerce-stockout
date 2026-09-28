import pandas as pd


def load_data(file_path):
    try:
        sales_data = pd.read_csv(file_path)
        print(f"Data loaded: {sales_data.shape}")
        return sales_data
    except Exception as e:
        print("Error loading dataset:", e)
        raise


def preprocess_data(sales_data):
    sales_data = sales_data.copy()

    # Convert date
    sales_data["Date"] = pd.to_datetime(sales_data["Date"])

    # Sort so previous/next day operations are correct
    sales_data = sales_data.sort_values(
        ["Store ID", "Product ID", "Date"]
    ).reset_index(drop=True)

    # Stockout on the following day
    next_day_inventory = (
        sales_data.groupby(["Store ID", "Product ID"])["Inventory Level"]
        .shift(-1)
    )

    next_day_demand = (
        sales_data.groupby(["Store ID", "Product ID"])["Demand"]
        .shift(-1)
    )

    sales_data["Stockout_Next_Day"] = (
        next_day_inventory < next_day_demand
    ).astype(int)

    # Last day of each Store + Product has no next-day data
    sales_data.loc[
        next_day_inventory.isna(),
        "Stockout_Next_Day"
    ] = pd.NA

    # Remove rows where tomorrow's outcome is unavailable
    sales_data = sales_data.dropna(
        subset=["Stockout_Next_Day"]
    ).copy()

    sales_data["Stockout_Next_Day"] = (
        sales_data["Stockout_Next_Day"].astype(int)
    )

    return sales_data