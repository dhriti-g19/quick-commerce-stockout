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

    # target variable
    sales_data["StockoutRisk_flag"] = (
        sales_data["Stock_On_Hand"] <= sales_data["Reorder_Level"]
    ).astype(int)

    # date conversion
    sales_data["Invoice_Date"] = pd.to_datetime(sales_data["Invoice_Date"])

    # time features
    sales_data["month"] = sales_data["Invoice_Date"].dt.month
    sales_data["day"] = sales_data["Invoice_Date"].dt.day
    sales_data["hour"] = sales_data["Invoice_Date"].dt.hour
    sales_data["weekday"] = sales_data["Invoice_Date"].dt.day_name()

    unnecessary_cols = [
        "Invoice_ID",
        "Invoice_Date",
        "Customer_Age",
        "Customer_Gender",
        "Revenue",
        "Cost",
        "Margin",
        "Margin_%"
    ]
    sales_data = sales_data.drop(unnecessary_cols, axis=1, errors="ignore")
    return sales_data