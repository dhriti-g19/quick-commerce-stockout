import joblib
import pandas as pd
from sklearn.metrics import f1_score

from src.data_preprocessing import load_data, preprocess_data
from src.feature import create_features
from src.train_model import (
    train_logistic_regression,
    train_random_forest
)
from src.evaluate_model import (
    evaluate_model,
    plot_feature_importance,
    plot_precision_recall
)


def find_best_threshold(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = 0

    for threshold in [i / 100 for i in range(10, 91)]:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_val, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def main():
    # 1. Load data
    sales_data = load_data("data/sales_data.csv")

    # 2. Create next-day stockout target
    sales_data = preprocess_data(sales_data)

    # 3. Create historical features
    sales_data = create_features(sales_data)

    # 4. Remove rows without a target
    sales_data = sales_data.dropna(
        subset=["Stockout_Next_Day"]
    ).copy()

    # 5. Select features
    features = [
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Price",
        "Discount",
        "Competitor Pricing",
        "Promotion",
        "Weather Condition",
        "Seasonality",
        "Epidemic",
        "Previous_Day_Sales",
        "Avg_7_Day_Sales",
        "Previous_Day_Demand",
        "Store ID",
        "Product ID",
        "Category",
        "Region"
    ]

    target = "Stockout_Next_Day"

    categorical_features = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    # 6. Encode categorical features
    X = sales_data[features].copy()
    y = sales_data[target].copy()

    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True
    )

    # 7. Chronological train / validation / test split
    train_mask = sales_data["Date"] < "2023-10-01"

    val_mask = (
        (sales_data["Date"] >= "2023-10-01") &
        (sales_data["Date"] < "2024-01-01")
    )

    test_mask = sales_data["Date"] >= "2024-01-01"

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Test data:", X_test.shape)

    # 8. Train models using training period only
    lr_model = train_logistic_regression(
        X_train,
        y_train
    )

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    # 9. Select Random Forest threshold using validation data
    rf_threshold, validation_f1 = find_best_threshold(
        rf_model,
        X_val,
        y_val
    )

    print(f"\nBest RF threshold: {rf_threshold:.2f}")
    print(f"Validation F1: {validation_f1:.3f}")

    # 10. Evaluate Random Forest on unseen test period
    evaluate_model(
        rf_model,
        X_test,
        y_test,
        threshold=rf_threshold
    )

    # 11. Feature importance
    plot_feature_importance(
        rf_model,
        X_train
    )

    # 12. Precision-recall curve
    plot_precision_recall(
        rf_model,
        X_test,
        y_test
    )

    # 13. Save model
    joblib.dump(
        rf_model,
        "models/stockout_forest_model.pkl"
    )

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()