import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.feature import create_features, encode_features
from src.train_model import split_data, train_logistic_regression, train_random_forest
from src.evaluate_model import (
    evaluate_model,
    plot_feature_importance,
    plot_roc_curve,
    plot_precision_recall,
    threshold_tuning
)


def main():
    # load data
    sales_data = load_data("data/sales_dataset.csv")
    # preprocess
    sales_data = preprocess_data(sales_data)
    # feature engineering
    sales_data = create_features(sales_data)
    # encoding
    X, y = encode_features(sales_data)
    # train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # train models
    log_model, scaler = train_logistic_regression(X_train, y_train)
    forest_model = train_random_forest(X_train, y_train)
    # evaluate Random Forest
    evaluate_model(forest_model, X_test, y_test)
    # feature importance
    plot_feature_importance(forest_model, X_train)

    # ROC curve
    plot_roc_curve(forest_model, X_test, y_test)
    # precision recall
    plot_precision_recall(forest_model, X_test, y_test)
    # threshold tuning
    threshold_tuning(forest_model, X_test, y_test, threshold=0.3)

    # save model
    joblib.dump(forest_model, "models/stockout_forest_model.pkl")
    print("\nModel saved")

if __name__ == "__main__":
    main()