import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\nModel evaluation")
    print(f"Threshold: {threshold}")

    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report")
    print(classification_report(y_test, y_pred))

    print("\nMetrics")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.3f}")

    return y_prob, y_pred


def plot_feature_importance(forest_model, X):
    feature_importances = pd.Series(
        forest_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 important features")
    print(feature_importances.head(10))

    plt.figure(figsize=(10, 6))
    feature_importances.head(10).plot(kind="bar")

    plt.title("Top stockout prediction features")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_precision_recall(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()