import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score
)

# basic evaluation
def evaluate_model(forest_model, X, y):
    y_pred = forest_model.predict(X)
    print("\nModel evaluation")

    print("\nAccuracy: ")
    print(accuracy_score(y, y_pred))

    print("\nPrediction results")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report")
    print(classification_report(y, y_pred))

# feature importance from random forest
def plot_feature_importance(forest_model, X):
    feature_importances = pd.Series(
        forest_model.feature_importances_,
        index=X.columns
    )

    feature_importances = feature_importances.sort_values(ascending=False)
    print("\nTop 10 important features")
    print(feature_importances.head(10))
    plt.figure(figsize=(10, 6))

    feature_importances.head(10).plot(kind="bar")

    plt.title("Top stockout risk features")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ROC curve
def plot_roc_curve(forest_model, X, y):
    y_prob = forest_model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

# precision recall curve
def plot_precision_recall(forest_model, X, y):
    y_prob = forest_model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

# try different probability threshold
def threshold_tuning(forest_model, X, y, threshold=0.3):
    y_prob = forest_model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nResults with threshold = {threshold}")
    print("\nPrediction results")
    print(confusion_matrix(y, y_pred))

    print("\nClassification report")
    print(classification_report(y, y_pred))