# Quick Commerce Stockout Prediction

The project predicts risk of product stockouts in a quick commerce setting using machine learning. It uses retail inventory, sales, and demand data to predict whether a product is likely to stockout on the following day.

## Dataset

The dataset used in this project is based on Retail Store Inventory and Demand Forecasting dataset available on Kaggle.
  
[Kaggle Dataset](https://doi.org/10.34740/kaggle/dsv/11895299)

The dataset is synthetically generated retail store data and is used in this project for a quick-commerce stockout prediction use case.

## Feature Engineering

Some key features created for stockout prediction:
- Previous day sales
- 7-day average sales
- Previous day demand

Categorical features such as store, product, category, region, weather condition, and seasonality are also used.

## Models Used

- Logistic Regression
- Random Forest Classifier

Class weighting is used to handle the imbalance between stockout and non-stockout cases.

## Model Evaluation

The data is split chronologically into training, validation, and test sets.

The model performance is evaluated using Confusion Matrix, Precision and Recall, F1 Score, PR-AUC, Precision-Recall Curve.

## How to Run

1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the training pipeline
```
python main.py
```