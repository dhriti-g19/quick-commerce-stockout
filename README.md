# Quick Commerce Stockout Prediction

The project predicts risk of product stockouts in a quick commerce setting using machine learning. It analyzes retail transaction data and inventory levels to identify situations where products may run out of stock.

## Dataset

The dataset used in this project is based on FMCG retail transaction data available on Kaggle.
  
[Kaggle Dataset](https://www.kaggle.com/code/arannayavadebnath/eda-of-fmcg-retail-transaction/notebook)

## Feature Engineering

Some key features created for stockout prediction:
- City-category average demand
- Days of inventory left
- Temporal features (month, hour, weekday)

## Models Used

- Logistic Regression
- Random Forest Classifier

## Model Evaluation

The model performance is evaluated using:
- Confusion Matrix
- Precision and Recall
- ROC Curve

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