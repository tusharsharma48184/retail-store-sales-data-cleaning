# Week 4 — Machine Learning Model Development and Evaluation
Prepared by: Tushar Sharma

## Problem
Predict `Total Spent` from `Price Per Unit`, `Quantity`, and available transaction descriptors. This is a regression task.

## Models
A baseline Linear Regression and a Random Forest Regressor are compared. Numeric and categorical preprocessing are handled in a pipeline. Random Forest hyperparameters are tuned with GridSearchCV and cross-validation. Final performance is measured on a held-out test set using MAE, RMSE, and R².

## Important limitation
`Total Spent` is often mathematically determined by `Price Per Unit × Quantity`. Therefore high predictive performance can partly reflect this formula rather than a difficult real-world forecasting problem. This dataset is used for learning and results should not be generalized without external validation.

## Run
`pip install -r requirements.txt`
`python src/week4_ml_model.py`
Or run the notebook from top to bottom.
