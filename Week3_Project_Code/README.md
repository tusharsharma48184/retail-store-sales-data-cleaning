# Week 3 — Statistical Modeling and Hypothesis Testing
Prepared by: Tushar Sharma

## Contents
- `notebooks/Week3_Statistical_Modeling.ipynb`: reproducible notebook
- `src/week3_statistical_modeling.py`: script version of the analysis
- `data/retail_store_sales_raw.csv`: source data
- `outputs/`: generated model summary/diagnostic plots when the script is run

## Research question
Among records with complete numeric values, is Quantity associated with Total Spent after accounting for Price Per Unit?

## Method
The analysis creates a multiple linear regression using Price Per Unit and Quantity as predictors of Total Spent. It checks missingness, fits the model on complete numeric rows, reports R-squared and coefficient p-values, and creates residual and Q-Q plots. Because Total Spent is generally constructed from unit price × quantity, interpret the model as a relationship check rather than an independent causal discovery.

## Run
`pip install -r requirements.txt`
`python src/week3_statistical_modeling.py`
Or open the notebook and run all cells.
