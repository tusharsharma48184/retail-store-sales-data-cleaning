# Week 6 — Integrated Data Science Project
Prepared by: Tushar Sharma

## Project question
Can retail transaction attributes help explain and predict recorded transaction spend, and what patterns are visible in categories, payment methods, locations, and time?

## Pipeline
1. Load and clean types
2. Data quality summary
3. Exploratory analysis and visualizations
4. Statistical linear regression with diagnostics
5. Machine learning model comparison and cross-validation
6. Save tables, charts, and reproducible outputs

## Important caveats
The target `Total Spent` is closely related to `Price Per Unit × Quantity`, so predictive scores may reflect this built-in formula. This is a learning project using a structured dataset, not a validated commercial forecasting system.

## Run
`pip install -r requirements.txt`
`python src/week6_integrated_project.py`
Or open `notebooks/Week6_Integrated_Data_Science_Project.ipynb` and run cells.
