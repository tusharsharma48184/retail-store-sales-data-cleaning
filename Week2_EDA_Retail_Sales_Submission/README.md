# Week 2 — Retail Store Sales EDA
**Prepared by:** Tushar Sharma

## Files
- `Week2_Retail_Sales_EDA.ipynb` — reproducible notebook with analysis code and commentary.
- `Week2_Retail_Sales_EDA_Report.pdf` — EDA report with visuals, findings, hypotheses, and limitations.
- `retail_store_sales_cleaned.csv` — cleaned dataset used for this analysis.
- `figures/` — chart images used in the report.

## Run
1. Install Python 3.
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in Jupyter Notebook or VS Code.
4. Run all cells from top to bottom.

## Main findings
- Dataset size: 12,575 rows × 11 columns.
- Recorded spend mean: 129.65; median: 108.50.
- Highest recorded category spend: Butchers (208,118.00).
- Welch's t-test comparing recorded spend by discount status: p = 0.799; not statistically significant at α = 0.05.

## Notes
Missing values are retained when they cannot be safely inferred. Findings are descriptive and do not establish causation. The dataset includes only part of January 2025, so 2025 should not be compared directly with complete years.
