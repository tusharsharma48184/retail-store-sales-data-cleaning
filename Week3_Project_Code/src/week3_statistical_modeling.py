from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "retail_store_sales_raw.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df[["Price Per Unit", "Quantity", "Total Spent"]].isna().sum())

model_df = df[["Price Per Unit", "Quantity", "Total Spent"]].dropna().copy()
# Exclude impossible zero values from division-based interpretations if present.
X = sm.add_constant(model_df[["Price Per Unit", "Quantity"]])
y = model_df["Total Spent"]
model = sm.OLS(y, X).fit()
print(model.summary())

with open(OUT / "regression_summary.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

# Hypothesis test: H0 coefficient for Quantity = 0; H1 coefficient != 0.
print("\nQuantity coefficient:", model.params["Quantity"])
print("Quantity coefficient p-value:", model.pvalues["Quantity"])
print("R-squared:", model.rsquared)

fitted = model.fittedvalues
residuals = model.resid
plt.figure(figsize=(7, 4.5))
plt.scatter(fitted, residuals, alpha=0.35)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted Total Spent")
plt.ylabel("Residual")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(OUT / "residuals_vs_fitted.png", dpi=160)
plt.close()

sm.qqplot(residuals, line="45", fit=True)
plt.title("Q-Q Plot of Regression Residuals")
plt.tight_layout()
plt.savefig(OUT / "residuals_qq_plot.png", dpi=160)
plt.close()

print("\nDiagnostic plots saved to:", OUT)
