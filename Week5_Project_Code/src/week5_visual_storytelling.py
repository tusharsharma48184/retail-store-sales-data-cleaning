from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "retail_store_sales_raw.csv")
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)

# 1. Distribution
fig, ax = plt.subplots(figsize=(8, 4.8))
df["Total Spent"].dropna().hist(bins=30, ax=ax)
ax.set(title="Distribution of Retail Transaction Spend", xlabel="Total Spent", ylabel="Transaction count")
fig.tight_layout(); fig.savefig(out / "01_spend_distribution.png", dpi=180); plt.close(fig)

# 2. Category totals
category = df.groupby("Category")["Total Spent"].sum(min_count=1).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
category.plot(kind="barh", ax=ax)
ax.set(title="Recorded Spend by Product Category", xlabel="Total Spent", ylabel="Category")
fig.tight_layout(); fig.savefig(out / "02_category_spend.png", dpi=180); plt.close(fig)

# 3. Quantity vs spend
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.scatter(df["Quantity"], df["Total Spent"], alpha=0.25)
ax.set(title="Quantity and Transaction Spend", xlabel="Quantity", ylabel="Total Spent")
fig.tight_layout(); fig.savefig(out / "03_quantity_vs_spend.png", dpi=180); plt.close(fig)

# 4. Discount comparison
df["Discount Applied"] = df["Discount Applied"].astype("string")
groups = [df.loc[df["Discount Applied"] == "False", "Total Spent"].dropna(),
          df.loc[df["Discount Applied"] == "True", "Total Spent"].dropna()]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.boxplot(groups, tick_labels=["No discount", "Discount applied"], showfliers=False)
ax.set(title="Spend by Recorded Discount Status", ylabel="Total Spent")
fig.tight_layout(); fig.savefig(out / "04_discount_comparison.png", dpi=180); plt.close(fig)

# 5. Monthly recorded spend
dated = df.dropna(subset=["Transaction Date", "Total Spent"]).copy()
dated["Month"] = dated["Transaction Date"].dt.to_period("M").astype(str)
monthly = dated.groupby("Month")["Total Spent"].sum()
fig, ax = plt.subplots(figsize=(10, 4.8))
monthly.plot(ax=ax)
ax.set(title="Monthly Recorded Retail Spend", xlabel="Month", ylabel="Total Spent")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(out / "05_monthly_spend.png", dpi=180); plt.close(fig)

# 6. Payment counts
counts = df["Payment Method"].value_counts()
fig, ax = plt.subplots(figsize=(7.5, 4.8))
counts.plot(kind="bar", ax=ax)
ax.set(title="Transaction Count by Payment Method", xlabel="Payment Method", ylabel="Number of transactions")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(out / "06_payment_method_counts.png", dpi=180); plt.close(fig)

print("Six visualization files saved in:", out)
print("\nNarrative prompts:")
print("- Spend distribution: compare mean and median; note missing totals.")
print("- Category totals: these are recorded spend, not profit.")
print("- Quantity vs spend: relationship is partly mathematical by construction.")
print("- Discount comparison: unknown discount statuses are excluded.")
print("- Monthly trend: the last date is partial; avoid overinterpreting incomplete periods.")
print("- Payment counts: transaction counts do not equal unique customers.")
