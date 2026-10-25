from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "retail_store_sales_raw.csv")

# 1. Basic cleaning and validation
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
df["Discount Applied"] = df["Discount Applied"].astype("boolean")
for col in ["Transaction ID", "Customer ID", "Category", "Item", "Payment Method", "Location"]:
    df[col] = df[col].astype("string").str.strip()

quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
})
quality.to_csv(OUT / "data_quality_summary.csv")
print("Dataset shape:", df.shape)
print("Exact duplicate rows:", df.duplicated().sum())
print(quality)

# 2. EDA outputs
df[["Price Per Unit", "Quantity", "Total Spent"]].describe().to_csv(OUT / "numeric_summary.csv")
category_spend = df.groupby("Category")["Total Spent"].sum(min_count=1).sort_values(ascending=False)
category_spend.to_csv(OUT / "category_recorded_spend.csv")
fig, ax = plt.subplots(figsize=(9, 5))
category_spend.sort_values().plot(kind="barh", ax=ax)
ax.set(title="Recorded Spend by Category", xlabel="Total Spent", ylabel="Category")
fig.tight_layout(); fig.savefig(OUT / "category_spend.png", dpi=160); plt.close(fig)

dated = df.dropna(subset=["Transaction Date", "Total Spent"]).copy()
dated["Month"] = dated["Transaction Date"].dt.to_period("M").astype(str)
monthly = dated.groupby("Month")["Total Spent"].sum()
monthly.to_csv(OUT / "monthly_recorded_spend.csv")
fig, ax = plt.subplots(figsize=(10, 4.5))
monthly.plot(ax=ax)
ax.set(title="Monthly Recorded Spend", xlabel="Month", ylabel="Total Spent")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(OUT / "monthly_spend.png", dpi=160); plt.close(fig)

# 3. Statistical model on complete numeric rows
numeric = df[["Price Per Unit", "Quantity", "Total Spent"]].dropna()
X_stat = sm.add_constant(numeric[["Price Per Unit", "Quantity"]])
stat_model = sm.OLS(numeric["Total Spent"], X_stat).fit()
with open(OUT / "statistical_model_summary.txt", "w", encoding="utf-8") as f:
    f.write(stat_model.summary().as_text())
print(stat_model.summary())

# 4. Machine learning model with train/test split and CV tuning
target = "Total Spent"
features = ["Price Per Unit", "Quantity", "Category", "Payment Method", "Location", "Discount Applied"]
model_df = df[features + [target]].dropna(subset=[target]).copy()
X, y = model_df[features], model_df[target]
num_features = ["Price Per Unit", "Quantity"]
cat_features = ["Category", "Payment Method", "Location", "Discount Applied"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
baseline = Pipeline([("preprocess", preprocess), ("model", LinearRegression())])
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)
rf = Pipeline([("preprocess", preprocess), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
grid = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10], "model__min_samples_leaf": [1, 3]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(rf, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
search.fit(X_train, y_train)
rf_pred = search.best_estimator_.predict(X_test)

def metrics(name, actual, predicted):
    return {"Model": name, "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)), "R2": r2_score(actual, predicted)}
results = pd.DataFrame([metrics("Linear Regression", y_test, base_pred),
                        metrics("Tuned Random Forest", y_test, rf_pred)])
results.to_csv(OUT / "model_evaluation.csv", index=False)
with open(OUT / "best_model_parameters.txt", "w", encoding="utf-8") as f:
    f.write(str(search.best_params_))
print("Best parameters:", search.best_params_)
print(results)

print("Integrated project outputs saved in:", OUT)
