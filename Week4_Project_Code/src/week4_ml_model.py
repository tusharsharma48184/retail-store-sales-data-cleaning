from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "retail_store_sales_raw.csv")
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

target = "Total Spent"
features = ["Price Per Unit", "Quantity", "Category", "Payment Method", "Location", "Discount Applied"]
model_df = df[features + [target]].dropna(subset=[target]).copy()
X = model_df[features]
y = model_df[target]

numeric_features = ["Price Per Unit", "Quantity"]
categorical_features = ["Category", "Payment Method", "Location", "Discount Applied"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = Pipeline([("preprocess", preprocess), ("model", LinearRegression())])
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

rf = Pipeline([("preprocess", preprocess),
               ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10],
    "model__min_samples_leaf": [1, 3]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(rf, param_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)

def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"Model": name, "MAE": mean_absolute_error(y_true, y_pred), "RMSE": rmse, "R2": r2_score(y_true, y_pred)}

results = pd.DataFrame([
    report("Linear Regression baseline", y_test, base_pred),
    report("Tuned Random Forest", y_test, pred)
])
print("Best parameters:", search.best_params_)
print(results.to_string(index=False))
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
results.to_csv(out / "model_comparison.csv", index=False)
with open(out / "best_parameters.txt", "w", encoding="utf-8") as f:
    f.write(str(search.best_params_))
