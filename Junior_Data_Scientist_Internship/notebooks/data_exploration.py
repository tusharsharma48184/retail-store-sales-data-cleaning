from pathlib import Path
import pandas as pd

# Step 1: Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Step 2: Locate the CSV file
csv_path = project_folder / "data" / "retail_store_sales.csv"

# Step 3: Read the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Step 4: Display the first five rows
print(df.head())

print("\nAll column Names:")
print(df.columns.tolist())

print("\nDataFrame Info:")
df.info()

print("\nDataFrame Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique Values in Category:")
print(df["Category"].value_counts(dropna=False))

print("\nNumerical Summary:")
print(df[["Price Per Unit", "Quantity", "Total Spent"]].describe())


print("\n1. Duplicate Transaction IDs:")
print(df["Transaction ID"].duplicated().sum())

print("\n2. Transaction Date Examples:")
print(df["Transaction Date"].head(10))

print("\n3. Discount Applied Values:")
print(df["Discount Applied"].value_counts(dropna=False))

print("\n4. Total Spent Mismatch Check:")
check = df["Price Per Unit"] * df["Quantity"]

mismatch = (
    df["Total Spent"].notna()
    & check.notna()
    & (df["Total Spent"] != check)
)

print("Mismatched rows:", mismatch.sum())


# Step 1: Create a separate copy for cleaning
df_clean = df.copy()

# Step 2: Remove unwanted spaces from text columns
text_columns = [
    "Transaction ID",
    "Customer ID",
    "Category",
    "Item",
    "Payment Method",
    "Location"
]

for col in text_columns:
    df_clean[col] = df_clean[col].str.strip()

# Step 3: Convert Transaction Date to datetime
df_clean["Transaction Date"] = pd.to_datetime(
    df_clean["Transaction Date"],
    errors="coerce"
)

# Step 4: Check the results
print("\nOriginal rows:", len(df))
print("Cleaned copy rows:", len(df_clean))

print("\nTransaction Date data type:")
print(df_clean["Transaction Date"].dtype)

print("\nMissing dates after conversion:")
print(df_clean["Transaction Date"].isna().sum())

print("\nMissing values after initial cleaning:")
print(df_clean.isnull().sum())

# Count missing values in each row
missing_per_row = df_clean.isnull().sum(axis=1)

print("\nMissing values per row:")
print(missing_per_row.value_counts().sort_index())

# Check rows with missing Item
print("\nRows with missing Item:")
print(df_clean[df_clean["Item"].isnull()][
    ["Category", "Item", "Price Per Unit", "Quantity", "Total Spent"]
].head(10))

# Check whether Total Spent can be calculated
can_calculate_total = (
    df_clean["Total Spent"].isnull()
    & df_clean["Price Per Unit"].notnull()
    & df_clean["Quantity"].notnull()
)

print("\nMissing totals with price and quantity available:")
print(can_calculate_total.sum())

# Check which missing prices might be recoverable
price_recoverable = (
    df_clean["Price Per Unit"].isna()
    & df_clean["Total Spent"].notna()
    & df_clean["Quantity"].notna()
    & df_clean["Quantity"].ne(0)
)

# Check which missing quantities might be recoverable
quantity_recoverable = (
    df_clean["Quantity"].isna()
    & df_clean["Total Spent"].notna()
    & df_clean["Price Per Unit"].notna()
    & df_clean["Price Per Unit"].ne(0)
)

print("\nMissing prices with enough information to calculate:")
print(price_recoverable.sum())

print("\nMissing quantities with enough information to calculate:")
print(quantity_recoverable.sum())

# Check whether each category has a single known item
items_per_category = (
    df_clean.dropna(subset=["Item"])
    .groupby("Category")["Item"]
    .nunique()
)

print("\nUnique known items per category:")
print(items_per_category)
# Step: Recover missing prices using Total Spent / Quantity

price_recoverable = (
    df_clean["Price Per Unit"].isna()
    & df_clean["Total Spent"].notna()
    & df_clean["Quantity"].notna()
    & df_clean["Quantity"].ne(0)
)

# Calculate missing prices only where the required values exist
df_clean.loc[price_recoverable, "Price Per Unit"] = (
    df_clean.loc[price_recoverable, "Total Spent"]
    / df_clean.loc[price_recoverable, "Quantity"]
)

print("\nMissing prices filled:")
print(price_recoverable.sum())

print("\nRemaining missing values:")
print(df_clean.isnull().sum())
# Step 3: Analyse remaining missing values

print("\n--- Missing Quantity Analysis ---")
print("Missing Quantity rows:")
print(df_clean[df_clean["Quantity"].isna()][
    ["Item", "Price Per Unit", "Quantity", "Total Spent"]
].head(10))

print("\n--- Missing Quantity and Total Spent Together ---")
print(
    "Both Quantity and Total Spent missing:",
    (
        df_clean["Quantity"].isna()
        & df_clean["Total Spent"].isna()
    ).sum()
)

print("\n--- Quantity Distribution ---")
print(df_clean["Quantity"].value_counts(dropna=False).sort_index())

print("\n--- Discount Applied Distribution ---")
print(df_clean["Discount Applied"].value_counts(dropna=False))
# Step 5: Check Discount Applied data type and values

print("\n--- Discount Data Type ---")
print(df_clean["Discount Applied"].dtype)

print("\n--- Exact Unique Discount Values ---")
print(df_clean["Discount Applied"].unique())

print("\n--- Data Types of All Columns ---")
print(df_clean.dtypes)
# Step 6: Convert Discount Applied to Boolean type
# Missing values will remain missing

df_clean["Discount Applied"] = df_clean["Discount Applied"].astype("boolean")

print("\n--- Discount Data Type After Cleaning ---")
print(df_clean["Discount Applied"].dtype)

print("\n--- Discount Values After Cleaning ---")
print(df_clean["Discount Applied"].value_counts(dropna=False))
# Step 7: Final data validation

print("\n--- Final Data Validation ---")

# 1. Check total rows and columns
print("\nDataset shape:")
print(df_clean.shape)

# 2. Check duplicate rows
print("\nDuplicate rows:")
print(df_clean.duplicated().sum())

# 3. Check missing values
print("\nMissing values in each column:")
print(df_clean.isna().sum())

# 4. Verify Total Spent = Price Per Unit * Quantity
valid_rows = df_clean.dropna(
    subset=["Price Per Unit", "Quantity", "Total Spent"]
)

mismatch_count = (
    ~(
        (valid_rows["Price Per Unit"] * valid_rows["Quantity"])
        .round(2)
        .eq(valid_rows["Total Spent"].round(2))
    )
).sum()

print("\nTotal Spent calculation mismatches:")
print(mismatch_count)

# 5. Check data types
print("\nFinal data types:")
print(df_clean.dtypes)
# Step 8: Save the cleaned dataset

output_path = "output/retail_store_sales_cleaned.csv"

df_clean.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("File location:", output_path)
print("Final dataset shape:", df_clean.shape)