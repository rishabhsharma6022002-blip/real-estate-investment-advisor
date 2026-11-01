import pandas as pd
import numpy as np

# Step 1: Load original dataset
df = pd.read_csv("data/raw/india_housing_prices.csv")

print("Original dataset shape:", df.shape)

# Step 2: Remove duplicate records
df = df.drop_duplicates()

# Step 3: Remove duplicate IDs if any
df = df.drop_duplicates(subset=["ID"])

# Step 4: Clean categorical text
categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns

for col in categorical_columns:
    df[col] = df[col].str.strip()

# Step 5: Identify invalid floor records
invalid_floor = df["Floor_No"] > df["Total_Floors"]

df["Floor_Data_Invalid"] = invalid_floor.astype(int)

print("Invalid floor records:", invalid_floor.sum())

# Replace inconsistent floor values with missing values
df.loc[invalid_floor, ["Floor_No", "Total_Floors"]] = np.nan

# Step 6: Fill missing numerical values with median
numerical_columns = df.select_dtypes(
    include="number"
).columns

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# Step 7: Preserve original price per sqft column
df.rename(
    columns={"Price_per_SqFt": "Price_per_SqFt_Lakhs"},
    inplace=True
)

# Step 8: Calculate price per sqft in rupees
df["Price_per_SqFt"] = (
    df["Price_in_Lakhs"] * 100000
) / df["Size_in_SqFt"]

# Step 9: Save cleaned dataset
output_path = "data/processed/cleaned_data.csv"

df.to_csv(output_path, index=False)

# Step 10: Final verification
print("\n========== PREPROCESSING RESULTS ==========")

print("Final dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nRemaining invalid floor records:")
print((df["Floor_No"] > df["Total_Floors"]).sum())

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)
