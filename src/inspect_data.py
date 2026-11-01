import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/raw/india_housing_prices.csv")

# 2. Display first five rows
print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

# 3. Check dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

# 4. Check column information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 5. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# 7. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 8. Identify numerical columns
print("\n========== NUMERICAL COLUMNS ==========")
print(df.select_dtypes(include="number").columns.tolist())

# 9. Identify categorical columns
print("\n========== CATEGORICAL COLUMNS ==========")
print(
    df.select_dtypes(include=["object", "str"]).columns.tolist()
)
