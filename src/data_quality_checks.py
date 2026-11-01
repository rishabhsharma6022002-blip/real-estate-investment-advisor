import pandas as pd

df = pd.read_csv("data/raw/india_housing_prices.csv")

# 1. Check invalid floor numbers
invalid_floors = df[df["Floor_No"] > df["Total_Floors"]]

print("Invalid floor records:", len(invalid_floors))

# 2. Check property age
print("\nYear Built range:")
print(df["Year_Built"].min(), df["Year_Built"].max())

print("\nProperty Age range:")
print(df["Age_of_Property"].min(), df["Age_of_Property"].max())

# 3. Check price per square foot calculation
expected_price = (
    df["Price_in_Lakhs"] * 100000
) / df["Size_in_SqFt"]

difference = (
    df["Price_per_SqFt"] - expected_price
).abs()

print("\nPrice per SqFt difference:")
print(difference.describe())

# 4. Check unique categorical values
categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# 5. Check unique property IDs
print("\nDuplicate IDs:", df["ID"].duplicated().sum())
