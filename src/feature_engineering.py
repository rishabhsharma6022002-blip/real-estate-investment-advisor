import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

print("Original shape:", df.shape)

# ---------------------------------
# TARGET 1: GOOD INVESTMENT
# ---------------------------------

# Calculate city-wise median price per sqft
city_median = df.groupby("City")["Price_per_SqFt"].transform("median")

# Create classification target
df["Good_Investment"] = (
    df["Price_per_SqFt"] <= city_median
).astype(int)

# ---------------------------------
# TARGET 2: FUTURE PROPERTY PRICE
# ---------------------------------

annual_growth_rate = 0.08
years = 5

df["Future_Price_5Y_Lakhs"] = (
    df["Price_in_Lakhs"] *
    (1 + annual_growth_rate) ** years
)

# ---------------------------------
# CHECK TARGET VARIABLES
# ---------------------------------

print("\nInvestment Classification:")
print(df["Good_Investment"].value_counts())

print("\nInvestment Class Percentages:")
print(df["Good_Investment"].value_counts(normalize=True) * 100)

print("\nFuture Price Statistics:")
print(df["Future_Price_5Y_Lakhs"].describe())

# ---------------------------------
# SAVE FEATURE-ENGINEERED DATA
# ---------------------------------

df.to_csv(
    "data/processed/feature_engineered_data.csv",
    index=False
)

print("\nFeature engineering completed!")
print("Saved: data/processed/feature_engineered_data.csv")
