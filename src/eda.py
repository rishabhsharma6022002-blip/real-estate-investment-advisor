import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Create output folder
import os
os.makedirs("reports/figures", exist_ok=True)

# 1. Property price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Price_in_Lakhs"], bins=40, kde=True)
plt.title("Distribution of Property Prices")
plt.xlabel("Price (Lakhs)")
plt.savefig("reports/figures/price_distribution.png")
plt.close()

# 2. Property size distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Size_in_SqFt"], bins=40, kde=True)
plt.title("Distribution of Property Sizes")
plt.xlabel("Size (SqFt)")
plt.savefig("reports/figures/size_distribution.png")
plt.close()

# 3. Property size vs price
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df.sample(5000, random_state=42),
    x="Size_in_SqFt",
    y="Price_in_Lakhs"
)
plt.title("Property Size vs Price")
plt.savefig("reports/figures/size_vs_price.png")
plt.close()

# 4. Average property price by property type
plt.figure(figsize=(9, 5))
sns.barplot(
    data=df,
    x="Property_Type",
    y="Price_in_Lakhs",
    errorbar=None
)
plt.title("Average Price by Property Type")
plt.savefig("reports/figures/property_type.png")
plt.close()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()

print("EDA completed successfully!")
print("All graphs saved inside reports/figures/")
