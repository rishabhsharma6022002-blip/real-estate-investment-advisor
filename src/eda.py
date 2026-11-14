import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Create output folder
os.makedirs("reports/figures", exist_ok=True)

# Consistent visualization style
sns.set_theme(style="whitegrid", context="notebook")


# ==========================================
# 1. PROPERTY PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Price_in_Lakhs",
    bins=40,
    kde=True
)

plt.title("Distribution of Property Prices")
plt.xlabel("Property Price (₹ Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()

plt.savefig(
    "reports/figures/price_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 2. PROPERTY SIZE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Size_in_SqFt",
    bins=40,
    kde=True
)

plt.title("Distribution of Property Sizes")
plt.xlabel("Property Size (Sq. Ft.)")
plt.ylabel("Number of Properties")
plt.tight_layout()

plt.savefig(
    "reports/figures/size_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 3. PROPERTY SIZE VS PRICE
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(5000, random_state=42),
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    alpha=0.35,
    s=20,
    edgecolor=None
)

plt.title("Property Size vs Property Price")
plt.xlabel("Property Size (Sq. Ft.)")
plt.ylabel("Property Price (₹ Lakhs)")
plt.tight_layout()

plt.savefig(
    "reports/figures/size_vs_price.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 4. AVERAGE PRICE BY PROPERTY TYPE
# ==========================================

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=df,
    x="Property_Type",
    y="Price_in_Lakhs",
    errorbar=None
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.title("Average Property Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Average Price (₹ Lakhs)")
plt.tight_layout()

plt.savefig(
    "reports/figures/property_type.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ==========================================
# 5. FEATURE CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(
    include="number"
).drop(
    columns=["ID"],
    errors="ignore"
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    center=0,
    vmin=-1,
    vmax=1
)

plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print("\nEDA completed successfully!")
print("All graphs saved inside reports/figures/")
