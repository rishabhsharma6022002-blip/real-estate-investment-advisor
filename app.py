import streamlit as st
import pandas as pd
import joblib
import os

from src.regression import predict_future_price

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/classification_model.pkl")

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/feature_engineered_data.csv")

df = load_data()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Property Investment",
        "Price Forecast",
        "Analytics Dashboard"
    ]
)

# Home page
if page == "Home":

    st.title("🏠 Real Estate Investment Advisor")

    st.subheader(
        "Predict Property Profitability & Future Value"
    )

    st.write(
        "An ML-powered application for analyzing "
        "real estate investment opportunities."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Properties", f"{len(df):,}")
    col2.metric("Cities", df["City"].nunique())
    col3.metric("States", df["State"].nunique())

    st.info(
        "Use the sidebar to analyze a property, "
        "estimate its future price, or explore market data."
    )

# Investment prediction page
elif page == "Property Investment":

    st.title("Property Investment Analysis")

    with st.form("property_form"):

        col1, col2 = st.columns(2)

        with col1:
            city = st.selectbox(
                "City",
                sorted(df["City"].unique())
            )

            property_type = st.selectbox(
                "Property Type",
                sorted(df["Property_Type"].unique())
            )

            bhk = st.selectbox(
                "BHK",
                sorted(df["BHK"].unique())
            )

            size = st.number_input(
                "Property Size (SqFt)",
                min_value=500,
                max_value=5000,
                value=1500
            )

            price = st.number_input(
                "Property Price (Lakhs)",
                min_value=10.0,
                max_value=500.0,
                value=60.0
            )

        with col2:
            state = df[df["City"] == city]["State"].mode()[0]

            locality = st.selectbox(
                "Locality",
                sorted(
                    df[df["City"] == city]["Locality"].unique()
                )
            )

            furnished = st.selectbox(
                "Furnished Status",
                sorted(df["Furnished_Status"].unique())
            )

            nearby_schools = st.number_input(
                "Nearby Schools",
                min_value=0,
                max_value=20,
                value=4
            )

            nearby_hospitals = st.number_input(
                "Nearby Hospitals",
                min_value=0,
                max_value=20,
                value=2
            )

        submitted = st.form_submit_button(
            "Analyze Investment"
        )

    if submitted:

        price_per_sqft = price * 100000 / size

        city_median = df[
            df["City"] == city
        ]["Price_per_SqFt"].median()

        # Estimate a plausible property age
        age = 10
        year_built = 2025 - age

        property_data = pd.DataFrame([{
            "State": state,
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Year_Built": year_built,
            "Furnished_Status": furnished,
            "Floor_No": 1,
            "Total_Floors": 5,
            "Age_of_Property": age,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Public_Transport_Accessibility": "Medium",
            "Parking_Space": "Yes",
            "Security": "Yes",
            "Amenities": "Gym, Pool",
            "Facing": "North",
            "Owner_Type": "Owner",
            "Availability_Status": "Ready_to_Move",
            "Floor_Data_Invalid": 0
        }])

        prediction = model.predict(property_data)[0]

        probability = model.predict_proba(
            property_data
        )[0][prediction]

        st.subheader("Investment Analysis")

        if prediction == 1:
            st.success("Good Investment (Relative Affordability)")
        else:
            st.warning("Not Classified as Good Investment")

        st.metric(
            "Model Confidence",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "This prediction uses synthetic-data-derived labels "
            "and should not be interpreted as a verified return forecast."
        )

        st.metric(
            "Price per SqFt",
            f"₹{price_per_sqft:,.2f}"
        )

        st.metric(
            "City Median Price per SqFt",
            f"₹{city_median:,.2f}"
        )

# Price forecast page
elif page == "Price Forecast":

    st.title("Five-Year Property Price Forecast")

    st.write(
        "Estimate future property value using "
        "different appreciation rates and durations."
    )

    col1, col2 = st.columns(2)

    with col1:
        current_price = st.number_input(
            "Current Property Price (Lakhs)",
            min_value=1.0,
            max_value=10000.0,
            value=60.0,
            step=5.0
        )

    with col2:
        duration = st.selectbox(
            "Forecast Duration (Years)",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=4
        )

    growth_rate = st.slider(
        "Annual Appreciation Assumption (%)",
        min_value=1,
        max_value=15,
        value=8
    )

    if st.button("Calculate Future Price"):

        future_price = predict_future_price(
            current_price,
            growth_rate / 100,
            duration
        )

        total_growth = (
            (future_price / current_price) - 1
        ) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Current Price",
            f"₹{current_price:.2f} Lakh"
        )

        col2.metric(
            "Estimated Future Price",
            f"₹{future_price:.2f} Lakh"
        )

        col3.metric(
            "Total Appreciation",
            f"{total_growth:.2f}%"
        )

        st.success(
            f"Estimated property value after {duration} years: "
            f"₹{future_price:.2f} lakh"
        )

        st.info(
            "This is an assumption-based financial scenario, "
            "not a machine-learning regression prediction."
        )


# Analytics page
elif page == "Analytics Dashboard":

    st.title("Real Estate Market Analytics")

    st.subheader("Average Property Price by City")

    city_prices = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(city_prices.head(10))

    st.subheader("Property Type Distribution")

    st.bar_chart(
        df["Property_Type"].value_counts()
    )

    st.subheader("Property Price Distribution")

price_bins = pd.cut(
    df["Price_in_Lakhs"],
    bins=10
)

price_distribution = (
    price_bins.value_counts(sort=False)
)

price_distribution.index = [
    f"Range {i+1:02d}"
    for i in range(len(price_distribution))
]

st.bar_chart(price_distribution)

st.caption(
    "Distribution of properties across 10 price ranges."
)

    

st.subheader("Dataset Preview")

st.dataframe(df.head(100))