import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from src.regression import predict_future_price

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "classification_model.pkl"


@st.cache_data
def create_demo_data(n_rows=420):
    """Generate illustrative sample records; no internship data is used."""
    rng = np.random.default_rng(42)
    city_state = {
        "Dehradun": "Uttarakhand",
        "Delhi": "Delhi",
        "Jaipur": "Rajasthan",
        "Lucknow": "Uttar Pradesh",
        "Pune": "Maharashtra",
        "Indore": "Madhya Pradesh",
        "Bengaluru": "Karnataka",
        "Hyderabad": "Telangana",
    }
    localities = {
        "Dehradun": ["Rajpur Road", "Sahastradhara Road", "Prem Nagar"],
        "Delhi": ["Rohini", "Dwarka", "Saket"],
        "Jaipur": ["Vaishali Nagar", "Jagatpura", "Mansarovar"],
        "Lucknow": ["Gomti Nagar", "Indira Nagar", "Aliganj"],
        "Pune": ["Wakad", "Hinjawadi", "Kothrud"],
        "Indore": ["Vijay Nagar", "Rau", "Palasia"],
        "Bengaluru": ["Whitefield", "Electronic City", "Yelahanka"],
        "Hyderabad": ["Kondapur", "Miyapur", "Gachibowli"],
    }
    city_base_price = {
        "Dehradun": 5200, "Delhi": 11500, "Jaipur": 4900, "Lucknow": 4300,
        "Pune": 8500, "Indore": 4700, "Bengaluru": 9200, "Hyderabad": 7800,
    }

    cities = np.array(list(city_state))
    selected_cities = rng.choice(cities, size=n_rows)
    sizes = rng.integers(550, 3201, size=n_rows)
    price_per_sqft = np.array(
        [city_base_price[c] for c in selected_cities], dtype=float
    ) * rng.lognormal(0, 0.22, size=n_rows)
    prices_lakhs = price_per_sqft * sizes / 100000
    total_floors = rng.integers(2, 18, n_rows)

    floor_numbers = np.array([
      rng.integers(0, floors) for floors in total_floors
    ])

    result = pd.DataFrame({
        "State": [city_state[c] for c in selected_cities],
        "Floor_No": floor_numbers,
        "Total_Floors": total_floors,
        "Age_of_Property": rng.integers(1, 31, n_rows),
        "City": selected_cities,
        "Locality": [rng.choice(localities[c]) for c in selected_cities],
        "Property_Type": rng.choice(
            ["Apartment", "Independent House", "Villa"], n_rows, p=[0.68, 0.22, 0.10]
        ),
        "BHK": rng.choice([1, 2, 3, 4], n_rows, p=[0.12, 0.42, 0.34, 0.12]),
        "Size_in_SqFt": sizes,
        "Price_in_Lakhs": np.round(prices_lakhs, 2),
        "Year_Built": rng.integers(1995, 2025, n_rows),
        "Furnished_Status": rng.choice(
            ["Unfurnished", "Semi-Furnished", "Furnished"], n_rows,
            p=[0.35, 0.45, 0.20]
        ),
        "Nearby_Schools": rng.integers(0, 11, n_rows),
        "Nearby_Hospitals": rng.integers(0, 8, n_rows),
        "Public_Transport_Accessibility": rng.choice(["Low", "Medium", "High"], n_rows),
        "Parking_Space": rng.choice(["Yes", "No"], n_rows, p=[0.75, 0.25]),
        "Security": rng.choice(["Yes", "No"], n_rows, p=[0.8, 0.2]),
        "Amenities": rng.choice(
            ["Gym", "Gym, Pool", "Garden", "Community Hall", "None"], n_rows
        ),
        "Facing": rng.choice(["North", "South", "East", "West"], n_rows),
        "Owner_Type": rng.choice(["Owner", "Builder", "Dealer"], n_rows),
        "Availability_Status": rng.choice(
            ["Ready_to_Move", "Under_Construction"], n_rows
        ),
        "Floor_Data_Invalid": np.zeros(n_rows, dtype=int),
    })
    result["Price_per_SqFt"] = (
        result["Price_in_Lakhs"] * 100000 / result["Size_in_SqFt"]
    )
    medians = result.groupby("City")["Price_per_SqFt"].transform("median")
    result["Good_Investment"] = (result["Price_per_SqFt"] <= medians).astype(int)
    return result


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            st.warning(f"Local model could not be loaded: {exc}")
    return None


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH), "local"
        except Exception as exc:
            st.warning(f"Local dataset could not be loaded: {exc}")
    return create_demo_data(), "demo"


df, data_source = load_data()
is_demo_data = data_source == "demo"

model = None if is_demo_data else load_model()
has_trained_model = model is not None

if is_demo_data:
    st.sidebar.info(
        "Demo mode: generated illustrative sample data is being used. "
        "The internship dataset is not included."
    )
if not has_trained_model:
    st.sidebar.warning(
        "No trained model file is available. The investment page will use a "
        "transparent city-median baseline, not the trained Random Forest."
    )

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Property Investment", "Price Forecast", "Analytics Dashboard"],
)


if page == "Home":
    st.title("🏠 Real Estate Investment Advisor")
    st.subheader("Explore Property Affordability & Future Value")
    st.write(
        "An interactive data science project for exploring residential "
        "property characteristics, relative affordability, and future-price scenarios."
    )
    if is_demo_data:
        st.warning(
            "**Public demo data:** The figures are generated sample records "
            "for demonstrating the application. They are not actual market "
            "statistics or records from the internship dataset."
        )
    col1, col2, col3 = st.columns(3)
    col1.metric("Properties in current dataset", f"{len(df):,}")
    col2.metric("Cities", f"{df['City'].nunique():,}")
    col3.metric("States", f"{df['State'].nunique():,}")
    st.info(
        "Use the sidebar to assess relative affordability, calculate a "
        "future-price scenario, or explore the sample market data."
    )


elif page == "Property Investment":
    st.title("Property Investment Analysis")
    st.caption(
        "This classification represents relative affordability within a city, "
        "not independently verified investment profitability."
    )

    with st.form("property_form"):
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("City", sorted(df["City"].dropna().unique()))
            city_df = df[df["City"] == city]
            property_type = st.selectbox(
                "Property Type", sorted(df["Property_Type"].dropna().unique())
            )
            bhk = st.selectbox("BHK", sorted(df["BHK"].dropna().unique().tolist()))
            size = st.number_input(
                "Property Size (SqFt)", min_value=500, max_value=5000, value=1500
            )
            price = st.number_input(
                "Property Price (Lakhs)", min_value=10.0, max_value=500.0, value=60.0
            )
        with col2:
            state = city_df["State"].mode().iloc[0]
            locality = st.selectbox(
                "Locality", sorted(city_df["Locality"].dropna().unique())
            )
            furnished = st.selectbox(
                "Furnished Status", sorted(df["Furnished_Status"].dropna().unique())
            )
            nearby_schools = st.number_input(
                "Nearby Schools", min_value=0, max_value=20, value=4
            )
            nearby_hospitals = st.number_input(
                "Nearby Hospitals", min_value=0, max_value=20, value=2
            )
        submitted = st.form_submit_button("Analyze Investment")

    if submitted:
        price_per_sqft = price * 100000 / size
        city_median = city_df["Price_per_SqFt"].median()
        age = 10
        property_data = pd.DataFrame([{
            "State": state,
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Year_Built": 2025 - age,
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
            "Floor_Data_Invalid": 0,
        }])

        st.subheader("Affordability Assessment")
        if has_trained_model:
            prediction = int(model.predict(property_data)[0])
            st.success("Trained Random Forest model used.")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(property_data)[0]
                st.metric(
                    "Model class probability",
                    f"{float(probabilities[prediction]) * 100:.2f}%"
                )
        else:
            # Public fallback: transparent rule, not machine-learning inference.
            prediction = int(price_per_sqft <= city_median)
            st.info(
                "Demo baseline: compares this property's price per sq ft with "
                "the selected city's median. No trained ML model was loaded."
            )

        if prediction == 1:
            st.success("Relatively affordable compared with the city median.")
        else:
            st.warning("Above the selected city's median price per sq ft.")

        metric1, metric2 = st.columns(2)
        metric1.metric("Property Price per SqFt", f"₹{price_per_sqft:,.2f}")
        metric2.metric("City Median Price per SqFt", f"₹{city_median:,.2f}")
        st.caption(
            "This is not a prediction of rental yield, capital gains, investment "
            "risk, or guaranteed profitability."
        )


elif page == "Price Forecast":
    st.title("Property Price Scenario Calculator")
    st.write(
        "Explore how a property's value could change under a selected annual "
        "appreciation assumption and forecast duration."
    )
    col1, col2 = st.columns(2)
    with col1:
        current_price = st.number_input(
            "Current Property Price (Lakhs)", min_value=1.0,
            max_value=10000.0, value=60.0, step=5.0
        )
    with col2:
        duration = st.selectbox(
            "Forecast Duration (Years)", list(range(1, 11)), index=4
        )
    growth_rate = st.slider(
        "Annual Appreciation Assumption (%)", min_value=1, max_value=15, value=8
    )
    if st.button("Calculate Future Price"):
        future_price = predict_future_price(
            current_price, growth_rate / 100, duration
        )
        total_growth = ((future_price / current_price) - 1) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"₹{current_price:.2f} Lakh")
        col2.metric("Illustrative Future Price", f"₹{future_price:.2f} Lakh")
        col3.metric("Total Assumed Appreciation", f"{total_growth:.2f}%")
        st.info(
            "This is a compound-growth scenario based on your chosen assumption. "
            "It is not an ML regression prediction or a guarantee of market prices."
        )


elif page == "Analytics Dashboard":
    st.title("Real Estate Market Analytics")
    if is_demo_data:
        st.warning(
            "These charts use generated illustrative demo data, not the private "
            "internship dataset."
        )

    st.subheader("Average Property Price by City")
    city_prices = (
        df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    )
    st.bar_chart(city_prices.head(10))

    st.subheader("Property Type Distribution")
    st.bar_chart(df["Property_Type"].value_counts())

    st.subheader("Property Price Distribution")
    price_bins = pd.cut(df["Price_in_Lakhs"], bins=10, duplicates="drop")
    price_distribution = price_bins.value_counts(sort=False)
    price_distribution.index = [
        f"Range {i + 1:02d}" for i in range(len(price_distribution))
    ]
    st.bar_chart(price_distribution)
    st.caption("Distribution of properties across price ranges.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)
