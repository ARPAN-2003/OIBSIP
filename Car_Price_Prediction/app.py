import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/car_data.csv")

model = joblib.load(
    "models/car_price_model.pkl"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#1E88E5;'>
        🚗 Car Price Prediction Using Machine Learning
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center;font-size:18px'>
    Predict the selling price of a used car using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Enter Car Details")

car_name = st.sidebar.selectbox(
    "Car Name",
    sorted(df["Car_Name"].unique())
)

year = st.sidebar.slider(
    "Manufacturing Year",
    int(df["Year"].min()),
    int(df["Year"].max()),
    2015
)

present_price = st.sidebar.number_input(
    "Present Price (Lakhs)",
    min_value=0.1,
    value=5.0
)

driven_kms = st.sidebar.number_input(
    "Driven Kilometers",
    min_value=0,
    value=30000
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["Fuel_Type"].unique())
)

selling_type = st.sidebar.selectbox(
    "Seller Type",
    sorted(df["Selling_type"].unique())
)

transmission = st.sidebar.selectbox(
    "Transmission",
    sorted(df["Transmission"].unique())
)

owner = st.sidebar.selectbox(
    "Owner",
    sorted(df["Owner"].unique())
)

# -----------------------------
# INPUT SUMMARY & DATASET INFO
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📋 Selected Inputs")

    input_df = pd.DataFrame({
        "Feature":[
            "Car Name",
            "Year",
            "Present Price",
            "Driven Kms",
            "Fuel Type",
            "Seller Type",
            "Transmission",
            "Owner"
        ],
        "Value":[
            car_name,
            year,
            present_price,
            driven_kms,
            fuel_type,
            selling_type,
            transmission,
            owner
        ]
    })

    st.dataframe(
        input_df,
        use_container_width=True
    )

with col2:

    st.subheader("📊 Dataset Information")

    st.metric(
        "Total Records",
        len(df)
    )

    st.metric(
        "Average Selling Price",
        f"₹ {df['Selling_Price'].mean():.2f} Lakh"
    )

    st.metric(
        "Maximum Selling Price",
        f"₹ {df['Selling_Price'].max():.2f} Lakh"
    )

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.divider()

st.subheader("🎯 Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("MAE", "0.599")

with m2:
    st.metric("RMSE", "0.914")

with m3:
    st.metric("R² Score", "0.964")

# -----------------------------
# PREDICTION
# -----------------------------
st.divider()

prediction = None

if st.button(
    "🚀 Predict Car Price",
    use_container_width=True
):

    sample = pd.DataFrame([{
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    prediction = model.predict(sample)[0]

    st.success(
        f"Estimated Selling Price : ₹ {prediction:.2f} Lakh"
    )

    p1, p2 = st.columns(2)

    with p1:
        st.metric(
            "Predicted Price",
            f"₹ {prediction:.2f} Lakh"
        )

    with p2:

        if prediction < 3:
            category = "Budget Car"
        elif prediction < 10:
            category = "Mid-Range Car"
        else:
            category = "Premium Car"

        st.metric(
            "Price Category",
            category
        )

    report = f"""
Car Price Prediction Report

Car Name : {car_name}
Year : {year}
Present Price : {present_price}
Driven Kms : {driven_kms}
Fuel Type : {fuel_type}
Seller Type : {selling_type}
Transmission : {transmission}
Owner : {owner}

Predicted Price : ₹ {prediction:.2f} Lakh
"""

    st.download_button(
        "📥 Download Prediction Report",
        report,
        file_name="prediction_report.txt"
    )

# -----------------------------
# SIMILAR CARS
# -----------------------------
if prediction is not None:

    st.divider()

    st.subheader("🚘 Top 10 Similar Cars")

    similar_cars = df.copy()

    similar_cars["distance"] = (
        abs(similar_cars["Year"] - year)
        +
        abs(similar_cars["Present_Price"] - present_price)
        +
        abs(similar_cars["Driven_kms"] - driven_kms)/10000
    )

    similar_cars = (
        similar_cars
        .sort_values("distance")
        .head(10)
    )

    st.dataframe(
        similar_cars.drop(
            columns=["distance"]
        ),
        use_container_width=True
    )

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.divider()

st.subheader("🔍 Search Dataset")

search_term = st.text_input(
    "Search Car Name"
)

filtered_df = df

if search_term:

    filtered_df = df[
        df["Car_Name"]
        .str.contains(
            search_term,
            case=False,
            na=False
        )
    ]

st.dataframe(
    filtered_df,
    use_container_width=True
)

# -----------------------------
# DATASET STATISTICS
# -----------------------------
st.divider()

st.subheader("📈 Dataset Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# -----------------------------
# PREDICTION VS ACTUAL
# -----------------------------
try:

    st.divider()

    st.subheader(
        "📉 Prediction vs Actual"
    )

    st.image(
        "outputs/prediction_vs_actual.png",
        use_container_width=True
    )

except:
    pass

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
try:

    st.divider()

    st.subheader("⭐ Feature Importance")

    st.image(
        "outputs/feature_importance.png",
        use_container_width=True
    )

except:
    pass

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
try:

    st.divider()

    st.subheader("🔥 Correlation Heatmap")

    st.image(
        "outputs/correlation_heatmap.png",
        use_container_width=True
    )

except:
    pass

# -----------------------------
# QUICK INSIGHTS
# -----------------------------
st.divider()

st.subheader("💡 Quick Insights")

st.info(
    f"""
    • Average Car Price : ₹ {df['Selling_Price'].mean():.2f} Lakh

    • Most Expensive Car : ₹ {df['Selling_Price'].max():.2f} Lakh

    • Oldest Model Year : {df['Year'].min()}

    • Latest Model Year : {df['Year'].max()}

    • Total Cars Analysed : {len(df)}
    """
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    Developed for Oasis Infobyte Data Science Internship Project
    </center>
    """,
    unsafe_allow_html=True
)
